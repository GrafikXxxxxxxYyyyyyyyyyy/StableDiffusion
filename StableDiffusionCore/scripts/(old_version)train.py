##################################################################################################################################



                    #########################################################################################
                    #       _____ ____      _    ___ _   _           ____   ____ ____  ___ ____ _____       #
                    # __/\_|_   _|  _ \    / \  |_ _| \ | |         / ___| / ___|  _ \|_ _|  _ \_   _|_/\__ #
                    # \    / | | | |_) |  / _ \  | ||  \| |         \___ \| |   | |_) || || |_) || | \    / #
                    # /_  _\ | | |  _ <  / ___ \ | || |\  |          ___) | |___|  _ < | ||  __/ | | /_  _\ #
                    #   \/   |_| |_| \_\/_/   \_\___|_| \_|         |____/ \____|_| \_\___|_|    |_|   \/   #
                    #########################################################################################



##################################################################################################################################
import torch
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.notebook import tqdm
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, AutoencoderTiny

# from pipeline.text2image_sd import StableDiffusionText2Image



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_lora_weights",
        action='store_true',
        help="Do we need to train LoRA instead of full models",
    )
    parser.add_argument(
        "--train_text_encoder",
        action='store_true',
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Last_trained_LoRA",
        help="Directory to store trained LoRA weights.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    

    args = parser.parse_args()

    return args


def show_images(images, rows=2, cols=2, isPIL=False):
    if isPIL:
        batch_size = len(images)
        grid_size = int(batch_size ** 0.5)

        if len(images) == 1:
            plt.imshow(images[0])
            plt.axis('off')
            plt.show()
            return 

        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*8, grid_size*8))
        fig.subplots_adjust(hspace=0, wspace=0)
        
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                axs[i, j].imshow(images[index])
                axs[i, j].axis('off')
                
        plt.show()
        return 
    
    batch_size, channels, height, width = images.shape
    grid_size = int(batch_size ** 0.5)

    if len(images) == 1:
        img = (images[0] / 2 + 0.5).clamp(0, 1)
        img = (img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(img)

        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return
    
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*8, grid_size*8))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            
            img = (images[index] / 2 + 0.5).clamp(0, 1)
            img = (img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            img = Image.fromarray(img)
            
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = load_dataset("MysticPixel42/erotic_teens_unconditional")
    processed_dataset = []
    for example in tqdm(dataset['train']):
        if len(processed_dataset) >= 10:
            break
            
        image = example['image']
        image = transform(image)
        caption = example['caption']
        processed_dataset.append({'image': image, 'caption': caption})

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch], dim=0)
        captions = [item['caption'] for item in batch]
        return {'pixel_values': images, 'captions': captions}

    train_loader = DataLoader(processed_dataset, args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(train_loader))
    show_images(batch["pixel_values"])
    print(batch['captions'], '\n', batch['pixel_values'].shape, '\n', len(batch['captions']))
    print(f"isLoRA : {args.use_lora_weights}\nisCLIP : {args.train_text_encoder}\n\n")

    sd = StableDiffusionText2Image(
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        ),
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
        ),
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae", 
            use_safetensors=True,
        ),
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet", 
            use_safetensors=True,
        ),
        scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler", 
        ),
        need_logger = False
    )
    sd.to(torch.device("cuda")) 
    print(sd.text_encoder.device, sd.vae.device, sd.unet.device)

    sd.finetune(train_loader, args)




from diffusers import StableDiffusionPipeline

StableDiffusionPipeline.from_pretrained()
StableDiffusionPipeline().load_lora_weights()