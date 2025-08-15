import os
import math
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from torchvision import transforms
from accelerate.utils import set_seed
from torch.utils.data import Dataset
from typing import Any, Optional, Union, Dict
from dataclasses import dataclass
from ..models.stable_diffusion import SDModelWrapper
from peft import LoraConfig
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from diffusers.training_utils import cast_training_params
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.optimization import get_scheduler
from torchvision.transforms.functional import crop



def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    # Убедимся, что тензор находится на CPU и значения в [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.detach().cpu().clamp(0, 1)
    # Преобразуем в [H, W, C] и в numpy
    image = tensor.permute(1, 2, 0).numpy()
    # Переводим в uint8
    image = (image * 255).astype("uint8")
    return Image.fromarray(image)



@dataclass
class SDTrainingArgs:
    seed: Optional[int] = None
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    mixed_precision: str = 'fp16'
    output_dir: str = 'sd-model-finetuned-lora'
    rank: int = 16
    train_text_encoder: bool = True
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    adam_weight_decay: float = 1e-2
    dataloader_num_workers: int = 0
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 100
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500
    resolution: int = 1024
    max_grad_norm: float = 1.0
    use_8bit_adam: bool = False




class SDLoRADataset(Dataset):
    def __init__(self, data_path: str = "data", target_size=(1024, 1024)):
        """
        Args:
            data_path (str): Путь к директории с данными (по умолчанию "data")
            transform (callable, optional): Трансформации для изображений
        """
        self.data_path = data_path
        self.random_crop = transforms.RandomCrop((3024,3024))
        self.transform = transforms.Compose([
            # transforms.RandomCrop((3024,3024)),
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.target_size = target_size
        
        # Собираем список всех файлов .jpg в директории
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        
        # Проверяем, что для каждого изображения есть соответствующий .txt файл
        self.valid_pairs = []
        for img_file in self.image_files:
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(os.path.join(data_path, txt_file)):
                self.valid_pairs.append((img_file, txt_file))
            else:
                print(f"Warning: No annotation file found for {img_file}")

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_file, txt_file = self.valid_pairs[idx]
        
        # Загружаем изображение
        img_path = os.path.join(self.data_path, img_file)
        image = Image.open(img_path).convert('RGB')
        original_sizes = image.size
        
        # Применяем трансформации, если они есть
        y1, x1, h, w = self.random_crop.get_params(
            image, (min(original_sizes[0], original_sizes[1]), min(original_sizes[0], original_sizes[1]))
        )
        image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        if self.transform is not None:
            image = self.transform(image)
        
        # Загружаем текст аннотации
        txt_path = os.path.join(self.data_path, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        return {
            'pixel_values': image,
            'original_sizes': original_sizes,
            'crops_coords_top_left': crop_top_left,
            'target_sizes': self.target_size,
            'caption': caption
        }



class SDLoRATrainer ():
    def __init__(
        self,
        model: SDModelWrapper,
        args: SDTrainingArgs,
        train_dataset: SDLoRADataset, 
    ):
        self.model = model
        self.args = args
        self.dataset = train_dataset


    def encode_prompt(self, prompt, prompt_2=None):
        tokenizers = [self.model.tokenizer]
        text_encoders = [self.model.text_encoder]
        prompt = [prompt] if isinstance(prompt, str) else prompt 
        prompts = [prompt] 

        # Define tokenizers and text encoders so
        # it can be used with sd15 and sdxl self.models
        if hasattr(self.model, "text_encoder_2") and hasattr(self.model, "tokenizer_2"):
            tokenizers = [self.model.tokenizer, self.model.tokenizer_2]
            text_encoders = [self.model.text_encoder, self.model.text_encoder_2]
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            # textual inversion: process multi-vector tokens if necessary
            prompts = [prompt, prompt_2]

        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
            )

            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = (
                prompt_embeds[-1][-2] 
                if hasattr(self.model, "text_encoder_2") else 
                prompt_embeds[0]
            )
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return prompt_embeds, pooled_prompt_embeds
    

    def compute_time_ids(self, original_size, crops_coords_top_left, target_size):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        return add_time_ids




    def train(self):
        # 1. Инициализируем акселератор
        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        print("Accelerator OK!", accelerator.device)


        # 2. Устанавливаем рандомный параметр
        if self.args.seed is not None:
            set_seed(self.args.seed)
        print("Seed OK!")


        # 3. Создаём директорию проекта
        if accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)
        print("Directory OK!")


        # 4. Отключаем градиенты для нетренеруемых параметров
        self.model.vae.requires_grad_(False)
        self.model.base.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        if hasattr(self.model, "text_encoder_2"):
            self.model.text_encoder_2.requires_grad_(False)
        print("Models OK!")


        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.model.base.to(accelerator.device, dtype=weight_dtype)
        # The VAE is always in float32 to avoid NaN losses.
        self.model.vae.to(accelerator.device, dtype=torch.float32)
        self.model.text_encoder.to(accelerator.device, dtype=weight_dtype)
        if hasattr(self.model, "text_encoder_2"):
            self.model.text_encoder_2.to(accelerator.device, dtype=weight_dtype)


        # 5. Загружаем в модели LoRA адаптеры
        # now we will add new LoRA weights to the attention layers
        # Set correct lora layers
        unet_lora_config = LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.model.base.add_adapter(unet_lora_config)

        # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
        if self.args.train_text_encoder:
            # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
            text_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.rank,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self.model.text_encoder.add_adapter(text_lora_config)
            if hasattr(self.model, "text_encoder_2"):
                self.model.text_encoder_2.add_adapter(text_lora_config)
        print("LoRA OK!")

        
        # 6. Make sure the trainable params are in float32.
        if self.args.mixed_precision == "fp16":
            models = [self.model.base]
            if self.args.train_text_encoder:
                if hasattr(self.model, "text_encoder_2"):
                    models.extend([self.model.text_encoder, self.model.text_encoder_2])
                else:
                    models.extend([self.model.text_encoder])
            cast_training_params(models, dtype=torch.float32)
        
        
        # 7. Создаём оптимизатор
        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.base.parameters()))
        if self.args.train_text_encoder:
            if hasattr(self.model, "text_encoder_2"):
                params_to_optimize = (
                    params_to_optimize
                    + list(filter(lambda p: p.requires_grad, self.model.text_encoder.parameters()))
                    + list(filter(lambda p: p.requires_grad, self.model.text_encoder_2.parameters()))
                )
            else:
                params_to_optimize = (
                    params_to_optimize
                    + list(filter(lambda p: p.requires_grad, self.model.text_encoder.parameters()))
                )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        print("Optimizer OK!")


        # 8. Создаём даталоудер
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crops_coords_top_left"] for example in examples]
            target_sizes = [example['target_sizes'] for example in examples]
            captions = [example['caption'] for example in examples]
            result = {
                "pixel_values": pixel_values,
                "original_sizes": original_sizes,
                "crops_coords_top_left": crop_top_lefts,
                'target_sizes': target_sizes,
                'captions': captions
            }
            return result

        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )
        print("Dataloader OK!")

        
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )
        print("lr scheduler OK!")


        # 9. Подготавливаем всё в акселератор
        if self.args.train_text_encoder:
            if hasattr(self.model, "text_encoder_2"):
                self.model.base, self.model.text_encoder, self.model.text_encoder_2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    self.model.base, self.model.text_encoder, self.model.text_encoder_2, optimizer, train_dataloader, lr_scheduler
                )
            else:
                self.model.base, self.model.text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    self.model.base, self.model.text_encoder, optimizer, train_dataloader, lr_scheduler
                )
        else:
            self.model.base, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.model.base, optimizer, train_dataloader, lr_scheduler
            )
        print("All prepared with Accelerator!")


        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        
        # 10. Тренировка модели
        global_step = 0
        first_epoch = 0 
        initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.base.train()
            if self.args.train_text_encoder:
                self.model.text_encoder.train()
                if hasattr(self.model, "text_encoder_2"):
                    self.model.text_encoder_2.train()

            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(self.model.base):
                    # Convert images to latent space
                    model_input = self.model.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    model_input = model_input * self.model.vae.config.scaling_factor
                    model_input = model_input.to(weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)

                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, self.model.scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = self.model.scheduler.add_noise(model_input, noise, timesteps)

                    ####################################################################################################
                    # time ids
                    unet_added_conditions = None
                    if hasattr(self.model, "text_encoder_2"):
                        add_time_ids = torch.cat(
                            [
                                self.compute_time_ids(s, c, t) 
                                for s, c, t in 
                                zip(batch["original_sizes"], batch["crops_coords_top_left"], batch['target_sizes'])
                            ]
                        )
                        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                        unet_added_conditions = {"time_ids": add_time_ids}

                    prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt=batch['captions'])

                    if hasattr(self.model, "text_encoder_2"):
                        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

                    # Predict the noise residual
                    model_pred = self.model.base(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]
                    ####################################################################################################

                    # Get the target for loss depending on the prediction type
                    if self.model.scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.model.scheduler.config.prediction_type == "v_prediction":
                        target = self.model.scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.model.scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(self.args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, self.args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0


                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.args.max_train_steps:
                    break


        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.model.base = accelerator.unwrap_model(self.model.base)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.model.base))

            if self.args.train_text_encoder:
                self.model.text_encoder = accelerator.unwrap_model(self.model.text_encoder)
                text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.model.text_encoder))
                if hasattr(self.model, "text_encoder_2"):
                    self.model.text_encoder_2 = accelerator.unwrap_model(self.model.text_encoder_2)
                    text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.model.text_encoder_2))

            else:
                text_encoder_lora_layers = None
                if hasattr(self.model, "text_encoder_2"):
                    text_encoder_2_lora_layers = None

            if hasattr(self.model, "text_encoder_2"):
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=self.args.output_dir,
                    unet_lora_layers=unet_lora_state_dict,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                    text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                    safe_serialization=True,
                )
            else:
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=self.args.output_dir,
                    unet_lora_layers=unet_lora_state_dict,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                    safe_serialization=True,
                )
            
            torch.cuda.empty_cache()


        accelerator.end_training()
                    

        