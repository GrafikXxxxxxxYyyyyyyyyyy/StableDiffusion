import time
import wandb
import torch
import argparse
import itertools 
import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torchvision import transforms
from accelerate import Accelerator
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, AutoencoderTiny #,PNDMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from typing import Any, Callable, Dict, List, Optional, Union



class StableDiffusionText2Image ():
    

    def _train_batch(
            self, 
            noisy_model_input, 
            target, 
            timesteps, 
            encoder_hidden_states, 
            optimizer,
            accelerator = None
        ):
        optimizer.zero_grad()
        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input,
            timesteps, 
            encoder_hidden_states, 
            return_dict=False
        )[0]
        # Compute loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        # Backprop
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        optimizer.step()

        return loss.cpu().item()
    

    def _train_epoch(
        self, 
        train_loader, 
        optimizer,
        accelerator=None,
    ):
        epoch_loss = 0
        total_items = 0
    
        for batch in tqdm(train_loader, desc="Batch"):
            # Get the text embeddings for conditioning
            captions = batch['captions']
            promt_embeddings, _ = self._encode_prompt(
                captions, 
                num_images_per_prompt=1, 
                do_classifier_free_guidance=False
            )

            # Convert images to latent space
            pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
            if isinstance(self.vae, AutoencoderKL):
                model_input = self.vae.encode(pixel_values).latent_dist.sample()
            else:
                # if using AutoencoderTiny
                model_input = self.vae.encode(pixel_values).latents
            model_input = model_input * self.vae.config.scaling_factor

            # Sample noise that we'll add to the model input
            noise = randn_tensor(model_input.shape, device=self.device, dtype=model_input.dtype)
            
            # Sample a random timestep for each image
            bsz = model_input.shape[0]
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device
            )
            timesteps = timesteps.long()

            # Add noise to the input image (forward diffusion process)
            noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)
            noisy_model_input = noisy_model_input.to(self.device)

            # After prepearing all variables make one training step
            batch_loss = self._train_batch(
                noisy_model_input, 
                noise, 
                timesteps, 
                promt_embeddings, 
                optimizer,
                accelerator
            )

            # Logging if logger provided
            if self.logger:
                self.logger.log({"Train loss ob batch": batch_loss})
            
            epoch_loss += batch_loss * len(batch["pixel_values"])
            total_items += len(batch["pixel_values"])

        return epoch_loss / total_items
    
    
    def finetune(
        self, 
        train_loader, 
        args, 
    ):
        """
        Pipeline for train diffusion model
        """
        # 1. Инитим акселератор и device
        accelerator = None
        device = self.device
        if not self.device == torch.device("cpu"):
            accelerator = Accelerator(
                mixed_precision=args.mixed_precision,
            )
            device = accelerator.device
            # Disable AMP for MPS.
            if torch.backends.mps.is_available():
                accelerator.native_amp = False


        # 2. Выбираем тип данных 
        weight_dtype = torch.float32
        if accelerator is not None and accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16


        # 3. Выбираем что и как обучаем
        self.vae.requires_grad_(False)
        self.vae.to(device, dtype=weight_dtype)
        if args.use_lora_weights:
            # We only train the additional adapter LoRA layers, so 
            # disable grads for UNet and cast to 'fp16'
            self.unet.requires_grad_(False)
            self.unet.to(device, dtype=weight_dtype)
            # same for text_encoder
            self.text_encoder.requires_grad_(False)
            self.text_encoder.to(device, dtype=weight_dtype)

            # now we will add new LoRA weights to the attention layers
            unet_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
            )
            self.unet.add_adapter(unet_lora_config)

            if args.train_text_encoder:
                text_lora_config = LoraConfig(
                    r=args.rank,
                    lora_alpha=args.rank,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                )
                self.text_encoder.add_adapter(text_lora_config)

            # Make sure the trainable params are in float32.
            if args.mixed_precision == "fp16":
                models = [self.unet]
                if args.train_text_encoder:
                    models.append(self.text_encoder)
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models, dtype=torch.float32)

            params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            if args.train_text_encoder:
                params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, self.text_encoder.parameters()))

        else:
            self.unet.requires_grad_(True)
            if not args.train_text_encoder:
                self.text_encoder.requires_grad_(False)
                self.text_encoder.to(device, dtype=weight_dtype)

            params_to_optimize = (
                itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) 
                if args.train_text_encoder 
                else self.unet.parameters()
            )

        # 3. Подготавливаем optimizer
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )


        # 4. Prepare everything with our `accelerator` if provided
        if accelerator is not None:
            if args.train_text_encoder:
                self.unet, self.text_encoder, optimizer, train_loader = accelerator.prepare(
                    self.unet, self.text_encoder, optimizer, train_loader
                )
            else:
                self.unet, optimizer, train_loader = accelerator.prepare(
                    self.unet, optimizer, train_loader
                )

        
        # 5. Traning loop
        progress_bar = tqdm(range(args.num_train_epochs), initial=0, desc="Epoch")
        for i in progress_bar:
            self.unet.train()
            if args.train_text_encoder:
                self.text_encoder.train()

            epoch_loss = self._train_epoch(train_loader, optimizer, accelerator)

            if self.logger:
                self.logger.log({"Train loss on epoch" : epoch_loss})

            progress_bar.set_postfix({'Epoch loss': epoch_loss})


        # 6. Save trained weights
        if args.use_lora_weights:
            # Save LoRA safetensors
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Unwrap UNet and convert it to diffusers style state_dict
                self.unet = accelerator.unwrap_model(self.unet)
                self.unet = self.unet.to(torch.float32)
                unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet))

                if args.train_text_encoder:
                    self.text_encoder = accelerator.unwrap_model(self.text_encoder)
                    text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.text_encoder))
                else:
                    text_encoder_state_dict = None

                # After preparing all state_dicts save LoRA weights using LoraLoaderMixin
                LoraLoaderMixin.save_lora_weights(
                    save_directory = "loras/" + args.output_dir,
                    unet_lora_layers = unet_lora_state_dict,
                    text_encoder_lora_layers = text_encoder_state_dict,
                )
        else:
            # Save full models safetensors
            self.save_pretrained("./models")

        if accelerator is not None:
            accelerator.end_training()

        return




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
        default=16,
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
        default=50
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
