import os
import math
import tqdm
import torch
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import Dataset
from typing import Any, Optional, Union
from dataclasses import dataclass
from ..models.stable_diffusion import SDModelWrapper
from peft import LoraConfig
from diffusers.training_utils import cast_training_params
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed



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
class TrainingArguments:
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




class SDLoRADataset(Dataset):
    pass



class SDLoRATrainer ():
    def __init__(
        self,
        model: SDModelWrapper,
        args: TrainingArguments,
        train_dataset: SDLoRADataset, 
    ):
        self.model = model
        self.args = args
        self.dataset = train_dataset


    def train(self):
        # 1. Инициализируем акселератор
        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            # log_with=args.report_to,
            # project_config=accelerator_project_config,
            # kwargs_handlers=[kwargs],
        )
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
        self.model.text_encoder.requires_grad_(False)
        self.model.text_encoder_2.requires_grad_(False)
        self.model.base.requires_grad_(False)
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
            self.model.text_encoder_2.add_adapter(text_lora_config)
        print("LoRA OK!")

        
        # 6. Make sure the trainable params are in float32.
        if self.args.mixed_precision == "fp16":
            models = [self.model.base]
            if self.args.train_text_encoder:
                models.extend([self.model.text_encoder, self.model.text_encoder_2])
            cast_training_params(models, dtype=torch.float32)
        
        
        # 7. Создаём оптимизатор
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.base.parameters()))
        if self.args.train_text_encoder:
            params_to_optimize = (
                params_to_optimize
                + list(filter(lambda p: p.requires_grad, self.model.text_encoder.parameters()))
                + list(filter(lambda p: p.requires_grad, self.model.text_encoder_2.parameters()))
            )
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        print("Optimizer OK!")


        # 8. Создаём даталоудер
        def collate_fn():
            pass

        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            # collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )


        # 9. Подготавливаем всё в акселератор
        if self.args.train_text_encoder:
            self.model.base, self.model.text_encoder, self.model.text_encoder_2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.model.base, self.model.text_encoder, self.model.text_encoder_2, optimizer, train_dataloader, lr_scheduler
            )
        else:
            self.model.base, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.model.base, optimizer, train_dataloader, lr_scheduler
            )


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
                self.model.text_encoder_2.train()

            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(self.model.base):
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"]
                    model_input = self.model.vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * self.model.vae.config.scaling_factor
                    model_input = model_input.to(weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)

                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, self.model.scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = self.model.scheduler.add_noise(model_input, noise, timesteps)

                    ####################################################################################################
                    # time ids
                    def compute_time_ids(original_size, crops_coords_top_left):
                        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                        target_size = (args.resolution, args.resolution)
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_time_ids = torch.tensor([add_time_ids])
                        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                        return add_time_ids

                    add_time_ids = torch.cat(
                        [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                    )

                    # Predict the noise residual
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                    )
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
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
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
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
            unet = unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

            if args.train_text_encoder:
                text_encoder_one = unwrap_model(text_encoder_one)
                text_encoder_two = unwrap_model(text_encoder_two)

                text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_one))
                text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_two))
            else:
                text_encoder_lora_layers = None
                text_encoder_2_lora_layers = None

            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            )

            del unet
            del text_encoder_one
            del text_encoder_two
            del text_encoder_lora_layers
            del text_encoder_2_lora_layers
            torch.cuda.empty_cache()


        accelerator.end_training()
                    

        