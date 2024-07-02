import time
# import wandb
import torch
import itertools 
import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from models.stable_diffusion import SDModelWrapper
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.peft_utils import scale_lora_layers
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin, LoraLoaderMixin

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline


class StableDiffusionText2Image ():
    def __init__(
        self, 
        do_cfg: bool = True,
        device: Optional[str] = None,
        output_type: Optional[str] = None,
    ):
        self.do_classifier_free_guidance = False
        if do_cfg:
            self.do_classifier_free_guidance = True

        self.device = torch.device("cpu")
        if device is not None:
            self.device = torch.device(device)

        self.output_type = "pt"
        if output_type is not None:
            self.output_type = output_type


    def __call__(
        self,
        model: SDModelWrapper,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 30,
        guidance_scale: Optional[float] = 6,
        num_images_per_prompt: Optional[int] = 1,
        denoising_end: Optional[float] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Inference pipeline
        """
        # Change device if different
        if model.device != self.device:
            model.to(self.device)

        # 0. Опциональный выбор размеров картинки
        height = height or model.base.config.sample_size * model.vae_scale_factor
        width = width or model.base.config.sample_size * model.vae_scale_factor


        # 1. Устанавливаем размер батча, исходя из кол-ва пришедших промптов
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError ("Пока нет возможности использовать уже готовые эмбеддинги промпта!\n")


        # 2. Получаем эмбеддинги промпта
        lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        # TODO: Переделать вообще всю логику взаимодействия с моделью на этом этапе
        # модель не должна гулять по методам класса 
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            model,
            prompt = prompt,
            prompt_2 = prompt_2,
            negative_prompt = negative_prompt,
            negative_prompt_2 = negative_prompt_2,
            num_images_per_prompt = num_images_per_prompt,
            lora_scale = lora_scale,
            clip_skip = clip_skip,
        )
        

        # 3. Создаёт расписание шумовых шагов и меняет планировщик шума модели если передан
        model.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = model.scheduler.timesteps

    
        # 4. Создадим стартовые (полностью шумные) латенты
        shape = (
            batch_size * num_images_per_prompt, 
            model.base.config.in_channels, 
            height // model.vae_scale_factor, 
            width // model.vae_scale_factor
        )
        latents = self._prepare_latents(
            shape,
            prompt_embeds.dtype,
            seed
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * model.scheduler.init_noise_sigma

        
        # 5. Если модель sdxl, то нужно добавить time ids & embeddings
        if hasattr(model, "text_encoder_2"):
            add_text_embeds = pooled_prompt_embeds

            add_time_ids = self._get_add_time_ids(
                (height, width),
                (0, 0),
                (height, width),
                prompt_embeds.dtype,
                model.base.config.addition_time_embed_dim,
                model.base.add_embedding.linear_1.in_features,
                model.text_encoder_2.config.projection_dim,
            )
            negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance:
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds.to(self.device), 
                "time_ids": add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)
            }
        else:
            added_cond_kwargs = None

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)


        # 6. Собственно сам denoising loop для инференса
        with torch.no_grad(): 
            latents = self.denoise_batch(
                model,
                denoising_end,
                timesteps, 
                latents,
                prompt_embeds,
                added_cond_kwargs,
                cross_attention_kwargs,
                guidance_scale,
            )

            # Опционально возвращаем либо картинку, либо латент
            if self.output_type == "pt":
                # unscale/denormalize the latents denormalize with the mean and std if available and not None
                has_latents_mean = hasattr(model.vae.config, "latents_mean") and model.vae.config.latents_mean is not None
                has_latents_std = hasattr(model.vae.config, "latents_std") and model.vae.config.latents_std is not None

                if has_latents_mean and has_latents_std:
                    latents_mean = torch.tensor(model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    latents_std = torch.tensor(model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    latents = latents * latents_std / model.vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / model.vae.config.scaling_factor

                images = model.vae.decode(latents, return_dict=False)[0]
            elif self.output_type == "latents":
                images = latents
            else:
                raise ValueError(f"Unknown output_type = '{self.output_type}'")

        return images
    

    def denoise_batch(
        self, 
        model: SDModelWrapper,
        denoising_end,
        timesteps, 
        latents,
        encoder_hidden_states,
        added_cond_kwargs,
        cross_attention_kwargs,
        guidance_scale
    ):
        """
        Denoising loop
        """
        # Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(model.scheduler.config.num_train_timesteps
                    - (denoising_end * model.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        for t in tqdm(timesteps):
            # Удваиваем количество латентов если работаем в режиме do_cfg=True 
            latent_model_input = latents
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            # По сути просто заглушка, которая не делает ничего с latent_model_input для DDPM/DDIM/PNDM 
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

            # Получаем предсказание шума моделью 
            noise_pred = model.base(
                latent_model_input,
                t,
                encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond

            # Вычисляем шумный латент с предыдущего шага x_t -> x_t-1
            latents = model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # # Получаем изображения
        # latents = latents.to(self.device)
        # latents = latents / model.vae.config.scaling_factor
        # images = model.vae.decode(latents, return_dict=False)[0]

        return latents


    # TODO: Reconstruct logic 
    def encode_prompt(
        self,
        model: SDModelWrapper,
        prompt: Optional[str] = None,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        num_images_per_prompt: int = 1,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(model.lora_loader, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale
            # dynamically adjust the LoRA scale
            if model.text_encoder is not None:
                scale_lora_layers(model.text_encoder, lora_scale)
            # sdxl
            if hasattr(model, "text_encoder_2") and model.text_encoder_2 is not None:
                scale_lora_layers(model.text_encoder_2, lora_scale)
        
        tokenizers = [model.tokenizer]
        text_encoders = [model.text_encoder]
        prompt = [prompt] if isinstance(prompt, str) else prompt 
        prompts = [prompt]       

        batch_size: int
        if prompt is not None:
            batch_size = len(prompt)
        else:
            raise ValueError ("Пока нет возможности использовать уже готовые эмбеддинги промпта!\n")
        
        # Define tokenizers and text encoders so
        # it can be used with sd15 and sdxl models
        if hasattr(model, "text_encoder_2") and hasattr(model, "tokenizer_2"):
            tokenizers = [model.tokenizer, model.tokenizer_2]
            text_encoders = [model.text_encoder, model.text_encoder_2]
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            # textual inversion: process multi-vector tokens if necessary
            prompts = [prompt, prompt_2]

        ################################################################################################################
        # Получим эмбеддинги промпта
        ################################################################################################################
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
            text_input_ids = text_input_ids.to(self.device)

            prompt_embeds = text_encoder(
                text_input_ids, output_hidden_states=True
            )

            pooled_prompt_embeds = prompt_embeds[0]
            # Сlip-skip работает немного по-разному для sdxl / sd15 
            if clip_skip is None:
                prompt_embeds = (
                    prompt_embeds.hidden_states[-2] 
                    if hasattr(model, "text_encoder_2") else 
                    prompt_embeds[0]
                )
            else:
                prompt_embeds = (
                    prompt_embeds.hidden_states[-(clip_skip + 2)]
                    if hasattr(model, "text_encoder_2") else
                    text_encoder.text_model.final_layer_norm(prompt_embeds[-1][-(clip_skip + 1)])
                )

            prompt_embeds_list.append(prompt_embeds)
        
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        ################################################################################################################


        negative_prompt_embeds = None
        if self.do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = (
                batch_size * [negative_prompt] 
                if isinstance(negative_prompt, str) else 
                negative_prompt
            )
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] 
                if isinstance(negative_prompt_2, str) else 
                negative_prompt_2
            )

            uncond_tokens: List[str]
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            ################################################################################################################
            # Получаем аналогично для негативного промпта
            ################################################################################################################
            negative_prompt_embeds_list = []
            negative_pooled_prompt_embeds = None
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input_ids = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                uncond_input_ids = uncond_input_ids.to(self.device)

                negative_prompt_embeds = text_encoder(
                    uncond_input_ids,
                    output_hidden_states=True,
                )

                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                if hasattr(model, "text_encoder_2"):
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                else:
                    negative_prompt_embeds = negative_prompt_embeds[0]
                
                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
            ################################################################################################################
        

        if hasattr(model, "text_encoder_2") and model.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=model.text_encoder_2.dtype, device=self.device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=model.base.dtype, device=self.device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)


        if self.do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if hasattr(model, "text_encoder_2") and model.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=model.text_encoder_2.dtype, device=self.device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=model.base.dtype, device=self.device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


        if hasattr(model, "text_encoder_2"):
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

            if self.do_classifier_free_guidance:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                    bs_embed * num_images_per_prompt, -1
                )

        # # Retrieve the original scale by scaling back the LoRA layers
        # if model.text_encoder is not None:
        #     if isinstance(model.lora_loader, StableDiffusionXLLoraLoaderMixin):
        #         unscale_lora_layers(model.text_encoder, lora_scale)
        # if model.text_encoder_2 is not None:
        #     if isinstance(model.lora_loader, StableDiffusionXLLoraLoaderMixin):
        #         unscale_lora_layers(model.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


    def _prepare_latents(
        self, 
        shape,
        dtype,  
        seed=None,
        latents=None,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        if latents is None:
            latents = randn_tensor(
                shape, 
                generator=generator,
                device=self.device, 
                dtype=dtype
            )
        else:
            latents = latents.to(self.device)

        return latents


    def _get_add_time_ids(
        self, 
        original_size, 
        crops_coords_top_left, 
        target_size, 
        dtype, 
        addition_time_embed_dim,
        expected_add_embed_dim,
        text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids