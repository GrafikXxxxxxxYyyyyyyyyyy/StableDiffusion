import PIL
import time
import torch
import inspect
import itertools 
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm.notebook import tqdm
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from StableDiffusionCore.sd_unified_model import StableDiffusionUnifiedModel


from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLInpaintPipeline,
    StableDiffusionUpscalePipeline,
)



class StableDiffusionMultitaskPipeline():
    def __init__(
        self, 
        do_cfg: bool = True,
        device: Optional[str] = None,
    ):
        self.do_cfg = False
        if do_cfg:
            self.do_cfg = True

        self.device = torch.device("cpu")
        if device:
            self.device = torch.device(device) 
        

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        addition_time_embed_dim,
        expected_add_embed_dim,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )

        if (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids


    ###########################################################################################################################################################################################
    # TODO: Добавить в измененную версию пайплайна
    ###########################################################################################################################################################################################
    # # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    # def prepare_ip_adapter_image_embeds(self, ip_adapter_image, device, num_images_per_prompt):
    #     if not isinstance(ip_adapter_image, list):
    #         ip_adapter_image = [ip_adapter_image]

    #     if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
    #         raise ValueError(
    #             f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
    #         )

    #     image_embeds = []
    #     for single_ip_adapter_image, image_proj_layer in zip(
    #         ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
    #     ):
    #         output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
    #         single_image_embeds, single_negative_image_embeds = self.encode_image(
    #             single_ip_adapter_image, device, 1, output_hidden_state
    #         )
    #         single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
    #         single_negative_image_embeds = torch.stack([single_negative_image_embeds] * num_images_per_prompt, dim=0)

    #         if self.do_classifier_free_guidance:
    #             single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
    #             single_image_embeds = single_image_embeds.to(device)

    #         image_embeds.append(single_image_embeds)

    #     return image_embeds
    ###########################################################################################################################################################################################


    ###########################################################################################################################################################################################
    # TODO: Добавить в измененную версию пайплайна
    ###########################################################################################################################################################################################
    # # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    # def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
    #     """
    #     See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    #     Args:
    #         timesteps (`torch.Tensor`):
    #             generate embedding vectors at these timesteps
    #         embedding_dim (`int`, *optional*, defaults to 512):
    #             dimension of the embeddings to generate
    #         dtype:
    #             data type of the generated embeddings

    #     Returns:
    #         `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    #     """
    #     assert len(w.shape) == 1
    #     w = w * 1000.0

    #     half_dim = embedding_dim // 2
    #     emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    #     emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    #     emb = w.to(dtype)[:, None] * emb[None, :]
    #     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    #     if embedding_dim % 2 == 1:  # zero pad
    #         emb = torch.nn.functional.pad(emb, (0, 1))
    #     assert emb.shape == (w.shape[0], embedding_dim)
    #     return emb
    ###########################################################################################################################################################################################



    @torch.no_grad()
    def __call__(
        self,
        model: StableDiffusionUnifiedModel,

        # text2image
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
            # positive_prompt: Optional[Union[str, List[str]]] = None,
            # positive_prompt_2: Optional[Union[str, List[str]]] = None,
            # positive_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: Optional[int] = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pt",
        
        # image2image
        image: PipelineImageInput = None,
        strength: float = 1.0,

        # inpaint
        mask_image: PipelineImageInput = None,

        # Refiner
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None,
        refiner_latents: Optional[torch.FloatTensor] = None,
            # ip_adapter_image: Optional[PipelineImageInput] = None,
    ):  
        ###############################################################################################
        # 1. Prepare constants
        ###############################################################################################
        batch_size: int
        prompt = prompt or ""
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        is_strength_max = strength == 1.0
        num_channels_latents = model.vae.config.latent_channels
        num_channels_unet = model.unet.config.in_channels
        ###############################################################################################


        
        ###############################################################################################
        # 2. Encode input prompt
        from StableDiffusionCore.models.sd_text_encoder import StableDiffusionTextEncoderModel
        ###############################################################################################
        # TODO: Добавить в аргументы опциональные готовые эмбеддинги
        prompt_embeds, pooled_prompt_embeds = model.text_encoder(
            prompt,
            prompt_2,
            prompt_3,
            num_images_per_prompt,
            lora_scale=None,
            clip_skip=None,
        )
        if self.do_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds = model.text_encoder(
                negative_prompt,
                negative_prompt_2,
                negative_prompt_3,
                num_images_per_prompt,
                lora_scale=None,
                clip_skip=None,
            )
            
            # # TODO: Positive prompt embeddings
            # positive_prompt_embeds, positive_pooled_prompt_embeds = model.text_encoder(
            #     positive_prompt,
            #     positive_prompt_2,
            #     positive_prompt_3,
            #     num_images_per_prompt,
            #     lora_scale=None,
            #     clip_skip=None,
            # )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        ###############################################################################################



        ###############################################################################################
        # 3. Get timesteps
        from StableDiffusionCore.models.sd_scheduler import StableDiffusionSchedulerModel
        ###############################################################################################
        timesteps, num_inference_steps = model.scheduler.prepare_timesteps(
            num_inference_steps,
            strength,
            self.device,
            denoising_start,
            denoising_end,
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        ###############################################################################################



        ###############################################################################################
        # 4. Prepare latents for the corresponding task
        from StableDiffusionCore.models.sd_vae import StableDiffusionVAEModel
        ###############################################################################################
        task: str
        latents: torch.Tensor
        if image is None and mask_image is None: # txt2img
            task = "txt2img"

            height = height or model.unet.config.sample_size * model.vae.scale_factor
            width = width or model.unet.config.sample_size * model.vae.scale_factor

            # prepare noisy latents
            latents = randn_tensor(
                shape=(
                    batch_size * num_images_per_prompt,
                    num_channels_latents, 
                    height // model.vae.scale_factor,
                    width // model.vae.scale_factor,
                ), 
                generator=generator, 
                device=self.device, 
                dtype=prompt_embeds.dtype,
            )
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * model.scheduler.init_noise_sigma
        elif mask_image is None: # img2img
            task = "img2img"

            # process and encode image
            image = model.image_processor.preprocess(image)
            if height and width:
                image = torch.nn.functional.interpolate(
                    image, size=(height, width)
                )
            image = image.to(device=self.device, dtype=prompt_embeds.dtype)
            image_latents = model.vae.encode(image, generator=generator)

            # add noise to latents
            noise = randn_tensor(
                shape=image_latents.shape, 
                generator=generator, 
                device=self.device, 
                dtype=prompt_embeds.dtype,
            )
            latents = model.scheduler.add_noise(image_latents, noise, latent_timestep)
        else: # inpaint
            task = "inpaint"

            # process mask and image
            initial_image = model.image_processor.preprocess(image)
            mask = model.mask_processor.preprocess(mask_image)
            if height and width:
                initial_image = torch.nn.functional.interpolate(
                    initial_image, size=(height, width)
                )
                mask = torch.nn.functional.interpolate(
                    mask, size=(height, width)
                )
            # masking the original image
            masked_image = initial_image * (mask < 0.5)

            # encode image
            initial_image = initial_image.to(device=self.device, dtype=prompt_embeds.dtype)
            image_latents = model.vae.encode(initial_image, generator)
            image_latents = image_latents.repeat(
                (batch_size * num_images_per_prompt) // image_latents.shape[0], 1, 1, 1
            )
            # encode masked image
            masked_image = masked_image.to(device=self.device, dtype=prompt_embeds.dtype)
            masked_image_latents = model.vae.encode(masked_image, generator)
            masked_image_latents = masked_image_latents.repeat(
                (batch_size * num_images_per_prompt) // masked_image_latents.shape[0], 1, 1, 1
            )
            # resize the mask to latents shape as we concatenate the mask to the latents
            mask = torch.nn.functional.interpolate(
                mask, size=(height // model.vae.scale_factor, width // model.vae.scale_factor)
            )
            mask = mask.repeat(
                (batch_size * num_images_per_prompt) // mask.shape[0], 1, 1, 1
            )

            # add noise to latents
            noise = randn_tensor(
                shape=image_latents.shape, 
                generator=generator, 
                device=self.device, 
                dtype=prompt_embeds.dtype
            )
            if is_strength_max:
                latents = noise * model.scheduler.init_noise_sigma
            else:
                latents = model.scheduler.add_noise(image_latents, noise, latent_timestep)
            
            # Применяем CFG
            if self.do_cfg:
                mask = torch.cat([mask] * 2)
                masked_image_latents = torch.cat([masked_image_latents] * 2) 

            # aligning device to prevent device errors when concating it with the latent model input
            image_latents = image_latents.to(device=self.device, dtype=prompt_embeds.dtype)
            mask = mask.to(device=self.device, dtype=prompt_embeds.dtype)
            masked_image_latents = masked_image_latents.to(device=self.device, dtype=prompt_embeds.dtype)
        ###############################################################################################
        # Если на вход пришли латенты с предыдущей стадии рефайнера, то просто используем их
        if refiner_latents:
            latents = refiner_latents



        ###############################################################################################
        # TODO: 5. Prepare additional arguments
        ###############################################################################################
        if "Prepare additional components":
            added_cond_kwargs = None
            
            # # TODO: IP_adapter
            # if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            #     image_embeds = self.prepare_ip_adapter_image_embeds(
            #         ip_adapter_image,
            #         ip_adapter_image_embeds,
            #         device,
            #         batch_size * num_images_per_prompt,
            #         self.do_classifier_free_guidance,
            #     )
            #     added_cond_kwargs["image_embeds"] = image_embeds
            
            if model.type == "sdxl":
                # time ids
                add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                    original_size=(height, width),
                    crops_coords_top_left=(0, 0),
                    target_size=(height, width),
                    negative_original_size=(height, width),
                    negative_crops_coords_top_left=(0, 0),
                    negative_target_size=(height, width),
                    addition_time_embed_dim=model.unet.config.addition_time_embed_dim,
                    expected_add_embed_dim=model.unet.add_embed_dim,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=model.text_encoder.text_encoder_projection_dim,
                )
                add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)

                if self.do_cfg:
                    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds.to(self.device), 
                    "time_ids": add_time_ids.to(self.device)
                }
            elif model.type == "sd3":
                raise ValueError(f"Model  type '{model.type}' cannot be used!")

            # # TODO: Optionally get Guidance Scale Embedding
            # timestep_cond = None
            # if self.unet.config.time_cond_proj_dim is not None:
            #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            #     timestep_cond = self.get_guidance_scale_embedding(
            #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            #     ).to(device=device, dtype=latents.dtype)
        ###############################################################################################
        


        ###############################################################################################
        # 6. Denoising loop
        from StableDiffusionCore.models.sd_unet import StableDiffusionUNetModel
        ###############################################################################################
        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_cfg else
                latents
            )
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

            # Объединяем латенты если модель для inpainting
            if model.unet.config.in_channels == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Получаем предсказание шума моделью 
            noise_pred = model.unet(
                latent_model_input,
                t,
                prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
            )

            # TODO: Переработать логику cfg для 3х промптов
            if self.do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond

            # Вычисляем шумный латент с предыдущего шага x_t -> x_t-1
            latents = model.scheduler.step(
                noise_pred, 
                t, 
                latents, 
            )
            
            # TODO: Перенести на сторону UNet модели и добавить параметр self.inpainting: bool
            # Если код в условии на 9 каналов не сработал, то это значит, что предсказание модели было получено при помощи
            # модели имеющей 4 канала и трюка с максикрованием шума на этапе денойзинга 
            if task == "inpaint" and model.unet.config.in_channels == 4:
                init_latents_proper = image_latents
                if self.do_cfg:
                    init_mask, _ = mask.chunk(2)
                else:
                    init_mask = mask

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = model.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents_proper + init_mask * latents                
        ###############################################################################################


        
        ###############################################################################################
        # 7. Опционально возвращаем либо картинку, либо латент
        ###############################################################################################
        if output_type == "pt":
            # unscale/denormalize the latents denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(model.vae.config, "latents_mean") and model.vae.config.latents_mean
            has_latents_std = hasattr(model.vae.config, "latents_std") and model.vae.config.latents_std

            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / model.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / model.vae.config.scaling_factor

            images = model.vae.decode(latents)
        elif output_type == "latents":
            images = latents
        else:
            raise ValueError(f"Unknown output_type = '{output_type}'")
        ###############################################################################################
        

        return images