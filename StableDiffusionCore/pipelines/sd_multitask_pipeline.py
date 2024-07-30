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


# from diffusers import (
#     StableDiffusionPipeline,
#     StableDiffusionXLPipeline, 
#     StableDiffusionXLImg2ImgPipeline, 
#     StableDiffusionXLInpaintPipeline,
#     StableDiffusionUpscalePipeline,
# )



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


    def prepare_text_to_image_latents(self, 
        batch_size, 
        num_channels_latents, 
        height, 
        width, 
        dtype, 
        generator, 
        latents=None
    ):
        shape = (
            batch_size, 
            num_channels_latents, 
            height, 
            width
        )

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


    def prepare_image_to_image_latents(
        self, 
        vae,
        scheduler,
        image, 
        timestep, 
        batch_size, 
        dtype, 
        generator=None, 
        add_noise=True
    ):  
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=self.device, dtype=dtype)

        # Если изображение уже в формате латентов (имеет 4 канала), то оно используется как начальные латенты.
        if image.shape[1] == 4:
            init_latents = image
        else:
            init_latents = vae.encode(
                image,
                dtype,
                generator,
            )

        # Тупо выравнивает размеры батчей
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)


        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(
                shape, 
                generator=generator, 
                device=self.device, 
                dtype=dtype
            )
            # get latents
            init_latents = scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents
        

    def prepare_inpaint_latents(
        self,
        vae, 
        scheduler,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        add_noise=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size, 
            num_channels_latents, 
            height, 
            width
        )
        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        # Если изображение уже имеет форму латентов, просто повторяем его для batch_size
        if image.shape[1] == 4:
            image_latents = image.to(device=self.device, dtype=dtype)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=self.device, dtype=dtype)
            # Кодируем изображение в латентное пространство с использованием VAE
            image_latents = vae.encode(image, generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # Если начальные латенты не заданы и требуется добавить шум
        if latents is None and add_noise:
            # Генерация шума
            noise = randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)
            # Если сила максимальная, используем шум как начальные латенты, иначе комбинируем изображение и шум
            latents = noise if is_strength_max else scheduler.add_noise(image_latents, noise, timestep)
            # Если используется чистый шум, масштабируем начальные латенты с учетом начальной сигмы шума
            latents = latents * scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:
            # Если начальные латенты заданы, а шум добавить надо, то просто масштабируем латенты
            noise = latents.to(self.device)
            latents = noise * scheduler.init_noise_sigma
        else:
            # Если шум не нужно добавлять, просто генерируем новый шум
            noise = randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)
            latents = image_latents.to(self.device)

        outputs = (latents,)
        if return_noise:
            outputs += (noise,)
        if return_image_latents:
            outputs += (image_latents,)

        return outputs   
        
        
    def prepare_mask_latents(
        self, 
        vae,
        mask, 
        masked_image, 
        batch_size, 
        height, 
        width, 
        dtype, 
        generator,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height, width)
        )
        mask = mask.to(device=self.device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if self.do_cfg else mask

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=self.device, dtype=dtype)
                masked_image_latents = vae.encode(masked_image, generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if self.do_cfg else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=self.device, dtype=dtype)

        return mask, masked_image_latents
    

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
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
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
        model: Optional[StableDiffusionUnifiedModel],
        # Refiner
        use_refiner: Optional[bool] = False,
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
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None,
        guidance_scale: float = 5.0,
        latents: Optional[torch.FloatTensor] = None,
        # ip_adapter_image: Optional[PipelineImageInput] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pt",
        # image2image
        image: PipelineImageInput = None,
        strength: float = 1.0,
        # inpaint
        mask_image: PipelineImageInput = None,
    ):  
        ###############################################################################################
        # 1. Prepare constants
        ###############################################################################################
        # Всегда переводим на обычную модель и опционально на рефайнер
        model.switch_denoising_model("base")
        if use_refiner:
            model.switch_denoising_model("refiner")

        height = height or model.unet.config.sample_size * model.vae.scale_factor
        width = width or model.unet.config.sample_size * model.vae.scale_factor

        batch_size: int
        prompt = prompt or ""
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        # dtype = 
        is_strength_max = strength == 1.0
        num_channels_latents = model.vae.config.latent_channels
        num_channels_unet = model.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        ###############################################################################################


        
        ###############################################################################################
        # 2. Encode input prompt
        from StableDiffusionCore.models.sd_text_encoder import StableDiffusionTextEncoderModel
        ###############################################################################################
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
        if image is None and mask_image is None: # txt2img
            self.task = "txt2img"

            latents = self.prepare_text_to_image_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height // model.vae.scale_factor, 
                width // model.vae.scale_factor,
                prompt_embeds.dtype,
                generator,
                latents,
            )
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * model.scheduler.init_noise_sigma

        elif mask_image is None: # img2img
            self.task = "img2img"

            # preprocess image
            image = model.image_processor.preprocess(image)

            add_noise = True if denoising_start is None else False
            latents = self.prepare_image_to_image_latents(
                model.vae,
                model.scheduler,
                image,
                latent_timestep,
                batch_size * num_images_per_prompt,
                prompt_embeds.dtype,
                generator,
                add_noise,
            )

            # Переопределяем размеры исходя из полученных латентов
            height, width = latents.shape[-2:]
            height = height * model.vae.scale_factor
            width = width * model.vae.scale_factor
        else: # inpaint
            self.task = "inpaint"

            # preprocess image
            init_image = model.image_processor.preprocess(
                image, 
                height=height, 
                width=width,
            ).to(dtype=torch.float32)

            # preprocess mask
            mask = model.mask_processor.preprocess(
                mask_image, 
                height=height, 
                width=width,
            )
            
            # create masked image
            masked_image = (
                None
                if init_image.shape[1] == 4 else
                init_image * (mask < 0.5)
            )
            
            latents_outputs = self.prepare_inpaint_latents(
                model.vae,
                model.scheduler,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height // model.vae.scale_factor, 
                width // model.vae.scale_factor,
                prompt_embeds.dtype,
                generator,
                latents,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                add_noise=False if denoising_start else True,
                return_noise=True,
                return_image_latents=return_image_latents,
            )

            if return_image_latents:
                latents, noise, image_latents = latents_outputs
            else:
                latents, noise = latents_outputs
        
            mask, masked_image_latents = self.prepare_mask_latents(
                model.vae,
                mask,
                masked_image,
                batch_size * num_images_per_prompt,
                height // model.vae.scale_factor, 
                width // model.vae.scale_factor,
                prompt_embeds.dtype,
                generator,
            )

            # check that sizes of mask, masked image and latents match
            if num_channels_unet == 9:
                num_channels_mask = mask.shape[1]
                num_channels_masked_image = masked_image_latents.shape[1]
                if num_channels_latents + num_channels_mask + num_channels_masked_image != model.unet.num_channels_unet:
                    raise ValueError(
                        f"Incorrect configuration settings! The config of `pipeline.unet`: {model.unet.config} expects"
                        f" {model.unet.num_channels_unet} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input."
                    )
            elif num_channels_unet != 4:
                raise ValueError(
                    f"The unet {model.unet.unet.__class__} should have either 4 or 9 input channels, not {model.unet.num_channels_unet}."
                )

            # Переопределяем размеры исходя из полученных латентов
            height, width = latents.shape[-2:]
            height = height * model.vae.scale_factor
            width = width * model.vae.scale_factor  
        ###############################################################################################



        ###############################################################################################
        # TODO: 5. Prepare additional arguments
        ###############################################################################################
        added_cond_kwargs = None
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
        for i, t in enumerate(timesteps):
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
            if self.task == "inpaint" and model.unet.config.in_channels == 4:
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