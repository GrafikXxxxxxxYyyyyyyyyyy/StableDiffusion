import PIL
import time
import torch
import inspect
import itertools 
import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from models.stable_diffusion import SDModelWrapper
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.peft_utils import scale_lora_layers
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin, LoraLoaderMixin

from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLInpaintPipeline,
)



def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def denoising_value_valid(dnv):
    return isinstance(dnv, float) and 0 < dnv < 1



class StableDiffusionUnifiedPipeline():
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

        self.model: SDModelWrapper = None

    
    @torch.no_grad()
    def __call__(
        self,
        # text2image
        model: Optional[SDModelWrapper],
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: Optional[int] = 1,
        num_inference_steps: int = 30,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        seed: Optional[int] = None,
        # image2image
        image: PipelineImageInput = None,
        strength: float = 1.0,
        denoising_start: Optional[float] = None,
        # inpaint
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        padding_mask_crop: Optional[int] = None,
    ):
        """
        Единый пайплайн для всех задач, которые должны выполнять наши модели 
        """
        if model.device != self.device:
            model.to(self.device)

        self.model = model

        # 0. Default height and width to unet
        height = height or self.model.base.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.base.config.sample_size * self.model.vae_scale_factor

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            # batch_size = prompt_embeds.shape[0]
            pass
        print(f"Batch size: {batch_size}")
        
        # 2. Encode input prompt
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        
        # 3. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler, 
            num_inference_steps, 
            self.device, 
        )

        ############################################################################################################################################
        # На промпт не смотрим (просто либо условная либо безусловная генерация)
        # Итого 6/8 вариантов развития событий, всё збс
        # Если нет картинки, то не может быть и маски --> text2image
        ############################################################################################################################################
        # Done!
        latents: torch.Tensor
        if image is None:
            # 4. Создадим стартовые (полностью шумные) латенты
            shape = (
                batch_size * num_images_per_prompt,
                self.model.base.config.in_channels,
                height // self.model.vae_scale_factor,
                width // self.model.vae_scale_factor,
            )
            latents = self.prepare_latents_txt2img(
                self.model.scheduler,
                shape,
                prompt_embeds.dtype,
                seed,
                latents,
            )
            print(f"Text to image latents: {latents.shape}")

            self.current_task = "Text2Image"
        else:
            ########################################################################################################################################
            # Если картинка есть, а маски нет --> image2image
            ########################################################################################################################################
            if mask_image is None:
                # 4. Preprocess image
                image = model.image_processor.preprocess(image)

                timesteps, num_inference_steps = self.get_timesteps(
                    self.model.scheduler,
                    num_inference_steps,
                    strength,
                    denoising_start if denoising_value_valid(denoising_start) else None,
                )
                latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

                # 6. Prepare latent variables
                add_noise = True if denoising_start is None else False
                latents = self.prepare_latents_img2img(
                    image,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    seed,
                    add_noise,
                )
                print(f"Image to image latents: {latents.shape}")

                # Переопределяем размеры исходя из полученных латентов
                height, width = latents.shape[-2:]
                height = height * model.vae_scale_factor
                width = width * model.vae_scale_factor

                self.current_task = "Image2Image"
            ########################################################################################################################################
            # Ну и если есть и картинка и маска --> inpainting
            ########################################################################################################################################
            else:
                timesteps, num_inference_steps = self.get_timesteps(
                    self.model.scheduler,
                    num_inference_steps,
                    strength,
                    denoising_start if denoising_value_valid(denoising_start) else None,
                )
                # check that number of inference steps is not < 1 - as this doesn't make sense
                if num_inference_steps < 1:
                    raise ValueError(
                        f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                        f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
                    )
                # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
                latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
                # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
                is_strength_max = strength == 1.0


                # 5. Preprocess mask and image
                #  Ваще не понятно, что это за параметр
                if padding_mask_crop is not None:
                    crops_coords = self.model.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
                    resize_mode = "fill"
                else:
                    crops_coords = None
                    resize_mode = "default"

                init_image = self.model.image_processor.preprocess(
                    image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
                )
                init_image = init_image.to(dtype=torch.float32)

                mask = self.model.mask_processor.preprocess(
                    mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
                )

                # Получаем маскированное изображение
                if masked_image_latents is not None:
                    masked_image = masked_image_latents
                elif init_image.shape[1] == 4:
                    # if images are in latent space, we can't mask it
                    masked_image = None
                else:
                    masked_image = init_image * (mask < 0.5)

                
                # 6. Prepare latent variables
                num_channels_latents = self.model.vae.config.latent_channels
                num_channels_unet = self.model.base.config.in_channels
                return_image_latents = num_channels_unet == 4

                add_noise = True if denoising_start is None else False
                shape = (
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height // self.model.vae_scale_factor,
                    width // self.model.vae_scale_factor,
                )
                latents_outputs = self.prepare_latents_inpaint(
                    self.model.scheduler,
                    shape,
                    prompt_embeds.dtype,
                    seed,
                    latents,
                    image=init_image,
                    timestep=latent_timestep,
                    is_strength_max=is_strength_max,
                    add_noise=add_noise,
                    return_noise=True,
                    return_image_latents=return_image_latents,
                )

                if return_image_latents:
                    latents, noise, image_latents = latents_outputs
                else:
                    latents, noise = latents_outputs


                # 7. Prepare mask latent variables
                mask, masked_image_latents = self.prepare_mask_latents(
                    mask,
                    masked_image,
                    batch_size * num_images_per_prompt,
                    height,
                    width,
                    prompt_embeds.dtype,
                    seed=None, 
                )


                # 8. Check that sizes of mask, masked image and latents match
                if num_channels_unet == 9:
                    # default case for runwayml/stable-diffusion-inpainting
                    num_channels_mask = mask.shape[1]
                    num_channels_masked_image = masked_image_latents.shape[1]
                    if num_channels_latents + num_channels_mask + num_channels_masked_image != self.model.base.config.in_channels:
                        raise ValueError(
                            f"Incorrect configuration settings! The config of `pipeline.unet`: {self.model.base.config} expects"
                            f" {self.model.base.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                            f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                            f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                            " `pipeline.unet` or your `mask_image` or `image` input."
                        )
                elif num_channels_unet != 4:
                    raise ValueError(
                        f"The unet {self.model.base.__class__} should have either 4 or 9 input channels, not {self.model.base.config.in_channels}."
                    )
                
                # Переопределяем размеры исходя из полученных латентов
                height, width = latents.shape[-2:]
                height = height * model.vae_scale_factor
                width = width * model.vae_scale_factor  
                print(f"Inpainting latents: {latents.shape, masked_image_latents.shape}")

                self.current_task = "Inpainting"
        ############################################################################################################################################

        # 9.1 Apply denoising_end
        if (
            denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.model.scheduler.config.num_train_timesteps
                    - (denoising_end * self.model.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]


        # Если модель sdxl, то нужно prepare added time ids & embeddings
        if hasattr(model, "text_encoder_2"):
            add_text_embeds = pooled_prompt_embeds

            # Убрал выбор параметров, захардкодил по размеру картинки 
            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                original_size=(height, width),
                crops_coords_top_left=(0, 0),
                target_size=(height, width),
                negative_original_size=(height, width),
                negative_crops_coords_top_left=(0, 0),
                negative_target_size=(height, width),
                addition_time_embed_dim=self.model.base.config.addition_time_embed_dim,
                expected_add_embed_dim=self.model.base.add_embedding.linear_1.in_features,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.model.text_encoder_2.config.projection_dim,
            )
            add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)

            if self.do_classifier_free_guidance:
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds.to(self.device), 
                "time_ids": add_time_ids.to(self.device)
            }
        else:
            added_cond_kwargs = None
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(self.device)


        # # TODO: Добавить когда-нибудь ip_adapter
        # if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        #     image_embeds = self.prepare_ip_adapter_image_embeds(
        #         ip_adapter_image,
        #         ip_adapter_image_embeds,
        #         device,
        #         batch_size * num_images_per_prompt,
        #         self.do_classifier_free_guidance,
        #     )
        #     added_cond_kwargs["image_embeds"] = image_embeds


        # TODO: Optionally get Guidance Scale Embedding
        # timestep_cond = None
        # if self.unet.config.time_cond_proj_dim is not None:
        #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        #     timestep_cond = self.get_guidance_scale_embedding(
        #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        #     ).to(device=device, dtype=latents.dtype)


        # Denoising loop
        latents = self.denoise_batch(
            latents,
            timesteps,
            prompt_embeds,
            denoising_start=denoising_start,
            denoising_end=denoising_end,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_scale=guidance_scale,
        )


        # Опционально возвращаем либо картинку, либо латент
        if self.output_type == "pt":
            # unscale/denormalize the latents denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.model.vae.config, "latents_mean") and self.model.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.model.vae.config, "latents_std") and self.model.vae.config.latents_std is not None

            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(self.model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / self.model.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.model.vae.config.scaling_factor

            images = self.model.vae.decode(latents, return_dict=False)[0]
        elif self.output_type == "latents":
            images = latents
        else:
            raise ValueError(f"Unknown output_type = '{self.output_type}'")

        return images
        
        
    # TODO: Reconstruct logic 
    def encode_prompt(
        self,
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
        if lora_scale is not None and isinstance(self.model.lora_loader, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale
            # dynamically adjust the LoRA scale
            if self.model.text_encoder is not None:
                scale_lora_layers(self.model.text_encoder, lora_scale)
            # sdxl
            if hasattr(self.model, "text_encoder_2") and self.model.text_encoder_2 is not None:
                scale_lora_layers(self.model.text_encoder_2, lora_scale)
        
        tokenizers = [self.model.tokenizer]
        text_encoders = [self.model.text_encoder]
        prompt = [prompt] if isinstance(prompt, str) else prompt 
        prompts = [prompt]       

        batch_size: int
        if prompt is not None:
            batch_size = len(prompt)
        else:
            raise ValueError ("Пока нет возможности использовать уже готовые эмбеддинги промпта!\n")
        
        # Define tokenizers and text encoders so
        # it can be used with sd15 and sdxl self.models
        if hasattr(self.model, "text_encoder_2") and hasattr(self.model, "tokenizer_2"):
            tokenizers = [self.model.tokenizer, self.model.tokenizer_2]
            text_encoders = [self.model.text_encoder, self.model.text_encoder_2]
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
                    if hasattr(self.model, "text_encoder_2") else 
                    prompt_embeds[0]
                )
            else:
                prompt_embeds = (
                    prompt_embeds.hidden_states[-(clip_skip + 2)]
                    if hasattr(self.model, "text_encoder_2") else
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
                if hasattr(self.model, "text_encoder_2"):
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                else:
                    negative_prompt_embeds = negative_prompt_embeds[0]
                
                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
            ################################################################################################################
        

        if hasattr(self.model, "text_encoder_2") and self.model.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.model.text_encoder_2.dtype, device=self.device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.model.base.dtype, device=self.device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)


        if self.do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if hasattr(self.model, "text_encoder_2") and self.model.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.model.text_encoder_2.dtype, device=self.device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.model.base.dtype, device=self.device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


        if hasattr(self.model, "text_encoder_2"):
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
        print(prompt_embeds.dtype, negative_prompt_embeds.dtype, pooled_prompt_embeds.dtype, negative_pooled_prompt_embeds.dtype)
        
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

 
    def get_timesteps(
        self, 
        scheduler,
        num_inference_steps, 
        strength, 
        denoising_start=None
    ):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = scheduler.timesteps[t_start * scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    scheduler.config.num_train_timesteps
                    - (denoising_start * scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps

        return timesteps, num_inference_steps - t_start


    def prepare_latents_txt2img(
        self, 
        scheduler,
        shape,
        dtype, 
        seed=None, 
        latents=None
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
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma

        return latents
    

    def prepare_latents_img2img(
        self, 
        image, 
        timestep, 
        batch_size, 
        num_images_per_prompt, 
        dtype, 
        seed=None, 
        add_noise=True
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=self.device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt
        generator = (
            torch.Generator(device=self.device).manual_seed(int(seed)) 
            if seed is not None else 
            None
        )

        # Если изображение уже в формате латентов (имеет 4 канала), то оно используется как начальные латенты.
        if image.shape[1] == 4:
            init_latents = image
        # Если изображение не в формате латентов, кодирует изображение в латентное представление с использованием VAE
        else:
            # Апкастим ВАЕ, если это необходимо
            if self.model.vae.config.force_upcast:
                image = image.float()
                self.model.vae.to(dtype=torch.float32)

            init_latents = retrieve_latents(self.model.vae.encode(image), generator=generator)

            if self.model.vae.config.force_upcast:
                self.model.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            init_latents = self.model.vae.config.scaling_factor * init_latents

        
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
            init_latents = self.model.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents


    # TODO: Удостовериться что все работает океечно, уже внутри UI
    def prepare_latents_inpaint(
        self,
        scheduler,
        shape,
        dtype,                          # Тип данных (например, torch.float32)
        seed=None,                      # Генератор для создания случайных чисел
        latents=None,                   # Начальные латенты (если заданы)
        image=None,                     # Входное изображение (если задано)
        timestep=None,                  # Временной шаг (если задан)
        is_strength_max=True,           # Флаг, указывающий, используется ли максимальная сила шума
        add_noise=True,                 # Флаг, указывающий, нужно ли добавлять шум
        return_noise=False,             # Флаг, указывающий, нужно ли возвращать шум
        return_image_latents=False,     # Флаг, указывающий, нужно ли возвращать латенты изображения
    ):        
        batch_size = shape[0]
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # Проверка, что при ненулевой силе требуются и image, и timestep
        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        # Если изображение уже имеет форму латентов, просто повторяем его для batch_size
        if image.shape[1] == 4:
            image_latents = image.to(device=self.device, dtype=dtype)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        # Иначе, если требуется вернуть латенты изображения или начальные латенты не заданы и сила не максимальная
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=self.device, dtype=dtype)
            # Кодируем изображение в латентное пространство с использованием VAE
            image_latents = self._encode_vae_image(image=image, generator=generator)
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

        # Создаем выходной кортеж с латентами
        outputs = (latents,)

        # Если требуется вернуть шум, добавляем его в кортеж
        if return_noise:
            outputs += (noise,)

        # Если требуется вернуть латенты изображения, добавляем их в кортеж
        if return_image_latents:
            outputs += (image_latents,)

        return outputs   
    

    def prepare_mask_latents(
        self, 
        mask, 
        masked_image, 
        batch_size, 
        height, 
        width, 
        dtype, 
        seed=None, 
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(mask, size=(height, width))
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

        mask = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=self.device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

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
                torch.cat([masked_image_latents] * 2) if self.do_classifier_free_guidance else masked_image_latents
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
    

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        dtype = image.dtype
        if self.model.vae.config.force_upcast:
            image = image.float()
            self.model.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.model.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.model.vae.encode(image), generator=generator)

        if self.model.vae.config.force_upcast:
            self.model.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        image_latents = self.model.vae.config.scaling_factor * image_latents

        return image_latents
    

    def denoise_batch(
        self, 
        latents: torch.FloatTensor,
        timesteps: List[int],
        encoder_hidden_states: torch.FloatTensor,
        mask: Optional[torch.FloatTensor] = None,
        image_latents: Optional[torch.FloatTensor] = None,
        masked_image_latents: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 7.5,
        # ip_adapter_image: Optional[PipelineImageInput] = None,
    ):
        # TODO: Переделать данную логику с учетом 9канальной UNet сети для инпеинтинга
        for i, t in enumerate(timesteps):
            # Удваиваем количество латентов если работаем в режиме do_cfg=True 
            latent_model_input = latents
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            # По сути просто заглушка, которая не делает ничего с latent_model_input для DDPM/DDIM/PNDM 
            latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

            if self.model.base.config.in_channels == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Получаем предсказание шума моделью 
            noise_pred = self.model.base(
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
            latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # А теперь самое весёлое
            # Если код в условии на 9 каналов не сработал, то это значит, что предсказание модели было получено при помощи
            # модели имеющей 4 канала и трюка с максикрованием шума на этапе денойзинга 
            # + надо не забыть вытащить image_latents
            if self.current_task == "Inpainting" and self.model.base.config.in_channels == 4:
                init_latents_proper = image_latents
                if self.do_classifier_free_guidance:
                    init_mask, _ = mask.chunk(2)
                else:
                    init_mask = mask

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.model.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents_proper + init_mask * latents                


        return latents
    










