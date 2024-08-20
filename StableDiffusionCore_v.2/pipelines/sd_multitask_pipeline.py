import PIL
import time
import torch
import inspect
import itertools 
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm.notebook import tqdm
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ..sd_unified_model import StableDiffusionModel
from ..models.sd_vae import StableDiffusionVAEInput, StableDiffusionVAEOutput
from ..models.sd_text_encoder import StableDiffusionTextEncoderInput, StableDiffusionTextEncoderOutput



########################################################################################################################
class StableDiffusionMultitaskPipelineOutput:
    def __init__(
        self, 
        latents: Optional[torch.Tensor] = None,
        denoising_start: Optional[float] = None,
        generated_images: Optional[torch.Tensor] = None,
        encoded_image: Optional[StableDiffusionVAEOutput] = None,
        encoded_prompt: Optional[StableDiffusionTextEncoderOutput] = None,
        negative_encoded_prompt: Optional[StableDiffusionTextEncoderOutput] = None,
    ):
        self.latents = latents
        self.encoded_image = encoded_image
        self.encoded_prompt = encoded_prompt
        self.denoising_start = denoising_start
        self.generated_images = generated_images
        self.negative_encoded_prompt = negative_encoded_prompt


    def __call__(
        self,
    ):
        pass
########################################################################################################################




########################################################################################################################
@dataclass
class StableDiffusionMultitaskPipelineInput(BaseOutput):
    # pre-stage models input
    vae_input: Optional[StableDiffusionVAEInput] = None
    text_encoder_input: Optional[StableDiffusionTextEncoderInput] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None

    # main-stage pipeline config
    do_cfg: bool = True
    strength: float = 1.0
    output_type: str = "pt"
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    num_images_per_prompt: int = 1
    denoising_end: Optional[float] = None

    # transposable elements
    return_extractor_out: bool = False
    prev_output: Optional[StableDiffusionMultitaskPipelineOutput] = None
########################################################################################################################




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
        aesthetic_score,
        negative_aesthetic_score,
        target_size,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        addition_time_embed_dim,
        expected_add_embed_dim,
        dtype,
        text_encoder_projection_dim,
        requires_aesthetics_score,
    ):
        if requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
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



    @torch.no_grad()
    def __call__(
        self,
        model: StableDiffusionModel,
        do_cfg: bool = True,
        strength: float = 1.0,
        output_type: str = "pt",
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        denoising_end: Optional[float] = None,
        vae_input: Optional[StableDiffusionVAEInput] = None,
        text_encoder_input: Optional[StableDiffusionTextEncoderInput] = None,
        prev_output: Optional[StableDiffusionMultitaskPipelineOutput] = None,
        **kwargs,
    ):  
        if "1. Prepare constants":
            _is_strength_max = strength == 1.0
            _unet_channels = model.unet.config.in_channels
            _latents_channels = model.vae.config.latent_channels       



        _batch_size: int
        prompt_embeds: torch.Tensor
        mask: Optional[torch.Tensor]
        image_latents: Optional[torch.Tensor]
        pooled_prompt_embeds: Optional[torch.Tensor]
        masked_image_latents: Optional[torch.Tensor]
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]]
        if "2. Prepare image_latents, prompt_embeds and added_cond_kwargs":
            if (
                prev_output is not None 
                and isinstance(prev_output, StableDiffusionMultitaskPipelineOutput)
            ):  
                encoded_image = (
                    prev_output.encoded_image
                    if (
                        prev_output.encoded_image is not None
                        and isinstance(prev_output.encoded_image, StableDiffusionVAEOutput)
                    ) else
                    model.vae(**vae_input)
                )        
                encoded_prompt = (
                    prev_output.encoded_prompt
                    if (
                        prev_output.encoded_prompt is not None 
                        and isinstance(prev_output.encoded_prompt, StableDiffusionTextEncoderOutput)
                    ) else
                    model.text_encoder(**text_encoder_input)
                )
            else:
                encoded_image = model.vae(**vae_input)
                encoded_prompt = model.text_encoder(**text_encoder_input)

            # get text_encoder embedding
            prompt_embeds, pooled_prompt_embeds = encoded_prompt(
                model.type,
                model.use_refiner, 
                num_images_per_prompt,
            )
            
            # apply classifier-free guidance
            if do_cfg:
                pass

            # get vae image_latents
            image_latents, mask, masked_image_latents = encoded_image(
                model.type,
                model.use_refiner, 
                num_images_per_prompt,
            )


            # if model.type == "sd15":
            #     added_cond_kwargs = None
            # elif model.type == "sdxl":
            #     # time ids
            #     add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            #         original_size = (height, width),
            #         crops_coords_top_left = (0, 0),
            #         aesthetic_score = aesthetic_score,
            #         negative_aesthetic_score = negative_aesthetic_score,
            #         target_size = (height, width),
            #         negative_original_size = (height, width),
            #         negative_crops_coords_top_left = (0, 0),
            #         negative_target_size = (height, width),
            #         addition_time_embed_dim = model.unet.config.addition_time_embed_dim,
            #         expected_add_embed_dim = model.unet.add_embed_dim,
            #         dtype = prompt_embeds.dtype,
            #         text_encoder_projection_dim = model.text_encoder.text_encoder_projection_dim,
            #         requires_aesthetics_score = model.use_refiner,
            #     )
            #     add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            #     add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)

            #     if self.do_cfg:
            #         add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

            #     added_cond_kwargs = {
            #         "text_embeds": pooled_prompt_embeds.to(self.device), 
            #         "time_ids": add_time_ids.to(self.device),
            #     }

            

        task: str
        timesteps: List[int]
        latents: torch.Tensor
        _batch_size = prompt_embeds.shape[0]
        _generator = (
            vae_input.generator
            if vae_input is not None else
            None   
        )
        _denoising_start = (
            prev_output.denoising_start
            if prev_output is not None else
            None
        )
        _denoising_end = denoising_end
        if "3. Prepare timesteps and noisy latents":
            timesteps, num_inference_steps = model.scheduler.prepare_timesteps(
                num_inference_steps,
                strength,
                self.device,
                _denoising_start,
                _denoising_end,
            )
            latent_timestep = timesteps[:1].repeat(_batch_size)


            if image_latents is None and masked_image_latents is None: # txt2img
                task = "txt2img"

                height = height or model.unet.config.sample_size * model.vae.scale_factor
                width = width or model.unet.config.sample_size * model.vae.scale_factor
                # prepare noisy latents
                latents = randn_tensor(
                    shape=(
                        _batch_size,
                        _latents_channels, 
                        height // model.vae.scale_factor,
                        width // model.vae.scale_factor,
                    ), 
                    generator=_generator, 
                    device=self.device, 
                    dtype=prompt_embeds.dtype,
                )
                # scale the initial noise by the standard deviation required by the scheduler
                latents = latents * model.scheduler.init_noise_sigma

            elif masked_image_latents is None: # img2img
                task = "img2img"
                # sample noise 
                noise = randn_tensor(
                    shape=image_latents.shape, 
                    generator=_generator, 
                    device=self.device, 
                    dtype=prompt_embeds.dtype,
                )
                # add noise to latents
                latents = model.scheduler.add_noise(
                    image_latents, 
                    noise, 
                    latent_timestep
                )

            else: # inpaint
                task = "inpaint"
                # sample noise 
                noise = randn_tensor(
                    shape=image_latents.shape, 
                    generator=_generator, 
                    device=self.device, 
                    dtype=prompt_embeds.dtype
                )
                # add noise to latents
                latents = (
                    noise * model.scheduler.init_noise_sigma
                    if _is_strength_max else
                    model.scheduler.add_noise(image_latents, noise, latent_timestep)
                )
                if do_cfg:
                    mask = torch.cat([mask] * 2)
                    masked_image_latents = torch.cat([masked_image_latents] * 2) 

                # aligning device to prevent device errors when concating it with the latent model input
                image_latents = image_latents.to(device=self.device, dtype=prompt_embeds.dtype)
                mask = mask.to(device=self.device, dtype=prompt_embeds.dtype)
                masked_image_latents = masked_image_latents.to(device=self.device, dtype=prompt_embeds.dtype)



        if "4. Denoising loop":
            cross_attention_kwargs = (
                None
                if text_encoder_input.lora_scale is None else
                {"scale": text_encoder_input.lora_scale}
            )

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if do_cfg else
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

                if self.do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond

                # Вычисляем шумный латент с предыдущего шага x_t -> x_t-1
                latents = model.scheduler.step(
                    noise_pred, 
                    t, 
                    latents, 
                )
                
                # максикрование шума 
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



        return StableDiffusionMultitaskPipelineOutput(
            latents=latents,
            generated_images=(
                model.vae.decode(latents)
                if _denoising_end is None else
                None
            ),
            denoising_start=(
                None
                if _denoising_end is None else
                _denoising_end
            ),
            encoded_image=encoded_image,
            encoded_prompt=encoded_prompt,
            negative_encoded_prompt=negative_encoded_prompt,
        )
