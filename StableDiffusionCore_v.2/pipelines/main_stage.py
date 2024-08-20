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
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ..sd_unified_model import StableDiffusionModel
# from .pre_stage import (
#     StableDiffusionExtractionPipelineOutput, 
#     StableDiffusionExtractionPipelineInput,
#     StableDiffusionTextEncoderOutput,
#     StableDiffusionTextEncoderInput, 
#     StableDiffusionVAEInput
# )


########################################################################################################################
class StableDiffusionMultitaskPipelineOutput:
    def __init__(
        self,
        refiner_steps: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        generated_images: Optional[torch.Tensor] = None,
        extractor_output: Optional[StableDiffusionExtractionPipelineOutput] = None,
    ):
        """
        Docstring
        """
        self.latents = latents
        self.refiner_steps = refiner_steps
        self.generated_images = generated_images
        self.extractor_output = extractor_output



    def __call__(
        self,
    ):
        """
        Docstring
        """
        pass
########################################################################################################################




########################################################################################################################
@dataclass
class StableDiffusionMultitaskPipelineInput(BaseOutput):
    """
    Docstring
    """
    extractor_input: Optional[StableDiffusionExtractionPipelineInput] = None

    # pipeline config
    strength: float = 1.0
    output_type: str = "pt"
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    num_images_per_prompt: int = 1
    denoising_end: Optional[float] = None
    denoising_start: Optional[float] = None

    # transposable elements
    return_extractor_out: bool = False
    prev_output: Optional[StableDiffusionMultitaskPipelineOutput] = None
########################################################################################################################




class StableDiffusionMultitaskPipeline:
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
        strength: float = 1.0,
        output_type: str = "pt",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None,
        return_extractor_out: bool = False,
        extractor_input: Optional[StableDiffusionExtractionPipelineInput] = None,
        prev_output: Optional[StableDiffusionMultitaskPipelineOutput] = None,
        **kwargs,
    ) -> StableDiffusionMultitaskPipelineOutput:  
        """
        Docstring
        """
        if "1. Prepare constants":
            is_strength_max = strength == 1.0
            num_channels_latents = model.vae.config.latent_channels
            num_channels_unet = model.unet.config.in_channels


        if "2. Encodes or get encoded prompt embeddings":
            batch_size: int
            prompt_embeds: torch.Tensor
            pooled_prompt_embeds: torch.Tensor
            encoded_prompt: Optional[StableDiffusionTextEncoderOutput]
            # if extractor_output provided
            if prev_output is not None: # collect data from it
                prompt_embeds, pooled_prompt_embeds = prev_output.extractor_output.encoded_prompt(
                    model.type,
                    model.use_refiner, 
                    num_images_per_prompt,
                )
            else: # create new embeddings arguments
                encoded_prompt = model.text_encoder(**extractor_input.text_encoder_input)
                prompt_embeds, pooled_prompt_embeds = encoded_prompt(
                    model.type,
                    model.use_refiner, 
                    num_images_per_prompt,
                )
            batch_size = prompt_embeds.shape[0]

            # classifier-free guidance
            if self.do_cfg:
                negative_prompt_embeds: torch.Tensor
                negative_pooled_prompt_embeds: torch.Tensor
                negative_encoded_prompt: Optional[StableDiffusionTextEncoderOutput]
                if extractor_output is not None:
                    # если нет негативного промпта, кодируем и добавляем его 
                    if extractor_output.negative_encoded_prompt is None:
                        negative_te_input = StableDiffusionTextEncoderInput(**extractor_input.text_encoder_input)
                        negative_te_input.prompt = extractor_input.negative_prompt
                        negative_te_input.prompt_2 = extractor_input.negative_prompt_2

                        extractor_output.negative_encoded_prompt = model.text_encoder(**negative_te_input)

                    negative_prompt_embeds, negative_pooled_prompt_embeds = extractor_output.negative_encoded_prompt(
                        model.type,
                        model.use_refiner, 
                        num_images_per_prompt,
                    )            
                else:
                    negative_te_input = StableDiffusionTextEncoderInput(**extractor_input.text_encoder_input)
                    negative_te_input.prompt = extractor_input.negative_prompt
                    negative_te_input.prompt_2 = extractor_input.negative_prompt_2
                    negative_encoded_prompt = model.text_encoder(**negative_te_input)

                    negative_prompt_embeds, negative_pooled_prompt_embeds = negative_encoded_prompt(
                        model.type,
                        model.use_refiner, 
                        num_images_per_prompt,
                    )            

                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            ############################################################################################################
            print(prompt_embeds.shape)
            print(pooled_prompt_embeds.shape)


        if "3. Get timesteps":
            timesteps, num_inference_steps = model.scheduler.prepare_timesteps(
                num_inference_steps,
                strength,
                self.device,
                denoising_start,
                denoising_end,
            )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)


        if "4. Encodes or get encoded image and mask latents":
            image_latents: Optional[torch.Tensor]
            mask: Optional[torch.Tensor]
            masked_image_latents: Optional[torch.Tensor]
            if transposon is not None and transposon.encoded_image is not None:
                image_latents, mask, masked_image_latents = transposon.encoded_image(
                    num_images_per_prompt,
                    device,
                    dtype,
                )

            generator = vae_input.generator

            task: str
            latents: torch.Tensor
            if image_latents is None and masked_image_latents is None: # txt2img
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
            elif masked_image_latents is None: # img2img
                task = "img2img"
                # sample noise 
                noise = randn_tensor(
                    shape=image_latents.shape, 
                    generator=generator, 
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
                    generator=generator, 
                    device=self.device, 
                    dtype=prompt_embeds.dtype
                )
                # add noise to latents
                latents = (
                    noise * model.scheduler.init_noise_sigma
                    if is_strength_max else
                    model.scheduler.add_noise(image_latents, noise, latent_timestep)
                )
                if self.do_cfg:
                    mask = torch.cat([mask] * 2)
                    masked_image_latents = torch.cat([masked_image_latents] * 2) 
                # aligning device to prevent device errors when concating it with the latent model input
                image_latents = image_latents.to(device=self.device, dtype=prompt_embeds.dtype)
                mask = mask.to(device=self.device, dtype=prompt_embeds.dtype)
                masked_image_latents = masked_image_latents.to(device=self.device, dtype=prompt_embeds.dtype)
            ###########################################################################################################

        # TODO: Доделать и перенести в экстрактор
        if "5. Prepare additional arguments":
            added_cond_kwargs = None
            
            if model.type == "sdxl":
                # time ids
                add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                    original_size = (height, width),
                    crops_coords_top_left = (0, 0),
                    aesthetic_score = aesthetic_score,
                    negative_aesthetic_score = negative_aesthetic_score,
                    target_size = (height, width),
                    negative_original_size = (height, width),
                    negative_crops_coords_top_left = (0, 0),
                    negative_target_size = (height, width),
                    addition_time_embed_dim = model.unet.config.addition_time_embed_dim,
                    expected_add_embed_dim = model.unet.add_embed_dim,
                    dtype = prompt_embeds.dtype,
                    text_encoder_projection_dim = model.text_encoder.text_encoder_projection_dim,
                    requires_aesthetics_score = model.use_refiner,
                )
                add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)

                if self.do_cfg:
                    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds.to(self.device), 
                    "time_ids": add_time_ids.to(self.device),
                }
            elif model.type == "sd3":
                raise ValueError(f"Model  type '{model.type}' cannot be used!")


        if "6. Denoising loop":
            cross_attention_kwargs = (
                None
                if te_input.lora_scale is None else
                {"scale": te_input.lora_scale}
            )

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

    
        # TODO: Переделать возвращение результатов, кажется что не нужно учитывать выходной тип
        # и можно определять что был использован рефайнер автоматически
        # if output_type == "pt":
        #     images = model.vae.decode(latents)

        #     return StableDiffusionMultitaskPipelineOutput(
        #         generated_images=images,
        #         latents=latents,
        #     )
        # elif output_type == "latents":
        #     # если запрос на латенты, значит работа в режиме refiner 
        #     # поэтому выполняется чуть более сложная логика
        #     current_denoising_end = transposon.denoising_end
            
            
        #     return StableDiffusionMultitaskPipelineOutput(
        #         latents=latents,
        #         denoising_start=current_denoising_end
        #     )
        # else:
        #     raise ValueError(f"Unknown output_type = '{output_type}'")
        
        return StableDiffusionMultitaskPipelineOutput()
    














@dataclass
class StableDiffusionMultitaskPipelineInput(BaseOutput):
    # extractor_input: Optional[StableDiffusionExtractionPipelineInput] = None

    strength: float = 1.0
    output_type: str = "pt"
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    num_images_per_prompt: int = 1
    return_extractor_out: bool = False
    denoising_end: Optional[float] = None
    denoising_start: Optional[float] = None

    # transposable elements
    vae_input: Optional[StableDiffusionVAEInput] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    text_encoder_input: Optional[StableDiffusionTextEncoderInput] = None
    prev_output: Optional[StableDiffusionMultitaskPipelineOutput] = None




@dataclass
class StableDiffusionExtractionPipelineInput(BaseOutput):
    vae_input: Optional[StableDiffusionVAEInput] = None
    text_encoder_input: Optional[StableDiffusionTextEncoderInput] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None