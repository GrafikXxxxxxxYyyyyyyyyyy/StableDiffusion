import torch
from PIL import Image
from tqdm.notebook import tqdm
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple



# ======================================================================================= #
class StableDiffusionPipelineOutput:
    def __init__(
        self,
        latents: torch.Tensor,
        denoising_start: Optional[float] = None,
        generated_images: Optional[torch.Tensor] = None,
    ):
        self.latents = latents
        self.denoising_start = denoising_start
        self.generated_images = generated_images


    def __call__(
        self,
    ):
        pass
# ======================================================================================= #



# /////////////////////////////////////////////////////////////////////////////////////// #
class StableDiffusionPipelineInput(BaseOutput):
    # TextEncoder input 
    te_input: StableDiffusionTextEncoderPipelineInput
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None

    # Vae input
    vae_input: StableDiffusionVaeEncoderPipelineInput

    # Procedure config
    do_cfg: bool = True
    strength: float = 1.0
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    num_images_per_prompt: int = 1
    denoising_end: Optional[float] = None
# /////////////////////////////////////////////////////////////////////////////////////// #

    
    
# ########################################################################################################## #
class StableDiffusionPipeline:
    def __call__(
        self,
        model: StableDiffusionModel,
        do_cfg: bool = True,
        strength: float = 1.0,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        denoising_end: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        te_input: Optional[StableDiffusionTextEncoderPipelineInput] = None,
        vae_input: Optional[StableDiffusionVaeEncoderPipelineInput] = None,
        **kwargs,
    ):
        # /////////////////////////////////////////////////////////////////////////////////////// #
        if "0. Инициализируем вспомогательные константы":
            device = model.device
            _denoising_end = denoising_end
            _denoising_start = denoising_start
        # /////////////////////////////////////////////////////////////////////////////////////// #


        batch_size: int
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: Optional[torch.Tensor]
        if "# 1. Получаем эмбеддинги промпта":
            prompt_processor = StableDiffusionTextEncoderPipeline()

            te_output = prompt_processor(
                model.text_encoder,
                **te_input,
            )
            prompt_embeds, pooled_prompt_embeds = te_output(
                model.type,
                model.use_refiner,
                num_images_per_prompt,
            )

            batch_size = prompt_embeds.shape[0]

            if do_cfg:
                negative_te_input = StableDiffusionTextEncoderPipelineInput(**te_input)
                negative_te_input.prompt = negative_prompt
                negative_te_input.prompt_2 = negative_prompt_2

                negative_te_output = prompt_processor(
                    model.text_encoder,
                    **negative_te_input,
                )
                negative_prompt_embeds, negative_pooled_prompt_embeds = negative_te_output(
                    model.type,
                    model.use_refiner,
                    num_images_per_prompt,
                )

                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)


        timesteps: torch.Tensor
        latent_timestep: torch.Tensor
        if "2. Подготавливаем шаги расшумления":
            timesteps, num_inference_steps = model.scheduler.prepare_timesteps(
                num_inference_steps,
                strength,
                self.device,
                _denoising_start,
                _denoising_end,
            )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)


        task: str
        latents: torch.Tensor
        if "3. Получаем латенты картинок, если те переданы":
            image_processor = StableDiffusionVaeEncoderPipeline()

            vae_output = image_processor(
                model.vae,
                **vae_input,
            )
            image_latents, mask, masked_image_latents = vae_output(
                batch_size,
                num_images_per_prompt,
            )

            if image_latents is None and masked_image_latents is None: # txt2img
                task = "txt2img"

            elif masked_image_latents is None: # img2img
                task = "img2img"
                
            else: # inpaint
                task = "inpaint"


        added_cond_kwargs: Optional[dict]
        if "4. Подготавливаем дополнительную условную информацию для UNet":
            added_cond_kwargs = None

        
        output_kwargs: dict 
        # ======================================================================================= #
        if "5. Выполняем цикл расшумления":
            cross_attention_kwargs = (
                None
                if te_input.lora_scale is None else
                {"scale": te_input.lora_scale}
            )

            # В цикле по шагам
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if do_cfg else
                    latents
                )
                latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

                if (
                    "# TODO: Перенести на сторону DiffusionPipeline"
                    and ""
                ):
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
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond

                # Вычисляем шумный латент с предыдущего шага x_t -> x_t-1
                # TODO: Перенести в StableDiffusionSchedulerPipeline
                latents = model.scheduler.step(
                    noise_pred, 
                    t, 
                    latents, 
                )
                
                # максикрование шума 
                if task == "inpaint" and model.unet.config.in_channels == 4:
                    init_latents_proper = image_latents
                    if do_cfg:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = model.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents
        # ======================================================================================= #

        return StableDiffusionPipelineOutput(**output_kwargs)


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
# ########################################################################################################## #





        
