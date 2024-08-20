import torch

from tqdm.notebook import tqdm
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ...models.models.diffusion_model import DiffusionModel
from .pipelines.denoiser_pipeline import DenoiserPipeline
from .vae_pipeline import VaePipelineOutput
from ..conditioner_pipeline import ConditionerPipelineOutput



class DiffusionPipelineInput(BaseOutput):
    # Чистые входы
    do_cfg: bool = True
    strength: float = 1.0
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    num_images_per_prompt: int = 1
    denoising_end: Optional[float] = None,
    denoising_start: Optional[float] = None,
        # image: Optional[PipelineImageInput] = None,
        # mask_image: Optional[PipelineImageInput] = None,
    # Выходы прошлых блоков
    vae_output: Optional[VaePipelineOutput] = None
    conditioner_output: Optional[ConditionerPipelineOutput] = None



class DiffusionPipelineOutput:
    def __init__(
        self,
    ):
        pass

    def __call__(
        self,
    ):
        pass



class DiffusionPipeline:
    """
    Пайплайн для процесса диффузии
    Цель: 
        Накинуть на латенты нужный уровень шума и произвести обратный диффузионный
        процесс расшумления, в результате возвращает расшумленные латенты
    """
    def __call__(
        self,
        model: DiffusionModel,
        # Основные параметры
        do_cfg: bool = True,
        strength: float = 1.0,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None,
        vae_output: Optional[VaePipelineOutput] = None,
        conditioner_output: Optional[ConditionerPipelineOutput] = None,
        **kwargs,
    ):
        """
        Docstring
        """
        if "0. Подготавливаем константы":
            device = model.device

        timesteps: torch.Tensor
        latent_timestep: torch.Tensor
        if "1. Подготавливаем шаги расшумления":
            timesteps, num_inference_steps = model.retrieve_timesteps(
                num_inference_steps,
                strength,
                denoising_end,
                denoising_start,
                device,
            )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)


        task: str
        latents: torch.Tensor
        if "2. Формируем латенты в зависимости от решаемой задачи":
            # Если передан выход VAE, значит латентная модель
            if vae_output is not None:
                image_latents, mask, masked_image_latents = vae_output(
                    batch_size,
                    num_images_per_prompt,
                    device,
                    dtype,
                )

            if image_latents is not None and mask is not None:
                "Inpainting"
            elif mask is None:
                "img2img"
            else:
                "txt2img"


        output_kwargs: dict 
        if "3. Выполняем цикл расшумления":
            # cross_attention_kwargs = (
            #     None
            #     if te_input.lora_scale is None else
            #     {"scale": te_input.lora_scale}
            # )

            # В цикле по шагам
            DENOISER_PIPELINE = DenoiserPipeline()
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if do_cfg else
                    latents
                )
                latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

                # if "Перенести на сторону DenoiserPipeline":
                #     # Объединяем латенты если модель для inpainting
                #     if model.denoiser.is_inpainting_model:
                #         latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                #     # Получаем предсказание шума моделью 
                #     DENOISER_OUTPUT = DENOISER_PIPELINE(
                #         model.denoiser,
                #         latent_model_input,
                #         t,
                #         prompt_embeds,
                #         timestep_cond=None,
                #         cross_attention_kwargs=cross_attention_kwargs,
                #         added_cond_kwargs=added_cond_kwargs,
                #     )
                    

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond

                # Вычисляем шумный латент с предыдущего шага x_t -> x_t-1
                latents = model.scheduler.step(
                    noise_pred, 
                    t, 
                    latents, 
                )
                
                # маскирование шума 
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


        


