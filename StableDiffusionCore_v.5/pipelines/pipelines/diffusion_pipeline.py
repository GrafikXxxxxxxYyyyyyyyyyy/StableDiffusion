import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ...models.models.diffusion_model import DiffusionModel



class DiffusionPipelineInput(BaseOutput):
    strength: float = 1.0
    num_inference_steps: int = 30
    num_images_per_prompt: int = 1
    denoising_end: Optional[float] = None
    denoising_start: Optional[float] = None



class DiffusionPipelineOutput:
    def __init__(
        self, 
        **kwargs,
    ):
        pass



class DiffusionPipeline:
    """
    Данный пайплайн нужен для того, чтобы просто провести процесс диффузии
    А именно выполняется следующий набор шагов
        1. Подготавливаются временные шаги
        2. Подготавливаются шумовые данные
        3. Выполняется backward дифузионный процесс
    """
    def __call__(
        self, 
        model: DiffusionModel,
        batch_size: int = 1,
        strength: float = 1.0,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None, 
        # TODO: Изображения и маски могут использоваться и без латентных моделей
            # image: PipelineImageInput = None,
            # mask_image: PipelineImageInput = None,
        # TODO: нужно добавить сюда ещё параметры, пришедшие с других блоков 
            # vae_output: Optional[VaePipelineOutput] = None,
            # conditioner_output: Optional[ConditionerPipelineOutput] = None,
        **kwargs,
    ):
        if "0. Подготавливаем константы":
            device = model.device

        timesteps: torch.Tensor
        initial_timestep: torch.Tensor
        if "1. Подготавливаем шаги расшумления":
            timesteps, num_inference_steps = model.retrieve_timesteps(
                num_inference_steps,
                strength,
                denoising_end,
                denoising_start,
                device,
            )
            initial_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)


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

                # Поэтому просто присваиваем соот-м значениям их латентные представления
                image = image_latents
                image_mask = mask
                masked_image = masked_image_latents

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


