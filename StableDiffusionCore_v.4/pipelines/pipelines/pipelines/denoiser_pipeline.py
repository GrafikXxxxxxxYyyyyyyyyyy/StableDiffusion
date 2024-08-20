import torch

from tqdm.notebook import tqdm
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ....models.models.models.denoiser_model import DenoiserModel



class DenoiserPipelineInput(BaseOutput):
    latents: torch.FloatTensor
    timestep: int
    encoder_hidden_states: torch.FloatTensor
    timestep_cond: Optional[torch.FloatTensor] = None
    added_cond_kwargs: Optional[Dict[str, Any]] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None



class DenoiserPipelineOutput:
    def __init__(
        self,
    ):
        pass



class DenoiserPipeline:
    """
    Пайплайн для использования модели предсказания шума
    Цель: 
        Получить на вход шумный латент + временной шаг + условие, а на 
        выходе вернуть тензор шума, который нужно снять с текущего шага

    Оговорки: 
        Поскольку я не до конца понимаю ща че ваще я конкретно хочу построить 
        то пускай в методе кол будут передаваться ваще все возможные параметры,
        из которых можно получить предсказание шума (я ебу а то как оно ваще)
    """
    def __call__(
        self,
        denoiser: DenoiserModel,
        latents: torch.FloatTensor,
        timestep: int,
        encoder_hidden_states: torch.FloatTensor,
        timestep_cond: Optional[torch.FloatTensor] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):     
        # В случае inpaint объединяем латенты
        if denoiser.is_inpainting_model:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)                        

        # Получаем предсказание шума моделью 
        # На этом этапе уже должен быть использован рефайнер в случае чего
        noise_pred = denoiser.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states,
            timestep_cond=None,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        )
        
        return noise_pred





















