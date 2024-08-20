import torch

from tqdm.notebook import tqdm
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# from ....models.models.models.denoiser_model import DenoiserModel



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
        predicted_noise: torch.Tensor,
        timestep: int,
        latents: torch.Tensor,
        **kwargs,
    ):
        self.noise = predicted_noise
        self.timestep = timestep
        

    # TODO: Подумать как реализовать 
    def __call__(
        self,
        **kwargs,
    ):
        pass



# TODO: Надо будет подумать как тут можн аккуратненько всё сделать
class DenoiserPipeline:
    """
    Пайплайн для использования модели предсказания шума
    Получает на вход все параметры, которые удалось собрать слоями выше
    и делает предсказание шума, исходя из этих параметров
    """
    def __call__(
        self,
        # denoiser: DenoiserModel,
        denoiser,
        # Unconditional params
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        # Conditional params
        encoder_hidden_states: torch.Tensor,
        # Optional params
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):     
        # TODO: Не оч понятно, куда логичнее впихнуть эту поебту
        # но кажется лучше оставить тут, потому что для обучения будет свой пайплайн
        if denoiser.is_inpainting_model:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)                        

        # Получаем предсказание шума моделью 
        # На этом этапе уже должен быть использован рефайнер в случае чего (!)
        noise_pred = denoiser.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states,
            timestep_cond=None,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        )
        
        return noise_pred







