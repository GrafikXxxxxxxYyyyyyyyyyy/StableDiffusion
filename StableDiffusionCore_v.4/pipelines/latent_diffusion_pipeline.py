import torch

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .conditioner_pipeline import ConditionerPipelineOutput
from .pipelines.vae_pipeline import (
    VaePipeline,
    VaePipelineInput,
    VaePipelineOutput
)
from .pipelines.diffusion_pipeline import (
    DiffusionPipeline,
    DiffusionPipelineInput,
    DiffusionPipelineOutput
)
from ..models.latent_diffusion_model import LatentDiffusionModel



class LatentDiffusionPipelineInput(BaseOutput):
    # Чистые входы
        # output_type: str = "pt"
    vae_input: Optional[VaePipelineInput] = None
    diffusion_input: Optional[DiffusionPipelineInput] = None
    # Выходы прошлых блоков
    conditioner_output: Optional[ConditionerPipelineOutput] = None



class LatentDiffusionPipelineOutput:
    def __init__(
        self,
    ):
        pass



class LatentDiffusionPipeline:
    def __call__(
        self,
        ldm: LatentDiffusionModel,
        vae_input: Optional[VaePipelineInput] = None,
        diffusion_input: Optional[DiffusionPipelineInput] = None,
        # Могут быть переданы условия генерации
        conditioner_output: Optional[ConditionerPipelineOutput] = None,
        **kwargs,
    ):
        """
        Docstring
        """
        mask: Optional[torch.Tensor]
        image_latents: Optional[torch.Tensor]
        masked_image_latents: Optional[torch.Tensor]
        if "VaeEncoderPipeline":
            VAE_PIPELINE = VaePipeline()
            VAE_OUTPUT = VAE_PIPELINE(
                ldm.vae,
                **vae_input,
            )
            
        
        latents: torch.Tensor
        if "DiffusionPipeline":
            DIFFUSION_PIPELINE = DiffusionPipeline()
            DIFFUSION_OUTPUT = DIFFUSION_PIPELINE(
                ldm.diffusion,
                **diffusion_input,
            )


        # TODO: Добавить пайпланы для кодирования и декодирования картинок
        # TODO: Сделать VaePipelineOutput сразу и для энкодера и для декодера
        if "VaeDecoderPipeline":
            pass


        return LatentDiffusionPipelineOutput()