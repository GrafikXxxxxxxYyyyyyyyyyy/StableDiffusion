import torch
from PIL import Image
from tqdm.notebook import tqdm
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .stable_diffusion_model import StableDiffusionModel
from .pipelines.conditioner_pipeline import (
    ConditionerPipeline, 
    ConditionerPipelineInput, 
    ConditionerPipelineOutput
)
from .pipelines.latent_diffusion_pipeline import (
    LatentDiffusionPipeline,
    LatentDiffusionPipelineInput, 
    LatentDiffusionPipelineOutput,
)



# /////////////////////////////////////////////////////////////////////////////////////// #
class StableDiffusionPipelineInput(BaseOutput):
    # ConditionerPipeline input 
    conditioner_input: Optional[ConditionerPipelineInput]
    ldm_input: Optional[LatentDiffusionPipelineInput]
# /////////////////////////////////////////////////////////////////////////////////////// #



class StableDiffusionPipelineOutput:
    def __init__(
        self,
    ):
        pass



class StableDiffusionPipeline:
    def __call__(
        self,
        model: StableDiffusionModel,
        do_cfg: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        ldm_input: Optional[LatentDiffusionPipelineInput] = None,
        conditioner_input: Optional[ConditionerPipelineInput] = None,
        **kwargs,
    ):
        batch_size: int
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: Optional[torch.Tensor]
        if "ConditionerPipeline":
            # TODO: По идее нужно сделать отдельный метод 
            # вместе с колл и унаследоваться от этого пайплайна
            CONDITIONER_PIPELINE = ConditionerPipeline()
            CONDITIONER_OUTPUT = CONDITIONER_PIPELINE(
                model=model.conditioner,
                **conditioner_input, 
            )
            # TODO: Добавить в LDM параметр key
            prompt_embeds, pooled_prompt_embeds = CONDITIONER_OUTPUT(**model.key)

        # if do_cfg:
        #     prompt_embeds: torch.Tensor
        #     pooled_prompt_embeds: Optional[torch.Tensor] 

        latents: torch.Tensor
        if "LatentDiffusionPipeline":
            LDM_PIPELINE = LatentDiffusionPipeline()
            LDM_OUTPUT = LDM_PIPELINE(model.ldm, **ldm_input)


        return StableDiffusionPipelineOutput()
        


