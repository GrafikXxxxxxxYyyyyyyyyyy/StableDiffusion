import torch

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .sd_unified_model import StableDiffusionModel
from .pipelines.main_stage import (
    StableDiffusionMultitaskPipelineInput, 
    StableDiffusionMultitaskPipelineOutput,
    StableDiffusionExtractionPipelineInput,
    StableDiffusionExtractionPipelineOutput,
)



class StableDiffusionPipelineInput(
    StableDiffusionMultitaskPipelineInput,
    StableDiffusionExtractionPipelineInput,
):
    pass



class StableDiffusionPipelineOutput:
    pass



class StableDiffusionPipeline:
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



    def __call__(
        self, 
        model: StableDiffusionModel, 
        input_kwargs: StableDiffusionPipelineInput,
        **kwargs,
    ) -> StableDiffusionPipelineOutput:
        pass       
    



    