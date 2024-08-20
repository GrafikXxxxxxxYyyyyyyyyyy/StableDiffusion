import torch

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .stable_diffusion_model import StableDiffusionModel
from .pipelines.vae_pipeline import VaePipelineInput


class StableDiffusionPipeline:
    def __call__(
        self,
        vae_input: Optional[VaePipelineInput] = None,
        **kwargs,
    ):
        # 1.
        pass