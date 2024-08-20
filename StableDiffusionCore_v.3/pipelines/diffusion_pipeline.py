import torch
from PIL import Image
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ..models.diffusion_model import DiffusionModel



class DiffusionPipelineInput(BaseOutput):
    pass


class DiffusionPipeline:
    def __call__(
        self, 
        diffuser: DiffusionModel,
        
    ):
        pass