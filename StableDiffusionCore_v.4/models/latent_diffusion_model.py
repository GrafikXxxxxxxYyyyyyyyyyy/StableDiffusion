import torch

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .models.vae_model import VaeModel
from .models.diffusion_model import DiffusionModel



class LatentDiffusionModel:
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        device: str = "cuda"
    ):  
        self.vae = VaeModel(
            model_path, 
            model_type, 
            device
        )

        self.diffusion = DiffusionModel(
            model_path, 
            model_type, 
            scheduler_name, 
            device
        )

    # @property
    # def

    
