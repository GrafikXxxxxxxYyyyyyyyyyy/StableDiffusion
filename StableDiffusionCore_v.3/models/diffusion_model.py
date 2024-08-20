import os
import torch
import numpy as np

from huggingface_hub import hf_hub_download
from diffusers.utils.peft_utils import delete_adapter_layers
from typing import Any, Callable, Dict, List, Optional, Union

from .models.unet_model import DiffusionUNetModel
from .models.scheduler_model import DiffusionSchedulerModel



class DiffusionModel:
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        scheduler_name: Optional[str] = None,
    ):
        model_type = model_type or "sd15"

        self.denoiser = DiffusionUNetModel(model_path, model_type, device)
        self.scheduler = DiffusionSchedulerModel(model_path, scheduler_name)
        self.to(device)

        # Инициализируем константы
        self.path = model_path
        self.type = model_type
        self.use_refiner = False


    @property
    def device(self):
        return self.denoiser.device

    @property
    def switch_to_base(self):
        self.use_refiner = False

    @property
    def switch_to_refiner(self):
        self.use_refiner = True


    def to(self, device, dtype=None):
        self.denoiser.to(device, dtype)
    


    def __call__(
        self,
    ):
        """
        Метод кол по сути должен выполнять одну полную итерацию из цикла расшумления
        """
        
        pass




