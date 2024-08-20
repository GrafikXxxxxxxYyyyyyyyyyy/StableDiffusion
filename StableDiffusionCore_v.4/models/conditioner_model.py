import torch

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from typing import Any, Callable, Dict, List, Optional, Union, Tuple



# "Обуславливатель" не кондиционер(!)
class ConditionerModel:
    """
    This class contains optional parts of different 
    stable diffusion's text encoder realisations
    """
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda",
    ):  
        model_type = model_type or "sd15"

        # Инициализируем модели
        if model_type == "sd15":
            # убираем из модели лишние части
            if hasattr(self, "tokenizer_2"):
                delattr(self, "tokenizer_2")
            if hasattr(self, "text_encoder_2"):
                delattr(self, "text_encoder_2")
            
            # инитим нужные
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, 
                subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_path, 
                subfolder="text_encoder", 
                torch_dtype=torch.float16,
                variant='fp16',
                use_safetensors=True
            )
        elif model_type == "sdxl":
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, 
                subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_path, 
                subfolder="text_encoder", 
                torch_dtype=torch.float16,
                variant='fp16',
                use_safetensors=True
            )
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder='tokenizer_2'
            )
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                subfolder='text_encoder_2', 
                torch_dtype=torch.float16,
                variant='fp16',
                use_safetensors=True
            )
        else:
            raise ValueError(f"Unknown model_type '{self.type}'")   
        self.to(device)

        # Инитим константы
        self.path = model_path
        self.type = model_type or "sd15"
        print(f"TextEncoder model has successfully loaded from '{model_path}' checkpoint!")

    @property
    def device(self):
        return self.text_encoder.device

    @property
    def text_encoder_projection_dim(self):
        return (
            self.text_encoder_2.config.projection_dim 
            if hasattr(self, "text_encoder_2") else
            None
        )


    def to(self, device, dtype=None):
        self.text_encoder = self.text_encoder.to(device, dtype=dtype)
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2 = self.text_encoder_2.to(device, dtype=dtype)


    def reload(self, 
        model_type: str,
        model_path: str,
        device: str = "cuda",
    ):
        self.__init__(
            model_path=model_path,
            model_type=model_type, 
            device=self.device,
        )




