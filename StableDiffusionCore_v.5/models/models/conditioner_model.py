import torch

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .models.text_encoder_model import TextEncoderModel
from .models.image_encoder_model import ImageEncoderModel


# "Обуславливатель" не кондиционер(!)
class ConditionerModel:
    """
    Класс представляет собой обёртку над обуславливающими моделями
    TextEncoder and ImageEncoder
    """
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda",
    ):  
        self.text_encoder = TextEncoderModel(model_path, model_type, device)
        # self.image_encoder = ImageEncoderModel(model_path, model_type, device)

        # Инитим константы
        self.path = model_path
        self.type = model_type or "sd15"


    def to(self, device, dtype=None):
        pass


