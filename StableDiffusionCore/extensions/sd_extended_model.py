import torch

from StableDiffusionCore.sd_unified_model import StableDiffusionUnifiedModel

from typing import Any, Callable, Dict, List, Optional, Union, Tuple



class StableDiffusionExtendedModel(StableDiffusionUnifiedModel):
    def __init__(
        self, 
    ):
        """
        По идее должна хранить в себе основную модель и все вспомогательные модели 
        Изменяет оригинальный метод __call__
        """
        
    pass