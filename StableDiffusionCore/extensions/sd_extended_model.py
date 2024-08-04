import torch

from StableDiffusionCore.sd_unified_model import StableDiffusionUnifiedModel
from StableDiffusionCore.extensions.models.controlnet_model import StableDiffusionControlNetModel

from typing import Any, Callable, Dict, List, Optional, Union, Tuple



class StableDiffusionExtendedModel(StableDiffusionUnifiedModel):
    def __init__(
        self,
        controlnets: Optional[str] = None,
        # model_path: Optional[str] = None,
        # model_type: Optional[str] = None,
        # model_name: Optional[str] = None,
        # scheduler_name: Optional[str] = None,
        # device: Optional[str] = "cuda"
        **kwargs,
    ):
        """
        По идее должна хранить в себе основную модель и все вспомогательные модели 
        Изменяет оригинальный метод __call__

        Input: 
            "controlnet" ([`ControlNetModel`] or `List[ControlNetModel]`):
                Provides additional conditioning to the `unet` during the denoising process. If you set multiple
                ControlNets as a list, the outputs from each ControlNet are added together to create one combined
                additional conditioning.
        """
        super(StableDiffusionExtendedModel, self).__init__(**kwargs)

        if controlnets:
            self.controlnets = [
                StableDiffusionControlNetModel(model_name=controlnet_name)
                for controlnet_name in controlnets
            ]
