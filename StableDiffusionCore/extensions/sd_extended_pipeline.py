import PIL
import time
import PIL.Image
import torch
import inspect
import itertools 
import numpy as np
import torch.nn.functional as F

from PIL import Image
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Tuple

from StableDiffusionCore.sd_unified_pipeline import StableDiffusionUnifiedPipeline
from StableDiffusionCore.extensions.sd_extended_model import StableDiffusionExtendedModel
from StableDiffusionCore.extensions.models.detailer_model import StableDiffusionDetailerModel
from StableDiffusionCore.extensions.pipelines.detailer_pipeline import StableDiffusionDetailerPipeline




class StableDiffusionExtendedPipeline():
    def __init__(
        self, 
        controlnets: Optional[List[dict]] = None,
        detailers: Optional[List[dict]] = None,
        do_cfg: bool = True,
        device: Optional[str] = None,
    ):  
        self.device = torch.device("cpu")
        if device:
            self.device = torch.device(device)  
        
        self.pre_stage: List[dict]
        # if controlnets and isinstance(controlnets, list):
        #     self.pre_stage = [

        #     ]

        self.main_stage = StableDiffusionUnifiedPipeline(
            do_cfg=do_cfg,
            device=self.device,
        )

        self.post_stage: List[Tuple[StableDiffusionDetailerModel, StableDiffusionDetailerPipeline, dict]]
        if detailers and isinstance(detailers, list):
            self.post_stage = [
                (
                    StableDiffusionDetailerModel(**detailers[key].pop("model")),
                    StableDiffusionDetailerPipeline(**detailers[key].pop("pipeline")),
                    {**detailers[key]},
                )
                for key
                in detailers
            ]

        self.model: StableDiffusionExtendedModel = None
        


    def __call__(
        self, 
        model: StableDiffusionExtendedModel, 
        refiner: Optional[str] = None,
        **kwargs
    ) -> List[PIL.Image.Image]:
        if model:
            self.model = model

        # Всякие там контрлнеты и прочая подготовительная хуйня
        if self.pre_stage:
            pass
        
        # Основная генерация картинки
        images = self.main_stage(self.model, refiner, **kwargs)

        # Всякая постобработка генерации
        if self.post_stage:
            for (detailer, pipeline, new_kwargs) in self.post_stage:
                kwargs = kwargs | new_kwargs

                images = pipeline(images, detailer, **kwargs)

        return images

       