import torch
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ...models.models.scheduler_model import StableDiffusionSchedulerModel



class StableDiffusionSchedulerPipelineInput(BaseOutput):
    noise: torch.FloatTensor
    sample: torch.FloatTensor
    timestep: Union[float, torch.FloatTensor]


    
class StableDiffusionSchedulerPipelineOutput:
    def __init__(
        self,
    ):
        pass



class StableDiffusionSchedulerPipeline:
    def __call__(
        self,
        scheduler: StableDiffusionSchedulerModel,
        noise: torch.FloatTensor,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        **kwargs,
    ):  
        """
        Проницаемый пайплайн, который может работать в обоих направлениях
        диффузионного процесса:
            1. Forward process: (x_0, t, ∑) -> x_t
            2. Backward process: x_t -> x_∆t
        """
        return scheduler.add_noise(sample, noise, timestep)
        return scheduler.step(noise, timestep, sample, return_dict=False)[0]