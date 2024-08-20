import torch

from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from typing import Any, Callable, Dict, List, Optional, Union, Tuple



class DiffusionModel:
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        device: str = "cuda",
    ):     
        # Сначала всегда инитиnся euler
        scheduler_name = scheduler_name or "euler"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder='scheduler'
        )
        # А затем, инициализируется из его конфига любой другой
        self.set_scheduler(scheduler_name)

        # Инициализируется денойзинг модель
        self.denoiser = DenoiserModel(model_path, model_type, device)


    @property
    def device(self):
        return self.denoiser.device


    def set_scheduler(self, scheduler_name):
        if hasattr(self, "scheduler_name") and self.scheduler_name == scheduler_name:
            return
        
        config = self.scheduler.config
        if scheduler_name == "DDIM":
            self.scheduler = DDIMScheduler.from_config(config)
        elif scheduler_name == "euler":
            self.scheduler = EulerDiscreteScheduler.from_config(config)
        elif scheduler_name == "euler_a":
            self.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        elif scheduler_name == "DPM++ 2M":
            self.scheduler = DPMSolverMultistepScheduler.from_config(config)
        elif scheduler_name == "DPM++ 2M Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
        elif scheduler_name == "DPM++ 2M SDE Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_config(
                config, se_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
            )
        elif scheduler_name == "PNDM":
            self.scheduler = PNDMScheduler.from_config(config)
        elif scheduler_name == "uni_pc":
            self.scheduler = UniPCMultistepScheduler.from_config(config)
        else:
            raise ValueError(f'Unknown scheduler name: {scheduler_name}')
        
        self.scheduler_name = scheduler_name
        print(f"Scheduler has successfully changed to '{scheduler_name}'")

    
    def retrieve_timesteps(
        self, 
        num_inference_steps: int, 
        strength: float = 1.0, 
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None,
        device: Union[str, torch.device] = None,
    ):
        # 1. 
            # # get the original timestep using init_timestep
            # init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            # t_start = max(num_inference_steps - init_timestep, 0)

            # self.scheduler.set_timesteps(num_inference_steps, device=device)
            # timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

            # return timesteps, len(timesteps)

        # 2. 
            # if denoising_start and denoising_value_valid(denoising_start):
            #     discrete_timestep_cutoff = int(
            #         round(
            #             self.config.num_train_timesteps
            #             - (denoising_start * self.config.num_train_timesteps)
            #         )
            #     )

            #     num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            #     # if the scheduler is a 2nd order scheduler we might have to do +1
            #     # because `num_inference_steps` might be even given that every timestep
            #     # (except the highest one) is duplicated. If `num_inference_steps` is even it would
            #     # mean that we cut the timesteps in the middle of the denoising step
            #     # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
            #     # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
            #     if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
            #         num_inference_steps = num_inference_steps + 1

            #     # because t_n+1 >= t_n, we slice the timesteps starting from the end
            #     timesteps = timesteps[-num_inference_steps:]

            # if (
            #     denoising_end
            #     and denoising_start
            #     and denoising_value_valid(denoising_end)
            #     and denoising_value_valid(denoising_start)
            #     and denoising_start >= denoising_end
            # ):
            #     raise ValueError(
            #         f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
            #         + f" {denoising_end} when using type float."
            #     )
            # elif denoising_end and denoising_value_valid(denoising_end):
            #     discrete_timestep_cutoff = int(
            #         round(
            #             self.config.num_train_timesteps
            #             - (denoising_end * self.config.num_train_timesteps)
            #         )
            #     )
            #     num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            #     timesteps = timesteps[:num_inference_steps]

            # return timesteps, num_inference_steps

        pass
    
    
    
    

