import torch

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)


def denoising_value_valid(dnv):
    return isinstance(dnv, float) and 0 < dnv < 1



class StableDiffusionSchedulerModel():
    def __init__(
        self,
        model_path: str,
        scheduler_name: Optional[str] = None,
    ):
        if scheduler_name is None:
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        else:
            if scheduler_name == "DDIM":
                self.scheduler = DDIMScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler'
                )
            elif scheduler_name == "euler":
                self.scheduler = EulerDiscreteScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler'
                )
            elif scheduler_name == "euler_a":
                self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler'
                )
            elif scheduler_name == "DPM++ 2M":
                self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler'
                )
            elif scheduler_name == "DPM++ 2M Karras":
                self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler',
                    use_karras_sigmas=True,
                )
            elif scheduler_name == "DPM++ 2M SDE Karras":
                self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler',
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++"
                )
            elif scheduler_name == "PNDM":
                self.scheduler = PNDMScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler'
                )
            elif scheduler_name == "uni_pc":
                self.scheduler = UniPCMultistepScheduler.from_pretrained(
                    model_path,
                    subfolder='scheduler'
                )
            else:
                raise ValueError(f'Unknown scheduler name: {scheduler_name}')
            
            print(f"'{scheduler_name}' scheduler has successfully initialized")



    @property
    def config(self):
        return self.scheduler.config
    
    @property
    def init_noise_sigma(self):
        return self.scheduler.init_noise_sigma


    def prepare_timesteps(
        self, 
        num_inference_steps: int, 
        strength: float = 1.0, 
        device: Union[str, torch.device] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
    ):
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
        )
        
        if denoising_start and denoising_value_valid(denoising_start):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            # if the scheduler is a 2nd order scheduler we might have to do +1
            # because `num_inference_steps` might be even given that every timestep
            # (except the highest one) is duplicated. If `num_inference_steps` is even it would
            # mean that we cut the timesteps in the middle of the denoising step
            # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
            # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]

        if (
            denoising_end
            and denoising_start
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        return timesteps, num_inference_steps


    def get_timesteps(
        self, 
        num_inference_steps: int, 
        strength: float = 1.0, 
        device: Union[str, torch.device] = None,
    ):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, len(timesteps)

    # # TODO: Протестировать
    # def apply_denosing_start(
    #     self, 
    #     timesteps: List[int],
    #     denoising_start: Optional[float] = None,
    # ):
    #     if denoising_start and denoising_value_valid(denoising_start):
    #         discrete_timestep_cutoff = int(
    #             round(
    #                 self.scheduler.config.num_train_timesteps
    #                 - (denoising_start * self.scheduler.config.num_train_timesteps)
    #             )
    #         )

    #         num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
    #         # if the scheduler is a 2nd order scheduler we might have to do +1
    #         # because `num_inference_steps` might be even given that every timestep
    #         # (except the highest one) is duplicated. If `num_inference_steps` is even it would
    #         # mean that we cut the timesteps in the middle of the denoising step
    #         # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
    #         # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
    #         if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
    #             num_inference_steps = num_inference_steps + 1

    #         # because t_n+1 >= t_n, we slice the timesteps starting from the end
    #         timesteps = timesteps[-num_inference_steps:]

    #         return timesteps, num_inference_steps
        
    #     return timesteps, len(timesteps)

    # # TODO: Протестировать    
    # def apply_denosing_end(
    #     self, 
    #     timesteps: List[int],
    #     denoising_start: Optional[float] = None,
    #     denoising_end: Optional[float] = None,
    # ):
    #     if (
    #         denoising_end
    #         and denoising_start
    #         and denoising_value_valid(denoising_end)
    #         and denoising_value_valid(denoising_start)
    #         and denoising_start >= denoising_end
    #     ):
    #         raise ValueError(
    #             f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
    #             + f" {denoising_end} when using type float."
    #         )
    #     elif denoising_end and denoising_value_valid(denoising_end):
    #         discrete_timestep_cutoff = int(
    #             round(
    #                 self.scheduler.config.num_train_timesteps
    #                 - (denoising_end * self.scheduler.config.num_train_timesteps)
    #             )
    #         )
    #         num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
    #         timesteps = timesteps[:num_inference_steps]

    #         return timesteps, num_inference_steps

    #     return timesteps, len(timesteps)


    def add_noise(
        self, 
        latents: torch.FloatTensor, 
        noise: torch.FloatTensor, 
        timestep: int
    ):
        return self.scheduler.add_noise(latents, noise, timestep)
    

    def scale_model_input(
        self,
        latents: torch.FloatTensor,
        timestep: int,
    ):
        return self.scheduler.scale_model_input(latents, timestep)
    

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
    ):
        return self.scheduler.step(model_output, timestep, sample, return_dict=False)[0]
    


