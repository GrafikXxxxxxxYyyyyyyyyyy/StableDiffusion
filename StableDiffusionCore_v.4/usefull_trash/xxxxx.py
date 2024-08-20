import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


from diffusers.utils import BaseOutput


########################################################################
# Это по сути просто вход UNet модели
########################################################################
class DenoiserPipelineInput(BaseOutput):
    latents: torch.FloatTensor
    timestep: int
    encoder_hidden_states: torch.FloatTensor
    timestep_cond: Optional[torch.FloatTensor] = None
    added_cond_kwargs: Optional[Dict[str, Any]] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
########################################################################



########################################################################
class DiffusionPipelineInput(BaseOutput):
    # Чистые входы
    do_cfg: bool = True
    strength: float = 1.0
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    num_images_per_prompt: int = 1
    denoising_end: Optional[float] = None,
    denoising_start: Optional[float] = None,
    # Выходы прошлых блоков
    vae_output: Optional[VaePipelineOutput] = None
    conditioner_output: Optional[ConditionerPipelineOutput] = None


class VaePipelineInput(BaseOutput):
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    mask_image: Optional[PipelineImageInput] = None
########################################################################



########################################################################
class ConditionerPipelineInput(BaseOutput):
    clip_skip: Optional[int] = None
    lora_scale: Optional[float] = None
    prompt: Optional[Union[str, List[str]]] = None
    prompt_2: Optional[Union[str, List[str]]] = None,


class LatentDiffusionPipelineInput(BaseOutput):
    # Чистые входы
    output_type: str = "pt"
    vae_input: Optional[VaePipelineInput] = None
    diffusion_input: Optional[DiffusionPipelineInput] = None
    # Выходы прошлых блоков
    conditioner_output: Optional[ConditionerPipelineOutput] = None
########################################################################



########################################################################
class StableDiffusionPipelineInput(BaseOutput):
    do_cfg: bool = True
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    conditioner_input: Optional[ConditionerPipelineInput]
    ldm_input: Optional[LatentDiffusionPipelineInput]
########################################################################
