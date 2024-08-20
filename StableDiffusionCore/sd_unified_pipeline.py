import torch

from dataclasses import dataclass
from collections import OrderedDict
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from StableDiffusionCore.sd_unified_model import StableDiffusionUnifiedModel



##############################################################################################################
# Input args
##############################################################################################################
@dataclass
class StableDiffusionPipelineInput(OrderedDict):
    """
    Docstring
    """
    seed: Optional[int]
    strength: float
    guidance_scale: float
    width: Optional[int]
    height: Optional[int]
    num_inference_steps: int
    num_images_per_prompt: int
    clip_skip: Optional[int]
    denoising_end: Optional[float]
    denoising_start: Optional[float]
    cross_attention_kwargs: Optional[float]

    # text2image
    prompt: Optional[Union[str, List[str]]]
    prompt_2: Optional[Union[str, List[str]]]
    prompt_embeds_1: torch.Tensor
    prompt_embeds_2: Optional[torch.Tensor]
    pooled_prompt_embeds: Optional[torch.Tensor]
    negative_prompt: Optional[Union[str, List[str]]]
    negative_prompt_2: Optional[Union[str, List[str]]]
    negative_prompt_embeds_1: Optional[torch.Tensor]
    negative_prompt_embeds_2: Optional[torch.Tensor]
    negative_pooled_prompt_embeds: Optional[torch.Tensor]

    # image2image
    image: Optional[PipelineImageInput]
    image_latents: Optional[torch.Tensor]

    # inpaint
    mask_image: Optional[PipelineImageInput]
    mask: Optional[torch.Tensor]
    masked_image_latents: Optional[torch.Tensor]
##############################################################################################################




##############################################################################################################
# Output args
##############################################################################################################
@dataclass
class StableDiffusionPipelineOutput(OrderedDict):
    output_type: str
    images: PipelineImageInput
    latents: Optional[torch.FloatTensor]

    # TODO: class StableDiffusionPipelineTransparent
    prompt_embeds_1: torch.Tensor
    prompt_embeds_2: Optional[torch.Tensor]
    pooled_prompt_embeds: Optional[torch.Tensor]
##############################################################################################################





# # TODO: Переделать логику вот этого пайплайна для IPAdapter + Refiner
# class StableDiffusionUnifiedPipeline():
#     def __init__(
#         self, 
#         do_cfg: bool = True,
#         device: Optional[str] = None,
#     ):
#         self.pipeline = StableDiffusionMultitaskPipeline(
#             do_cfg,
#             device,
#         )


#     def __call__(
#         self, 
#         model: StableDiffusionUnifiedModel, 
#         refiner: Optional[str] = None,
#         # ip_adapter_image: Optional[PipelineImageInput] = None,
#         **kwargs
#     ):  
#         pass       
  

    