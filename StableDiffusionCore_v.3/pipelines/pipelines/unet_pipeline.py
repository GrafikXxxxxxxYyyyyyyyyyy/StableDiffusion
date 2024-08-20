import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# from ...models.models.unet_model import StableDiffusionUNetModel



# class StableDiffusionUNetPipelineInput:
#     pass



# class StableDiffusionUNetPipeline:
#     def __call__(
#         self, 
#         unet: StableDiffusionUNetModel,
#         latents: torch.FloatTensor,
#         timestep: int,
#         encoder_hidden_states: torch.FloatTensor,
#         timestep_cond: Optional[torch.FloatTensor] = None,
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#         added_cond_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> torch.FloatTensor:                
#         return self.unet.__call__(
#             latents,
#             timestep,
#             encoder_hidden_states,
#             timestep_cond=timestep_cond,
#             cross_attention_kwargs=cross_attention_kwargs,
#             added_cond_kwargs=added_cond_kwargs,
#             return_dict=False,
#         )[0]