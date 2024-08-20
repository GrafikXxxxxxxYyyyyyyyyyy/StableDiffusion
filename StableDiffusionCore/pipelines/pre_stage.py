# import PIL
# import time
# import torch

# from PIL import Image
# from tqdm.notebook import tqdm
# from dataclasses import dataclass
# from diffusers.image_processor import PipelineImageInput
# from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# from StableDiffusionCore.sd_unified_model import StableDiffusionUnifiedModel
# # from StableDiffusionCore.models.sd_vae import StableDiffusionVAELatentsHolder
# # from StableDiffusionCore.models.sd_text_encoder import StableDiffusionTextEmbeddingsHolder
# from StableDiffusionCore.models.sd_text_encoder import (
#     StableDiffusionTextEncoderInput, 
#     StableDiffusionTextEncoderOutput
# )
# from StableDiffusionCore.models.sd_vae import (
#     StableDiffusionVAEInput,
#     StableDiffusionVAEOutput
# )



# @dataclass
# class StableDiffusionExtractionPipelineInput(
#     StableDiffusionTextEncoderInput, 
#     StableDiffusionVAEInput
# ):
#     # TextEncoder
#     prompt: Optional[Union[str, List[str]]]
#     prompt_2: Optional[Union[str, List[str]]]
#     negative_prompt: Optional[Union[str, List[str]]]
#     negative_prompt_2: Optional[Union[str, List[str]]]
#     cross_attention_kwargs: Optional[Dict[str, Any]]
#     clip_skip: Optional[int]

#     # VAE
#     height: Optional[int] = None,
#     width: Optional[int] = None,
#     image: Optional[PipelineImageInput] = None, # TODO: Переделать в PIL
#     mask_image: Optional[PipelineImageInput] = None,
#     generator: Optional[torch.Generator] = None,     



# class StableDiffusionExtractionPipelineOutput(
#     StableDiffusionTextEncoderOutput,
#     StableDiffusionVAEOutput
# ):
    



# class StableDiffusionExtractionPipeline:
#     def __init__(
#         self, 
#         do_cfg: bool = True,
#         device: Optional[str] = None,
#     ):
#         self.do_cfg = False
#         if do_cfg:
#             self.do_cfg = True

#         self.device = torch.device("cpu")
#         if device:
#             self.device = torch.device(device) 



#     def __call__(
#         self,
#         model: StableDiffusionUnifiedModel,
#         # TextEncoder
#         prompt: Optional[Union[str, List[str]]] = None,
#         prompt_2: Optional[Union[str, List[str]]] = None,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         negative_prompt_2: Optional[Union[str, List[str]]] = None,
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#         clip_skip: Optional[int] = None,
#         # VAE
#         height: Optional[int] = None,
#         width: Optional[int] = None,
#         image: Optional[PipelineImageInput] = None, # TODO: Переделать в PIL
#         mask_image: Optional[PipelineImageInput] = None,
#         generator: Optional[torch.Generator] = None, 
    
#     ) -> StableDiffusionPipelineOutput:
#         """
#         Docstring
#         """
#         output_kwargs = {}

#         # prepare constants 
#         batch_size = 1
#         if prompt is not None and isinstance(prompt, list):
#             batch_size = len(prompt)
#         height = height or model.unet.config.sample_size * model.vae.scale_factor
#         width = width or model.unet.config.sample_size * model.vae.scale_factor
    
#         # encode input prompt
#         output_kwargs["embeddings_holder"] = model.text_encoder(
#             prompt,
#             prompt_2,
#             batch_size,
#             lora_scale=None,
#             clip_skip=clip_skip,
#         )
#         if self.do_cfg:
#             output_kwargs["negative_embeddings_holder"] = model.text_encoder(
#                 negative_prompt,
#                 negative_prompt_2,
#                 batch_size,
#                 lora_scale=None,
#                 clip_skip=clip_skip,
#             )

#         # encode images and masks
#         output_kwargs["vae_latents_holder"] = model.vae(
#             image, 
#             mask_image,
#             height,
#             width,
#             generator
#         )

#         return StableDiffusionPipelineOutput(**output_kwargs)

