import torch

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# from ..models.vae_model import VaeModel



class VaePipelineInput(BaseOutput):
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    mask_image: Optional[PipelineImageInput] = None



class VaePipelineOutput:
    def __init__(
        self, 
        image_latents: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        masked_image_latents: Optional[torch.Tensor] = None,
    ):
        self.image_latents = image_latents
        self.mask = mask
        self.masked_image_latents = masked_image_latents


    def __call__(
        self,
        batch_size,
        num_images_per_prompt,
        device=None,
        dtype=None,
    ) -> Tuple[Optional[torch.Tensor]]:
        image_latents = (
            None 
            if self.image_latents is None else
            self.image_latents.repeat(
                (batch_size * num_images_per_prompt) // self.image_latents.shape[0], 1, 1, 1
            ).to(device=device, dtype=dtype)
        )

        mask = (
            None
            if self.mask is None else
            self.mask.repeat(
                (batch_size * num_images_per_prompt) // self.mask.shape[0], 1, 1, 1
            ).to(device=device, dtype=dtype)
        )
        
        masked_image_latents = (
            None   
            if self.masked_image_latents is None else
            self.masked_image_latents.repeat(
                (batch_size * num_images_per_prompt) // self.masked_image_latents.shape[0], 1, 1, 1
            ).to(device=device, dtype=dtype)
        )

        return (image_latents, mask, masked_image_latents)



class VaePipeline:
    def __call__(
        self,
        # vae: VaeModel,
        vae,
        width: Optional[int] = None,
        height: Optional[int] = None,
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[PipelineImageInput] = None,
        **kwargs,
    ):
        """
        Кодирует картинку и маску в латентное представление, если те переданы
        """
        # Тянем генератор из аргументов (если он передан) 
        # поскольку с тем же самым генератором должен работать и TextEncoder
        # кстати для TE тоже генератор нужно будет брать из кваргов
        _generator = kwargs.get("generator", None)

        output_kwargs = {}
        if image is not None:
            image = vae.image_processor.preprocess(image)
            image = image.to(
                device=vae.device, 
                dtype=vae.dtype
            )
            if height and width:
                # resize if sizes provided
                image = torch.nn.functional.interpolate(
                    image, 
                    size=(height, width)
                )
            output_kwargs["image_latents"] = vae.encode(
                image, 
                _generator    
            )

            if mask_image is not None:
                mask = vae.mask_processor.preprocess(mask_image)
                mask = mask.to(
                    device=vae.device, 
                    dtype=vae.dtype
                )
                if height and width:
                    # resize if sizes provided
                    mask = torch.nn.functional.interpolate(
                        mask, 
                        size=(height, width)
                    )
                masked_image = image * (mask < 0.5)
                output_kwargs["masked_image_latents"] = vae.encode(
                    masked_image, 
                    _generator
                )
                output_kwargs["mask"] = torch.nn.functional.interpolate(
                    mask, 
                    size=(
                        height // vae.scale_factor, 
                        width // vae.scale_factor
                    )
                )

        return VaePipelineOutput(**output_kwargs)




