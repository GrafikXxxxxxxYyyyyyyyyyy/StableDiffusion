import torch
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Dict, List, Optional, Union, Tuple
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput

from ..models.vae_model import StableDiffusionVaeModel



########################################################################################################
# Encoder
########################################################################################################
@dataclass
class StableDiffusionVaeEncoderPipelineInput(BaseOutput):
    image: Optional[PipelineImageInput] = None
    mask_image: Optional[PipelineImageInput] = None
    height: Optional[int] = None
    width: Optional[int] = None
    generator: Optional[torch.Generator] = None



class StableDiffusionVaeEncoderPipelineOutput:
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
                (batch_size * num_images_per_prompt) // image_latents.shape[0], 1, 1, 1
            ).to(device=device, dtype=dtype)
        )

        mask = (
            None
            if self.mask is None else
            self.mask.repeat(
                (batch_size * num_images_per_prompt) // mask.shape[0], 1, 1, 1
            ).to(device=device, dtype=dtype)
        )
        
        masked_image_latents = (
            None   
            if self.masked_image_latents is None else
            self.masked_image_latents.repeat(
                (batch_size * num_images_per_prompt) // masked_image_latents.shape[0], 1, 1, 1
            ).to(device=device, dtype=dtype)
        )

        return (image_latents, mask, masked_image_latents)



class StableDiffusionVaeEncoderPipeline:
    def __call__(
        self,
        vae: StableDiffusionVaeModel,
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[PipelineImageInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """
        Кодирует картинку и маску в латентное представление, если те переданы
        """        
        output_kwargs: dict
        if image is not None:
            # process and encode image
            image = self.image_processor.preprocess(image)
            image = image.to(
                device=self.device, dtype=self.dtype
            )

            if height and width:
                # resize if sizes provided
                image = torch.nn.functional.interpolate(
                    image, size=(height, width)
                )
            output_kwargs["image_latents"] = self._encode(
                image, generator
            )

            if mask_image is not None:
                mask = self.mask_processor.preprocess(mask_image)
                mask = mask.to(
                    device=self.device, dtype=self.dtype
                )

                if height and width:
                    # resize if sizes provided
                    mask = torch.nn.functional.interpolate(
                        mask, size=(height, width)
                    )
                masked_image = image * (mask < 0.5)
                output_kwargs["masked_image_latents"] = self.encode(
                    masked_image, generator
                )
                output_kwargs["mask"] = torch.nn.functional.interpolate(
                    mask, 
                    size=(
                        height // vae.scale_factor, 
                        width // vae.scale_factor
                    )
                )
        
        return StableDiffusionVaeEncoderPipelineOutput(**output_kwargs)
########################################################################################################






########################################################################################################
# Decoder
########################################################################################################
@dataclass
class StableDiffusionVaeDecoderPipelineInput(BaseOutput):
    pass



class StableDiffusionVaeDecoderPipelineOutput:
    pass



class StableDiffusionVaeDecoderPipeline:  
    pass
########################################################################################################