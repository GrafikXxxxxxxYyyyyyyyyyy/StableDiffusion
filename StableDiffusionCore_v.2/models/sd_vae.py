import torch

from dataclasses import dataclass
from diffusers import AutoencoderKL
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput



def retrieve_latents(
    encoder_output: torch.Tensor, 
    generator: Optional[torch.Generator] = None, 
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")



##############################################################################################################################
# Base VAE model
##############################################################################################################################
class StableDiffusionVaeModel:
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda"
    ):  
        # Инициализируем модель, можно добавить выбор различных VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_path, 
            subfolder="vae",
            torch_dtype=torch.float16,
            variant='fp16', 
            use_safetensors=True
        )
        self.to(device)

        # Инициализируем функциональные классы
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.scale_factor, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True,
        )

        # Инициализируем ключ модели
        self.type = model_type or "sd15"
        self.path = model_path
        print(f"VAE model has successfully loaded from '{model_path}' checkpoint!")

    @property
    def config(self) -> int:
        return self.vae.config
    
    @property
    def dtype(self) -> type:
        return self.vae.dtype

    @property
    def device(self) -> torch.device:
        return self.vae.device

    @property
    def scale_factor(self) -> int:
        return 2 ** (len(self.config.block_out_channels) - 1)
    

    def to(self, device, dtype=None):
        self.vae.to(device=device, dtype=dtype)

    
    def encode(
        self, 
        image: torch.Tensor, 
        generator: Optional[torch.Generator] = None, 
    ) -> torch.Tensor:  
        """
        По сути просто обёртка над методом .encode() оригинального энкодера, 
        которая делает upcast vae при необходимости
        """
        _dtype = image.dtype

        if self.config.force_upcast:
            image = image.float()
            self.to(dtype=torch.float32)

        latents = retrieve_latents(
            self.vae.encode(image), 
            generator=generator
        )
        latents = latents.to(_dtype)
        latents = self.config.scaling_factor * latents

        if self.config.force_upcast:
            self.to(dtype=_dtype)

        return latents
    

    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:  
        # unscale/denormalize the latents denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.config, "latents_mean") and self.config.latents_mean
        has_latents_std = hasattr(self.config, "latents_std") and self.config.latents_std

        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.config.latents_mean).view(1, 4, 1, 1).to(
                latents.device, latents.dtype
            )
            latents_std = torch.tensor(self.config.latents_std).view(1, 4, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_std / self.config.scaling_factor + latents_mean
        else:
            latents = latents / self.config.scaling_factor

        images = self.vae.decode(latents, return_dict=False)[0]

        return images
##############################################################################################################################
    




##############################################################################################################################
# VAE Encoder class
##############################################################################################################################
class StableDiffusionVaeEncoderInput(BaseOutput):
    image: Optional[PipelineImageInput] = None
    mask_image: Optional[PipelineImageInput] = None
    height: Optional[int] = None
    width: Optional[int] = None
    generator: Optional[torch.Generator] = None




class StableDiffusionVaeEncoderOutput:
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
        



class StableDiffusionVaeEncoder(StableDiffusionVaeModel):
    def __call__(
        self, 
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[PipelineImageInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> StableDiffusionVaeEncoderOutput:
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
            output_kwargs["image_latents"] = self.encode(
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
                        height // self.scale_factor, 
                        width // self.scale_factor
                    )
                )
        
        return StableDiffusionVaeEncoderOutput(**output_kwargs)
##############################################################################################################################





##############################################################################################################################
# VAE Decoder class
##############################################################################################################################
class StableDiffusionVaeDecoderInput(BaseOutput):
    latents: Optional[torch.Tensor] = None




class StableDiffusionVaeDecoderOutput:
    def __init__(
        self
    ):
        pass


    def __call__(
        self
    ):
        pass





class StableDiffusionVaeDecoder(StableDiffusionVaeModel):
    def __call__(
        self, 
        latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> StableDiffusionVaeDecoderOutput:
##############################################################################################################################




##############################################################################################################################
# Stable Diffusion VAE wrapper
##############################################################################################################################
class StableDiffusionVae