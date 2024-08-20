import torch

from dataclasses import dataclass
from diffusers import AutoencoderKL
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union
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



@dataclass
class StableDiffusionVAEInput(BaseOutput):
    image: Optional[PipelineImageInput] = None
    mask_image: Optional[PipelineImageInput] = None
    height: Optional[int] = None
    width: Optional[int] = None
    generator: Optional[torch.Generator] = None



class StableDiffusionVAEOutput:
    def __init__(
        self, 
        image_latents: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        masked_image_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.image_latents = image_latents
        self.mask = mask
        self.masked_image_latents = masked_image_latents

    
    def get_image_latents(
        self,
        num_images_per_prompt,
        device,
        dtype,
    ):
        # image_latents = image_latents.repeat(
        #     (batch_size * num_images_per_prompt) // image_latents.shape[0], 1, 1, 1
        # )
            
        # masked_image_latents = masked_image_latents.repeat(
        #     (batch_size * num_images_per_prompt) // masked_image_latents.shape[0], 1, 1, 1
        # )

        # mask = mask.repeat(
        #     (batch_size * num_images_per_prompt) // mask.shape[0], 1, 1, 1
        # )
        pass
    



class StableDiffusionVAE:
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda"
    ):  
        # Можно добавить выбор различных VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_path, 
            subfolder="vae",
            torch_dtype=torch.float16,
            variant='fp16', 
            use_safetensors=True
        )
        self.to(device)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.scale_factor, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True,
        )
        self.type = model_type or "sd15"
        self.path = model_path
        print(f"VAE model has successfully loaded from '{model_path}' checkpoint!")


    def to(self, device):
        self.vae.to(device)
    
    @property
    def config(self) -> int:
        return self.vae.config

    @property
    def scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)
    

    def encode(
        self, 
        image: torch.Tensor, 
        generator: Optional[torch.Generator] = None, 
    ) -> torch.Tensor:  
        dtype = image.dtype
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        if self.vae.config.force_upcast:
            self.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents


    def decode(
        self,
        latents: torch.Tensor,
    ):  
        # unscale/denormalize the latents denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.config, "latents_mean") and self.config.latents_mean
        has_latents_std = hasattr(self.config, "latents_std") and self.config.latents_std

        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents_std = torch.tensor(self.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents = latents * latents_std / self.config.scaling_factor + latents_mean
        else:
            latents = latents / self.config.scaling_factor

        return self.vae.decode(latents, return_dict=False)[0]
    

    
    def __call__(
        self, 
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[PipelineImageInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Optional[StableDiffusionVAEOutput]:
        """
        Кодирует картинку и маску в латентное представление, если те переданы
        """        
        device = self.device
        dtype = self.vae.dtype

        output_kwargs: dict
        if image is not None and mask_image is not None: # inpaint
            # process mask and image
            image = self.image_processor.preprocess(image)
            mask = self.mask_processor.preprocess(mask_image)
            masked_image = image * (mask < 0.5)
            image = image.to(device=device, dtype=dtype)
            mask = mask.to(device=device, dtype=dtype)
            masked_image = masked_image.to(device=device, dtype=dtype)
            # resize if need
            if height and width:
                image = torch.nn.functional.interpolate(
                    image, size=(height, width)
                )
                mask = torch.nn.functional.interpolate(
                    mask, size=(height, width)
                )
            # encode image and masked_image and stack with the reshaped mask
            output_kwargs["image_latents"] = self.encode(image, generator)
            output_kwargs["mask"] = torch.nn.functional.interpolate(
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask, size=(height // self.scale_factor, width // self.scale_factor)
            )
            output_kwargs["masked_image_latents"] = self.encode(masked_image, generator)
        elif image is not None: # img2img
            # process and encode image
            image = self.image_processor.preprocess(image)
            image = image.to(device, dtype)
            # resize if need
            if height and width:
                image = torch.nn.functional.interpolate(
                    image, size=(height, width)
                )
            # encode image 
            output_kwargs["image_latents"] = self.encode(image, generator)
        else:
            # если text2image или передана только маска без картинки,
            # то VAE нечего кодировать ==> return None
            return 

        return StableDiffusionVAEOutput(**output_kwargs)
























 # # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
# def upcast_vae(self):
#     dtype = self.vae.dtype
#     self.vae.to(dtype=torch.float32)
#     use_torch_2_0_or_xformers = isinstance(
#         self.vae.decoder.mid_block.attentions[0].processor,
#         (
#             AttnProcessor2_0,
#             XFormersAttnProcessor,
#             LoRAXFormersAttnProcessor,
#             LoRAAttnProcessor2_0,
#         ),
#     )
#     # if xformers or torch_2_0 is used attention block does not need
#     # to be in float32 which can save lots of memory
#     if use_torch_2_0_or_xformers:
#         self.vae.post_quant_conv.to(dtype)
#         self.vae.decoder.conv_in.to(dtype)
#         self.vae.decoder.mid_block.to(dtype)
