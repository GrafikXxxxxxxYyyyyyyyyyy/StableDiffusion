import torch
from diffusers import (
    AutoencoderKL,
)
from typing import Any, Callable, Dict, List, Optional, Union



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



class StableDiffusionVAEModel():
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
    ):  
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
        return self.vae.decode(latents, return_dict=False)[0]
    
    
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


    
    def __call__(
        self, 
        image: torch.Tensor, 
        generator: Optional[torch.Generator] = None, 
    ) -> torch.Tensor:

        pass




