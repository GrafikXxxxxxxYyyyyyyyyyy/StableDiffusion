import torch

from diffusers import UNet2DConditionModel
from typing import Any, Callable, Dict, List, Optional, Union



class DiffusionUNetModel:
    # /////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda"
    ):  
        # Инитим константы
        self.path = model_path
        self.type = model_type or "sd15"

        # Можно сделать выбор различных архитектур под SD3 + Video
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, 
            subfolder='unet', 
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True
        )
        self.to(device)
        print(f"UNet model has successfully loaded from '{model_path}' checkpoint!")
    # /////////////////////////////////////////////////////////////////////////////////////// #
        
    @property
    def config(self):
        return self.unet.config
    
    @property
    def device(self):
        return self.unet.device
    
    @property
    def is_inpainting_model(self):
        return self.unet.config.in_channels == 9
    
    @property
    def is_latent_model(self):
        return self.unet.config.in_channels == 4
    
    @property
    def add_embed_dim(self):
        return self.unet.add_embedding.linear_1.in_features


    def to(
        self, 
        device,
        dtype=None,
    ):
        self.unet.to(device, dtype=dtype)


    def reload(
        self, 
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda",
    ):
        self.__init__(
            model_path=model_path,
            device=device,
            model_type=model_type, 
        )


    # ======================================================================================= #
    def __call__(
        self, 
        latents: torch.FloatTensor,
        timestep: int,
        encoder_hidden_states: torch.FloatTensor,
        timestep_cond: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.FloatTensor:                
        return self.unet.__call__(
            latents,
            timestep,
            encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    # ======================================================================================= #




