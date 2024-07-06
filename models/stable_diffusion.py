import os
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torchvision import transforms
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    LoraLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.peft_utils import delete_adapter_layers, set_adapter_layers

from typing import Any, Callable, Dict, List, Optional, Union



class SDModelWrapper():
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cuda"
    ):  
        """
        Единая обёртка над всеми моделями StableDiffusion
        По умолчанию если не указаны никакие поля, грузит sd15 от runwayml/stable-diffusion-v1-5
        """
        if model_type is None or model_type == 'sd15':
            if hasattr(self, "text_encoder_2"):
                delattr(self, "text_encoder_2")
            if hasattr(self, "tokenizer_2"):
                delattr(self, "tokenizer_2")
        else:
            self.text_encoder_2 = (
                self.text_encoder_2 
                if hasattr(self, "text_encoder_2") else
                None
            )
            self.tokenizer_2 = (
                self.tokenizer_2 
                if hasattr(self, "tokenizer_2") else
                None
            )
            # self.load_refiner()
        
        # Если не указан чекпоинт, то должны быть указаны тип модели и имя чекпоинта
        if ckpt_path is None:
            type = model_type or "sd15"
            # Если имя не указано, то грузим по дефолту
            if model_name is None:
                if type == "sd15":
                    ckpt_path = "runwayml/stable-diffusion-v1-5"
                elif type == "sdxl":
                    ckpt_path = "stabilityai/stable-diffusion-xl-base-1.0"
                else:
                    raise ValueError(f"Unknown type {type}")
            else:
                ckpt_path = f"GrafikXxxxxxxYyyyyyyyyyy/{type}_{model_name}"     
        
        # Грузим модель из выбранного чекпоинта
        self.load_hf_checkpoint(ckpt_path)
        self.to(device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.lora_loader = StableDiffusionXLLoraLoaderMixin()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.scheduler_name = 'euler'
        self.type = model_type or "sd15"
        self.name = model_name or None
        self.path = ckpt_path


    def load_hf_checkpoint(self, ckpt_path):
        if hasattr(self, 'path') and self.path == ckpt_path:
            return
    
        self.vae = AutoencoderKL.from_pretrained(
            ckpt_path, 
            subfolder="vae",
            torch_dtype=torch.float16,
            variant='fp16', 
            use_safetensors=True
        )
        self.base = UNet2DConditionModel.from_pretrained(
            ckpt_path, 
            subfolder='unet', 
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            ckpt_path, 
            subfolder="text_encoder", 
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            ckpt_path, 
            subfolder="tokenizer"
        )
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            ckpt_path,
            subfolder='scheduler'
        )

        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                ckpt_path,
                subfolder='text_encoder_2', 
                torch_dtype=torch.float16,
                variant='fp16',
                use_safetensors=True
            )
        if hasattr(self, "tokenizer_2"):
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                ckpt_path,
                subfolder='tokenizer_2'
            )
        
        self.path = ckpt_path
        print(f"StableDiffusion model successfully loaded from '{ckpt_path}' checkpoint")
        print(f"All LoRA adapters deleted")


    def load_refiner(self, refiner: UNet2DConditionModel = None):
        if refiner is None:
            if not hasattr(self, 'refiner'):
                refiner = UNet2DConditionModel.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0", 
                    subfolder='unet', 
                    torch_dtype=torch.float16,
                    variant='fp16',
                    use_safetensors=True
                )
                self.refiner = refiner
        else:
            self.refiner = refiner


    def reload(self, 
        ckpt_path:Optional[str] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        self.__init__(
            ckpt_path=ckpt_path, 
            model_name=model_name, 
            model_type=model_type, 
            device=self.device
        )


    def to(self, device):
        self.vae.to(device)
        self.base.to(device)
        self.text_encoder.to(device)
        if hasattr(self, 'text_encoder_2'):
            self.text_encoder_2.to(device)
        if hasattr(self, 'refiner'):
            self.refiner.to(device)
        
        self.device = torch.device(device)


    def set_scheduler(self, scheduler_name):
        if hasattr(self, 'scheduler_name') and self.scheduler_name == scheduler_name:
            return
        
        config = self.scheduler.config

        if scheduler_name == "DDIM":
            self.scheduler = DDIMScheduler.from_config(config)
        elif scheduler_name == "euler":
            self.scheduler = EulerDiscreteScheduler.from_config(config)
        elif scheduler_name == "euler_a":
            self.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        elif scheduler_name == "DPM++ 2M":
            self.scheduler = DPMSolverMultistepScheduler.from_config(config)
        elif scheduler_name == "DPM++ 2M Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
        elif scheduler_name == "DPM++ 2M SDE Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_config(
                config, se_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
            )
        elif scheduler_name == "PNDM":
            self.scheduler = PNDMScheduler.from_config(config)
        elif scheduler_name == "uni_pc":
            self.scheduler = UniPCMultistepScheduler.from_config(config)
        else:
            raise ValueError(f'Unknown scheduler name: {scheduler_name}')
        
        self.scheduler_name = scheduler_name
        print(f"Scheduler has successfully changed to '{scheduler_name}'")


    def load_loras(self, loras):
        if isinstance(loras, str):
            loras = {loras: 1.0}
        elif isinstance(loras, list):
            loras = {lora_name: 1.0 for lora_name in loras}
        elif isinstance(loras, dict):
            loras = loras
        
        current_adapters_list = [] if self.get_list_adapters() == {} else self.get_list_adapters()['base']
        new_adapters_list = list(loras.keys())

         # Удаление адаптеров только если списки адаптеров не совпадают
        if set(current_adapters_list) != set(new_adapters_list):
            self.delete_adapters(current_adapters_list)
    
            for lora_name in new_adapters_list:
                lora_path = f"models/loras/{self.type}_{lora_name}.safetensors"
                # Если нужной лоры нет в скачанных, то качаем
                if not os.path.exists(lora_path):
                    hf_hub_download(
                        repo_id = 'GrafikXxxxxxxYyyyyyyyyyy/loras',
                        filename = f"{self.type}_{lora_name}.safetensors",
                        local_dir = lora_path,
                    )

                self.load_lora_weights(lora_path, adapter_name=lora_name)

        print(f"LoRA adapters has successfully changed to:\n{loras}")
        self.set_adapters(list(loras.keys()), list(loras.values()))
    

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_loader.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.base.config,
            **kwargs,
        )
        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.lora_loader.load_lora_into_unet(
            state_dict, 
            network_alphas=network_alphas, 
            unet=self.base, 
            adapter_name=adapter_name,
        )

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.lora_loader.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                # lora_scale=self.lora_scale,
                adapter_name=adapter_name,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.lora_loader.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                # lora_scale=self.lora_scale,
                adapter_name=adapter_name,
            )


    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        adapter_weights: Optional[List[float]] = None,
    ):
        # Handle the UNET
        self.base.set_adapters(adapter_names, adapter_weights)
        # Handle the Text Encoder
        if hasattr(self, "text_encoder"):
            self.lora_loader.set_adapters_for_text_encoder(adapter_names, self.text_encoder, adapter_weights)
        if hasattr(self, "text_encoder_2"):
            self.lora_loader.set_adapters_for_text_encoder(adapter_names, self.text_encoder_2, adapter_weights)


    def delete_adapters(self, adapter_names: Union[List[str], str]):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Delete unet adapters
        self.base.delete_adapters(adapter_names)
        # Delete text encoder adapters
        for adapter_name in adapter_names:
            if hasattr(self, "text_encoder"):
                delete_adapter_layers(self.text_encoder, adapter_name)
            if hasattr(self, "text_encoder_2"):
                delete_adapter_layers(self.text_encoder_2, adapter_name)


    def get_list_adapters(self) -> Dict[str, List[str]]:
        set_adapters = {}
        if hasattr(self.base, "peft_config"):
            set_adapters["base"] = list(self.base.peft_config.keys())
        if hasattr(self.text_encoder, "peft_config"):
            set_adapters["text_encoder"] = list(self.text_encoder.peft_config.keys())
        if hasattr(self, "text_encoder_2") and hasattr(self.text_encoder_2, "peft_config"):
            set_adapters["text_encoder_2"] = list(self.text_encoder_2.peft_config.keys())

        return set_adapters
    
    
    # TODO: Дописать функцию сохранения лор после обучения
    # def save_lora_weights(
    #     cls,
    #     save_directory: Union[str, os.PathLike],
    #     unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
    #     text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
    #     text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
    #     is_main_process: bool = True,
    #     weight_name: str = None,
    #     save_function: Callable = None,
    #     safe_serialization: bool = True,
    # ):
    #     state_dict = {}

    #     def pack_weights(layers, prefix):
    #         layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
    #         layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
    #         return layers_state_dict

    #     if not (unet_lora_layers or text_encoder_lora_layers or text_encoder_2_lora_layers):
    #         raise ValueError(
    #             "You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`."
    #         )

    #     if unet_lora_layers:
    #         state_dict.update(pack_weights(unet_lora_layers, "unet"))

    #     if text_encoder_lora_layers and text_encoder_2_lora_layers:
    #         state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
    #         state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

    #     cls.write_lora_layers(
    #         state_dict=state_dict,
    #         save_directory=save_directory,
    #         is_main_process=is_main_process,
    #         weight_name=weight_name,
    #         save_function=save_function,
    #         safe_serialization=safe_serialization,
    #     )















