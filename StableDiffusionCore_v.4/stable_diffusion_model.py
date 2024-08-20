import os
import torch
import numpy as np

from huggingface_hub import hf_hub_download
from diffusers.utils.peft_utils import delete_adapter_layers
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.loaders import IPAdapterMixin, StableDiffusionXLLoraLoaderMixin

from .models.conditioner_model import ConditionerModels
from .models.latent_diffusion_model import LatentDiffusionModel



class StableDiffusionModel:
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
    ):  
        self.device = torch.device(device)

        model_type = model_type or "sd15"
        author = "GrafikXxxxxxxYyyyyyyyyyy"
        if model_path is None:
            if model_name is None:
                if model_type == "sd15":
                    model_path = "runwayml/stable-diffusion-v1-5"
                elif model_type == "sdxl":
                    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
                else:
                    raise ValueError(f"Unknown type {type}")
            else:
                model_path = f"{author}/{model_type}_{model_name}"     

        self.vae = StableDiffusionVaeModel(model_path, model_type, device)
        self.text_encoder = StableDiffusionTextEncoderModel(model_path, model_type, device)

            # TODO: Добавить функционал рефайнера на сторону DiffusionModel
            # self.base = DiffusionModel()
            # self.scheduler = StableDiffusionSchedulerModel(model_path, scheduler_name)
            # if model_type == "sdxl": # Если SDXL то добавляется ещё и Refiner
            #     print(f"Loading refiner...")
            #     self.refiner = StableDiffusionUNetModel(
            #         model_path="stabilityai/stable-diffusion-xl-refiner-1.0", 
            #         model_type="sdxl", 
            #         device=device
            #     )
            # else: # в противном случае наоборот удаляется
            #     if hasattr(self, "refiner"):
            #         delattr(self, "refiner")

        # Инициализируем функциональные классы
        self.lora_loader = StableDiffusionXLLoraLoaderMixin()

        # Инициализируем константы
        self.path = model_path
        self.type = model_type
        self.name = model_name
            # self.scheduler_name = scheduler_name or "euler"
            # self.use_refiner: bool = False


    def reload(self, 
        device: str = "cuda",
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
    ):
        self.__init__(
            device=device,
            model_path=model_path,
            model_type=model_type, 
            model_name=model_name,
            scheduler_name=scheduler_name,
        )


# def switch_denoising_model(self, denoising_model: str):
#     if denoising_model == "base":
#         self.use_refiner = False
#     elif denoising_model == "refiner":
#         self.use_refiner = True
#     else:
#         raise ValueError('')

# @property
# def unet(self):
#     return (
#         self.refiner 
#         if self.use_refiner and self.type == "sdxl" else
#         self.base
#     )


# def set_scheduler(self, scheduler_name):
#     if self.scheduler_name != scheduler_name:
#         self.scheduler = StableDiffusionSchedulerModel(self.path, scheduler_name) 


    def load_loras(
        self, 
        loras: Union[str, List[str], Dict[str, float]]
    ):
        """
        Загружает LoRA в выбранную модель StableDiffusion
        """
        if isinstance(loras, str):
            loras = {loras: 1.0}
        elif isinstance(loras, list):
            loras = {lora_name: 1.0 for lora_name in loras}
        elif isinstance(loras, dict):
            loras = loras
        
        current_adapters_list = (
            [] 
            if self.get_list_adapters() == {} else
            self.get_list_adapters()["unet"]
        )
        new_adapters_list = list(loras.keys())

         # Удаление адаптеров только если списки адаптеров не совпадают
        if set(current_adapters_list) != set(new_adapters_list):
            self.delete_adapters(current_adapters_list)
    
            for lora_name in new_adapters_list:
                lora_path = f"models/loras/{self.type}_{lora_name}.safetensors"
                # Если нужной лоры нет в скачанных, то качаем
                if not os.path.exists(lora_path):
                    hf_hub_download(
                        repo_id = f'{self.author}/loras',
                        filename = f"{self.type}_{lora_name}.safetensors",
                        local_dir = lora_path,
                    )

                self.load_lora_weights(lora_path, adapter_name=lora_name)

            self.set_adapters(list(loras.keys()), list(loras.values()))
            print(f"LoRA adapters has successfully changed to:\n{loras}")
    

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Delete unet adapters
        self.unet.unet.delete_adapters(adapter_names)
        # Delete text encoder adapters
        for adapter_name in adapter_names:
            for text_encoder in self.text_encoder.text_encoders:
                delete_adapter_layers(text_encoder, adapter_name)


    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_loader.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.lora_loader.load_lora_into_unet(
            state_dict, 
            network_alphas=network_alphas, 
            unet=self.unet.unet, 
            adapter_name=adapter_name,
        )

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.lora_loader.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder.text_encoders[0],
                prefix="text_encoder",
                # lora_scale=self.lora_scale,
                adapter_name=adapter_name,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.lora_loader.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder.text_encoders[1],
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
        self.unet.unet.set_adapters(adapter_names, adapter_weights)
        # Handle the Text Encoder
        for text_encoder in self.text_encoder.text_encoders:
            self.lora_loader.set_adapters_for_text_encoder(adapter_names, text_encoder, adapter_weights)

    
    def get_list_adapters(self) -> Dict[str, List[str]]:
        set_adapters = {}
        if hasattr(self.unet.unet, "peft_config"):
            set_adapters["unet"] = list(self.unet.unet.peft_config.keys())
        for i, text_encoder in enumerate(self.text_encoder.text_encoders):
            if hasattr(text_encoder, "peft_config"):
                set_adapters[f"text_encoder_{i+1}"] = list(text_encoder.peft_config.keys())

        return set_adapters













