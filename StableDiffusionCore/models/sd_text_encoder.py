import torch
from dataclasses import dataclass
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from typing import Any, Callable, Dict, List, Optional, Union, Tuple



@dataclass
class StableDiffusionTextEncoderOutput:
    prompt_embeds_1: torch.Tensor
    prompt_embeds_2: Optional[torch.Tensor]
    pooled_prompt_embeds: Optional[torch.Tensor]

    def get_embeddings(
        self,
        model_type: str,
        use_refiner: bool = False,
        num_images_per_prompt: int = 1,
    ):
        """
        По сути просто формирует и выплёвывает эмбеддинги для нужной модельки
        """
        if model_type == "sd15":
            bs_embed, seq_len, _ = self.prompt_embeds_1.shape
            prompt_embeds = self.prompt_embeds_1.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            return (prompt_embeds, None)
        elif model_type == "sdxl":
            prompt_embeds = (
                self.prompt_embeds_2
                if use_refiner else
                torch.concat([self.prompt_embeds_1, self.prompt_embeds_2], dim=-1)
            )

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            pooled_prompt_embeds = self.pooled_prompt_embeds_2.repeat(1, num_images_per_prompt)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)

            return (prompt_embeds, pooled_prompt_embeds)
        elif model_type == "sd3":
            pass
        else:
            raise ValueError(f"Unknown model type '{model_type}'")



class StableDiffusionTextEncoderModel():
    """
    This class contains optional parts of different 
    stable diffusion's text encoder realisations
    """
    tokenizer: CLIPTokenizer
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection]
    tokenizer_2: Optional[CLIPTokenizer]
    text_encoder_2: Optional[CLIPTextModelWithProjection]

    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda",
    ):  
        # Инициализируем константы
        self.path = model_path
        self.type = model_type or "sd15"
        self.device = torch.device(device)

        # Инициализируем модели
        if self.type == "sd15":
            # убираем из модели лишние части
            if hasattr(self, "tokenizer_2"):
                delattr(self, "tokenizer_2")
            if hasattr(self, "text_encoder_2"):
                delattr(self, "text_encoder_2")
            
            # инитим нужные
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, 
                subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_path, 
                subfolder="text_encoder", 
                torch_dtype=torch.float16,
                variant='fp16',
                use_safetensors=True
            )
        elif self.type == "sdxl":
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, 
                subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_path, 
                subfolder="text_encoder", 
                torch_dtype=torch.float16,
                variant='fp16',
                use_safetensors=True
            )
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder='tokenizer_2'
            )
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                subfolder='text_encoder_2', 
                torch_dtype=torch.float16,
                variant='fp16',
                use_safetensors=True
            )
        elif self.type == "sd3":
            pass
        else:
            raise ValueError(f"Unknown model_type '{self.type}'")   
        
        self.to(device)

    @property
    def text_encoder_projection_dim(self):
        return (
            self.text_encoder_2.config.projection_dim 
            if hasattr(self, "text_encoder_2") else
            None
        )


    def to(self, device):
        self.text_encoder = self.text_encoder.to(device)
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2 = self.text_encoder_2.to(device)
        # if hasattr(self, "text_encoder_3"):
        #     self.text_encoder_3 = self.text_encoder_3.to(device)
        
        self.device = torch.device(device)


    def reload(self, 
        model_type: str,
        model_path: str,
        device: str = "cuda",
        do_cfg: bool = True,
    ):
        self.__init__(
            model_path=model_path,
            model_type=model_type, 
            device=self.device,
        )


    def __call__(
        self, 
        batch_size: int,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.
        """
        prompt = prompt or ""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        # Проверка, что размеры промптов совпадают (для негатива в большей степени)
        if batch_size != len(prompt):
            prompt = batch_size * prompt

        # set lora scale so that monkey patched LoRA function of text encoder can correctly access it
        if lora_scale is not None:
            [
                scale_lora_layers(text_encoder, lora_scale) 
                for text_encoder in text_encoders
            ] 

        # Получаем эмбеддинги 
        output_embeds = {}
        if self.type == "sd15":
            text_input_ids = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_input_ids = text_input_ids.to(self.device)            

            encoder_output = self.text_encoder(
                text_input_ids, output_hidden_states=True
            )

            output_embeds["prompt_embeds_1"] = (
                self.text_encoder.text_model.final_layer_norm(encoder_output[-1][-(clip_skip + 1)])
                if clip_skip is not None else
                encoder_output[0]
            )
            output_embeds["pooled_prompt_embeds_1"] = encoder_output[0]

        elif self.type == "sdxl":
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            
            prompts = [prompt, prompt_2]
            tokenizers = [self.tokenizer, self.tokenizer_2]
            text_encoders = [self.text_encoder, self.text_encoder_2]
            for k, (prompt, tokenizer, text_encoder) in enumerate(zip(prompts, tokenizers, text_encoders)):
                text_input_ids = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                text_input_ids = text_input_ids.to(self.device)

                encoder_output = text_encoder(
                    text_input_ids, 
                    output_hidden_states=True
                )

                output_embeds[f"prompt_embeds_{k+1}"] = (
                    encoder_output.hidden_states[-(clip_skip + 2)]
                    if clip_skip is not None else
                    encoder_output.hidden_states[-2]
                )
                output_embeds[f"pooled_prompt_embeds_{k+1}"] = encoder_output[0]
            
        elif self.type == "sd3":
            pass
        else: 
            raise ValueError(f"Unknown type {self.type}")
        
        # Retrieve the original scale by scaling back the LoRA layers
        if lora_scale is not None:
            [
                unscale_lora_layers(text_encoder, lora_scale)
                for text_encoder in text_encoders
            ]
        
        return StableDiffusionTextEncoderOutput(**output_embeds)


