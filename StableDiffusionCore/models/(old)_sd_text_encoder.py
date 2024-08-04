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



class StableDiffusionTextEncoderOutput():
    def __init__(
        self,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds_2: Optional[torch.Tensor] = None,
    ):
        self.prompt_embeds = prompt_embeds
        self.prompt_embeds_2 = prompt_embeds_2
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.pooled_prompt_embeds_2 = pooled_prompt_embeds_2


    def __call__(
        self,
        model_type: str,
        use_refiner: bool = False,
        num_images_per_prompt: int = 1,
    ):
        """
        По сути просто формирует и выплёвывает эмбеддинги для нужной модельки
        """
        if model_type == "sd15":
            pass
        elif model_type == "sdxl":
            pass
        elif model_type == "sd3":
            pass
        else:
            raise ValueError(f"Unknown model type '{model_type}'")




class StableDiffusionTextEncoderModel():
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda",
    ):  
        # Инитим константы
        self.path = model_path
        self.type = model_type or "sd15"
        self.device = torch.device(device)

        # Инитим модели
        self.tokenizers: list[CLIPTokenizer]
        self.text_encoders: list[Union[CLIPTextModel, CLIPTextModelWithProjection]]
        if model_type == "sd15":
            self.tokenizers = [
                CLIPTokenizer.from_pretrained(
                    model_path, 
                    subfolder="tokenizer"
                )
            ]
            self.text_encoders = [
                CLIPTextModel.from_pretrained(
                    model_path, 
                    subfolder="text_encoder", 
                    torch_dtype=torch.float16,
                    variant='fp16',
                    use_safetensors=True
                )
            ]
        elif model_type == "sdxl":
            self.tokenizers = [
                CLIPTokenizer.from_pretrained(
                    model_path, 
                    subfolder="tokenizer"
                ),
                CLIPTokenizer.from_pretrained(
                    model_path,
                    subfolder='tokenizer_2'
                )
            ]
            self.text_encoders = [
                CLIPTextModel.from_pretrained(
                    model_path, 
                    subfolder="text_encoder", 
                    torch_dtype=torch.float16,
                    variant='fp16',
                    use_safetensors=True
                ),
                CLIPTextModelWithProjection.from_pretrained(
                    model_path,
                    subfolder='text_encoder_2', 
                    torch_dtype=torch.float16,
                    variant='fp16',
                    use_safetensors=True
                )
            ]
        elif model_type == "sd3":
            pass
        else:
            raise ValueError(f"Unknown model type '{model_type}'")
        self.to(device)
        print(f"TextEncoder model has successfully loaded from '{model_path}' checkpoint!")


    def to(self, device):
        for text_encoder in self.text_encoders:
            text_encoder.to(device)
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


    @property
    def text_encoder_projection_dim(self):
        return (
            self.text_encoders[1].config.projection_dim 
            if self.type == "sdxl" else
            None
        )


    def normalize_prompts_to_list(
        self, 
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[int] = None,
    ):
        prompt = prompt or ""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Проверка, что размеры промптов совпадают (для негатива в большей степени)
        if batch_size != len(prompt):
            prompt = batch_size * prompt
        
        if self.type == "sd15":
            prompts = [prompt]    
        elif self.type == "sdxl":
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompts = [prompt, prompt_2]
        elif self.type == "sd3":
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompts = [prompt, prompt_2, prompt_3]
        else: 
            raise ValueError(f"Unknown type {self.type}")

        return prompts
    
            
    def encode_prompt(
        self,
        prompts: List[List[str]],
        clip_skip: Optional[int] = None,
        num_images_per_prompt: int = 1,
        lora_scale: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Encodes the prompt into text encoder hidden states.
        """
        # set lora scale so that monkey patched LoRA function of text encoder can correctly access it
        if lora_scale:
            [
                scale_lora_layers(text_encoder, lora_scale) 
                for text_encoder 
                in self.text_encoders
            ]

        # Получаем эмбеддинги с каждой модели
        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for prompt, tokenizer, text_encoder in zip(prompts, self.tokenizers, self.text_encoders):
            # Токенизируем
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_input_ids = text_input_ids.to(self.device)

            # Получаем эмбеддинги
            prompt_embeds = text_encoder(
                text_input_ids, output_hidden_states=True
            )

            pooled_prompt_embeds = prompt_embeds[0]

            if clip_skip:
                if self.type == "sd15":
                    prompt_embeds = text_encoder.text_model.final_layer_norm(
                        prompt_embeds[-1][-(clip_skip + 1)]
                    )
                elif self.type == "sdxl":
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
                elif self.type == "sd3":
                    # prompt_embeds = ...
                    pass
            else:
                if self.type == "sd15":
                    prompt_embeds = prompt_embeds[0]
                elif self.type == "sdxl":
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                elif self.type == "sd3":
                    # prompt_embeds = ...
                    pass

            prompt_embeds_list.append(prompt_embeds)
        
        # TODO: Вынести тут получение эмбеддингов только SDXL
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)


        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if self.type == "sdxl":
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)


        # Retrieve the original scale by scaling back the LoRA layers
        if lora_scale:
            [
                unscale_lora_layers(text_encoder, lora_scale)
                for text_encoder 
                in self.text_encoders
            ]
        
        return prompt_embeds, pooled_prompt_embeds
        

    
    def __call__(
        self, 
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        По сути просто предствляет из себя переделанные методы DiffusionPipeline.encode_prompt()
        """
        batch_size: int
        prompt = prompt or ""
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        # 1. Подготавливаем промпты
        prompts = self.normalize_prompts_to_list(
            prompt,
            prompt_2,
            prompt_3,
            batch_size,
        )

        # 2. Получаем эмбеддинги
        # set lora scale so that monkey patched LoRA function of text encoder can correctly access it
        if lora_scale:
            [
                scale_lora_layers(text_encoder, lora_scale) 
                for text_encoder 
                in self.text_encoders
            ]

        # Получаем эмбеддинги с каждой модели
        # prompt_embeds_list = []
        # pooled_prompt_embeds = None
        kwargs = {}
        for k, (prompt, tokenizer, text_encoder) in enumerate(zip(prompts, self.tokenizers, self.text_encoders)):
            # Токенизируем
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_input_ids = text_input_ids.to(self.device)

            # Получаем эмбеддинги
            prompt_embeds = text_encoder(
                text_input_ids, output_hidden_states=True
            )
            kwargs["pr"]

            pooled_prompt_embeds = prompt_embeds[0]

        
        # Retrieve the original scale by scaling back the LoRA layers
        if lora_scale:
            [
                unscale_lora_layers(text_encoder, lora_scale)
                for text_encoder 
                in self.text_encoders
            ]

        return prompt_embeds, pooled_prompt_embeds








