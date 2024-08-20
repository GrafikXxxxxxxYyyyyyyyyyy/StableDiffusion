import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers

from ..models.text_encoder_model import StableDiffusionTextEncoderModel



class StableDiffusionTextEncoderPipelineInput(BaseOutput):
    prompt: Optional[Union[str, List[str]]] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    clip_skip: Optional[int] = None
    lora_scale: Optional[float] = None




class StableDiffusionTextEncoderPipelineOutput:
    def __init__(
        self,
        prompt_embeds_1: torch.Tensor,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        self.prompt_embeds_1 = prompt_embeds_1
        self.prompt_embeds_2 = prompt_embeds_2
        self.pooled_prompt_embeds = pooled_prompt_embeds


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
            bs_embed, seq_len, _ = self.prompt_embeds_1.shape

            prompt_embeds = self.prompt_embeds_1
            prompt_embeds = self.prompt_embeds_1.repeat(
                1, num_images_per_prompt, 1
            )
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

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
            pooled_prompt_embeds = self.pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)

            return (prompt_embeds, pooled_prompt_embeds)
        
        else:
            raise ValueError(f"Unknown model type '{model_type}'")




class StableDiffusionTextEncoderPipeline:
    def __call__(
        self, 
        te_model: StableDiffusionTextEncoderModel,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        **kwargs, 
    ) -> StableDiffusionTextEncoderPipelineOutput:
        prompt = prompt or ""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        if te_model.type == "sd15":
            prompts = [prompt]   
            tokenizers = [te_model.tokenizer]
            text_encoders = [te_model.text_encoder] 
        elif te_model.type == "sdxl":
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            
            prompts = [prompt, prompt_2]
            tokenizers = [te_model.tokenizer, te_model.tokenizer_2]
            text_encoders = [te_model.text_encoder, te_model.text_encoder_2]
        else: 
            raise ValueError(f"Unknown type {te_model.type}")

        # set lora scale so that monkey patched LoRA function of text encoder can correctly access it
        if lora_scale is not None:
            [
                scale_lora_layers(text_encoder, lora_scale) 
                for text_encoder in text_encoders
            ] 
        
        output_kwargs: dict
        for k, (prompt, tokenizer, text_encoder) in enumerate(zip(prompts, tokenizers, text_encoders)):
            # tokenize
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_input_ids = text_input_ids.to(te_model.device)

            # get embeddings
            encoder_output = text_encoder(
                text_input_ids, 
                output_hidden_states=True
            )

            if te_model.type == "sd15":
                output_kwargs[f"prompt_embeds_{k+1}"] = (
                    te_model.text_encoder.text_model.final_layer_norm(encoder_output[-1][-(clip_skip + 1)])
                    if clip_skip is not None else
                    encoder_output[0]
                )
            elif te_model.type == "sdxl":
                output_kwargs[f"prompt_embeds_{k+1}"] = (
                    encoder_output.hidden_states[-(clip_skip + 2)]
                    if clip_skip is not None else
                    encoder_output.hidden_states[-2]
                )                
                # contains pooled embeds from only te_model.text_encoder_2 after loop 
                output_kwargs[f"pooled_prompt_embeds"] = encoder_output[0]
            else: 
                raise ValueError(f"Unknown type {te_model.type}")
        
        # Retrieve the original scale by scaling back the LoRA layers
        if lora_scale is not None:
            [
                unscale_lora_layers(text_encoder, lora_scale)
                for text_encoder in text_encoders
            ]
        
        return StableDiffusionTextEncoderPipelineOutput(**output_kwargs)
