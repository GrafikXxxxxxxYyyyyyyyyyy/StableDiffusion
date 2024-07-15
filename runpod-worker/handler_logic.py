import os
import sys
import wandb
import torch
import base64
import runpod
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_upload, rp_cleanup
from models.stable_diffusion import SDModelWrapper
from pipelines.sd_unified_pipeline import StableDiffusionUnifiedPipeline


def convert_pt_to_numpy(images: torch.Tensor) -> List[np.ndarray]:
    np_images = []
    for idx in range(len(images)):
        img = (images[idx] / 2 + 0.5).clamp(0, 1)
        img = (img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()

        np_images.append(img)

    return np_images


class Handler():
    def __init__(self, device="cuda"): 
        self.device = device
        self.last_id = '0'
        self.model: SDModelWrapper = None


    def __call__(self, model: SDModelWrapper, job_input, job_id):
        """
        Принимает на вход модель, с которой нужно работать и 
        запрос от пользователя, который нужно обработать моделью
        """
        self.model = model
        self.last_id = job_id

        # 1. Устанавливает режим работы (по-умолчанию inference)
        mode = "inference"
        if "mode" in list(job_input.keys()):
            mode = job_input.pop('mode')

        # 2. Перенастраивает модель
        if "model" in list(job_input.keys()):
            self.maybe_reload_model(job_input.pop('model'))
        
        # 3. Получает параметры запуска
        request = {}
        if "params" in list(job_input.keys()):
            request = job_input.pop('params')

        # Run!
        response = {}
        if mode == "inference":
            if "prompt" not in list(job_input.keys()):
                raise ValueError(f"Request must contain 'prompt' field working in '{mode}' mode!")
            
            request = {**job_input, **request}
            if "seed" not in request:
                request["seed"] = np.random.randint(0, 1000000000)
                
            response = self.inference_mode(request)
            response["seed"] = request["seed"]
        elif mode == "train":
            pass
        else:
            raise ValueError(f"Unknown mode '{mode}")
        
        return response



    def maybe_reload_model(self, model_config):
        # Если указан чекпоинт, то грузит из него
        if "ckpt_path" in list(model_config.keys()):
            self.model.reload(ckpt_path=model_config["ckpt_path"])            
        else:
            ckpt_type = None
            ckpt_name = None
            if "type" in list(model_config.keys()):
                ckpt_type = model_config.pop('type')
            if "name" in list(model_config.keys()):
                ckpt_name = model_config.pop('name')
            self.model.reload(model_name=ckpt_name, model_type=ckpt_type)
        
        loras = {}
        if "loras" in list(model_config.keys()):
            loras = model_config.pop("loras")
        self.model.load_loras(loras)

        if "scheduler" in list(model_config.keys()):
            scheduler_name = model_config.pop("scheduler")
        self.model.set_scheduler(scheduler_name)



    def inference_mode(self, inference_config) -> dict:
        """
        inference_config example:
        {
            prompt: Union[str, List[str]],
            prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 30,
            guidance_scale: Optional[float] = 6,
            num_images_per_prompt: Optional[int] = 1,
            denoising_end: Optional[float] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
            seed: Optional[int] = None,
        }
        """
        # Init inference pipeline
        pipeline = StableDiffusionUnifiedPipeline(do_cfg=True, device=self.device)

        # Обработка изображений если те присутствуют в конфиге
        if "image" in list(inference_config.keys()):
            inference_config["image"] = Image.open(io.BytesIO(base64.b64decode(inference_config["image"])))
        if "mask_image" in list(inference_config.keys()):
            inference_config["mask_image"] = Image.open(io.BytesIO(base64.b64decode(inference_config["mask_image"])))

        images = pipeline(self.model, **inference_config)
        if isinstance(images, torch.Tensor):
            images = convert_pt_to_numpy(images)
        
        base64_images = []
        for img in images:
            img = np.ascontiguousarray(img)
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_images.append(base64_str)

        response = {
            "images": base64_images
        }

        return response
    