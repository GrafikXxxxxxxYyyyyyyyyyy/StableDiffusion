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


def save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)

    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        plt.imsave(image_path, image)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])

    return image_urls


class Handler():
    def __init__(self, device="cuda"): 
        self.inference_step = 0
        self.train_step = 0
        self.constructor_step = 0
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

        # 1. Устанавливает режим работы
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

        elif mode == "inference_wandb":
            if "prompt" not in list(job_input.keys()):
                raise ValueError(f"Request must contain 'prompt' field working in '{mode}' mode!")
            
            request = {**job_input, **request}
            if "seed" not in request:
                request["seed"] = np.random.randint(0, 1000000000)
                
            response = self.inference_wandb_mode(model, request)
        
        elif mode == "constructor":
            variables = {}
            if "variable" not in list(job_input.keys()):
                raise ValueError(f"Request must contain 'variable' field working in '{mode}' mode!")
            variables = job_input.pop('variable')
            
            if "prompt" not in list(job_input.keys()):
                raise ValueError(f"Request must contain 'prompt' field working in '{mode}' mode!")
            
            request = {**job_input, **request}
            if "seed" not in request:
                request["seed"] = np.random.randint(0, 1000000000)
            
            response = self.constructor_mode(model, request, variables)

        elif mode == "train":
            # TODO: Create serverless train pipeline
            pass

        else:
            raise ValueError(f"Unknown mode '{mode}")
        
        return response


    def maybe_reload_model(self, model_config):
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
    
    
    def inference_wandb_mode(self, inference_config) -> dict:
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

        # Inference
        # TODO: Написать логику работы с refiner
        images = pipeline(self.model, **inference_config)
        images = convert_pt_to_numpy(images)
        
        # Log results with W&B
        with wandb.init(project="AvaGen_endpoint", name=f'infrence_run_{self.inference_step}') as run:
            # Берём из модели название планировщика шума
            inference_config['scheduler_name'] = self.model.scheduler_name
            self._log_inference(images, inference_config)
            self.inference_step += 1
            run_url = run.get_url()
        wandb.finish()

        image_urls = save_and_upload_images(images, self.last_id)
        response = {
            "wandb_url": run_url,
            "images": image_urls,
        }

        return response


    def constructor_mode(self, constructor_config, variables) -> dict:
        pipeline = StableDiffusionUnifiedPipeline(do_cfg=True, device=self.device)

        if "schedulers" not in variables:
            variables["schedulers"] = ["DPM++ 2M SDE Karras"]
        if "lora_scales" not in variables:
            variables["lora_scales"] = [0.7]
        if "num_inference_steps" not in variables:
            variables["num_inference_steps"] = [30]
        if "guidance_scale" not in variables:
            variables["guidance_scale"] = [7]

        lora_name = list(self.last_adapters)[0]

        with wandb.init(project="AvaGen_endpoint", name=f'constructor_run_{self.constructor_step}') as run:
            self.constructor_step += 1
            run_url = run.get_url()

            for scheduler_name in variables["schedulers"]:
                self.model.set_scheduler(scheduler_name)

                for lora_scale in variables["lora_scales"]:
                    self.model.set_adapters(lora_name, lora_scale)

                    keys = ["Num steps"]
                    keys.extend([f"CFG scale: {val}" for val in variables["guidance_scale"]])
                    steps_guidance = wandb.Table(keys)
                    for steps in variables["num_inference_steps"]:
                        batch_images = []
                        for guidance_scale in variables["guidance_scale"]:
                            images = pipeline(
                                self.model, 
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                **constructor_config
                            )
                            images = convert_pt_to_numpy(images)
                            batch_images.append([wandb.Image(img) for img in images])
                        
                        steps_guidance.add_data(steps, *batch_images)

                    wandb.log({f"Scheduler: '{scheduler_name}' / LoRA scale: '{float(lora_scale)}'": steps_guidance})
        wandb.finish()

        # Create responce
        response = {
            "wandb_url": run_url
        }
        
        return response

    
    def _log_inference(self, images, config):
        log_info = wandb.Table([
            "Prompt", "Negative prompt", "Prompt 2", "Negative prompt 2", "Scheduler", "Steps", "Guidance scale", "Clip_skip", "Seed"
        ])

        num_images_per_prompt = 1
        if 'num_images_per_prompt' in config:
            num_images_per_prompt = config['num_images_per_prompt']

        prompts = config['prompt']
        if isinstance(config['prompt'], str):
            prompts = [config['prompt']]

        negative_prompts = [""] * len(prompts)
        if 'negative_prompt' in config:
            negative_prompts = (
                [config['negative_prompt']]
                if isinstance(config['negative_prompt'], str) else
                config['negative_prompt']      
            )

        prompts_2 = prompts
        if 'prompt_2' in config:
            prompts_2 = (
                [config['prompt_2']]
                if isinstance(config['prompt_2'], str) else
                config['prompt_2']      
            )

        negative_prompts_2 = negative_prompts
        if 'negative_prompt_2' in config:
            negative_prompts = (
                [config['negative_prompt_2']]
                if isinstance(config['negative_prompt_2'], str) else
                config['negative_prompt_2']
            )

        # if len(negative_prompts) != len(prompts):
        #     negative_prompts *= len(prompts)

        img_id = 0
        for idx in range(len(prompts)):
            log_info.add_data(
                prompts[idx],
                negative_prompts[idx] or "",
                prompts_2[idx] or "",
                negative_prompts_2[idx] or "",
                config['scheduler_name'],
                config["num_inference_steps"] if "num_inference_steps" in config else 50,
                config["guidance_scale"] if "guidance_scale" in config else 7.5,
                config["clip_skip"] if "clip_skip" in config else None,
                config["seed"]
            )

            for n in range(1, num_images_per_prompt + 1):
                wandb.log({f"Prompt:{idx+1}": wandb.Image(images[img_id])})
                img_id += 1

        wandb.log({f"Inference results": log_info})

