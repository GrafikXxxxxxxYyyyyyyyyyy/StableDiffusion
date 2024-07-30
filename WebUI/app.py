import io
import os
import json
import base64
import requests
import numpy as np
import gradio as gr

from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

load_dotenv('.venv/.env')

class App():
    def __init__(
        self, 
        endpoint_id: str = "gvgo0fvy4o79m8", 
        hf_author: str = "GrafikXxxxxxxYyyyyyyyyyy",
    ):
        self.endpoint_id = endpoint_id

        self.available_ckpt = self._get_avaliable_checkpoints(hf_author=hf_author)
        self.available_loras = self._get_avaliable_loras(hf_path=f"{hf_author}/loras")
        self.available_schedulers = ["DDIM", "euler", "euler_a", "DPM++ 2M SDE Karras"]

        self.launch()


    def launch(self, **kwargs):
        layout_block = gr.Blocks()
        with layout_block as demo:
            with gr.Tab("Inference"):
                inference_params = {}
                ####################################################################################################
                # Not renderebel elements
                ####################################################################################################
                with gr.Accordion(label="Configuration", open=False):
                    with gr.Accordion(label="Model:", open=True):
                        with gr.Row():
                            model_type = gr.Radio(
                                ["sd15", "sdxl"], 
                                label="Stable Diffusion model type:", 
                                info="Which model architecture should be used?",
                                value="sd15",
                                interactive=True,
                                key="model_type",
                            )
                            inference_params["model_type"] = model_type

                            lora_count = gr.Slider(
                                minimum=0,
                                maximum=4,
                                value=0,
                                step=1,
                                visible=True,
                                interactive=True,
                                label="Number of using LoRAs:", 
                                info="How many LoRAs should be used?",
                                key="lora_count",
                            )
                            inference_params["lora_count"] = lora_count

                            ip_adapter = gr.Dropdown(
                                choices=["None"],
                                value="None",
                                visible=True,
                                interactive=True,
                                label="IP-Adapter:", 
                                info="Which style you want to use?",
                                key="ip_adapter"   
                            )
                            inference_params["ip_adapter"] = ip_adapter


                    with gr.Accordion(label="Pipeline:", open=True):
                        with gr.Row():
                            task = gr.Radio(
                                ["Text-To-Image", "Image-To-Image", "Inpainting"],
                                label="Generation task:",
                                value="Text-To-Image",
                                info="Which conditional generation task you want to solve?",
                                interactive=True,
                                visible=True,
                                key="task",
                            )
                            inference_params["task"] = task

                            prompt_examples_count = gr.Slider(
                                label="Choose number of prompt examples:",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                visible=True,
                                interactive=True,
                                key="prompt_examples_count",
                            )
                            inference_params["prompt_examples_count"] = prompt_examples_count

                            detailers = gr.CheckboxGroup(
                                ["FaceDetailer", "ArmDetailer"],
                                label="Additional detailer:",
                                info="You can apply additional detailer to generation task",
                                interactive=True,
                                visible=True,
                                key="detailers",
                            )
                ####################################################################################################



                ####################################################################################################
                # Renderebel elements
                ####################################################################################################
                @gr.render(inputs=[
                    model_type, 
                    lora_count,
                    task, 
                    prompt_examples_count,
                    detailers,
                ])
                def rendered_elements(
                    type=model_type, 
                    lora_count=lora_count, 
                    task=task, 
                    count=prompt_examples_count,
                    detailers=detailers, 
                ):  
                    # Задание всех настроек
                    with gr.Accordion(label="Settings", open=True):
                        with gr.Accordion(label="Model:", open=True):
                            # Чекпоинт и планировщик шума
                            with gr.Row():
                                inference_params["model_name"] = gr.Dropdown(
                                    self.available_ckpt[type],
                                    label="Checkpoint", 
                                    info="Which model checkpoint you want to use?",
                                    interactive=True,
                                    visible=True,
                                    key="model_name",
                                )

                                inference_params["scheduler"] = gr.Dropdown(
                                    self.available_schedulers, 
                                    label="Scheduler:", 
                                    info="Which scheduler you want to use for denoising?",
                                    key="scheduler",
                                )

                                if type == "sdxl":
                                    inference_params["use_refiner"] = gr.Radio(
                                        ["None", "Ensemble of experts", "Two-stage"], 
                                        label="For SDXL models you can choose refiner:", 
                                        info="Which refiner extended pipeline should be used?",
                                        key="use_refiner",
                                    )

                            # Все нужные лоры
                            with gr.Row():
                                loras = []
                                scales = []
                                for i in range(lora_count):
                                    with gr.Group():
                                        with gr.Column():
                                            loras.append(gr.Dropdown(
                                                self.available_loras[type],
                                                label="LoRA", 
                                                info="Which LoRA you want to use?",
                                                interactive=True,
                                                visible=True,
                                            ))

                                            scales.append(gr.Slider(
                                                minimum=0.0,
                                                maximum=1.2,
                                                value=0.7,
                                                step=0.01,
                                                visible=True,
                                                interactive=True,
                                            ))


                        with gr.Accordion(label="Pipeline extensions:", open=True):
                            # Если нужно использовать детайлеры
                            if detailers is not None:
                                for detailer in detailers:
                                    if detailer == "FaceDetailer":
                                        inference_params[detailer] = gr.State(value="face_detailer")
                                        with gr.Accordion(label="FaceDetailer", open=False):
                                            model_path = gr.Dropdown(
                                                choices=["Bingsu/adetailer"],
                                                value="Bingsu/adetailer",
                                                visible=True,
                                                interactive=True,
                                                label="Model:", 
                                                info="Which face detailer you want to use?",
                                            )
                                            inference_params[f"{detailer}_model_path"] = model_path

                                            confidence = gr.Slider(
                                                label="Confidence",
                                                minimum=0,
                                                maximum=1.0,
                                                value=0.3,
                                                step=0.01,
                                                visible=True,
                                                interactive=True,
                                            )
                                            inference_params[f"{detailer}_confidence"] = confidence

                                            mask_dilation = gr.Slider(
                                                label="Mask delation",
                                                minimum=0,
                                                maximum=12,
                                                value=4,
                                                step=1,
                                                visible=True,
                                                interactive=True,
                                            )
                                            inference_params[f"{detailer}_mask_dilation"] = mask_dilation

                                            mask_blur = gr.Slider(
                                                label="Mask blur",
                                                minimum=0,
                                                maximum=12,
                                                value=4,
                                                step=1,
                                                visible=True,
                                                interactive=True,
                                            )
                                            inference_params[f"{detailer}_mask_blur"] = mask_blur

                                            mask_padding = gr.Slider(
                                                label="Mask padding",
                                                minimum=0,
                                                maximum=64,
                                                value=32,
                                                step=1,
                                                visible=True,
                                                interactive=True,
                                            )
                                            inference_params[f"{detailer}_mask_padding"] = mask_padding

                                            strength = gr.Slider(
                                                label="Inpaint strength",
                                                minimum=0,
                                                maximum=1.0,
                                                value=0.4,
                                                step=0.01,
                                                visible=True,
                                                interactive=True,
                                            )
                                            inference_params[f"{detailer}_strength"] = strength

                                            face_detailer_prompt = gr.Textbox(
                                                label=f"Detailer prompt:",
                                            )
                                            inference_params[f"{detailer}_prompt"] = face_detailer_prompt
                                            
                                            face_detaile_negative_prompt = gr.Textbox(
                                                label=f"Detailer negative prompt:",
                                            )
                                            inference_params[f"{detailer}_negative_prompt"] = face_detaile_negative_prompt
                                    elif detailer == "ArmDetailer":
                                        inference_params[detailer] = "arm_detailer"
                                        with gr.Accordion(label="ArmDetailer", open=False):

                                            pass


                        with gr.Accordion(label="Pipeline:", open=True):
                            with gr.Row():
                                with gr.Group():
                                    inference_params["num_inference_steps"] = gr.Slider(
                                        label="Choose number of inference steps",
                                        minimum=0,
                                        maximum=100,
                                        value=30,
                                        step=1,
                                        visible=True,
                                        interactive=True,
                                        key='num_inference_steps'
                                    )

                                    inference_params["guidance_scale"] = gr.Slider(
                                        label="Choose guidadnce scale:",
                                        minimum=0,
                                        maximum=15,
                                        value=7,
                                        step=0.1,
                                        visible=True,
                                        interactive=True,
                                        key="guidance_scale",
                                    )

                                    inference_params["cross_attention_kwargs"] = gr.Slider(
                                        label="Select LoRA's strength which apply to the text encoder:",
                                        minimum=0,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.01,
                                        visible=True,
                                        interactive=True,
                                        key="cross_attention_kwargs",
                                    )

                                    inference_params["clip_skip"] = gr.Slider(
                                        label="Clip skip:",
                                        minimum=0,
                                        maximum=4,
                                        value=0,
                                        step=1,
                                        visible=True,
                                        interactive=True,
                                        key="clip_skip",
                                    )

                                    inference_params["seed"] = gr.Slider(
                                        label="Seed:",
                                        minimum=-1,
                                        maximum=1000000000,
                                        value=-1,
                                        step=1,
                                        visible=True,
                                        interactive=True,
                                        key="seed",
                                    )

                                    inference_params["width"] = gr.Slider(
                                        label="Width:",
                                        minimum=256,
                                        maximum=2048,
                                        value=768,
                                        step=8,
                                        visible=True,
                                        interactive=True,
                                        key="width",
                                    )
                                    
                                    inference_params["height"] = gr.Slider(
                                        label="Height:",
                                        minimum=256,
                                        maximum=2048,
                                        value=768,
                                        step=8,
                                        visible=True,
                                        interactive=True,
                                        key="height",
                                    )

                                    inference_params["num_images_per_prompt"] = gr.Slider(
                                        label="Batch size:",
                                        minimum=1,
                                        maximum=16,
                                        value=1,
                                        step=1,
                                        visible=True,
                                        interactive=True,
                                        key="num_images_per_prompt",
                                    )

                                    if task is not None and task != "Text-To-Image":
                                        inference_params["strength"] = gr.Slider(
                                            label="Strength:",
                                            minimum=0,
                                            maximum=1.0,
                                            value=1.0,
                                            step=0.01,
                                            visible=True,
                                            interactive=True,
                                            key="strength",
                                        )
                                
                                if task is not None and task != "Text-To-Image":
                                    with gr.Accordion(label="Input image:", open=True):
                                        inference_params["input_image"] = gr.ImageEditor(
                                            sources=('upload'), 
                                            layers=False,
                                            brush=(
                                                gr.Brush(colors=["#FFFFFF"], color_mode="fixed")
                                                if task == "Inpainting" else
                                                None
                                            ),
                                            type='numpy',
                                            key="input_image",
                                        )


                    # Задание промптов
                    with gr.Accordion(label="Promts:", open=True):
                        prompts = []
                        negative_prompts = []
                        with gr.Row():
                            for i in range(int(count)):
                                with gr.Column():
                                    prompts.append(gr.Textbox(
                                        label=f"Prompt example {i+1}:",
                                        key=f"prompt_{i}",
                                    ))
                                    negative_prompts.append(gr.Textbox(
                                        label=f"Negative prompt example {i+1}:",
                                        key=f"negative_prompt_{i}",
                                    ))
                            
                    # Кнопка для отправки реквеста
                    GENERATE = gr.Button(
                        value="Generate",
                        visible=True,
                        interactive=True,
                    )
                        
                    # Аутпут полученных результатов
                    with gr.Row():
                        outputs = []
                        for i in range(int(count)):
                            outputs.append(gr.Gallery(
                                label="Generated images", 
                                show_label=True, 
                                object_fit="contain", 
                                height="auto",
                            ))
                
                    # Сохраняем ключи и значения для передачи в конфиг
                    btn_input = [
                        gr.State(value=list(inference_params.keys())),
                    ] + list(inference_params.values()) + loras + scales + prompts + negative_prompts

                    GENERATE.click(
                        self.create_and_send_request,
                        inputs=btn_input,
                        outputs=outputs,
                    )
                ####################################################################################################
                            

            with gr.Tab("Train"):
                gr.Markdown("Когда-нибудь я сделаю UI и для обучения моделей")
                gr.Markdown("Не то что бы нам очень надо, но просто интересно дать за щёку Kohya_ss")
        
        demo.launch(**kwargs)


    def create_and_send_request(self, keys, *args):
        # 1. Извлекаем переданные параметры обратно
        inference_params = dict(zip(keys, list(args[:len(keys)])))
        lora_count = inference_params["lora_count"]
        task = inference_params["task"]
        prompt_examples_count = int(inference_params["prompt_examples_count"])
        batch_size = int(inference_params["num_images_per_prompt"],)
        

        # 2. Собираем конфиг модели
        lora_names, lora_scales = [], []
        for i in range(lora_count):
            if args[len(keys)+i] != []:
                lora_names.append(args[len(keys)+i])
                lora_scales.append(args[len(keys)+i+lora_count])
        loras = dict(zip(lora_names, lora_scales))
            
        model = {
            "type": inference_params["model_type"],
            "name": inference_params["model_name"],
            "scheduler": inference_params["scheduler"],
        }
        if loras != {}:
            model["loras"] = loras


        # 3. Собираем расширения пайплайна
        pipeline_extensions = {
            # "detailers": {
            #     "face_detailer": {
            #         "model": {
            #             "repo_id": "Bingsu/adetailer",
            #             "filename": "face_yolov8n.pt",
            #             "confidence": 0.3
            #         },

            #         "params": {
            #             "mask_dilation": 4,
            #             "mask_blur": 4,
            #             "mask_padding": 32,
            #             "strength": 0.4
            #         },
                    
            #         "face_detailer_prompt": [],
            #         "face_detaile_negative_prompt": []
            #     },  
                
            #     "arm_detailer": {
            #         "model": {

            #         },
            #         "params": {
                        
            #         },
            #         "arm_detailer_prompt": [],
            #         "arm_detaile_negative_prompt": []
            #     }
            # },

            # "controlnets": {},
        }
        detailers = {}
        if "FaceDetailer" in list(inference_params.keys()):
            detailers[inference_params["FaceDetailer"]] = {
                "model": {
                    "model_path": inference_params["FaceDetailer_model_path"],
                    "confidence": inference_params["FaceDetailer_confidence"],
                },
                "params": {
                    "mask_dilation": inference_params["FaceDetailer_mask_dilation"],
                    "mask_blur": inference_params["FaceDetailer_mask_blur"],
                    "mask_padding": inference_params["FaceDetailer_mask_padding"],
                    "strength": inference_params["FaceDetailer_strength"],
                },
                "face_detailer_prompt": inference_params["FaceDetailer_prompt"],
                "face_detailer_negative_prompt": inference_params["FaceDetailer_negative_prompt"],
            }
        if "ArmDetailer" in list(inference_params.keys()):
            pass
        
        pipeline_extensions = {}
        if detailers != {}:
            pipeline_extensions["detailers"] = detailers


        # 4. Соберём конфиг базовых параметров генерации
        params = {
            "num_inference_steps": inference_params["num_inference_steps"],
            "guidance_scale": inference_params["guidance_scale"],
            "cross_attention_kwargs": {"scale": inference_params["cross_attention_kwargs"]},
            "height": inference_params["height"],
            "width": inference_params["width"],
            "num_images_per_prompt": batch_size,
        }
        if inference_params["clip_skip"] != 0:
            params["clip_skip"] = inference_params["clip_skip"]
        if inference_params["seed"] != -1:
            params["seed"] = inference_params["seed"]
        if task == "Image-To-Image":
            # Берём картинку c рисуночками -> "composite"
            img = np.ascontiguousarray(inference_params["input_image"]["composite"])
            pil_img = Image.fromarray(img).convert("RGB")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            # Добавляем необходимые параметры
            params["image"] = base64_str
            params["strength"] = inference_params["strength"]
        if task == "Inpainting":
            # Берём саму картинку
            img = np.ascontiguousarray(inference_params["input_image"]["background"])
            pil_img = Image.fromarray(img).convert("RGB")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            # Берём маску
            img_mask = np.ascontiguousarray(inference_params["input_image"]["layers"][0])
            pil_img_mask = Image.fromarray(img_mask).convert("RGB")
            buffer_mask = io.BytesIO()
            pil_img_mask.save(buffer_mask, format="JPEG")
            base64_str_mask = base64.b64encode(buffer_mask.getvalue()).decode('utf-8')
            # Добавляем необходимые параметры
            params["image"] = base64_str
            params["mask_image"] = base64_str_mask
            params["strength"] = inference_params["strength"]


        # 5. Собираем промпты
        prompt, negative_prompt = [], []
        for i in range(prompt_examples_count) :
            prompt.append(args[len(keys) + 2*lora_count + i])
            negative_prompt.append(args[len(keys) + 2*lora_count + prompt_examples_count + i])


        # 6. Собираем полностью весь конфиг для генерации
        input_request = {
            "model": model,
            "pipeline": params,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        if pipeline_extensions != {}:
            input_request["pipeline_extensions"] = pipeline_extensions


        # 7. Отправляем запрос
        self._save_last_request({"input": input_request})
        # response = self.request_to_endpoint({"input": input_request})
        

        # # 8. Обрабатываем полученные результаты
        # # TODO: Переделать логику с учётом prompt_examples_count и batch_size
        # gallery = []
        # for k in range(prompt_examples_count):
        #     images = []
        #     subgallery = response.json()['output']['images'][k*batch_size : (k+1)*batch_size]
        #     for i, base64_string in enumerate(subgallery):
        #         img = Image.open(BytesIO(base64.b64decode(base64_string)))
        #         images.append([img, f"{prompt[k]} {i}"])

        #     gallery.append(images)

        # # gallery = [[images[i], f"{prompt[i]}"] for i in range(len(prompt))]
        # return gallery
        return list()


    def request_to_endpoint(self, request):
        api_key = os.getenv("RUNPOD_API_KEY")
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
        headers = {
            "accept": "application/json",
            "authorization": api_key,
            "content-type": "application/json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(request)) 

        return response
        

    def _save_last_request(self, request):
        with open("test_input.json", 'w') as file:
            json.dump(request, file)
    

    def _get_avaliable_checkpoints(self, hf_author):
        available_ckpt = {
            "sd15": [],
            "sdxl": [],
            # "sd3": [],
        }

        hf_api = HfApi()
        all_avaliable_models = hf_api.list_models(author=hf_author)

        for model in all_avaliable_models:
            # Это тот же самый ckpt_path, который юзается в SDModelWrapper
            ckpt_path = model.id

            if ckpt_path.startswith(f"{hf_author}/sd15_"):
                available_ckpt["sd15"].append(ckpt_path[len(f"{hf_author}/sd15_"): ])
            elif ckpt_path.startswith(f"{hf_author}/sdxl_"):
                available_ckpt["sdxl"].append(ckpt_path[len(f"{hf_author}/sdxl_"): ])
            elif ckpt_path.startswith(f"{hf_author}/sd3_"):
                available_ckpt["sd3"].append(ckpt_path[len(f"{hf_author}/sd3_"): ])


        return available_ckpt


    def _get_avaliable_loras(self, hf_path):
        avaliable_loras = {
            "sd15": [],
            "sdxl": [], 
            # "sd3": [],
        }

        fs = HfFileSystem()
        files = fs.ls(hf_path, detail=False)

        for file in files:
            if file.endswith(".safetensors"):
                if file.startswith(f"{hf_path}/sd15_"):
                    avaliable_loras["sd15"].append(file[len(f"{hf_path}/sd15_"): -len(".safetensors")])
                elif file.startswith(f"{hf_path}/sdxl_"):
                    avaliable_loras["sdxl"].append(file[len(f"{hf_path}/sdxl_"): -len(".safetensors")])
                elif file.startswith(f"{hf_path}/sd3_"):
                    avaliable_loras["sd3"].append(file[len(f"{hf_path}/sd3_"): -len(".safetensors")]) 

        return avaliable_loras

        

app = App()


