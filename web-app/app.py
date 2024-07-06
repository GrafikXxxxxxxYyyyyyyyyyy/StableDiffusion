import io
import os
import json
import base64
import requests
import numpy as np
import gradio as gr
from PIL import Image


class App():
    def __init__(self):
        self.endpoint_id = ""
        self.available_ckpt = {
            "sd15": ["Default", "NeverEndingDream"],
            "sdxl": ["Default", "Juggernaut"], 
            "sd3": []
        }
        self.available_loras = {
            "sd15": ['ciri', 'makima'],
            "sdxl": ['melanie', 'tsunade'], 
            "sd3": []
        }
        self.available_schedulers = ["DDIM", "euler", "euler_a", "DPM++ 2M SDE Karras"]

        self.launch()


    def launch(self):
        layout_block = gr.Blocks()
        with layout_block as demo:
            with gr.Tab("Inference"):
                inference_params = {}
                ####################################################################################################
                # Not renderebel elements
                ####################################################################################################
                with gr.Blocks(title="Global configuration"):
                    with gr.Accordion(label="Global configuration:", open=True):
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

                            with gr.Column():
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
                ####################################################################################################


                ####################################################################################################
                # Renderebel elements
                ####################################################################################################
                @gr.render(inputs=[model_type, lora_count, prompt_examples_count, task])
                def rendered_elements(type=model_type, lora_count=lora_count, prompt_examples_count=prompt_examples_count, task=task):
                    with gr.Blocks(title="Model configuration"):
                        with gr.Accordion(label="Model configuration:", open=True):
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

                            # Все нужные параметры для выбранной задачи
                            with gr.Row():
                                with gr.Accordion(label="Generation parameters:", open=True):
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
                                            layers=True if task == "Inpainting" else False,
                                            type='pil',
                                            key="input_image",
                                        )

                        # Задание промптов
                        with gr.Accordion(label="Promts:", open=True):
                            prompts = []
                            negative_prompts = []
                            with gr.Row():
                                for i in range(int(prompt_examples_count)):
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
                            for i in range(int(prompt_examples_count)):
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
                        # outputs=outputs,
                        outputs=[],
                    )
                ####################################################################################################
                            


            with gr.Tab("Train"):
                gr.Markdown("Когда-нибудь я сделаю UI и для обучения моделей")
                gr.Markdown("Не то что бы нам очень надо, но просто интересно дать за щёку Kohya_ss")
        
        # Launch the Application!
        demo.launch()


    def create_and_send_request(self, keys, *args):
        inference_params = dict(zip(keys, list(args[:len(keys)])))
        lora_count = inference_params["lora_count"]
        task = inference_params["task"]
        prompt_examples_count = inference_params["prompt_examples_count"]
        
        # Учитываем выбранные лоры
        lora_names, lora_scales = [], []
        for i in range(lora_count):
            if args[len(keys)+i] != []:
                lora_names.append(args[len(keys)+i])
                lora_scales.append(args[len(keys)+i+lora_count])
        loras = dict(zip(lora_names, lora_scales))
            
        # Соберём конфиг модели
        model = {
            "type": inference_params["model_type"],
            "name": inference_params["model_name"],
            "scheduler": inference_params["scheduler"],
        }
        if loras != {}:
            model["loras"] = loras

        # Соберём конфиг базовых параметров генерации
        params = {
            "num_inference_steps": inference_params["num_inference_steps"],
            "guidance_scale": inference_params["guidance_scale"],
            "cross_attention_kwargs": {"scale": inference_params["cross_attention_kwargs"]},
            "height": inference_params["height"],
            "width": inference_params["width"],
            "num_images_per_prompt": inference_params["num_images_per_prompt"],
        }
        if inference_params["clip_skip"] != 0:
            params["clip_skip"] = inference_params["clip_skip"]
        if inference_params["seed"] != -1:
            params["seed"] = inference_params["seed"]
        # if task != "Text-To-Image":
        #     pil_img = inference_params["input_image"]["layers"][0].convert("RGB")
        #     buffer = io.BytesIO()
        #     pil_img.save(buffer, format="JPEG")s
        #     base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        #     params["image"] = base64_str
        #     params["strength"] = inference_params["strength"]

        #     if task == "Inpainting":
        #         pil_img = inference_params["input_image"]["layers"][1].convert("RGB")
        #         buffer = io.BytesIO()
        #         pil_img.save(buffer, format="JPEG")
        #         base64_str_mask = base64.b64encode(buffer.getvalue()).decode('utf-8')

        #         params["mask_image"] = base64_str_mask

        # Собираем промпты
        prompt, negative_prompt = [], []
        for i in range(prompt_examples_count) :
            prompt.append(args[len(keys) + 2*lora_count + i])
            negative_prompt.append(args[len(keys) + 2*lora_count + prompt_examples_count + i])

        # Собираем полностью весь конфиг для генерации
        input_request = {
            "model": model,
            "params": params,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        # response = self.request_to_endpoint({"input": input_request})
        self.save_last_request({"input": input_request})

    
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
        

    def save_last_request(self, request):
        with open("input_request.json", 'w') as file:
            json.dump(request, file)



app = App()


