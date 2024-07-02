import gradio as gr


def generate_images():
    return [["https://via.placeholder.com/150", "Example Image 1"], 
            ["https://via.placeholder.com/150", "Example Image 2"]] 


class App():
    def __init__(self):
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
                ####################################################################################################
                # Блок выбора модели для генерации
                ####################################################################################################
                with gr.Blocks(title="Model Selector"):
                    with gr.Group():
                        model_type = gr.Radio(
                            ["sd15", "sdxl", "sd3"], 
                            label="Stable Diffusion model type:", 
                            info="Which model architecture should be used?",
                            key=1,
                        )
                        
                        with gr.Accordion(label="Generation procedure configuration", open=False):
                            with gr.Row():
                                with gr.Column():
                                    model_name = gr.Dropdown(
                                        [],
                                        label="Checkpoint", 
                                        info="Which model checkpoint you want to use?",
                                        interactive=True,
                                        visible=False,
                                        key=2, 
                                    )

                                    # use_refiner = gr.Radio(
                                    #     ["None", "Ensemble of experts", "Two-stage"], 
                                    #     label="For SDXL models you can choose refiner:", 
                                    #     info="Which refiner extended pipeline should be used?",
                                    #     key=3,
                                    # )

                                selected_loras = gr.CheckboxGroup(
                                    [],
                                    label="Available LoRA adapters:", 
                                    info="Which LoRA adapter you want to use?",
                                    interactive=True,
                                    visible=False,
                                    key=4,
                                )


                            lora_sliders_block = gr.Column(visible=False)

                            # @gr.render(inputs=[selected_loras])
                            # def lora_sliders(loras=selected_loras, type=model_type):
                            #     for lora_name in loras:
                            #         weight = gr.Slider(
                            #             0, 
                            #             2.0, 
                            #             step=0.01, 
                            #             label=lora_name, 
                            #             interactive=True, 
                            #             key=lora_name,
                            #         )

                            with gr.Column():
                                scheduler = gr.Dropdown(
                                    self.available_schedulers, 
                                    label="Scheduler:", 
                                    info="Which scheduler you want to use for denoising?"
                                )

                def update_model(type: str):
                    checkpoints = self.available_ckpt[type]
                    loras = self.available_loras[type]
                    
                    return {
                        model_name: gr.update(choices=checkpoints, visible=bool(checkpoints)),
                        selected_loras: gr.update(choices=loras, visible=bool(loras)),
                        lora_sliders_block: gr.update(visible=True)
                    }

                model_type.change(
                    update_model,
                    inputs=model_type,
                    outputs=[model_name, selected_loras, lora_sliders_block]
                    # outputs=[model_name, selected_loras]
                )

                def update_sliders(loras):
                    new_sliders = [
                        gr.Slider(0, 2.0, 0.01, label=lora_name, visible=True)
                        for lora_name in loras
                    ]
                    return {
                        lora_sliders_block: gr.update(new_sliders, visible=bool(new_sliders)),
                    }
                
                selected_loras.change(
                    update_sliders,
                    inputs=selected_loras,
                    outputs=lora_sliders_block
                )
                ####################################################################################################


                ####################################################################################################
                # Блок для выбора параметров генерации
                ####################################################################################################
                with gr.Blocks(title="Params selector"):
                    with gr.Group():
                        task = gr.Radio(
                            ["Text-To-Image", "Image-To-Image", "Inpainting"],
                            label="Generation task:",
                            info="Which conditional generation task you want to solve?",
                            interactive=True,
                            visible=True,
                        )

                        with gr.Accordion(label="Generation procedure configuration", open=False):
                            @gr.render(inputs=task)
                            def generation_params_renderer(task=task):
                                with gr.Row():
                                    # with gr.Column():
                                    with gr.Blocks() and gr.Group():
                                        num_inference_steps = gr.Slider(
                                            label="Choose number of inference steps",
                                            minimum=0,
                                            maximum=100,
                                            step=1,
                                            visible=True,
                                            interactive=True,
                                        )

                                        guidance_scale = gr.Slider(
                                            label="Choose guidadnce scale:",
                                            minimum=0,
                                            maximum=15,
                                            step=0.1,
                                            visible=True,
                                            interactive=True,
                                        )

                                        cross_attention_kwargs = gr.Slider(
                                            label="Select LoRA's strength which apply to the text encoder:",
                                            minimum=0,
                                            maximum=1.0,
                                            step=0.01,
                                            visible=True,
                                            interactive=True,
                                        )

                                        clip_skip = gr.Slider(
                                            label="Select number of text encoder layers to be skipped:",
                                            minimum=0,
                                            maximum=4,
                                            step=1,
                                            visible=True,
                                            interactive=True,
                                        )

                                        seed = gr.Slider(
                                            label="Seed for generating pictures:",
                                            minimum=-1,
                                            maximum=1000000000,
                                            step=1,
                                            visible=True,
                                            interactive=True,
                                        )

                                        width = gr.Slider(
                                            label="Width:",
                                            minimum=256,
                                            maximum=2048,
                                            step=1,
                                            visible=True,
                                            interactive=True,
                                        )
                                        
                                        height = gr.Slider(
                                            label="Height:",
                                            minimum=256,
                                            maximum=2048,
                                            step=1,
                                            visible=True,
                                            interactive=True,
                                        )

                                        num_images_per_prompt = gr.Slider(
                                            label="Choose the number of images per prompt:",
                                            minimum=1,
                                            maximum=16,
                                            step=1,
                                            visible=True,
                                            interactive=True,
                                        )

                                        if task is not None and task != "Text-To-Image":
                                            strength = gr.Slider(
                                                label="Strength:",
                                                minimum=0,
                                                maximum=1.0,
                                                step=0.01,
                                                visible=True,
                                                interactive=True,
                                            )

                                    if task is not None and task != "Text-To-Image":
                                        with gr.Column():
                                            input_image = gr.ImageEditor(
                                                sources='upload',
                                                height='auto',
                                                label="Upload",
                                            )
                                                    

                            # with gr.Row():
                            #     with gr.Column():
                            #         with gr.Group():
                            #             num_inference_steps = gr.Slider(
                            #                 label="Choose number of inference steps",
                            #                 minimum=0,
                            #                 maximum=100,
                            #                 step=1,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )

                            #             guidance_scale = gr.Slider(
                            #                 label="Choose guidadnce scale:",
                            #                 minimum=0,
                            #                 maximum=15,
                            #                 step=0.1,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )

                            #             cross_attention_kwargs = gr.Slider(
                            #                 label="Select LoRA's strength which apply to the text encoder:",
                            #                 minimum=0,
                            #                 maximum=1.0,
                            #                 step=0.01,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )

                            #             clip_skip = gr.Slider(
                            #                 label="Select number of text encoder layers to be skipped:",
                            #                 minimum=0,
                            #                 maximum=4,
                            #                 step=1,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )

                            #             seed = gr.Slider(
                            #                 label="Seed for generating pictures:",
                            #                 minimum=-1,
                            #                 maximum=1000000000,
                            #                 step=1,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )

                            #             width = gr.Slider(
                            #                 label="Width:",
                            #                 minimum=256,
                            #                 maximum=2048,
                            #                 step=1,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )
                                        
                            #             height = gr.Slider(
                            #                 label="Height:",
                            #                 minimum=256,
                            #                 maximum=2048,
                            #                 step=1,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )

                            #             num_images_per_prompt = gr.Slider(
                            #                 label="Choose the number of images per prompt:",
                            #                 minimum=1,
                            #                 maximum=16,
                            #                 step=1,
                            #                 visible=True,
                            #                 interactive=True,
                            #             )

                            #             strength = gr.Slider(
                            #                 label="Strength:",
                            #                 minimum=0,
                            #                 maximum=1.0,
                            #                 step=0.01,
                            #                 visible=False,
                            #                 interactive=False,
                            #             )

                            #     @gr.render(inputs=task)
                            #     def extra_parameters_renderer(task=task):
                            #         if task is not None and task != "Text-To-Image":                                   
                            #             with gr.Column():
                            #                 input_image = gr.ImageEditor(
                            #                     sources='upload',
                            #                     # tool='sketch',
                            #                     label="Upload",
                            #                 )

                            #             return {
                            #                 strength: gr.update(visible=True, interactive=True)
                            #             }
                ####################################################################################################


                ####################################################################################################
                # Блок для выбора количества промптов и количество различных вариантов генерации
                ####################################################################################################

                ####################################################################################################

                
                ####################################################################################################
                # Тупо отдельная кнопочка для генерации
                ####################################################################################################
                with gr.Blocks(title="Generate button"):
                    GENERATE = gr.Button(
                        value="Generate",
                        visible=True,
                        interactive=True,
                    )
                ####################################################################################################
                
                
                ####################################################################################################
                # TODO: ВОТ этот блок тоже имеет смысл сделать адаптивным
                # чтобы количество галерей картинок зависило от количества промптов
                ####################################################################################################
                with gr.Blocks(title="Output results"):
                    output = gr.Gallery(
                        label="Generated images", 
                        show_label=True, 
                        object_fit="contain", 
                        height="auto",
                    )

                    GENERATE.click(fn=generate_images, outputs=output)
                ####################################################################################################
                    
            
            with gr.Tab("Train"):
                gr.Markdown("Когда-нибудь я сделаю UI и для обучения моделей")
                gr.Markdown("Не то что бы нам очень надо, но просто интересно дать за щёку Kohya_ss")
        
        # Launch the Application!
        demo.launch()



app = App()
