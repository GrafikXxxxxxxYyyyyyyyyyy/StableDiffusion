import io
import os
import json
import base64
import requests
import numpy as np
import gradio as gr

from PIL import Image
from io import BytesIO

from pipeline_extensions.detailer.detailer_model import DetailerModel
from pipeline_extensions.detailer.detailer_pipeline import DetailerPipeline, convert_pt_to_pil

class App():
    def __init__(
        self, 
    ):
        self.launch()


    def launch(self, **kwargs):
        layout_block = gr.Blocks()
        with layout_block as demo:
            with gr.Row():
                input = gr.Image(
                    sources=('upload'), 
                    type='pil',
                    key="input_image",
                )
                
                output_mask = gr.Gallery(
                    label="Generated images", 
                    show_label=True, 
                    object_fit="contain", 
                    height="auto",
                )

                output_pipeline = gr.Gallery(
                    label="Generated images", 
                    show_label=True, 
                    object_fit="contain", 
                    height="auto",
                )

            face_detailer = gr.Button(
                value="Face detector",
                visible=True,
                interactive=True,
            )
            
            face_detailer.click(
                self.face_detailer_masks,
                inputs=input,
                outputs=[output_mask, output_pipeline],
            )

        demo.launch(**kwargs)


    def face_detailer_masks(self, image: Image.Image):
        detector = DetailerModel()
        masks = detector(image)
        gallery_1 = []
        for i, mask in enumerate(masks):
            gallery_1.append([mask, f"mask_{i+1}"])

        pipeline = DetailerPipeline()
        gallery_2 = []
        for i, mask in enumerate(masks):
            mask = mask.convert("L")
            # Отступ
            mask = pipeline.mask_dilate(mask)
            # Calculates the bounding box of the non-zero regions in the
            bbox = mask.getbbox()
            # Размываем маску
            mask = pipeline.mask_gaussian_blur(mask)
            bbox_padded = pipeline.bbox_padding(bbox, image.size)
            print("padded dim:", bbox_padded)
            # Получаем обрезанное изображение и его маску
            crop_image = image.crop(bbox_padded)
            crop_mask = mask.crop(bbox_padded)

            gallery_2.append([crop_image, f"pipeline_image"])
            gallery_2.append([crop_mask, f"pipeline_mask"])

        return (gallery_1, gallery_2)

App()