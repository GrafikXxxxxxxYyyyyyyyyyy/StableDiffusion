import inspect
from typing import Any, Callable, Iterable, List, Mapping, Optional

# Собственно проглядывается ещё старая структура, которую я и начал менять посреди установки рефайнера
# from StableDiffusion.sd import StableDiffusionUnifiedPipeline
# from pipeline_extensions.detailer.detailer_model import DetailerModel

from dataclasses import dataclass
import cv2
import torch
import PIL
import numpy as np
from diffusers.utils import BaseOutput
from PIL import Image, ImageFilter, ImageOps
from ultralytics import YOLO 
from pathlib import Path


def convert_pt_to_pil(images: torch.Tensor) -> List[PIL.Image.Image]:
    pil_images = []
    for idx in range(len(images)):
        img = (images[idx] / 2 + 0.5).clamp(0, 1)
        img = (img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        pil_image = Image.fromarray(np.ascontiguousarray(img))
        pil_images.append(pil_image)

    return pil_images



@dataclass
class ADOutput(BaseOutput):
    images: list[Image.Image]
    init_images: list[Image.Image]



class StableDiffusionDetailerPipeline():
    def __init__(
        self, 
        mask_dilation: int = 4,
        mask_blur: int = 4,
        mask_padding: int = 32, 
        strength: float = 0.4,
    ):  
        self.mask_dilation = mask_dilation
        self.mask_blur = mask_blur
        self.mask_padding = mask_padding
        self.strength = strength

        self.pipeline: StableDiffusionUnifiedPipeline = None


    def __call__(
        self, 
        images: List[Image.Image],
        detailer_model: DetailerModel,
        **inference_kwargs,
    ):  
        """
        Пайплайн для любого типа детейлера
        Принимает на вход изображения и производит детализацию соответствующих частей
        За которые отвечает переданная модель детектора
        """
        init_images = []
        final_images = []
        # Цикл по переданным картинкам
        for i, init_image in enumerate(images):
            init_images.append(init_image.copy())
            final_image = None

            # Получаем маски от детектора
            masks = detailer_model(init_image)
            # Цикл по макскам найденных на картинке объектов
            for k, mask in enumerate(masks):
                mask = mask.convert("L")
                # Отступ
                mask = self.mask_dilate(mask)
                # Calculates the bounding box of the non-zero regions in the
                bbox = mask.getbbox()
                # Размываем маску
                mask = self.mask_gaussian_blur(mask)
                bbox_padded = self.bbox_padding(bbox, init_image.size)
                print("padded dim:", bbox_padded)

                ########################################################
                # Inpainting mask region
                ########################################################
                # Вырезаем из изображения лицо и его маску
                crop_image = init_image.crop(bbox_padded)
                crop_mask = mask.crop(bbox_padded)

                # Добавляем вырезанные изображения в Inpainting pipeline
                inference_kwargs["image"] = crop_image
                inference_kwargs["mask_image"] = crop_mask
                inference_kwargs["strength"] = self.strength
                
                print(f"Detailer '{detailer_model.type}' inpainting...")
                inpaint_output = pipeline(**inference_kwargs)
                if isinstance(inpaint_output, torch.Tensor):
                    inpaint_output = convert_pt_to_pil(inpaint_output)
                ########################################################
                
                inpaint_image = inpaint_output[0]
                final_image = self.composite(
                    init_image,
                    mask,
                    inpaint_image,
                    bbox_padded,
                )
                init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)
    

    def mask_dilate(
        self, 
        image: Image.Image, 
    ) -> Image.Image:
        if self.mask_dilation <= 0:
            return image

        arr = np.array(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.mask_dilation, self.mask_dilation))
        dilated = cv2.dilate(arr, kernel, iterations=1)

        return Image.fromarray(dilated)
    

    def mask_gaussian_blur(
        self, 
        image: Image.Image, 
    ) -> Image.Image:
        if self.mask_blur <= 0:
            return image

        blur = ImageFilter.GaussianBlur(self.mask_blur)
        return image.filter(blur)
    

    def bbox_padding(
        self,
        bbox: tuple[int, int, int, int], 
        image_size: tuple[int, int], 
    ) -> tuple[int, int, int, int]:
        if self.mask_padding <= 0:
            return bbox

        arr = np.array(bbox).reshape(2, 2)
        arr[0] -= self.mask_padding
        arr[1] += self.mask_padding
        arr = np.clip(arr, (0, 0), image_size)
        
        return tuple(arr.flatten())
    

    def composite(
        init: Image.Image,
        mask: Image.Image,
        gen: Image.Image,
        bbox_padded: tuple[int, int, int, int],
    ) -> Image.Image:
        img_masked = Image.new("RGBa", init.size)
        img_masked.paste(
            init.convert("RGBA").convert("RGBa"),
            mask=ImageOps.invert(mask),
        )
        img_masked = img_masked.convert("RGBA")

        size = (
            bbox_padded[2] - bbox_padded[0],
            bbox_padded[3] - bbox_padded[1],
        )
        resized = gen.resize(size)

        output = Image.new("RGBA", init.size)
        output.paste(resized, bbox_padded)
        output.alpha_composite(img_masked)

        return output.convert("RGB")





