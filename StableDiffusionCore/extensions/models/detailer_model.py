import cv2
import inspect
import numpy as np

from pathlib import Path
from ultralytics import YOLO 
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from torchvision.transforms.functional import to_pil_image
from typing import Any, Callable, Iterable, List, Mapping, Optional



class StableDiffusionDetailerModel():
    def __init__(
        self, 
        model_type: str = "face_detailer",
        model_path: str = "Bingsu/adetailer", 
        filename: str = "face_yolov8n.pt",
        confidence: float = 0.3,
    ):  
        if model_type == "face_detailer":
            model_path = hf_hub_download(model_path, filename)
            self.model = YOLO(model_path)    

        self.type = model_type
        self.confidence = confidence    


    def __call__(
        self, 
        image: Image.Image, 
    ):     
        """
        Возвращает маски с детектированными объектами
        """
        # Проверка типа входного изображения
        if not isinstance(image, Image.Image):
            raise ValueError("Входное изображение должно быть типа PIL.Image.Image")
        
        # Получаем прогноз детектора
        pred = self.model(image, conf=self.confidence)

        # Вытаскиваем из этого прогноза координаты ббокса 
        # с детектированными на изображении объектами
        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        if bboxes.size == 0:
            return None

        # Если детектор не создал маски, то создадим
        if pred[0].masks is None:
            masks = []
            for bbox in bboxes:
                # Создаёт чёрную картинку нужного размера
                mask = Image.new("L", image.size, "black")
                # Хз каким механизмом, но типа позволяет на этой картинке что-то отрисовать
                mask_draw = ImageDraw.Draw(mask)
                # Рисует в ббоксе белый прямоугольник
                mask_draw.rectangle(bbox, fill="white")
                masks.append(mask)
        else:
            masks = pred[0].masks.data
            masks = [to_pil_image(masks[i], mode="L").resize(image.size) for i in range(masks.shape[0])]
    
        return masks
    







