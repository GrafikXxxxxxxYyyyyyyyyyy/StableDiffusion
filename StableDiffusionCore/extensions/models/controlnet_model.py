from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


class StableDiffusionControlNetModel():
    def __init__(
        self, 
        model_type: str,
        control_model: str = "canny",
        confidence: float = 0.3,
    ):  
        if control_model == "canny":
            self.controlnet = ControlNetModel.from_pretrained(
                # model_path, 
                use_safetensors=True,
            )