from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from StableDiffusionCore.sd_unified_model import StableDiffusionUnifiedModel
from StableDiffusionCore.pipelines.sd_multitask_pipeline import StableDiffusionMultitaskPipeline



class StableDiffusionUnifiedPipeline():
    def __init__(
        self, 
        do_cfg: bool = True,
        device: Optional[str] = None,
    ):
        self.pipeline = StableDiffusionMultitaskPipeline(
            do_cfg,
            device,
        )


    def __call__(
        self, 
        model: StableDiffusionUnifiedModel, 
        refiner: Optional[str] = None,
        **kwargs
    ):  
        """
        По сути просто оборачивает пайплайн рефайнера
        """
        if refiner is None:
            model.switch_denoising_model("base")
            print("Processing default pipeline...")
            return self.pipeline(
                model,
                **kwargs,
            )
        elif refiner == "Ensemble of experts":
            print("Processing 'Ensemble of experts' pipeline...")
            model.switch_denoising_model("base")
            kwargs["denoising_end"] = 0.8
            latents = self.pipeline(
                model, 
                output_type='latents',
                **kwargs,
            )

            model.switch_denoising_model("refiner")
            kwargs["denoising_start"] = kwargs.pop("denoising_end")
            return self.pipeline(
                model,
                refiner_latents=latents,   
                **kwargs,
            )
        elif refiner ==  "Two-stage":
            print("Processing 'Two-stage' pipeline...")
            model.switch_denoising_model("base")
            image = self.pipeline(
                model, 
                output_type='pt',
                **kwargs,
            )

            model.switch_denoising_model("refiner")
            return self.pipeline(
                model,
                output_type='pt',
                image=image,
                **kwargs,
            )