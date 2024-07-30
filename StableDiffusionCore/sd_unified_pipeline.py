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

        self.model: StableDiffusionUnifiedModel = None



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
            print("Processing default pipeline...")
            return self.pipeline(
                model,
                **kwargs,
            )
        elif refiner == "Ensemble of experts":
            print("Processing 'Ensemble of experts' pipeline...")
            latents = self.pipeline(
                model, 
                denoising_end=0.8,
                output_type='latents',
                **kwargs,
            )

            return self.pipeline(
                model,
                use_refiner=True,
                denoising_start=0.8,
                image=latents,   
                **kwargs,
            )
        elif refiner ==  "Two-stage":
            print("Processing 'Two-stage' pipeline...")
            image = self.pipeline(
                model, 
                output_type='pt',
                **kwargs,
            )

            return self.pipeline(
                model,
                use_refiner=True,
                output_type='pt',
                image=image,   
                **kwargs,
            )