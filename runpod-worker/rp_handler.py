import os
import sys
import torch
import runpod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handler_logic import Handler
from StableDiffusionCore.sd_unified_model import StableDiffusionUnifiedModel

torch.cuda.empty_cache()

# Select device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize default model
MODEL = StableDiffusionUnifiedModel(device=DEVICE, model_type='sdxl', model_name='Juggernaut')

# Initialize Handler class
HANDLER = Handler(device=DEVICE)

@torch.inference_mode()
def trigger_fn(request):
    '''
    Generates response from user request using Handler multifunctional class

    Request structure:
        request = {
            "id": str,
            "input": {
                "mode": Optional[str],
                "model": Optional[dict],
                "params": Optional[dict],
                "prompt": Union[str, list]
                ...
            }
        }
    '''
    return HANDLER(MODEL, request["input"], request["id"])

runpod.serverless.start({"handler": trigger_fn})