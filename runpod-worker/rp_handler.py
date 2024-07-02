import os
import sys
import torch
import runpod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handler_logic import Handler
from models.stable_diffusion import SDModelWrapper
from huggingface_hub import HfFileSystem, hf_hub_download

torch.cuda.empty_cache()

# Select device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Опять же пока не решено, что делать с хранилищем, пусть будет так
# def preload_models(checkpoits: list, need_loras=True):
#     # Подгружаем в кеш все модели из тех что имеем
#     model: SDModelWrapper
    # for ckpt in checkpoits:
    #     model = SDModelWrapper(device=DEVICE, model_type='sdxl', model_name=ckpt)
#     # Аналогично поступаем с лорами если надо
#     if need_loras:
#         hf_path = "OnMoon/loras"
#         files = HfFileSystem().ls(hf_path, detail=False)
#         for file in files:
#             if file.endswith(".safetensors") and file.startswith(f"{hf_path}/sdxl_"):
#                 lora_weights = hf_hub_download(
#                     repo_id = hf_path,
#                     filename = file[len(f"{hf_path}/"):],
#                 )
# preload_models(['AutismMix', 'Juggernaut', 'AnimaPencil'], need_loras=False)

# Initialize default model
MODEL = SDModelWrapper(device=DEVICE, model_type='sdxl', model_name='AnimaPencil')
# models = {
#     "ckpt_url": SDModelWrapper("ckpt_url", device=DEVICE)
# }

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