import time
import wandb
import torch
import argparse
import itertools 
import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torchvision import transforms
from accelerate import Accelerator
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, AutoencoderTiny #,PNDMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from typing import Any, Callable, Dict, List, Optional, Union


# TODO: Добавить prior preservation
class StableDiffusionTrainer ():
    def __init__(self):
        pass


    def __call__(
        self, 
        train_loader,
        mixed_precision
        # args, 
    ):
        """
        Pipeline for train diffusion model
        """
        # 1. Инитим акселератор и device
        accelerator = None
        device = self.device
        if not self.device == torch.device("cpu"):
            accelerator = Accelerator(
                mixed_precision=mixed_precision,
            )
            device = accelerator.device
            # Disable AMP for MPS.
            if torch.backends.mps.is_available():
                accelerator.native_amp = False

        