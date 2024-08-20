import torch

from typing import Any, Callable, Dict, List, Optional, Union, Tuple



class ConditionedDiffusionModel:
    """
    Представляет из себя обёртку над обусловленной какой-то информацией 
    диффузионной моделью, предоставляет функционал для взаимодействия, как
    с самой диффузионной моделью, так и с обуславливающей
    """
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        device: str = "cuda",
    ):
        pass



    