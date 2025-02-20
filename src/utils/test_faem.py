import csv
import pickle
import math
import random
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from faesm.esm import FAEsmForMaskedLM
from esm import Alphabet

class ModelSelector:

    MODEL_MAPPING = {
        "8M":   {"model_name": "facebook/esm2_t6_8M_UR50D"},
        "35M":  {"model_name": "facebook/esm2_t12_35M_UR50D"},
        "150M": {"model_name": "facebook/esm2_t30_150M_UR50D"},
        "650M": {"model_name": "facebook/esm2_t33_650M_UR50D"},
        "3B":   {"model_name": "facebook/esm2_t36_3B_UR50D"},
        "15B":  {"model_name": "facebook/esm2_t48_15B_UR50D"}}
    
    def __init__(self, param_size: str):
        try:
            model_info = self.MODEL_MAPPING[param_size]
        except KeyError:
            raise ValueError(param_size)
        self.model = FAEsmForMaskedLM.from_pretrained(model_info["model_name"])
        self.alphabet = Alphabet.from_architecture("ESM-1")


model = ModelSelector("8M").model
mask_token = model.tokenizer.mask_token        
mask_token_id = model.tokenizer.mask_token_id     

print(mask_token, mask_token_id)