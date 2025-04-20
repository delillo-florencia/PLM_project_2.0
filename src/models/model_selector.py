import contextlib
import io
_f = io.StringIO()
with contextlib.redirect_stdout(_f), contextlib.redirect_stderr(_f):
    from faesm.esm import FAEsmForMaskedLM
    from faesm import *


class ModelSelector:

    MODEL_MAPPING = {
        "8M":   {"model_name": "facebook/esm2_t6_8M_UR50D"},
        "35M":  {"model_name": "facebook/esm2_t12_35M_UR50D"},
        "150M": {"model_name": "facebook/esm2_t30_150M_UR50D"},
        "650M": {"model_name": "facebook/esm2_t33_650M_UR50D"},
        "3B":   {"model_name": "facebook/esm2_t36_3B_UR50D"},
        "15B":  {"model_name": "facebook/esm2_t48_15B_UR50D"}}
    
    def __init__(self, param_size: str, use_fa: bool):
        try:
            model_info = self.MODEL_MAPPING[param_size]
        except KeyError:
            raise ValueError(param_size)
        self.model = FAEsmForMaskedLM.from_pretrained(model_info["model_name"], use_fa=use_fa)

