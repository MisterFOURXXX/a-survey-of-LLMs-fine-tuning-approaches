"""Model + tokenizer loading helpers (transformers 4.45 – 4.47 compatible)."""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from src.utils.version import DTYPE_KWARG


def load_tokenizer(model_name: str, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def _base_kwargs(dtype, device_map, quantize):
    kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        DTYPE_KWARG: dtype,           # `dtype` on new transformers, `torch_dtype` on old
    }
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return kwargs


def load_causal_lm(
    model_name: str,
    quantize: bool = False,
    dtype=torch.float16,
    device_map: str = "auto",
):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        **_base_kwargs(dtype, device_map, quantize),
    )


def load_reward_model(
    model_name: str,
    num_labels: int = 1,
    quantize: bool = False,
    dtype=torch.float16,
    device_map: str = "auto",
):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        **_base_kwargs(dtype, device_map, quantize),
    )