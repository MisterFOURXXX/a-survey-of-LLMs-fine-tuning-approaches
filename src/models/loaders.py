"""Model + tokenizer loaders — version-aware dtype, explicit PAD/BOS/EOS."""

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
    # Assign AFTER loading so we don't fight the model config defaults.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def _align_special_tokens(model, tokenizer):
    """Align model config and generation config with the tokenizer so
    transformers doesn't print the 'tokenizer has new PAD/BOS/EOS tokens'
    warning at generate time."""
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        gen.pad_token_id = tokenizer.pad_token_id
        gen.eos_token_id = tokenizer.eos_token_id
        gen.bos_token_id = tokenizer.bos_token_id


def _base_kwargs(dtype, device_map, quantize):
    kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        DTYPE_KWARG: dtype,
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
    dtype="auto",
    device_map=None,                      # <-- None, not "auto"
    tokenizer=None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_base_kwargs(dtype, device_map, quantize),
    )
    if tokenizer is not None:
        _align_special_tokens(model, tokenizer)
    model.config.use_cache = False
    return model


def load_reward_model(
    model_name: str,
    num_labels: int = 1,
    quantize: bool = False,
    dtype="auto",
    device_map=None,                      # <-- None, not "auto"
    tokenizer=None,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        **_base_kwargs(dtype, device_map, quantize),
    )
    if tokenizer is not None:
        _align_special_tokens(model, tokenizer)
    model.config.use_cache = False
    return model