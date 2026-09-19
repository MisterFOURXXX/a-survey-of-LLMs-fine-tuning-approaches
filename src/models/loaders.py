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
    # Assign AFTER loading so we don't fight the model config's defaults.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def _align_special_tokens(model, tokenizer):
    """Make the model config agree with the tokenizer up front, which silences
    the 'tokenizer has new PAD/BOS/EOS tokens' warning."""
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id


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
    dtype=torch.float16,
    device_map: str = "auto",
    tokenizer=None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_base_kwargs(dtype, device_map, quantize),
    )
    if tokenizer is not None:
        _align_special_tokens(model, tokenizer)
    return model


def load_reward_model(
    model_name: str,
    num_labels: int = 1,
    quantize: bool = False,
    dtype=torch.float16,
    device_map: str = "auto",
    tokenizer=None,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        **_base_kwargs(dtype, device_map, quantize),
    )
    if tokenizer is not None:
        _align_special_tokens(model, tokenizer)
    return model