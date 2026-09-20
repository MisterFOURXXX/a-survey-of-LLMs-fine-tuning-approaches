"""Model + tokenizer loaders — version-aware dtype, explicit PAD/BOS/EOS."""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from src.utils.version import DTYPE_KWARG
import json, os

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
    device_map=None,                     
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


def load_model_for_inference(path: str, dtype="auto", device_map: str = "auto"):
    """Load a full model OR a PEFT/LoRA adapter directory for inference.

    Returns
    -------
    (model, base_model_id) : tuple
        `base_model_id` is the tokenizer source — the adapter dir if it has
        tokenizer files, otherwise the base model ID from adapter_config.json.
    """
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path") or cfg.get("base_model_name_or_path")
        if not base_id:
            raise ValueError(f"adapter_config.json in {path} has no base_model_name_or_path")

        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
            **{DTYPE_KWARG: dtype},
        )
        model = PeftModel.from_pretrained(base, path, device_map=device_map)
        model = model.merge_and_unload()          # fold LoRA into base for fast eval
        tokenizer_source = (
            path if os.path.exists(os.path.join(path, "tokenizer_config.json"))
            else base_id
        )
        return model, tokenizer_source

    # Full model directory
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
        **{DTYPE_KWARG: dtype},
    )
    return model, path