import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)


def load_tokenizer(model_name: str, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_causal_lm(
    model_name: str,
    quantize: bool = False,
    dtype=torch.float16,
    device_map: str = "auto",
):
    kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_reward_model(
    model_name: str,
    num_labels: int = 1,
    quantize: bool = False,
    dtype=torch.float16,
    device_map: str = "auto",
):
    kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "num_labels": num_labels,
    }

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)