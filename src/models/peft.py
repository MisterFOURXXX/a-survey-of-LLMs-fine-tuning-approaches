"""PEFT / LoRA helpers."""

from peft import LoraConfig, get_peft_model, TaskType


def _resolve_targets(model, user_targets):
    """Fallback to a broader list if the model doesn't expose the requested
    module names (e.g. Gemma vs Llama vs Qwen)."""
    if user_targets:
        return user_targets
    # Try the common Llama/Gemma/Qwen names; PEFT will ignore missing ones.
    return ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules=None,
    task_type=TaskType.CAUSAL_LM,
):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=_resolve_targets(model, target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model