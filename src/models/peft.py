"""PEFT / LoRA helpers — hardened against peft 0.19's strict torchao check."""

import importlib.util

from peft import LoraConfig, get_peft_model, TaskType


def _torchao_present() -> bool:
    return importlib.util.find_spec("torchao") is not None


def _resolve_targets(model, user_targets):
    if user_targets:
        return user_targets
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


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

    try:
        model = get_peft_model(model, config)
    except ImportError as e:
        if "torchao" in str(e).lower() and _torchao_present():
            raise ImportError(
                "peft detected an incompatible `torchao` install. This "
                "project does not use torchao. Fix:\n\n"
                "    !pip uninstall -y torchao\n"
                "    # then: Runtime -> Restart Session\n"
            ) from e
        raise

    model.print_trainable_parameters()
    return model