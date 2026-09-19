"""PEFT / LoRA helpers — hardened against peft 0.19's strict torchao check
and against the frozen-embedding + reentrant-gradient-checkpointing bug.
"""

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


def _enable_input_require_grads(model):
    """Force the input embeddings to require grad.

    With PEFT the base model is frozen, so the embedding outputs have
    requires_grad=False. Reentrant gradient checkpointing then cannot rebuild
    the autograd graph during the backward pass, producing:

        UserWarning: None of the inputs have requires_grad=True.
        RuntimeError: element 0 of tensors does not require grad ...

    enable_input_require_grads() registers a forward hook that flips
    requires_grad=True on the embedding output, which fixes the reentrant
    path. It is a no-op when gradient checkpointing is disabled.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        emb = model.get_input_embeddings()
        emb.register_forward_hook(make_inputs_require_grad)


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

    # --- KEY FIX: input embeddings must require grad for reentrant GC ---
    _enable_input_require_grads(model)

    model.print_trainable_parameters()
    return model