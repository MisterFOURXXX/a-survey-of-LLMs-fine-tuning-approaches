"""Central place that builds TRL config objects with the correct parameter
names for the installed TRL + transformers versions.

Works on trl 0.12 → 0.15 and transformers 4.40 → 4.60.
"""

from __future__ import annotations

from typing import Any

from src.utils.version import (
    SFT_USES_MAX_LENGTH,
    USES_EVAL_STRATEGY,
    HAS_MASK_TRUNCATED,
    HAS_SFT, HAS_DPO, HAS_REWARD, HAS_GRPO,
    TRL_VERSION,
)

# ---------------------------------------------------------------------------
# Conditional imports so a missing class does not break the whole module.
# ---------------------------------------------------------------------------
if HAS_SFT:
    from trl import SFTConfig
else:
    SFTConfig = None

if HAS_DPO:
    from trl import DPOConfig
else:
    DPOConfig = None

if HAS_REWARD:
    from trl import RewardConfig
else:
    RewardConfig = None

if HAS_GRPO:
    from trl import GRPOConfig
else:
    GRPOConfig = None


def _require(klass, name: str, min_trl: str):
    if klass is None:
        raise ImportError(
            f"`{name}` is not available in the installed TRL version "
            f"({TRL_VERSION}). Upgrade with:\n"
            f'    pip install --upgrade "trl>={min_trl}"\n'
            f"and restart the kernel."
        )


def _eval_kwarg(strategy: str) -> dict[str, str]:
    """Pick the right evaluation-strategy kwarg for the installed transformers."""
    return (
        {"eval_strategy": strategy}
        if USES_EVAL_STRATEGY
        else {"evaluation_strategy": strategy}
    )


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def build_sft_config(
    output_dir: str,
    max_seq_length: int = 256,
    dataset_text_field: str = "text",
    packing: bool = False,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 50,
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    fp16: bool = False,
    bf16: bool = False,
    report_to: str = "none",
    seed: int = 42,
    gradient_checkpointing: bool = True,
) -> "SFTConfig":
    """SFTConfig for trl 0.14.x.

    Notes
    -----
    * TRL <0.16 uses `max_seq_length`. Passing `max_length` raises
      `TypeError: SFTConfig.__init__() got an unexpected keyword argument
      'max_length'`.
    * transformers >=4.46 uses `eval_strategy` (not `evaluation_strategy`).
    * `remove_unused_columns=True` is required so the default
      DataCollatorForLanguageModeling does not try to tensorize the raw
      `text` column (→ ValueError: too many dimensions 'str').
    """
    _require(SFTConfig, "SFTConfig", "0.12.0")

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=fp16,
        bf16=bf16,
        report_to=report_to,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        # TRL SFT-specific fields
        dataset_text_field=dataset_text_field,
        packing=packing,
        # Do NOT keep the raw string column — the default collator
        # cannot handle `text` as a list of str.
        remove_unused_columns=True,
    )

    # Version-aware evaluation-strategy kwarg
    kwargs.update(_eval_kwarg(eval_strategy))

    # Version-aware max-length kwarg: trl 0.14 → max_seq_length
    if SFT_USES_MAX_LENGTH:
        kwargs["max_length"] = max_seq_length
    else:
        kwargs["max_seq_length"] = max_seq_length

    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
def build_dpo_config(
    output_dir: str,
    beta: float = 0.1,
    max_length: int = 512,
    max_prompt_length: int = 256,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-6,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    eval_strategy: str = "steps",
    eval_steps: int = 100,
    save_strategy: str = "steps",
    save_steps: int = 100,
    logging_steps: int = 50,
    fp16: bool = False,
    bf16: bool = False,
    report_to: str = "none",
    seed: int = 42,
):
    _require(DPOConfig, "DPOConfig", "0.12.0")

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        fp16=fp16,
        bf16=bf16,
        report_to=report_to,
        seed=seed,
        # DPO needs the raw prompt/chosen/rejected columns to reach the trainer
        remove_unused_columns=False,
    )
    kwargs.update(_eval_kwarg(eval_strategy))
    return DPOConfig(**kwargs)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
def build_reward_config(
    output_dir: str,
    max_length: int = 512,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 10,
    bf16: bool = True,
    fp16: bool = False,
    report_to: str = "none",
    seed: int = 42,
    gradient_checkpointing: bool = True,
):
    _require(RewardConfig, "RewardConfig", "0.12.0")

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
    )
    kwargs.update(_eval_kwarg(eval_strategy))
    return RewardConfig(**kwargs)


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
def build_grpo_config(
    output_dir: str,
    num_generations: int = 4,
    max_completion_length: int = 256,
    max_prompt_length: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.95,
    beta: float = 0.04,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-6,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    eval_strategy: str = "steps",
    eval_steps: int = 50,
    save_strategy: str = "steps",
    save_steps: int = 50,
    logging_steps: int = 10,
    bf16: bool = True,
    fp16: bool = False,
    report_to: str = "none",
    seed: int = 42,
    gradient_checkpointing: bool = True,
):
    _require(GRPOConfig, "GRPOConfig", "0.14.0")

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        temperature=temperature,
        top_p=top_p,
        beta=beta,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        # GRPO's reward functions read `reference` / `ground_truth` columns
        remove_unused_columns=False,
    )
    if HAS_MASK_TRUNCATED:
        kwargs["mask_truncated_completions"] = True
    kwargs.update(_eval_kwarg(eval_strategy))
    return GRPOConfig(**kwargs)