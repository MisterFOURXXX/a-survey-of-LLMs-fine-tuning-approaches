"""TRL config builders — correct kwarg names for trl 0.14.x + transformers 4.57.

Step-based strategy defaults are tuned for small datasets (< 500 rows) so
training loss is actually logged and evaluation actually runs.
"""

from __future__ import annotations
from typing import Any

from src.utils.version import (
    SFT_USES_MAX_LENGTH, USES_EVAL_STRATEGY, HAS_MASK_TRUNCATED,
    HAS_GRPO_TOP_P,
    HAS_SFT, HAS_DPO, HAS_REWARD, HAS_GRPO, TRL_VERSION,
)

if HAS_SFT:    from trl import SFTConfig
else:          SFTConfig = None
if HAS_DPO:    from trl import DPOConfig
else:          DPOConfig = None
if HAS_REWARD: from trl import RewardConfig
else:          RewardConfig = None
if HAS_GRPO:   from trl import GRPOConfig
else:          GRPOConfig = None


def _require(klass, name, min_trl):
    if klass is None:
        raise ImportError(
            f"`{name}` is not available in TRL {TRL_VERSION}. "
            f'Upgrade: pip install --upgrade "trl>={min_trl}"'
        )


def _eval_kwarg(strategy: str) -> dict:
    return ({"eval_strategy": strategy} if USES_EVAL_STRATEGY
            else {"evaluation_strategy": strategy})


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def build_sft_config(
    output_dir, max_seq_length=256, dataset_text_field="text", packing=False,
    num_train_epochs=3, per_device_train_batch_size=4, per_device_eval_batch_size=4,
    gradient_accumulation_steps=2, learning_rate=2e-4, weight_decay=0.01,
    warmup_ratio=0.1, lr_scheduler_type="cosine",
    eval_strategy="epoch", save_strategy="epoch",
    logging_steps=5,                     # <-- was 50; small enough for short runs
    save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    greater_is_better=False, fp16=False, bf16=False, report_to="none",
    seed=42, gradient_checkpointing=True,
):
    _require(SFTConfig, "SFTConfig", "0.12.0")
    kwargs = dict(
        output_dir=output_dir, num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio, lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy, logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better, fp16=fp16, bf16=bf16,
        report_to=report_to, seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        dataset_text_field=dataset_text_field, packing=packing,
        remove_unused_columns=True,
    )
    kwargs.update(_eval_kwarg(eval_strategy))
    if SFT_USES_MAX_LENGTH:
        kwargs["max_length"] = max_seq_length
    else:
        kwargs["max_seq_length"] = max_seq_length
    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
def build_dpo_config(
    output_dir, beta=0.1, max_length=512, max_prompt_length=256,
    num_train_epochs=3, per_device_train_batch_size=2, per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=1e-6, weight_decay=0.01,
    warmup_ratio=0.1, lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=5,                        # <-- was 100; must be <= total steps
    save_strategy="steps",
    save_steps=5,                        # <-- was 100
    logging_steps=1,                     # <-- was 50; log every step
    fp16=False, bf16=False, report_to="none", seed=42,
    log_completions=False,               # <-- hides the huge rich table
):
    _require(DPOConfig, "DPOConfig", "0.12.0")
    kwargs = dict(
        output_dir=output_dir, beta=beta, max_length=max_length,
        max_prompt_length=max_prompt_length, num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio, lr_scheduler_type=lr_scheduler_type,
        eval_steps=eval_steps, save_strategy=save_strategy,
        save_steps=save_steps, logging_steps=logging_steps,
        fp16=fp16, bf16=bf16, report_to=report_to, seed=seed,
        remove_unused_columns=False,
        log_completions=log_completions,
    )
    kwargs.update(_eval_kwarg(eval_strategy))
    return DPOConfig(**kwargs)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
def build_reward_config(
    output_dir, max_length=512, num_train_epochs=3,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=4, learning_rate=2e-5, weight_decay=0.01,
    warmup_ratio=0.1, lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=10,                       # <-- was epoch (rely on eval_steps now)
    save_strategy="steps",
    save_steps=10,
    logging_steps=1,                     # <-- was 10
    bf16=True, fp16=False,
    report_to="none", seed=42, gradient_checkpointing=True,
    log_completions=False,
):
    _require(RewardConfig, "RewardConfig", "0.12.0")
    kwargs = dict(
        output_dir=output_dir, max_length=max_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio, lr_scheduler_type=lr_scheduler_type,
        eval_steps=eval_steps, save_strategy=save_strategy,
        save_steps=save_steps, logging_steps=logging_steps,
        bf16=bf16, fp16=fp16, report_to=report_to, seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
        log_completions=log_completions,
    )
    kwargs.update(_eval_kwarg(eval_strategy))
    return RewardConfig(**kwargs)


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
def build_grpo_config(
    output_dir, num_generations=4, max_completion_length=256,
    max_prompt_length=256, temperature=0.9,
    beta=0.04, num_train_epochs=3, per_device_train_batch_size=2,
    per_device_eval_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=1e-6, weight_decay=0.01, warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=5,                        # <-- was 50
    save_strategy="steps",
    save_steps=5,                        # <-- was 50
    logging_steps=1,                     # <-- was 10
    bf16=True, fp16=False, report_to="none", seed=42,
    gradient_checkpointing=True,
):
    _require(GRPOConfig, "GRPOConfig", "0.14.0")
    kwargs = dict(
        output_dir=output_dir, num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length, temperature=temperature,
        beta=beta, num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio, lr_scheduler_type=lr_scheduler_type,
        eval_steps=eval_steps, save_strategy=save_strategy,
        save_steps=save_steps, logging_steps=logging_steps,
        bf16=bf16, fp16=fp16, report_to=report_to, seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
    )
    if HAS_MASK_TRUNCATED:
        kwargs["mask_truncated_completions"] = True
    if HAS_GRPO_TOP_P:
        kwargs["top_p"] = 0.95
    kwargs.update(_eval_kwarg(eval_strategy))
    return GRPOConfig(**kwargs)