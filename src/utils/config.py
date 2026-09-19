"""Central place that builds TRL config objects with the correct parameter
names for the installed TRL version. Every trainer file imports from here."""

from trl import SFTConfig, DPOConfig, RewardConfig, GRPOConfig

from src.utils.version import (
    SFT_USES_MAX_LENGTH,
    HAS_MASK_TRUNCATED,
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
    gradient_checkpointing: bool = False,
):
    """Return an SFTConfig that works on TRL 0.12+ (and older with fallback)."""
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        eval_strategy=eval_strategy,
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
        remove_unused_columns=False,
        dataset_text_field=dataset_text_field,
        packing=packing,
    )

    # The ONE thing that differs across TRL versions:
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
    return DPOConfig(
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
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        fp16=fp16,
        bf16=bf16,
        report_to=report_to,
        seed=seed,
        remove_unused_columns=False,
    )


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
    return RewardConfig(
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
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
    )


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
    kwargs = dict(
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
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
    )
    if HAS_MASK_TRUNCATED:
        kwargs["mask_truncated_completions"] = True
    return GRPOConfig(**kwargs)