"""GRPO (Group Relative Policy Optimization) training."""

import logging
logging.getLogger("trl.trainer.reward_trainer").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.models.loaders import load_tokenizer, load_causal_lm
from src.models.peft import apply_lora
from src.utils.config import build_grpo_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS, HAS_GRPO, TRL_VERSION

if HAS_GRPO:
    from trl import GRPOTrainer
else:
    GRPOTrainer = None


def train_grpo(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    use_lora: bool = True,
    qlora: bool = False,
    num_generations: int = 4,
    max_completion_length: int = 256,
    max_prompt_length: int = 256,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 1e-6,
    reward_funcs=None,
):
    if GRPOTrainer is None:
        raise ImportError(
            f"GRPOTrainer is not available in TRL {TRL_VERSION}. "
            f'Install with: pip install --upgrade "trl>=0.14.0"'
        )

    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(
        model_name, quantize=qlora, dtype="auto", tokenizer=tokenizer
    )
    if use_lora or qlora:
        model = apply_lora(model)
    model.config.use_cache = False

    grpo_config = build_grpo_config(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
    )

    extra = (
        {"processing_class": tokenizer}
        if TRAINER_USES_PROCESSING_CLASS
        else {"tokenizer": tokenizer}
    )

    trainer_kwargs = dict(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **extra,
    )
    if reward_funcs is not None:
        trainer_kwargs["reward_funcs"] = reward_funcs

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    return trainer