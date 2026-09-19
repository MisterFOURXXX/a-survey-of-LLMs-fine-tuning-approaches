"""Reward-model training."""

import logging
logging.getLogger("trl.trainer.reward_trainer").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.models.loaders import load_tokenizer, load_reward_model
from src.models.peft import apply_lora
from src.utils.config import build_reward_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS, HAS_REWARD, TRL_VERSION

if HAS_REWARD:
    from trl import RewardTrainer
else:
    RewardTrainer = None


def train_reward(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    use_lora: bool = True,
    max_length: int = 512,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 4,
    lr: float = 2e-5,
    logging_steps: int = 1,
    eval_steps: int = 5,
    save_steps: int = 10,
):
    if RewardTrainer is None:
        raise ImportError(
            f"RewardTrainer is not available in TRL {TRL_VERSION}. "
            f'Install with: pip install --upgrade "trl>=0.12.0"'
        )

    tokenizer = load_tokenizer(model_name)
    model = load_reward_model(
        model_name, tokenizer=tokenizer, dtype="auto"
    )
    if use_lora:
        model = apply_lora(model, task_type=None)
    model.config.use_cache = False

    reward_config = build_reward_config(
        output_dir=output_dir,
        max_length=max_length,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logning_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        learning_rate=lr,
    )

    extra = (
        {"processing_class": tokenizer}
        if TRAINER_USES_PROCESSING_CLASS
        else {"tokenizer": tokenizer}
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **extra,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer