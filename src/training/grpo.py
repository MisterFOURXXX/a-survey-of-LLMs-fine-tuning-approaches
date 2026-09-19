"""GRPO — Group Relative Policy Optimization.

Raises a clear error only if the user *calls* train_grpo() on a TRL version
that does not ship GRPO (i.e. anything below 0.14.0).
"""

from src.models.loaders import load_tokenizer, load_causal_lm
from src.utils.config import build_grpo_config
from src.utils.version import (
    TRAINER_USES_PROCESSING_CLASS,
    HAS_GRPO,
    TRL_VERSION,
)

# Conditional import — safe on old TRL.
if HAS_GRPO:
    from trl import GRPOTrainer
else:
    GRPOTrainer = None


def reward_length_and_reasoning(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0] if isinstance(completion, list) else completion
        words = text.split()
        length_score = min(len(words) / 40.0, 1.0)
        reasoning = 0.3 if any(
            w in text.lower()
            for w in ["step", "first", "then", "therefore", "because"]
        ) else 0.0
        rewards.append(length_score + reasoning)
    return rewards


def train_grpo(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 1e-6,
    num_generations: int = 4,
    max_completion_length: int = 256,
    reward_funcs=None,
):
    if GRPOTrainer is None:
        raise ImportError(
            f"`GRPOTrainer` is not available in the installed TRL version "
            f"({TRL_VERSION}). Please upgrade with:\n"
            f"    pip install --upgrade \"trl>=0.14.0\"\n"
            f"and restart the kernel."
        )

    if reward_funcs is None:
        reward_funcs = reward_length_and_reasoning

    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = load_causal_lm(model_name, quantize=False, dtype="auto")

    grpo_config = build_grpo_config(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
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

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        **extra,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer