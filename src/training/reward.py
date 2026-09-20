"""Reward-model training (Bradley-Terry sequence classification).

This is the *trainer*. For GRPO's reward *functions*, see
`src/training/reward_funcs.py`.
"""

import logging

logging.getLogger("trl.trainer.reward_trainer").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.models.loaders import load_tokenizer, load_reward_model
from src.models.peft import apply_lora
from src.utils.config import build_reward_config
from src.utils.config_loader import resolve_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS, HAS_REWARD, TRL_VERSION

if HAS_REWARD:
    from trl import RewardTrainer
else:
    RewardTrainer = None


def train_reward(cfg, train_dataset, eval_dataset):
    """Train a Bradley-Terry reward model from a YAML config dict or path.

    Reads every value from `cfg` (see `configs/reward.yaml`):
      model_name, output_dir, use_lora, lora.{r,alpha,dropout,target_modules},
      max_length, epochs, batch_size, grad_accum, lr, weight_decay,
      warmup_ratio, eval_strategy, eval_steps, save_strategy, save_steps,
      logging_steps, bf16, fp16, report_to, seed, gradient_checkpointing.
    """
    cfg = resolve_config(cfg)

    if RewardTrainer is None:
        raise ImportError(
            f"RewardTrainer is not available in TRL {TRL_VERSION}. "
            f'Install with: pip install --upgrade "trl>=0.12.0"'
        )

    model_name = cfg["model_name"]
    output_dir = cfg["output_dir"]
    use_lora   = cfg.get("use_lora", True)
    lora_cfg   = cfg.get("lora", {})

    tokenizer = load_tokenizer(model_name)
    model = load_reward_model(model_name, tokenizer=tokenizer, dtype="auto")
    if use_lora:
        model = apply_lora(
            model,
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            task_type=None,          # classification head, not causal-LM
        )
    model.config.use_cache = False

    reward_config = build_reward_config(
        output_dir=output_dir,
        max_length=cfg.get("max_length", 512),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 10),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 10),
        logging_steps=cfg.get("logging_steps", 1),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
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