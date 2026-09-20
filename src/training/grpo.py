"""GRPO (Group Relative Policy Optimization) driven by a YAML config."""

import logging

logging.getLogger("trl.trainer.reward_trainer").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.models.loaders import load_tokenizer, load_causal_lm
from src.models.peft import apply_lora
from src.training.reward_funcs import resolve_reward_funcs
from src.utils.config import build_grpo_config
from src.utils.config_loader import resolve_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS, HAS_GRPO, TRL_VERSION

if HAS_GRPO:
    from trl import GRPOTrainer
else:
    GRPOTrainer = None


def train_grpo(cfg, train_dataset, eval_dataset):
    cfg = resolve_config(cfg)

    if GRPOTrainer is None:
        raise ImportError(
            f"GRPOTrainer is not available in TRL {TRL_VERSION}. "
            f'Install with: pip install --upgrade "trl>=0.14.0"'
        )

    model_name = cfg["model_name"]
    output_dir = cfg["output_dir"]
    use_lora   = cfg.get("use_lora", True)
    qlora      = cfg.get("qlora", False)
    lora_cfg   = cfg.get("lora", {})

    reward_funcs = resolve_reward_funcs(cfg.get("reward_funcs"))
    if not reward_funcs:
        raise ValueError(
            "GRPO requires at least one reward function. Set "
            "`reward_funcs:` in configs/grpo.yaml."
        )

    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(
        model_name, quantize=qlora, dtype="auto", tokenizer=tokenizer
    )
    if use_lora or qlora:
        model = apply_lora(
            model,
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
        )
    model.config.use_cache = False

    grpo_config = build_grpo_config(
        output_dir=output_dir,
        num_generations=cfg.get("num_generations", 4),
        max_completion_length=cfg.get("max_completion_length", 256),
        max_prompt_length=cfg.get("max_prompt_length", 256),
        temperature=cfg.get("temperature", 0.9),
        beta=cfg.get("beta", 0.04),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 5),
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

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **extra,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer