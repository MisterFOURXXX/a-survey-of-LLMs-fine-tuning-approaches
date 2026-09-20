"""Direct Preference Optimization (DPO) driven by a YAML config."""

from src.models.loaders import load_tokenizer, load_causal_lm
from src.models.peft import apply_lora
from src.utils.config import build_dpo_config
from src.utils.config_loader import resolve_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS, HAS_DPO, TRL_VERSION

if HAS_DPO:
    from trl import DPOTrainer
else:
    DPOTrainer = None


def train_dpo(cfg, train_dataset, eval_dataset):
    cfg = resolve_config(cfg)

    if DPOTrainer is None:
        raise ImportError(
            f"DPOTrainer is not available in TRL {TRL_VERSION}. "
            f'Install with: pip install --upgrade "trl>=0.12.0"'
        )

    model_name = cfg["model_name"]
    output_dir = cfg["output_dir"]
    use_lora   = cfg.get("use_lora", True)
    qlora      = cfg.get("qlora", False)
    lora_cfg   = cfg.get("lora", {})

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

    dpo_config = build_dpo_config(
        output_dir=output_dir,
        beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 256),
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
        save_steps=cfg.get("save_steps", 5),
        logging_steps=cfg.get("logging_steps", 1),
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )

    extra = (
        {"processing_class": tokenizer}
        if TRAINER_USES_PROCESSING_CLASS
        else {"tokenizer": tokenizer}
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **extra,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer