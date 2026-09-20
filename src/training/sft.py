"""Supervised Fine-Tuning (SFT) driven by a YAML config."""

from src.models.loaders import load_tokenizer, load_causal_lm
from src.models.peft import apply_lora
from src.utils.config import build_sft_config
from src.utils.config_loader import resolve_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS, HAS_SFT, TRL_VERSION

if HAS_SFT:
    from trl import SFTTrainer
else:
    SFTTrainer = None


def train_sft(cfg, train_dataset, eval_dataset):
    cfg = resolve_config(cfg)

    if SFTTrainer is None:
        raise ImportError(
            f"SFTTrainer is not available in TRL {TRL_VERSION}. "
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

    sft_config = build_sft_config(
        output_dir=output_dir,
        max_seq_length=cfg["max_seq_length"],
        dataset_text_field=cfg["dataset_text_field"],
        packing=cfg.get("packing", False),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        logging_steps=cfg.get("logging_steps", 5),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
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

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **extra,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer