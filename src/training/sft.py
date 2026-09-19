"""Supervised Fine-Tuning (SFT) with optional LoRA / QLoRA."""

from trl import SFTTrainer

from src.models.loaders import load_tokenizer, load_causal_lm
from src.models.peft import apply_lora
from src.utils.config import build_sft_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS


def train_sft(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    use_lora: bool = True,
    qlora: bool = False,
    max_seq_length: int = 256,
    dataset_text_field: str = "text",
    packing: bool = False,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 2,
    lr: float = 2e-4,
    fp16: bool = False,
    bf16: bool = False,
):
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name, quantize=qlora, dtype="auto")

    if use_lora or qlora:
        model = apply_lora(model)

    sft_config = build_sft_config(
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        dataset_text_field=dataset_text_field,
        packing=packing,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,
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