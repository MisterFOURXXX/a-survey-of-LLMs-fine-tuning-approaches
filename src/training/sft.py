from transformers import TrainingArguments
from trl import SFTTrainer
from src.models.loaders import load_tokenizer, load_causal_lm
from src.models.peft import apply_lora


def train_sft(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    use_lora: bool = True,
    qlora: bool = False,
    max_seq_length: int = 256,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 2,
    lr: float = 2e-4,
):
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name, quantize=qlora)

    if use_lora or qlora:
        model = apply_lora(model)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        fp16=not qlora,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer