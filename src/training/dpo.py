"""DPO (Direct Preference Optimization) trainer wrapper."""

from trl import DPOTrainer, DPOConfig

from src.models.loaders import load_tokenizer, load_causal_lm


def train_dpo(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    beta: float = 0.1,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 1e-6,
    max_length: int = 256,
    max_prompt_length: int = 128,
):
    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = load_causal_lm(model_name)
    ref_model = load_causal_lm(model_name)

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer