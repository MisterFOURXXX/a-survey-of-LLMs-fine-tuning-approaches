"""Bradley-Terry reward modeling trainer wrapper."""

from trl import RewardTrainer, RewardConfig

from src.models.loaders import load_tokenizer, load_reward_model


def train_reward(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 4,
    lr: float = 2e-5,
    max_length: int = 256,
):
    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = load_reward_model(model_name, num_labels=1)

    args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        max_length=max_length,
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer