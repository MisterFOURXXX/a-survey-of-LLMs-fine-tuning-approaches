"""Bradley–Terry reward modeling."""

from trl import RewardTrainer

from src.models.loaders import load_tokenizer, load_reward_model
from src.utils.config import build_reward_config


def train_reward(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 4,
    lr: float = 2e-5,
    max_length: int = 512,
    bf16: bool = True,
):
    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = load_reward_model(model_name, num_labels=1, dtype="auto")

    reward_config = build_reward_config(
        output_dir=output_dir,
        max_length=max_length,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=bf16,
    )

    # RewardTrainer already uses `processing_class` in TRL 0.12+
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer