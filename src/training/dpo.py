"""Direct Preference Optimization (DPO)."""

from trl import DPOTrainer

from src.models.loaders import load_tokenizer, load_causal_lm
from src.utils.config import build_dpo_config
from src.utils.version import TRAINER_USES_PROCESSING_CLASS


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
    max_length: int = 512,
    max_prompt_length: int = 256,
):
    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = load_causal_lm(model_name, dtype="auto")
    ref_model = load_causal_lm(model_name, dtype="auto")

    dpo_config = build_dpo_config(
        output_dir=output_dir,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
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

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **extra,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer