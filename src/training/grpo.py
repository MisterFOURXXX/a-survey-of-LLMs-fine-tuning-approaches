"""GRPO (Group Relative Policy Optimization) trainer wrapper."""

from trl import GRPOTrainer, GRPOConfig

from src.models.loaders import load_tokenizer, load_causal_lm


def reward_length_and_reasoning(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0] if isinstance(completion, list) else completion
        words = text.split()
        length_score = min(len(words) / 40, 1.0)
        reasoning_score = 0.3 if any(
            w in text.lower() for w in ["step", "first", "then", "therefore", "because"]
        ) else 0.0
        rewards.append(length_score + reasoning_score)
    return rewards


def train_grpo(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 1e-6,
    num_generations: int = 4,
    max_completion_length: int = 256,
):
    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = load_causal_lm(model_name, quantize=True)

    args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=0.9,
        top_p=0.95,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_length_and_reasoning,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer