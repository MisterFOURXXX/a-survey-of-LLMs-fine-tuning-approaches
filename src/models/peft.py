from peft import LoraConfig, get_peft_model, TaskType


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules=None,
    task_type=TaskType.CAUSAL_LM,
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model