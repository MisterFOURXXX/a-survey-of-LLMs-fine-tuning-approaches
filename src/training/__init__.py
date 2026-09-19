"""Training subpackage. Uses lazy imports so one broken module doesn't break all."""

__all__ = ["train_sft", "train_dpo", "train_reward", "train_grpo"]


def train_sft(*args, **kwargs):
    from .sft import train_sft as _fn
    return _fn(*args, **kwargs)


def train_dpo(*args, **kwargs):
    from .dpo import train_dpo as _fn
    return _fn(*args, **kwargs)


def train_reward(*args, **kwargs):
    from .reward import train_reward as _fn
    return _fn(*args, **kwargs)


def train_grpo(*args, **kwargs):
    from .grpo import train_grpo as _fn
    return _fn(*args, **kwargs)