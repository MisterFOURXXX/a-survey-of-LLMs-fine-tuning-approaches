"""Detect installed library versions and expose feature flags."""

import importlib
import importlib.metadata as md


def _ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "0.0.0"


def _parse(v: str):
    parts = []
    for p in v.split("."):
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


TRL_VERSION = _ver("trl")
TRANSFORMERS_VERSION = _ver("transformers")
PEFT_VERSION = _ver("peft")

TRL = _parse(TRL_VERSION)
TRANSFORMERS = _parse(TRANSFORMERS_VERSION)
PEFT = _parse(PEFT_VERSION)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

# TRL >= 0.12 renamed `max_seq_length` -> `max_length` in SFTConfig
SFT_USES_MAX_LENGTH = TRL >= (0, 12, 0)

# TRL >= 0.12 trainers prefer `processing_class` over `tokenizer`
TRAINER_USES_PROCESSING_CLASS = TRL >= (0, 12, 0)

# TRL >= 0.13 added `mask_truncated_completions`
HAS_MASK_TRUNCATED = TRL >= (0, 13, 0)

# transformers >= 4.46 renamed `torch_dtype` kwarg to `dtype`
DTYPE_KWARG = "dtype" if TRANSFORMERS >= (4, 46, 0) else "torch_dtype"


# ---------------------------------------------------------------------------
# Which trainers are actually importable?
# ---------------------------------------------------------------------------
def _has(module_path: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(module_path)
        return hasattr(mod, attr)
    except ImportError:
        return False


HAS_SFT = _has("trl", "SFTTrainer") and _has("trl", "SFTConfig")
HAS_DPO = _has("trl", "DPOTrainer") and _has("trl", "DPOConfig")
HAS_REWARD = _has("trl", "RewardTrainer") and _has("trl", "RewardConfig")
HAS_GRPO = _has("trl", "GRPOTrainer") and _has("trl", "GRPOConfig")


def banner() -> str:
    return (
        f"trl={TRL_VERSION} transformers={TRANSFORMERS_VERSION} "
        f"peft={PEFT_VERSION} | "
        f"SFT={HAS_SFT} DPO={HAS_DPO} Reward={HAS_REWARD} GRPO={HAS_GRPO}"
    )