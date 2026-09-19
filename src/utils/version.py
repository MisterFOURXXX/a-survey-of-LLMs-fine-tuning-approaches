"""Detect installed library versions and expose feature flags.

IMPORTANT: this module must never fail to import. Every probe is wrapped in a
broad `except Exception` because TRL's lazy import machinery can raise
RuntimeError (not just ImportError) when a transitive dependency such as
torchvision is broken.
"""

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
SFT_USES_MAX_LENGTH = TRL >= (0, 12, 0)
TRAINER_USES_PROCESSING_CLASS = TRL >= (0, 12, 0)
HAS_MASK_TRUNCATED = TRL >= (0, 13, 0)
DTYPE_KWARG = "dtype" if TRANSFORMERS >= (4, 46, 0) else "torch_dtype"


# ---------------------------------------------------------------------------
# Trainer availability — NEVER let a probe exception escape.
# ---------------------------------------------------------------------------
def _probe(module_path: str, attr: str) -> bool:
    """
    Return True only if `attr` on `module_path` can be *touched* successfully.
    Catches every exception type because TRL's __getattr__ raises RuntimeError
    (not ImportError) when a transitive dep such as torchvision is broken.
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return False

    try:
        getattr(mod, attr)
    except Exception:
        return False

    return True


HAS_SFT = _probe("trl", "SFTTrainer") and _probe("trl", "SFTConfig")
HAS_DPO = _probe("trl", "DPOTrainer") and _probe("trl", "DPOConfig")
HAS_REWARD = _probe("trl", "RewardTrainer") and _probe("trl", "RewardConfig")
HAS_GRPO = _probe("trl", "GRPOTrainer") and _probe("trl", "GRPOConfig")


def banner() -> str:
    return (
        f"trl={TRL_VERSION} transformers={TRANSFORMERS_VERSION} "
        f"peft={PEFT_VERSION} | "
        f"SFT={HAS_SFT} DPO={HAS_DPO} Reward={HAS_REWARD} GRPO={HAS_GRPO}"
    )