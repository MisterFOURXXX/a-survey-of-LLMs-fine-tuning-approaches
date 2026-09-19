"""Version detection used to pick the right TRL / transformers keyword names.

All flags are derived from the installed package versions, so the same repo
works on trl 0.12 -> 0.16+ and transformers 4.40 -> 4.60+.
"""

from __future__ import annotations

import transformers

try:
    import trl
except ImportError:
    trl = None


def _v(mod) -> tuple[int, int, int]:
    if mod is None:
        return (0, 0, 0)
    parts = []
    for tok in mod.__version__.split(".")[:3]:
        num = "".join(ch for ch in tok if ch.isdigit())
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


TRANSFORMERS_VERSION = _v(transformers)
TRL_VERSION = _v(trl)

# ---- feature flags -------------------------------------------------------
# transformers >= 4.46 renamed:
#   evaluation_strategy -> eval_strategy
#   torch_dtype         -> dtype
USES_EVAL_STRATEGY = TRANSFORMERS_VERSION >= (4, 46, 0)
DTYPE_KWARG = "dtype" if TRANSFORMERS_VERSION >= (4, 46, 0) else "torch_dtype"

# TRL renamed max_seq_length -> max_length in 0.16.
# On 0.14.x SFTConfig does NOT accept `max_length`, so this MUST be False.
SFT_USES_MAX_LENGTH = TRL_VERSION >= (0, 16, 0)

# SFTTrainer accepts `processing_class` (not `tokenizer`) starting in trl 0.12
TRAINER_USES_PROCESSING_CLASS = TRL_VERSION >= (0, 12, 0)

# GRPOConfig gained mask_truncated_completions in 0.15
HAS_MASK_TRUNCATED = TRL_VERSION >= (0, 15, 0)

# GRPOConfig gained top_p/top_k in 0.15; on 0.14.0 they are NOT accepted
HAS_GRPO_TOP_P = TRL_VERSION >= (0, 15, 0)

# Trainer availability
HAS_SFT    = trl is not None and hasattr(trl, "SFTTrainer")
HAS_DPO    = trl is not None and hasattr(trl, "DPOTrainer")
HAS_REWARD = trl is not None and hasattr(trl, "RewardTrainer")
HAS_GRPO   = trl is not None and hasattr(trl, "GRPOTrainer")


def banner() -> str:
    tv = ".".join(map(str, TRANSFORMERS_VERSION))
    rv = ".".join(map(str, TRL_VERSION))
    return (
        f"trl={rv} transformers={tv} | "
        f"SFT={HAS_SFT} DPO={HAS_DPO} Reward={HAS_REWARD} GRPO={HAS_GRPO}"
    )