"""Reward functions for GRPO.

Kept separate from `src/training/reward.py` (the reward-model trainer) on
purpose: this module exposes pure Python callables, that one trains a
sequence-classification model. Each function has the signature

    fn(completions: list[str], **kwargs) -> list[float]
"""

from __future__ import annotations


def length(completions, **kwargs):
    """Soft length reward: longer answers score higher up to a 200-word cap."""
    return [min(len(c.split()), 200) / 200.0 for c in completions]


def format(completions, **kwargs):
    """Reward answers that look complete (non-empty, ends cleanly or has code)."""
    out = []
    for c in completions:
        s = 0.0
        if c and c.strip():
            s += 0.5
        if "```" in c or c.rstrip().endswith((".", "!", "?")):
            s += 0.5
        out.append(s)
    return out


def no_repetition(completions, **kwargs):
    """Penalise n-gram repetition."""
    out = []
    for c in completions:
        toks = c.split()
        if len(toks) < 8:
            out.append(0.0)
            continue
        tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        out.append(len(set(tri)) / max(len(tri), 1))
    return out


REGISTRY = {
    "length": length,
    "format": format,
    "no_repetition": no_repetition,
}


def resolve_reward_funcs(names) -> list:
    """Turn YAML entries into callables.

    Entries can be:
      * a registered name (e.g. "length"),
      * a callable (kept as-is),
      * a path/ID string not in the registry (passed through; TRL will load
        it as a reward model).
    """
    if names is None:
        return [length, format, no_repetition]
    if callable(names):
        return [names]

    resolved = []
    for n in names:
        if callable(n):
            resolved.append(n)
        elif isinstance(n, str) and n in REGISTRY:
            resolved.append(REGISTRY[n])
        elif isinstance(n, str):
            # Not a registry name → assume it's a reward-model path/ID.
            resolved.append(n)
        else:
            raise ValueError(f"Unsupported reward_funcs entry: {n!r}")
    return resolved