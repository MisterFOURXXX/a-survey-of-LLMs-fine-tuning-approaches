"""Load YAML configs with `defaults:` inheritance.

A config file may declare:

    defaults: [base, other]

Each name resolves to `<configs_dir>/<name>.yaml`; the parents are deep-merged
in the given order, and the current file's own keys override them.
"""

from __future__ import annotations

import os
from copy import deepcopy

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k == "defaults":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, configs_dir: str | None = None) -> dict:
    """Load `path` and resolve its `defaults:` chain.

    `configs_dir` defaults to the directory containing `path`, so parents
    referenced by name are looked up next to the calling file.
    """
    configs_dir = configs_dir or os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    merged: dict = {}
    for parent in raw.get("defaults", []) or []:
        parent_path = os.path.join(configs_dir, f"{parent}.yaml")
        merged = _deep_merge(merged, load_config(parent_path, configs_dir))

    merged = _deep_merge(merged, raw)
    merged.pop("defaults", None)
    return merged


def resolve_config(cfg_or_path) -> dict:
    """Accept a dict (returned as-is) or a path to a YAML file."""
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    return load_config(str(cfg_or_path))