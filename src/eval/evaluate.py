"""Evaluate models listed in a YAML config."""

from __future__ import annotations

import gc
import os

import pandas as pd
import torch

from src.data.preprocess import load_stackoverflow, split_qa, make_eval_frame
from src.eval.generate import generate_predictions
from src.eval.metrics  import compute_all_metrics
from src.models.loaders import load_model_for_inference, load_tokenizer
from src.utils.config_loader import resolve_config


def evaluate_models(cfg) -> pd.DataFrame:
    cfg = resolve_config(cfg)

    df = load_stackoverflow(cfg["data_dir"])
    _, _, test_df = split_qa(df)
    eval_df = make_eval_frame(test_df).head(cfg.get("num_samples", 20))

    rows = []
    for name, path in cfg["models"].items():
        if not os.path.isdir(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        try:
            model, tok_src = load_model_for_inference(
                path, dtype="auto", device_map="auto"
            )
            tok = load_tokenizer(tok_src, padding_side="left")
            preds = generate_predictions(
                model, tok,
                eval_df["prompt"].tolist(),
                max_new_tokens=cfg.get("max_new_tokens", 128),
                batch_size=cfg.get("batch_size", 4),
                max_input_length=cfg.get("max_input_length", 512),
            )
            metrics = compute_all_metrics(preds, eval_df["reference"].tolist())
            metrics["method"] = name
            rows.append(metrics)
            print(f"[OK] {name}: bleu={metrics['bleu']:.4f} "
                  f"rougeL={metrics['rougeL']:.4f} EM={metrics['exact_match']:.4f}")
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    return pd.DataFrame(rows)