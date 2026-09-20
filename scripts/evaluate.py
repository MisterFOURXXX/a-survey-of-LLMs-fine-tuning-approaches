"""CLI: evaluate every model in configs/eval.yaml."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from src.utils.config_loader import load_config
from src.eval.evaluate import evaluate_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate_models(cfg)

    if results.empty:
        print("No models were evaluated.")
        return

    print("\n", results.to_string(index=False))

    metrics = ["bleu", "rougeL", "exact_match"]
    ax = results.set_index("method")[metrics].plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1)
    plt.tight_layout()

    out = cfg.get("output_png", "outputs/evaluation_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()