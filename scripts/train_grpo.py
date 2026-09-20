"""CLI: train GRPO from a YAML config."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.data.preprocess import load_stackoverflow, split_qa, make_grpo_prompts
from src.data.dataset import make_train_val_datasets
from src.training.grpo import train_grpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    df = load_stackoverflow(cfg["data_dir"])
    train_df, val_df, _ = split_qa(df, seed=cfg.get("seed", 42))

    grpo_df = make_grpo_prompts(train_df)
    train_ds, val_ds = make_train_val_datasets(grpo_df, frac=0.8, seed=cfg.get("seed", 42))

    train_grpo(cfg, train_ds, val_ds)


if __name__ == "__main__":
    main()