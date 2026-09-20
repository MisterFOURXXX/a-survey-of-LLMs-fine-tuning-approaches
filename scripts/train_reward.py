"""CLI: train a Bradley-Terry reward model from a YAML config."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.data.preprocess import load_stackoverflow, split_qa, make_reward_pairs
from src.data.dataset import make_train_val_datasets
from src.training.reward import train_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reward.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    df = load_stackoverflow(cfg["data_dir"])
    train_df, val_df, _ = split_qa(df, seed=cfg.get("seed", 42))

    rew_df = make_reward_pairs(train_df)
    train_ds, val_ds = make_train_val_datasets(rew_df, frac=0.8, seed=cfg.get("seed", 42))

    train_reward(cfg, train_ds, val_ds)


if __name__ == "__main__":
    main()