"""CLI: train SFT from a YAML config."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset

from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.data.preprocess import load_stackoverflow, split_qa, make_sft_dataframe
from src.training.sft import train_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_lora.yaml",
                        help="Path to a YAML config (see configs/).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    df = load_stackoverflow(cfg["data_dir"])
    train_df, val_df, _ = split_qa(df, seed=cfg.get("seed", 42))

    train_ds = Dataset.from_pandas(make_sft_dataframe(train_df), preserve_index=False)
    val_ds   = Dataset.from_pandas(make_sft_dataframe(val_df),   preserve_index=False)

    train_sft(cfg, train_ds, val_ds)


if __name__ == "__main__":
    main()