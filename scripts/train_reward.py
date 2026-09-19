"""CLI: Bradley-Terry reward modeling."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.data.preprocess import load_stackoverflow, split_qa
from src.data.dataset import build_reward_datasets
from src.training.reward import train_reward


def main():
    parser = argparse.ArgumentParser(description="Train a Bradley-Terry reward model.")
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/reward")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    # --- NEW ---
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)

    df = load_stackoverflow(args.raw_dir)
    train_df, val_df, _ = split_qa(df, seed=args.seed)

    datasets = build_reward_datasets(train_df, val_df, seed=args.seed)

    train_reward(
        model_name=args.model_name,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()