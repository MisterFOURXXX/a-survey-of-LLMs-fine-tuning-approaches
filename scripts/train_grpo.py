"""CLI: GRPO (Group Relative Policy Optimization)."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.data.preprocess import load_stackoverflow, split_qa
from src.data.dataset import build_grpo_datasets
from src.training.grpo import train_grpo


def main():
    parser = argparse.ArgumentParser(description="Train with GRPO on StackSample.")
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/grpo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    df = load_stackoverflow(args.raw_dir)
    train_df, val_df, _ = split_qa(df, seed=args.seed)

    datasets = build_grpo_datasets(train_df, val_df)

    train_grpo(
        model_name=args.model_name,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
    )


if __name__ == "__main__":
    main()