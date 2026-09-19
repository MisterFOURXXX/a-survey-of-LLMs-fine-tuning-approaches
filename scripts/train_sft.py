"""CLI: Supervised Fine-Tuning (SFT) with optional LoRA / QLoRA."""

import argparse
import os
import sys

# Make the repo root importable when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset

from src.utils.seed import set_seed
from src.data.preprocess import load_stackoverflow, split_qa, make_sft_dataframe
from src.training.sft import train_sft


def main():
    parser = argparse.ArgumentParser(description="Train SFT / LoRA / QLoRA on StackSample.")
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--raw_dir", default=None,
                        help="Path to folder with Questions.csv and Answers.csv. "
                             "If omitted, auto-detect (Kaggle / local / env STACKSAMPLE_DIR).")
    parser.add_argument("--output_dir", default="outputs/sft-lora")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA and run full fine-tune.")
    parser.add_argument("--qlora", action="store_true",
                        help="Enable 4-bit quantization + LoRA (QLoRA).")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    df = load_stackoverflow(args.raw_dir)
    train_df, val_df, _ = split_qa(df, seed=args.seed)

    train_ds = Dataset.from_pandas(make_sft_dataframe(train_df), preserve_index=False)
    val_ds = Dataset.from_pandas(make_sft_dataframe(val_df), preserve_index=False)

    train_sft(
        model_name=args.model_name,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir=args.output_dir,
        use_lora=not args.no_lora,
        qlora=args.qlora,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()