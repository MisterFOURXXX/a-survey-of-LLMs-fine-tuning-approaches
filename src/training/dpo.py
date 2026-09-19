import argparse
from src.data.preprocess import load_stackoverflow, split_qa
from src.data.dataset import build_dpo_datasets
from src.training.dpo import train_dpo
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train DPO on StackOverflow Q&A.")
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--raw_dir", default="data/raw/stacksample")
    parser.add_argument("--output_dir", default="outputs/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    df = load_stackoverflow(args.raw_dir)
    train_df, val_df, _ = split_qa(df, seed=args.seed)

    datasets = build_dpo_datasets(train_df, val_df, seed=args.seed)

    train_dpo(
        model_name=args.model_name,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        output_dir=args.output_dir,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()