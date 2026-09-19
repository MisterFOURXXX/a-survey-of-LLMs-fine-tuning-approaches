import argparse, os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from src.utils.seed import set_seed
from src.utils.version import banner
from src.data.preprocess import load_stackoverflow, split_qa, make_sft_dataframe
from src.training.sft import train_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/sft-lora")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print("Environment:", banner())
    set_seed(42)

    df = load_stackoverflow(args.raw_dir)
    train_df, val_df, _ = split_qa(df)

    train_ds = Dataset.from_pandas(make_sft_dataframe(train_df), preserve_index=False)
    val_ds = Dataset.from_pandas(make_sft_dataframe(val_df), preserve_index=False)

    train_sft(
        model_name=args.model_name,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir=args.output_dir,
        use_lora=True,
        qlora=args.qlora,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()