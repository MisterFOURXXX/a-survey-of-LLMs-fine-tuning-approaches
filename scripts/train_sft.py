import argparse
from datasets import Dataset
from src.data.preprocess import (
    load_stackoverflow,
    split_qa,
    make_sft_dataframe,
)
from src.training.sft import train_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--raw_dir", default="data/raw/stacksample")
    parser.add_argument("--output_dir", default="outputs/sft-lora")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()

    df = load_stackoverflow(args.raw_dir)
    train_df, val_df, _ = split_qa(df)

    train_ds = Dataset.from_pandas(make_sft_dataframe(train_df))
    val_ds = Dataset.from_pandas(make_sft_dataframe(val_df))

    train_sft(
        model_name=args.model_name,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        qlora=args.qlora,
    )


if __name__ == "__main__":
    main()