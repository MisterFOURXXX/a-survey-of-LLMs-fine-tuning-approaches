import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from src.data.preprocess import load_stackoverflow, split_qa, make_sft_dataframe
from src.eval.generate import generate_predictions
from src.eval.metrics import compute_all_metrics
from src.utils.io import save_json, ensure_dir


def evaluate_model(path: str, prompts: list[str], references: list[str], max_new_tokens: int = 64):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        trust_remote_code=True,
    )

    preds = generate_predictions(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
    return compute_all_metrics(preds, references)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple trained models.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to trained model output directories.",
    )
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--raw_dir", default="data/raw/stacksample")
    parser.add_argument("--output", default="outputs/evaluation_comparison.csv")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_stackoverflow(args.raw_dir)
    _, _, test_df = split_qa(df, seed=args.seed)
    test_ds = Dataset.from_pandas(make_sft_dataframe(test_df), preserve_index=False)

    prompts = test_ds["text"][: args.num_samples]
    references = [p.split("Answer:")[-1].strip() for p in prompts]

    names = args.names if args.names else [Path(p).name for p in args.models]
    rows = []

    for name, path in zip(names, args.models):
        try:
            metrics = evaluate_model(path, prompts, references)
            metrics["method"] = name
            metrics["path"] = str(path)
            rows.append(metrics)
            print(f"[OK] {name}: {metrics}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    results = pd.DataFrame(rows)
    ensure_dir(Path(args.output).parent)
    results.to_csv(args.output, index=False)
    save_json(rows, Path(args.output).with_suffix(".json"))
    print(f"Saved comparison to {args.output}")
    print(results)


if __name__ == "__main__":
    main()