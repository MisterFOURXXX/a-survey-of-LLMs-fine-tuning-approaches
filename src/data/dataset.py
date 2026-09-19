"""Dataset builders and Hugging Face Dataset wrappers."""

from datasets import Dataset, DatasetDict
import pandas as pd

from .preprocess import (
    make_sft_dataframe,
    make_preference_pairs,
    make_reward_pairs,
    make_grpo_prompts,
    split_qa,
)


def build_sft_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_pandas(make_sft_dataframe(train_df), preserve_index=False),
        "validation": Dataset.from_pandas(make_sft_dataframe(val_df), preserve_index=False),
    })


def build_dpo_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = 42,
) -> DatasetDict:
    train_pref = make_preference_pairs(train_df)
    val_pref = make_preference_pairs(val_df)

    if train_pref.empty:
        raise ValueError("No preference pairs could be built from the training data.")
    if val_pref.empty:
        train_pref, val_pref = _fallback_split(train_pref, seed)

    return DatasetDict({
        "train": Dataset.from_pandas(train_pref, preserve_index=False),
        "validation": Dataset.from_pandas(val_pref, preserve_index=False),
    })


def build_reward_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = 42,
) -> DatasetDict:
    train_pairs = make_reward_pairs(train_df)
    val_pairs = make_reward_pairs(val_df)

    if train_pairs.empty:
        raise ValueError("No reward pairs could be built from the training data.")
    if val_pairs.empty:
        train_pairs, val_pairs = _fallback_split(train_pairs, seed)

    return DatasetDict({
        "train": Dataset.from_pandas(train_pairs, preserve_index=False),
        "validation": Dataset.from_pandas(val_pairs, preserve_index=False),
    })


def build_grpo_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> DatasetDict:
    train_prompts = make_grpo_prompts(train_df)
    val_prompts = make_grpo_prompts(val_df)

    if val_prompts.empty:
        train_prompts, val_prompts = _fallback_split(train_prompts, seed=42)

    return DatasetDict({
        "train": Dataset.from_pandas(train_prompts, preserve_index=False),
        "validation": Dataset.from_pandas(val_prompts, preserve_index=False),
    })


def _fallback_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """If no validation pairs exist, split the training pairs 90/10."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = max(1, int(0.9 * len(shuffled)))
    return shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]