from pathlib import Path
import os
import polars as pl
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# Candidate locations, checked in order. First hit that contains
# "Questions.csv" and "Answers.csv" wins.
_DEFAULT_CANDIDATES = [
    # Local repo layout
    "data/raw/stacksample",
    "../data/raw/stacksample",
    # Kaggle "Datasets" input (new path scheme)
    "/kaggle/input/stacksample",
    "/kaggle/input/datasets/stackoverflow/stacksample",
    # Kaggle "Datasets" input (old path scheme)
    "/kaggle/input/stackoverflow-stacksample",
    # Colab / user home
    "~/data/raw/stacksample",
]


def _resolve_raw_dir(raw_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory that actually contains Questions.csv / Answers.csv.

    Priority:
      1. Environment variable STACKSAMPLE_DIR (if set)
      2. The explicit `raw_dir` argument (if provided)
      3. A list of well-known candidate locations
    """
    candidates: list[Path] = []

    env_dir = os.environ.get("STACKSAMPLE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    if raw_dir is not None:
        candidates.append(Path(raw_dir).expanduser())

    candidates.extend(Path(p).expanduser() for p in _DEFAULT_CANDIDATES)

    for cand in candidates:
        if cand.is_dir() and (cand / "Questions.csv").exists() and (cand / "Answers.csv").exists():
            return cand

    # Fallback: return the first candidate that exists (even if empty),
    # or the first candidate path so the error message is meaningful.
    for cand in candidates:
        if cand.is_dir():
            return cand

    raise FileNotFoundError(
        "Could not find the StackSample dataset. Checked:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nDownload instructions: see data/DOWNLOAD_DATA.txt "
          "or set the STACKSAMPLE_DIR environment variable."
    )


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_html(text: str | None) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_stackoverflow(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load and clean the StackSample Q&A dataset.

    Parameters
    ----------
    raw_dir : str or Path or None
        Path to the folder containing Questions.csv and Answers.csv.
        If None, common Kaggle/Colab/local paths are auto-detected.
        You can also set the STACKSAMPLE_DIR environment variable.
    """
    raw = _resolve_raw_dir(raw_dir)
    print(f"[load_stackoverflow] Using data directory: {raw}")

    questions = pl.read_csv(
        raw / "Questions.csv",
        encoding="utf8-lossy",
        columns=["Id", "Title", "Body", "Score"],
    ).filter(pl.col("Score") > 5)

    answers = pl.read_csv(
        raw / "Answers.csv",
        encoding="utf8-lossy",
        columns=["Id", "ParentId", "Body", "Score"],
    ).filter(pl.col("Score") > 5)

    questions = questions.sort("Score", descending=True).head(100) # was 200

    questions = questions.with_columns([
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8),
        pl.col("Title").str.strip_chars(),
    ])

    answers = answers.with_columns(
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8)
    )

    qa = answers.join(
        questions,
        left_on="ParentId",
        right_on="Id",
        how="inner",
    ).select([
        pl.col("ParentId").alias("question_id"),
        pl.col("Title").alias("question_title"),
        pl.col("Body_right").alias("question_body"),
        pl.col("Id").alias("answer_id"),
        pl.col("Body").alias("answer"),
        pl.col("Score").alias("answer_score"),
    ])

    df = qa.to_pandas().dropna()
    df = df[df["answer"].str.len() > 10]
    df = df[df["question_body"].str.len() > 10]

    print(f"[load_stackoverflow] Loaded {len(df)} Q&A rows")
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_qa(df: pd.DataFrame, test_size=0.3, val_size=0.5, seed=42):
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=seed)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Task-specific data builders
# ---------------------------------------------------------------------------
def format_sft_text(row) -> str:
    return (
        f"Question: {row['question_title']} {row['question_body']}\n"
        f"Answer: {row['answer']}"
    )


def make_sft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out.apply(format_sft_text, axis=1)
    return out[["text"]]


def make_preference_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _qid, group in df.groupby("question_id"):
        if len(group) < 2:
            continue

        group = group.sort_values("answer_score", ascending=False)
        chosen = group.iloc[0]
        rejected = group.iloc[-1]

        if chosen["answer_score"] > rejected["answer_score"]:
            prompt = f"Question: {chosen['question_title']} {chosen['question_body']}\nAnswer:"
            rows.append({
                "prompt": prompt,
                "chosen": chosen["answer"],
                "rejected": rejected["answer"],
                "chosen_score": chosen["answer_score"],
                "rejected_score": rejected["answer_score"],
            })
    return pd.DataFrame(rows)


def make_reward_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _qid, group in df.groupby("question_id"):
        if len(group) < 2:
            continue

        group = group.reset_index(drop=True)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group.iloc[i], group.iloc[j]
                if a["answer_score"] == b["answer_score"]:
                    continue

                if a["answer_score"] > b["answer_score"]:
                    chosen, rejected = a, b
                else:
                    chosen, rejected = b, a

                prompt = (
                    f"Question: {chosen['question_title']} "
                    f"{chosen['question_body']}\nAnswer:"
                )
                rows.append({
                    "chosen": f"{prompt} {chosen['answer']}",
                    "rejected": f"{prompt} {rejected['answer']}",
                })
    return pd.DataFrame(rows)


def make_grpo_prompts(df: pd.DataFrame) -> pd.DataFrame:
    prompts = []
    for _, row in df.iterrows():
        prompts.append({
            "prompt": f"Question: {row['question_title']} {row['question_body']}\nAnswer:",
            "reference": row["answer"],
        })
    return pd.DataFrame(prompts).drop_duplicates(subset=["prompt"])