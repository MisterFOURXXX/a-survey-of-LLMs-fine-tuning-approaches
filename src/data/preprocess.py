from pathlib import Path
import polars as pl
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def load_stackoverflow(raw_dir: str = "data/raw/stacksample") -> pd.DataFrame:
    raw = "/kaggle/input/datasets/stackoverflow/stacksample" #Path(raw_dir)

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

    questions = questions.sort("Score", descending=True).head(200)

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
    return df


def split_qa(df: pd.DataFrame, test_size=0.3, val_size=0.5, seed=42):
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=seed)
    return train_df, val_df, test_df


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
    for qid, group in df.groupby("question_id"):
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
    for qid, group in df.groupby("question_id"):
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

                prompt = f"Question: {chosen['question_title']} {chosen['question_body']}\nAnswer:"
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