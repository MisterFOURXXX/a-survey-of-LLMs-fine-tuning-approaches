"""BLEU / ROUGE-L / Exact Match, hardened for short or empty strings."""

from __future__ import annotations

from typing import Iterable


def _normalise(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _safe_corpus(preds, refs):
    pairs = [(p, r) for p, r in zip(preds, refs) if str(r).strip()]
    if not pairs:
        return [], []
    return [p for p, _ in pairs], [r for _, r in pairs]


def compute_bleu(preds: Iterable[str], refs: Iterable[str]) -> float:
    preds, refs = _safe_corpus(preds, refs)
    if not preds:
        return 0.0
    try:
        from sacrebleu.metrics import BLEU
        # sacrebleu expects refs as list-of-lists: [[ref1], [ref2], ...]
        score = BLEU(effective_order=True).corpus_score(preds, [refs]).score
        return float(score) / 100.0
    except Exception:
        return 0.0


def compute_rouge_l(preds: Iterable[str], refs: Iterable[str]) -> float:
    preds, refs = _safe_corpus(preds, refs)
    if not preds:
        return 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(r, p)["rougeL"].fmeasure
            for p, r in zip(preds, refs)
        ]
        return float(sum(scores)) / max(len(scores), 1)
    except Exception:
        return 0.0


def compute_exact_match(preds: Iterable[str], refs: Iterable[str]) -> float:
    preds, refs = list(preds), list(refs)
    if not preds:
        return 0.0
    hits = sum(1 for p, r in zip(preds, refs) if _normalise(p) == _normalise(r))
    return hits / max(len(preds), 1)


def compute_all_metrics(preds, refs) -> dict:
    preds = [str(p) for p in preds]
    refs = [str(r) for r in refs]
    return {
        "bleu": compute_bleu(preds, refs),
        "rougeL": compute_rouge_l(preds, refs),
        "exact_match": compute_exact_match(preds, refs),
        "n": len(preds),
    }