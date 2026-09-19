import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
exact_match = evaluate.load("exact_match")


def compute_all_metrics(predictions, references):
    em = exact_match.compute(
        predictions=predictions,
        references=references,
    )["exact_match"]

    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )["bleu"]

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "exact_match": em,
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
    }