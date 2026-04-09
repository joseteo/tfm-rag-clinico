"""Métricas de evaluación para el sistema RAG."""
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """Calcula ROUGE-1 y ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
) -> list[float]:
    """Calcula BERTScore F1 para un lote de predicciones."""
    _, _, f1 = bert_score_fn(
        predictions, references, lang=lang, verbose=False
    )
    return f1.tolist()


def compute_weighted_score(metrics: dict[str, float]) -> float:
    """Calcula el score ponderado según los pesos del TFM.

    Pesos: ROUGE-1 20%, ROUGE-L 20%, BERTScore 30%,
           Faithfulness 15%, Retrieval Precision 15%.
    """
    weights = {
        "rouge1_f1": 0.20,
        "rougeL_f1": 0.20,
        "bertscore_f1": 0.30,
        "faithfulness": 0.15,
        "retrieval_precision": 0.15,
    }
    total = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            total += metrics[metric] * weight
    return total
