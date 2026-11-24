"""Juiz de sentimento: TF-IDF + regressão logística em 3 classes.

O juiz é o **instrumento de medida** da métrica principal do charter, a coerência com a nota: ele
lê uma resenha gerada e diz se ela soa negativa, neutra ou positiva. Por isso ele é treinado só em
resenhas reais (``train``) e avaliado na ``validation``. A acurácia dele é o teto da coerência que
conseguimos medir: o gerador não pode parecer mais coerente do que o juiz é capaz de enxergar.

Uso (normalmente pelo ``dvc repro``, que lê os parâmetros de ``params.yaml``)::

    python -m nano_review.baseline.judge --train data/gold/train.parquet \\
        --validation data/gold/validation.parquet --model models/judge.joblib \\
        --metrics reports/metrics/judge.json --ngram-max 2 --min-df 2 --c 1.0
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

from nano_review.data.clean import dedupe_key

# A ordem importa: é a ordem das linhas e colunas da matriz de confusão no JSON.
CLASSES = ["negative", "neutral", "positive"]
# Rótulo para texto sem nenhuma palavra; não é uma classe do juiz, então nunca é "coerente".
EMPTY = "empty"


def rating_to_sentiment(rating: int) -> str:
    """Agrupa a nota em 3 classes, como define o charter: 1-2 negativo, 3 neutro, 4-5 positivo.

    Por que 3 classes e não 5: separar 4 de 5 (ou 1 de 2) pelo texto é difícil até para uma
    pessoa. Um juiz de 5 classes erraria muito nessas fronteiras, e esses erros apareceriam como
    "incoerência" do gerador.
    """
    if not 1 <= rating <= 5:
        raise ValueError(f"nota fora de 1 a 5: {rating}")
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def build_judge(ngram_max: int, min_df: int, c: float) -> Pipeline:
    """Monta o juiz sem treinar: o vetorizador e o classificador viajam juntos num Pipeline.

    Juntos porque o vocabulário do TF-IDF faz parte do modelo: ele é aprendido no treino e
    precisa ser exatamente o mesmo na hora de julgar uma resenha gerada.
    """
    vectorizer = TfidfVectorizer(
        # Caixa não muda o sentimento ("ÓTIMO" e "ótimo"), e juntar as duas formas dá mais
        # exemplos para cada palavra.
        lowercase=True,
        # Bigramas pegam negação e intensidade: "não recomendo" e "muito bom" viram uma
        # característica só, que uma palavra isolada não captura.
        ngram_range=(1, ngram_max),
        # Termo que aparece em uma resenha só não generaliza; descartá-lo encolhe o vocabulário.
        min_df=min_df,
        # log(1 + contagem): a 5ª repetição de "bom" não pode pesar 5 vezes a primeira.
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        C=c,
        # A classe neutra é a menor (12%). Sem o peso, o juiz quase nunca diria "neutro", e toda
        # amostra de nota 3 pareceria incoerente por culpa do juiz, e não do gerador.
        class_weight="balanced",
        max_iter=1000,
    )
    return Pipeline([("tfidf", vectorizer), ("logreg", classifier)])


def predict_sentiment(judge: Pipeline, texts: Sequence[str]) -> list[str]:
    """Julga uma lista de textos. É a função que a avaliação do gerador vai chamar.

    Texto sem nenhuma palavra ("", "!!!", "....") não tem sentimento. Sem esta regra, o juiz o
    chamaria de "positive", a classe que ele escolhe quando não há indício nenhum, e isso daria
    acerto grátis nas notas 4 e 5 e erro automático nas notas 1 a 3 (achado da revisão de código).
    """
    texts = list(texts)
    labels = [str(label) for label in judge.predict(texts)]
    return [label if dedupe_key(text) else EMPTY for text, label in zip(texts, labels, strict=True)]


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, object]:
    """Acurácia, F1 macro, recall por classe e matriz de confusão.

    A média **macro** dá o mesmo peso às 3 classes. A acurácia sozinha seria dominada pela classe
    positiva (60% das resenhas): um juiz que errasse quase todo neutro ainda teria acurácia alta.
    """
    recalls = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)), 4
        ),
        "recall": {name: round(float(r), 4) for name, r in zip(CLASSES, recalls, strict=True)},
        # Linha = classe verdadeira, coluna = classe prevista, na ordem de CLASSES.
        "confusion_matrix": [[int(v) for v in row] for row in matrix],
    }


def majority_baseline(y_train: Sequence[str], n: int) -> list[str]:
    """Piso trivial: responde sempre a classe mais comum do treino, sem ler o texto.

    Se o juiz não superar isto com folga, ele não está lendo o texto de verdade.
    """
    most_common = pd.Series(list(y_train)).mode().iloc[0]
    return [str(most_common)] * n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--ngram-max", type=int, required=True)
    parser.add_argument("--min-df", type=int, required=True)
    parser.add_argument("--c", type=float, required=True)
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    y_train = train["overall_rating"].map(rating_to_sentiment).tolist()
    y_val = validation["overall_rating"].map(rating_to_sentiment).tolist()

    judge = build_judge(args.ngram_max, args.min_df, args.c)
    judge.fit(train["review_text"].tolist(), y_train)

    metrics = {
        "judge": classification_metrics(
            y_val, predict_sentiment(judge, validation["review_text"].tolist())
        ),
        "majority_baseline": classification_metrics(y_val, majority_baseline(y_train, len(y_val))),
        "vocabulary_size": len(judge.named_steps["tfidf"].vocabulary_),
    }

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(judge, args.model)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
