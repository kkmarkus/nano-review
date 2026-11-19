"""Treino do Transformer: o mesmo ciclo do bigrama (forward, loss, backward, atualização) em escala.

O que este script acrescenta ao laço do notebook ``03_forward_backward.ipynb``, e por quê:

- **Warmup e decaimento cosseno do LR:** no começo os pesos são aleatórios e os gradientes,
  instáveis; um LR alto logo de saída pode jogar o modelo para uma região ruim. Depois do warmup,
  o LR cai devagar até 10% do máximo, para os últimos passos fazerem ajustes finos.
- **Corte da norma do gradiente (1,0):** um lote estranho pode gerar um gradiente enorme e desfazer
  em um passo o que levou centenas para aprender. O corte limita o tamanho do passo.
- **Weight decay só nas matrizes:** encolher um pouco os pesos a cada passo é uma regularização
  contra decorar o treino. Viés e LayerNorm ficam de fora, porque não multiplicam a entrada e
  encolhê-los só atrapalha.
- **Avaliação periódica na validação e checkpoint do melhor modelo:** a loss de treino sempre cai;
  a de validação é que diz se o modelo está generalizando. O modelo salvo é o do melhor ponto da
  validação, e não o do último passo.
- **Early stopping (paciência):** se a validação passa ``patience`` avaliações seguidas sem
  melhorar, o treino para. Continuar só gastaria CPU, e o checkpoint salvo já é o do melhor ponto.
- **MLflow:** cada treino vira um *run*, com os parâmetros, a curva e os artefatos. Com três ou mais
  configurações, a memória não basta para lembrar qual gerou qual resultado; o painel compara.

Uso (normalmente pelo ``dvc repro``, que lê os parâmetros de ``params.yaml``)::

    python -m nano_review.training.train --train data/curated/train.parquet \\
        --validation data/curated/validation.parquet --tokenizer models/tokenizer.json \\
        --checkpoint models/transformer.pt --metrics reports/metrics/train.json \\
        --log reports/metrics/train_log.csv --figure reports/figures/train_curve.png \\
        --samples reports/samples/transformer_first.jsonl [hiperparâmetros...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import mlflow
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from nano_review.baseline.ngram import BOS, EOS, rating_token
from nano_review.data.dataset import IGNORE_INDEX, ReviewDataset, make_loader
from nano_review.models.loss import language_model_loss
from nano_review.models.transformer import (
    TransformerConfig,
    TransformerLanguageModel,
    count_parameters,
    generate,
)
from nano_review.reproducibility import set_seed
from nano_review.tokenization.bpe import decode, special_id


def learning_rate_at(step: int, max_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Warmup linear até ``max_lr`` e depois cosseno até 10% dele, no último passo."""
    min_lr = 0.1 * max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # O último passo é max_steps - 1; o denominador faz o progresso chegar a 1 exatamente nele.
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps - 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(
    model: torch.nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.AdamW:
    """AdamW com weight decay só nos parâmetros com 2 ou mais dimensões (as matrizes)."""
    matrices = [p for p in model.parameters() if p.dim() >= 2]
    others = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": matrices, "weight_decay": weight_decay},
            {"params": others, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


@torch.no_grad()
def evaluate(model: TransformerLanguageModel, loader: DataLoader[list[int]]) -> float:
    """Loss média **por token** em todo o loader.

    Somar a loss de cada lote multiplicada pelo número de tokens dele, e só no fim dividir, dá a
    média por token. A média das médias dos lotes daria peso igual a um lote de resenhas curtas e a
    um de resenhas longas.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        n = int((y != IGNORE_INDEX).sum())
        total += language_model_loss(model(x), y).item() * n
        count += n
    model.train(was_training)
    return total / count


def _forever(loader: DataLoader[list[int]]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Percorre o loader época após época; cada época sorteia uma ordem nova."""
    while True:
        yield from loader


def save_curve(log: list[dict[str, float]], path: Path) -> None:
    """Curva de treino e validação, no padrão visual dos notebooks."""
    import matplotlib

    matplotlib.use("Agg")  # sem janela: o script roda no terminal e no dvc repro
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in log]
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, label, color in [
        ("train_loss", "treino", "#2a78d6"),
        ("val_loss", "validação", "#eb6834"),
    ]:
        values = [row[key] for row in log]
        ax.plot(steps, values, color=color, linewidth=2, label=label)
        ax.annotate(
            label,
            (steps[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color="#52514e",
            fontsize=9,
        )
    ax.set_title("Transformer: loss por token", fontsize=11, fontweight="bold")
    ax.set_xlabel("passo", color="#52514e")
    ax.set_ylabel("entropia cruzada", color="#52514e")
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Nome do experimento no MLflow: todas as runs do projeto ficam juntas, para comparar.
MLFLOW_EXPERIMENT = "nano-review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    for name in (
        "train",
        "validation",
        "tokenizer",
        "checkpoint",
        "metrics",
        "log",
        "figure",
        "samples",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "block-size",
        "d-model",
        "n-layers",
        "n-heads",
        "batch-size",
        "warmup-steps",
        "max-steps",
        "eval-every",
        "eval-reviews",
        "patience",
        "samples-per-rating",
        "seed",
    ):
        parser.add_argument(f"--{name}", type=int, required=True)
    for name in ("dropout", "learning-rate", "weight-decay"):
        parser.add_argument(f"--{name}", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # Dentro de um `dvc exp run`, o DVC põe o nome do experimento nesta variável; assim o run do
    # MLflow e o experimento do DVC têm o mesmo nome e dá para cruzar os dois.
    run_name = os.environ.get("DVC_EXP_NAME", "dvc-repro")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        _train(args, run.info.run_id)


def _train(args: argparse.Namespace, run_id: str) -> None:
    set_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    mlflow.log_param("vocab_size", tokenizer.get_vocab_size())
    train_set = ReviewDataset(pd.read_parquet(args.train), tokenizer, args.block_size)
    validation = pd.read_parquet(args.validation)
    # Durante o treino, a validação é medida numa amostra fixa, para a avaliação não custar
    # mais que o próprio treino. No fim, o melhor modelo é medido na validação inteira.
    validation_sample = validation.sample(n=args.eval_reviews, random_state=args.seed)
    eval_loader = make_loader(
        ReviewDataset(validation_sample, tokenizer, args.block_size), tokenizer, 128, False, 0
    )
    full_validation_set = ReviewDataset(validation, tokenizer, args.block_size)
    train_loader = make_loader(train_set, tokenizer, args.batch_size, True, args.seed)

    config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model = TransformerLanguageModel(config)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    print(f"parâmetros: {count_parameters(model):,} | resenhas de treino: {len(train_set):,}")

    log: list[dict[str, float]] = []
    best_val, best_step = math.inf, -1
    evals_without_improvement = 0
    stopped_early = False
    # Soma da loss vezes o número de tokens, para a curva de treino ser média por token, como a
    # de validação. (A de treino é medida com dropout ligado; a de validação, sem.)
    window_loss_sum, window_tokens = 0.0, 0
    batches = _forever(train_loader)
    start = time.time()
    steps_run = 0
    for step in range(args.max_steps):
        learning_rate = learning_rate_at(
            step, args.learning_rate, args.warmup_steps, args.max_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        x, y = next(batches)
        loss = language_model_loss(model(x), y)  # forward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # backward
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()  # atualização
        steps_run = step + 1
        n_tokens = int((y != IGNORE_INDEX).sum())
        window_loss_sum += loss.item() * n_tokens
        window_tokens += n_tokens

        is_last = step == args.max_steps - 1
        if (step + 1) % args.eval_every == 0 or is_last:
            val_loss = evaluate(model, eval_loader)
            row = {
                "step": step + 1,
                "train_loss": round(window_loss_sum / window_tokens, 4),
                "val_loss": round(val_loss, 4),
                "learning_rate": round(learning_rate, 7),
                "grad_norm": round(grad_norm, 4),
                "elapsed_seconds": round(time.time() - start, 1),
            }
            log.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "step"}, step=step + 1)
            window_loss_sum, window_tokens = 0.0, 0
            print(json.dumps(row))
            if val_loss < best_val:
                best_val, best_step = val_loss, step + 1
                evals_without_improvement = 0
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"config": asdict(config), "state_dict": model.state_dict(), "step": step + 1},
                    args.checkpoint,
                )
            else:
                evals_without_improvement += 1
                # patience 0 desliga o early stopping: o treino vai até max_steps.
                if args.patience and evals_without_improvement >= args.patience:
                    stopped_early = True
                    print(f"early stopping: {args.patience} avaliações sem melhorar")
                    break

    train_seconds = time.time() - start
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    full_val_loss = evaluate(model, make_loader(full_validation_set, tokenizer, 128, False, 0))

    # Amostras de sanidade: só para ler e ver se o texto faz sentido. A avaliação formal
    # (coerência, cópia, diversidade, perplexity por palavra) é do estágio evaluate.
    generator = torch.Generator().manual_seed(args.seed)
    bos, eos = special_id(tokenizer, BOS), special_id(tokenizer, EOS)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("w", encoding="utf-8") as f:
        for rating in range(1, 6):
            prefix = [bos, special_id(tokenizer, rating_token(rating))]
            for _ in range(args.samples_per_rating):
                ids = generate(model, prefix, eos, generator=generator)
                text = decode(tokenizer, ids)
                f.write(json.dumps({"requested_rating": rating, "text": text}, ensure_ascii=False))
                f.write("\n")

    metrics = {
        "parameters": count_parameters(model),
        "train_reviews": len(train_set),
        "train_reviews_dropped_over_block": train_set.n_dropped,
        "max_steps": args.max_steps,
        "steps": steps_run,
        "stopped_early": stopped_early,
        "train_seconds": round(train_seconds, 1),
        "best_step": best_step,
        "best_val_loss_sample": round(best_val, 4),
        # Loss por token BPE na validação (só as resenhas que cabem no block_size). Ela inclui a
        # amostra que escolheu o checkpoint, então é levemente otimista; o número sem viés é o do
        # teste, na Etapa 4. A perplexity por palavra do H1 sai do estágio evaluate.
        "val_loss": round(full_val_loss, 4),
        "val_perplexity_per_token": round(math.exp(full_val_loss), 2),
        # O estágio evaluate lê este id e grava H1 a H6 no mesmo run do MLflow.
        "mlflow_run_id": run_id,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    with args.log.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    save_curve(log, args.figure)
    # Prefixo "final_": a val_loss final é da validação inteira e não pode cair na mesma curva
    # da val_loss do treino, que é da amostra fixa (bug achado pelo teste de ponta a ponta).
    mlflow.log_metrics(
        {f"final_{k}": float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    )
    for path in (args.metrics, args.log, args.figure, args.samples):
        mlflow.log_artifact(str(path))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
