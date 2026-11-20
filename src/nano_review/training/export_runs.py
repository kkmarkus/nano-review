"""Exporta as runs do MLflow para CSV, versionado no Git.

O MLflow deste projeto é local: o ``mlflow.db`` e a pasta ``mlruns/`` ficam fora do Git. Sem este
export, quem clona o repositório não vê os experimentos. Ele grava dois arquivos:

- ``runs.csv``: uma linha por run, com o nome (o mesmo do ``dvc exp``), os parâmetros e o valor
  final de cada métrica (H1 a H6, tempo, passos...);
- ``curves.csv``: as métricas registradas passo a passo (``train_loss``, ``val_loss``...), no
  formato longo ``run_name, metric, step, value``, para redesenhar e comparar as curvas.

Uso (na pasta do projeto, onde está o ``mlflow.db``)::

    python -m nano_review.training.export_runs --out-dir reports/experiments
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from nano_review.training.train import MLFLOW_EXPERIMENT, configure_mlflow


def runs_table(runs: list[Run]) -> pd.DataFrame:
    """Uma linha por run: identificação, parâmetros (``param.*``) e métricas (``metric.*``)."""
    rows = []
    for run in runs:
        row: dict[str, object] = {
            "run_name": run.info.run_name,
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": pd.Timestamp(run.info.start_time, unit="ms", tz="UTC").isoformat(),
        }
        row.update({f"param.{k}": v for k, v in sorted(run.data.params.items())})
        row.update({f"metric.{k}": v for k, v in sorted(run.data.metrics.items())})
        rows.append(row)
    return pd.DataFrame(rows)


def curves_table(client: MlflowClient, runs: list[Run]) -> pd.DataFrame:
    """Histórico das métricas com mais de um ponto, isto é, as registradas a cada avaliação."""
    rows: list[tuple[str, str, int, float]] = []
    for run in runs:
        for name in sorted(run.data.metrics):
            history = client.get_metric_history(run.info.run_id, name)
            if len(history) < 2:
                continue  # métrica final (um ponto só): já está no runs.csv
            rows.extend(
                (run.info.run_name, name, point.step, point.value)
                for point in sorted(history, key=lambda point: point.step)
            )
    return pd.DataFrame(rows, columns=["run_name", "metric", "step", "value"])


def export(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê o experimento ``nano-review`` e grava ``runs.csv`` e ``curves.csv`` em ``out_dir``."""
    configure_mlflow()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise ValueError(f"o MLflow não tem o experimento {MLFLOW_EXPERIMENT!r}")
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])
    runs_df = runs_table(list(runs))
    curves_df = curves_table(client, list(runs))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(out_dir / "runs.csv", index=False)
    curves_df.to_csv(out_dir / "curves.csv", index=False)
    return runs_df, curves_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs_df, curves_df = export(args.out_dir)
    print(f"{len(runs_df)} runs e {len(curves_df)} pontos de curva gravados em {args.out_dir}")


if __name__ == "__main__":
    main()
