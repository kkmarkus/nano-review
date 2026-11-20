"""O export das runs do MLflow para CSV: o que o Git guarda dos experimentos."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from nano_review.training.export_runs import export
from nano_review.training.train import MLFLOW_EXPERIMENT


@pytest.mark.usefixtures("isolated_mlflow")
def test_export_writes_one_row_per_run_and_the_step_curves(tmp_path: Path) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    for name, learning_rate, losses in (("run0", 3e-4, [5.0, 4.5]), ("e2", 1e-3, [4.8, 4.0])):
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"learning_rate": learning_rate})
            for step, loss in zip((250, 500), losses, strict=True):
                mlflow.log_metric("val_loss", loss, step=step)
            mlflow.log_metric("validation.word_perplexity", 100.0 if name == "run0" else 80.0)

    runs, curves = export(tmp_path / "experiments")

    assert runs["run_name"].tolist() == ["run0", "e2"]  # na ordem em que rodaram
    assert runs["param.learning_rate"].tolist() == ["0.0003", "0.001"]
    assert runs["metric.validation.word_perplexity"].tolist() == [100.0, 80.0]
    assert runs["metric.val_loss"].tolist() == [4.5, 4.0]  # o último valor da curva
    # Só a métrica com mais de um ponto vira curva; a final fica de fora.
    assert set(curves["metric"]) == {"val_loss"}
    e2 = curves[curves["run_name"] == "e2"]
    assert e2[["step", "value"]].values.tolist() == [[250, 4.8], [500, 4.0]]
    # O que foi gravado é o mesmo que foi devolvido.
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "experiments" / "curves.csv"), curves)
