"""Fixtures compartilhadas pelos testes."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """MLflow numa pasta temporária: os testes nunca escrevem no ``mlflow.db`` do projeto.

    O ``chdir`` leva junto a pasta ``mlruns/`` dos artefatos, que o MLflow cria relativa à pasta
    atual.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    return uri
