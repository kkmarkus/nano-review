"""O lote cego da rubrica (H3): sorteio, cegueira, chave separada e o resumo das notas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nano_review.evaluation.rubric import (
    CRITERIA,
    SHEET_COLUMNS,
    batch_paths,
    build_batch,
    exclude_seen,
    load_source,
    normalize_spaces,
    read_sheet,
    real_reviews,
    summarize,
    write_batch,
)
from nano_review.tokenization.bpe import train_tokenizer


def _pool(name: str, per_rating: int = 30) -> pd.DataFrame:
    rows = [(r, f"{name} nota {r} amostra {i}") for r in range(1, 6) for i in range(per_rating)]
    return pd.DataFrame(rows, columns=["requested_rating", "text"])


SOURCES = {name: _pool(name) for name in ("trigram", "transformer", "real")}


def test_batch_has_n_per_rating_from_each_source_and_hides_the_origin() -> None:
    sheet, key = build_batch(SOURCES, per_rating=20, seed=0)
    assert tuple(sheet.columns) == SHEET_COLUMNS  # o formato de docs/rubric.md, sem "model"
    assert len(sheet) == 300
    assert sheet["sample_id"].is_unique
    assert set(sheet["sample_id"]) == set(key["sample_id"])
    assert (sheet[list(CRITERIA)] == "").all().all()  # as notas começam vazias

    merged = sheet.merge(key, on="sample_id")
    counts = merged.groupby(["model", "requested_rating"]).size()
    assert (counts == 20).all() and len(counts) == 15
    # Cada texto veio da fonte e da nota que a chave diz.
    assert all(
        text.startswith(f"{model} nota {rating} ")
        for text, model, rating in merged[["text", "model", "requested_rating"]].itertuples(
            index=False
        )
    )


def test_batch_is_repeatable_and_shuffled() -> None:
    sheet, key = build_batch(SOURCES, per_rating=20, seed=0)
    again_sheet, again_key = build_batch(SOURCES, per_rating=20, seed=0)
    pd.testing.assert_frame_equal(sheet, again_sheet)
    pd.testing.assert_frame_equal(key, again_key)

    other, _ = build_batch(SOURCES, per_rating=20, seed=1)
    assert not other["text"].equals(sheet["text"])
    # Embaralhado: a ordem das fontes não pode ser a ordem em que foram sorteadas.
    assert key["model"].tolist() != sorted(key["model"].tolist())


def test_batch_needs_enough_samples_for_every_rating() -> None:
    short = {"trigram": _pool("trigram", per_rating=5)}
    with pytest.raises(ValueError, match="nota 1"):
        build_batch(short, per_rating=20, seed=0)


def test_line_breaks_and_extra_spaces_do_not_reveal_the_source() -> None:
    assert normalize_spaces("Gostei.\n\nRecomendo   muito. ") == "Gostei. Recomendo muito."


def test_real_reviews_keep_only_what_fits_in_the_context_like_training() -> None:
    texts = ["Produto bom.", "Muito ruim " * 30]
    tokenizer = train_tokenizer(texts * 5, vocab_size=60)
    validation = pd.DataFrame({"review_text": texts, "overall_rating": [5, 1]})
    real = real_reviews(validation, tokenizer, block_size=12)
    assert real.to_dict("records") == [{"requested_rating": 5, "text": "Produto bom."}]


def test_writing_refuses_to_overwrite_a_batch(tmp_path: Path) -> None:
    sheet, key = build_batch(SOURCES, per_rating=2, seed=0)
    write_batch(tmp_path, sheet, key, {"seed": 0})
    sheet_path, key_path, _ = batch_paths(tmp_path)
    assert sheet_path.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM: acentos certos na planilha
    assert "model" not in read_sheet(sheet_path).columns
    with pytest.raises(FileExistsError):
        write_batch(tmp_path, sheet, key, {"seed": 0})


def _scored(rows: list[tuple[str, int, int, int, int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sample_id", "requested_rating", *CRITERIA, "model"])


def test_summary_counts_bad_samples_by_hand() -> None:
    table = _scored(
        [
            ("s1", 1, 3, 3, 3, 3, "trigram"),
            ("s2", 5, 1, 2, 3, 3, "trigram"),  # nota 1 na fluência: ruim
            ("s3", 1, 3, 3, 2, 3, "transformer"),
            ("s4", 5, 2, 3, 3, 3, "transformer"),
        ]
    )
    # Célula de observação vazia, lida do CSV, chega como NaN (None aqui).
    notes = pd.Series(["", "loop", None, ""], dtype="object")
    sheet = table.drop(columns="model").assign(text="x", notes=notes)
    result = summarize(sheet, table[["sample_id", "model"]])

    sources = result["sources"]
    assert isinstance(sources, dict)
    assert sources["trigram"]["bad_fraction"] == 0.5
    assert sources["transformer"]["bad_fraction"] == 0.0
    assert sources["trigram"]["means"]["fluency"] == 2.0  # (3 + 1) / 2
    assert sources["trigram"]["by_rating"]["5"]["bad_fraction"] == 1.0
    assert sources["trigram"]["with_notes"] == 1
    assert result["H3"] == {
        "rule": "fração de amostras ruins do Transformer menor que a do trigrama",
        "transformer": 0.0,
        "trigram": 0.5,
        "passed": True,
    }


@pytest.mark.parametrize("value", ["", "4", "0", "bom"])
def test_summary_rejects_missing_or_out_of_range_scores(value: str) -> None:
    sheet, key = build_batch(SOURCES, per_rating=1, seed=0)
    for column in CRITERIA:
        sheet[column] = "3"
    sheet.loc[0, "coherence"] = value
    first = str(sheet["sample_id"].iloc[0])
    with pytest.raises(ValueError, match=f"coherence: {first}"):
        summarize(sheet, key)


def test_summary_needs_the_key_to_match_the_sheet() -> None:
    sheet, key = build_batch(SOURCES, per_rating=1, seed=0)
    for column in CRITERIA:
        sheet[column] = 3
    with pytest.raises(ValueError, match="mesmos sample_id"):
        summarize(sheet, key.iloc[1:])


def test_reads_sheets_saved_with_semicolons(tmp_path: Path) -> None:
    # O Excel em português salva CSV com ponto e vírgula, porque a vírgula é o separador decimal.
    sheet, _ = build_batch(SOURCES, per_rating=1, seed=0)
    path = tmp_path / "sheet.csv"
    sheet.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    assert tuple(read_sheet(path).columns) == SHEET_COLUMNS


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def test_sources_come_from_jsonl_with_an_optional_label(tmp_path: Path) -> None:
    path = _jsonl(
        tmp_path / "samples.jsonl",
        [
            {"label": "t1.0-k0", "requested_rating": 5, "text": "Bom."},
            {"label": "t1.0-k50", "requested_rating": 1, "text": "Ruim."},
        ],
    )
    name, df = load_source(f"padrao={path}@t1.0-k0")
    assert name == "padrao" and df.to_dict("records") == [{"requested_rating": 5, "text": "Bom."}]
    _, everything = load_source(f"todas={path}")
    assert len(everything) == 2
    with pytest.raises(ValueError, match="label"):
        load_source(f"x={path}@t0.7-k20")
    with pytest.raises(ValueError, match="fonte inválida"):
        load_source(str(path))


def test_texts_already_seen_are_excluded_ignoring_case_and_punctuation() -> None:
    pool = pd.DataFrame(
        {"requested_rating": [5, 5, 1], "text": ["Ótimo produto!", "Chegou rápido", "Péssimo"]}
    )
    kept = exclude_seen(pool, {"ótimo produto", "péssimo"})
    assert kept["text"].tolist() == ["Chegou rápido"]


def test_a_second_batch_has_its_own_files_and_version(tmp_path: Path) -> None:
    sheet, key = build_batch(SOURCES, per_rating=1, seed=0)
    write_batch(tmp_path, sheet, key, {"seed": 0})  # v1
    write_batch(tmp_path, sheet, key, {"seed": 0}, suffix="_v2")  # não colide com o v1
    assert [p.name for p in batch_paths(tmp_path, "_v2")] == [
        "sheet_v2.csv",
        "key_v2.csv",
        "batch_v2.json",
    ]
    for column in CRITERIA:
        sheet[column] = 3
    assert summarize(sheet, key, version="1.1")["rubric_version"] == "1.1"
