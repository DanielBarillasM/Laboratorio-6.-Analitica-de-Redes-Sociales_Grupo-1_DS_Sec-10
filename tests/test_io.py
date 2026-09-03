from pathlib import Path

import pandas as pd

from lab6_social.io import detect_table_format, parse_count, read_table, repair_video_ids


def test_detects_true_csv(tmp_path: Path):
    path = tmp_path / "table.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert detect_table_format(path) == "csv"
    assert read_table(path).shape == (1, 2)


def test_detects_excel_even_with_csv_extension(tmp_path: Path):
    excel_path = tmp_path / "source.xlsx"
    misleading_path = tmp_path / "source.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_excel(excel_path, index=False)
    excel_path.rename(misleading_path)
    assert detect_table_format(misleading_path) == "excel"
    assert read_table(misleading_path).to_dict("records") == [{"a": 1, "b": 2}]


def test_repair_video_id_from_url():
    frame = pd.DataFrame({"video_id": [None], "video_url": ["https://www.youtube.com/watch?v=-ABC123"]})
    repaired, audit = repair_video_ids(frame)
    assert repaired.loc[0, "video_id"] == "-ABC123"
    assert audit.loc[0, "video_id_recuperado"] == "-ABC123"


def test_strip_formula_marker():
    frame = pd.DataFrame({"video_id": ["=-ABC123"], "video_url": ["https://youtu.be/-ABC123"]})
    repaired, audit = repair_video_ids(frame)
    assert repaired.loc[0, "video_id"] == "-ABC123"
    assert audit.empty


def test_parse_count_variants():
    assert parse_count(" ") == 0
    assert parse_count("1,250 vistas") == 1250
    assert parse_count("1.5 K") == 1500
    assert parse_count("2 mil") == 2000
