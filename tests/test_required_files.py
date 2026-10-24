from pathlib import Path

import holdout_analysis as ha


def test_pairs_and_tff_exist():
    root = Path(__file__).resolve().parents[1]
    assert ha.TFF_PATH.exists() or (root / ha.TFF_PATH.name).exists()
    missing = [p for p in ha.PAIRS if not (root / f"combined_{p}_daily.csv").exists()]
    assert missing == [], f"Missing combined daily files: {missing}"


def test_combined_files_have_expected_columns():
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    required = {"date", "daily_open", "daily_high", "daily_low", "daily_close"}
    df = pd.read_csv(root / "combined_GBPAUD_daily.csv", nrows=5)
    assert required.issubset(set(df.columns))
