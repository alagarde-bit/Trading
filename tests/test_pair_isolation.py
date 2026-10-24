import numpy as np
import pandas as pd

import holdout_analysis as ha


def _two_pair_bars():
    idx = [10, 11, 12, 13, 14, 15]
    df = pd.DataFrame(
        {
            "pair": ["GBPAUD", "USDJPY", "GBPAUD", "USDJPY", "GBPAUD", "USDJPY"],
            "daily_close": [1.90, 150.0, 1.91, 151.0, 1.92, 149.0],
            "daily_high": [1.905, 151.0, 1.915, 152.0, 1.925, 150.0],
            "daily_low": [1.895, 149.0, 1.905, 150.0, 1.915, 148.0],
        },
        index=idx,
    )
    return df


def test_next_5d_return_does_not_cross_pairs():
    df = pd.DataFrame(
        {
            "pair": ["A"] * 6 + ["B"] * 6,
            "daily_close": [1.0, 1.0, 1.0, 1.0, 1.0, 1.10, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0],
        }
    )
    df["next_5d_return"] = df.groupby("pair")["daily_close"].transform(lambda s: s.shift(-5) / s - 1)
    a0 = df.loc[0, "next_5d_return"]
    b0 = df.loc[6, "next_5d_return"]
    assert abs(a0 - 0.10) < 1e-9
    assert abs(b0 - (-0.10)) < 1e-9


def test_check_surpass_uses_same_pair_prices():
    df = _two_pair_bars()
    row = df.loc[10]
    # Small predicted move; USDJPY prices must not be used as "future" GBPAUD bars
    surpass, pnl, reason, exit_idx = ha.check_surpass_and_pnl(
        row, df, pred_return=0.02, days=2, trade_size=10000, stop_loss_mult=1.5
    )
    assert reason in ("target", "stop", "expire", "no_data")
    # PnL as a fraction of notional should stay FX-sized, not 150/1.9
    assert abs(pnl) < 5000


def test_relative_features_per_pair_pct_change():
    df = pd.DataFrame(
        {
            "pair": ["A", "A", "B", "B"],
            "daily_close": [1.0, 1.1, 100.0, 110.0],
            "daily_open": [1.0, 1.05, 100.0, 105.0],
            "daily_high": [1.02, 1.12, 102.0, 112.0],
            "daily_low": [0.99, 1.04, 99.0, 104.0],
            "daily_sma_20": [1.0, 1.0, 100.0, 100.0],
            "daily_sma_50": [1.0, 1.0, 100.0, 100.0],
            "daily_sma_200": [1.0, 1.0, 100.0, 100.0],
            "weekly_high": [1.12, 1.12, 112.0, 112.0],
            "weekly_low": [0.99, 0.99, 99.0, 99.0],
            "weekly_close": [1.1, 1.1, 110.0, 110.0],
            "weekly_open": [1.0, 1.0, 100.0, 100.0],
            "weekly_sma_4": [1.0, 1.0, 100.0, 100.0],
            "weekly_sma_10": [1.0, 1.0, 100.0, 100.0],
            "weekly_sma_40": [1.0, 1.0, 100.0, 100.0],
            "monthly_high": [1.12, 1.12, 112.0, 112.0],
            "monthly_low": [0.99, 0.99, 99.0, 99.0],
            "monthly_close": [1.1, 1.1, 110.0, 110.0],
            "monthly_open": [1.0, 1.0, 100.0, 100.0],
            "monthly_sma_1": [1.0, 1.0, 100.0, 100.0],
            "monthly_sma_3": [1.0, 1.0, 100.0, 100.0],
            "monthly_sma_10": [1.0, 1.0, 100.0, 100.0],
            "daily_rsi": [50.0, 55.0, 50.0, 55.0],
            "weekly_rsi": [50.0, 50.0, 50.0, 50.0],
            "monthly_rsi": [50.0, 50.0, 50.0, 50.0],
        }
    )
    out = df.groupby("pair", group_keys=False).apply(ha.compute_relative_features)
    a_ret = out.loc[out["pair"] == "A", "daily_return"].iloc[1]
    b_ret = out.loc[out["pair"] == "B", "daily_return"].iloc[1]
    assert abs(a_ret - 0.1) < 1e-9
    assert abs(b_ret - 0.1) < 1e-9


def test_pattern_group_hammer_is_reversal():
    assert ha._pattern_to_group("Hammer") == "reversal"
    assert ha._pattern_to_group("BearishFlag_NN") == "continuation"
