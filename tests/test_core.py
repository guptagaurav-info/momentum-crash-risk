import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mcr import analysis as A


def test_signal_skips_last_month():
    px = pd.DataFrame({"a": np.arange(1.0, 15.0)}, index=pd.date_range("2000-01-31", periods=14, freq="ME"))
    s = A.momentum_signal(px)
    assert np.isclose(s["a"].iloc[13], px["a"].iloc[12] / px["a"].iloc[1] - 1)


def test_bear_state_uses_only_past():
    m = pd.Series([-0.05] * 30 + [0.5], index=pd.date_range("2000-01-31", periods=31, freq="ME"))
    b = A.bear_state(m, 24)
    assert bool(b.iloc[-1])  # the +50% month itself does not switch its own state


def test_vol_scaling_uses_lagged_vol():
    idx = pd.bdate_range("2000-01-03", "2001-12-31"); d = pd.Series(0.01 * np.sign(np.sin(np.arange(len(idx)))), index=idx)
    m = (1 + d).resample("ME").prod() - 1
    vs, w = A.vol_scaled(m, d, 0.12, 126)
    rv = d.rolling(126, min_periods=100).std().mul(np.sqrt(252)).resample("ME").last()
    assert np.isclose(w.iloc[-1], min(2.0, 0.12 / rv.iloc[-2]))


def test_deciles_long_short():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME"); cols = [f"s{i}" for i in range(300)]
    sig = pd.DataFrame(np.tile(np.arange(300.0), (3, 1)), index=idx, columns=cols)
    ret = pd.DataFrame(np.tile(np.arange(300.0) / 3000, (3, 1)), index=idx, columns=cols)
    uni = pd.DataFrame(True, index=idx, columns=cols)
    d = A.decile_returns(sig, ret, uni)
    assert (d["WML"] > 0).all() and np.isclose(d["WML"].iloc[0], ret.iloc[0, 270:].mean() - ret.iloc[0, :30].mean())
