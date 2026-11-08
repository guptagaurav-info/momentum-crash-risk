"""Momentum portfolios, crash states and volatility scaling."""
import numpy as np
import pandas as pd


def momentum_signal(px):
    """12-1 momentum: return from month t-12 to t-1, skipping the most recent month."""
    return px.shift(1) / px.shift(12) - 1


def universe(size, top=1000):
    """The `top` stocks by trailing 12-month average daily dollar volume (adjusted price x adjusted volume = real dollars)."""
    rk = size.rank(axis=1, ascending=False)
    return rk <= top


def decile_returns(sig, ret, uni, n=10):
    """Equal-weighted decile returns for month t+1, formed at the end of month t (liquid top-1000 universe)."""
    out = {}
    for t in sig.index[:-1]:
        s = sig.loc[t].where(uni.loc[t]).dropna()
        if len(s) < 200:
            continue
        nxt = sig.index[sig.index.get_loc(t) + 1]
        q = pd.qcut(s.rank(method="first"), n, labels=False) + 1
        r = ret.loc[nxt, s.index]; w = pd.Series(1.0, index=s.index)
        row = {}
        for k in range(1, n + 1):
            idx = q.index[q == k]
            rr = r[idx]; ww = w[idx].where(rr.notna())
            row[k] = (rr * ww).sum() / ww.sum() if ww.sum() > 0 else np.nan
        mkt_w = w.where(r.notna()); row["MKT"] = (r * mkt_w).sum() / mkt_w.sum()
        out[nxt] = row
    d = pd.DataFrame(out).T.sort_index()
    d["WML"] = d[10] - d[1]
    return d


def stats(x, periods=12):
    x = x.dropna()
    ann = x.mean() * periods; vol = x.std() * np.sqrt(periods)
    w = (1 + x).cumprod(); dd = (w / w.cummax() - 1).min()
    return {"ann_ret": ann, "ann_vol": vol, "sharpe": ann / vol if vol else np.nan,
            "max_dd": dd, "skew": x.skew(), "worst_month": x.min(), "n": len(x)}


def bear_state(mkt, months=24):
    """Daniel-Moskowitz bear indicator: cumulative market return over the past 24 months is negative (known at formation)."""
    cum = (1 + mkt).rolling(months).apply(np.prod, raw=True) - 1
    return (cum.shift(1) < 0)


def vol_scaled(wml_m, daily_wml, target=0.12, lookback=126):
    """Barroso & Santa-Clara: scale next month's exposure by target / trailing 6-month realised daily vol."""
    rv = daily_wml.rolling(lookback, min_periods=100).std() * np.sqrt(252)
    rv_m = rv.resample("ME").last().shift(1).reindex(wml_m.index)
    w = (target / rv_m).clip(upper=2.0)
    return wml_m * w, w


def extreme_members(sig, uni, n=10):
    """Winner (top decile) and loser (bottom decile) symbols formed at each month-end t, held in month t+1."""
    out = {}
    idx = sig.index
    for i, t in enumerate(idx[:-1]):
        s = sig.loc[t].where(uni.loc[t]).dropna()
        if len(s) < 200:
            continue
        q = pd.qcut(s.rank(method="first"), n, labels=False) + 1
        out[idx[i + 1]] = (list(q.index[q == n]), list(q.index[q == 1]))
    return out


def daily_wml(members, dret):
    """Daily equal-weighted winners-minus-losers return inside each holding month (rebalanced daily to equal weight)."""
    parts = []
    for m, (w, l) in members.items():
        days = dret.loc[(dret.index > m - pd.offsets.MonthEnd(1)) & (dret.index <= m)]
        if days.empty:
            continue
        rw = days.reindex(columns=w).mean(axis=1); rl = days.reindex(columns=l).mean(axis=1)
        parts.append(rw - rl)
    return pd.concat(parts).sort_index()
