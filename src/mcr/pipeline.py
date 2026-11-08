"""Run everything: python -m mcr.pipeline -> reports/metrics.json + reports/figures/*.png"""
import json, os, pickle
import yaml
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import data, analysis as A

NAVY, GREY, GOLD, RED, TEAL = "#1F3A5F", "#8A8F98", "#9A7F2E", "#B5473A", "#2A7F7F"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
FIG = "reports/figures/"


def daily_extremes(members, path="data/raw/stock_prices.parquet"):
    if os.path.exists("data/daily_ret_extremes.parquet"):
        return pd.read_parquet("data/daily_ret_extremes.parquet")
    syms = {s for w, l in members.values() for s in w + l}
    f = pq.ParquetFile(path); parts = []
    for i in range(f.num_row_groups):
        d = f.read_row_group(i, columns=["symbol", "report_date", "close"]).to_pandas()
        d = d[d.symbol.isin(syms)]
        if len(d):
            d["close"] = d.close.astype("float32"); parts.append(d)
    d = pd.concat(parts); d["d"] = pd.to_datetime(d.report_date)
    dp = d.pivot_table(index="d", columns="symbol", values="close", aggfunc="last").sort_index()
    r = dp.pct_change(fill_method=None)
    r = r.mask(r.abs() > 0.9).astype("float32")  # isolated bad daily prints
    r.to_parquet("data/daily_ret_extremes.parquet")
    return r


def r3(d):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in d.items()}


def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    px, nd = data.monthly_panel(); ret = data.clean_returns(px, nd)
    dv = data.dollar_volume_12m().reindex(index=px.index, columns=px.columns)
    uni = A.universe(dv, cfg["universe_top"]) & ret.notna()
    sig = A.momentum_signal(px).loc[cfg["start"]:]
    dec = A.decile_returns(sig, ret, uni, cfg["deciles"]); dec.columns = [str(c) for c in dec.columns]
    mem = A.extreme_members(sig, uni, cfg["deciles"])
    dwml = A.daily_wml(mem, daily_extremes(mem))
    wml, mkt = dec["WML"], dec["MKT"]
    vs, w = A.vol_scaled(wml, dwml, cfg["vol_target"], cfg["vol_lookback_days"]); w = w.clip(upper=cfg["max_leverage"])
    keep = vs.notna(); bear = A.bear_state(mkt, cfg["bear_months"]).reindex(wml.index).fillna(False)
    up = mkt > 0
    split = cfg["split"]; B = str(pd.Timestamp(split) + pd.offsets.MonthEnd(1))[:10]
    beta = {}
    for nm, m in [("bear", bear), ("bull", ~bear)]:
        X = np.c_[np.ones(m.sum()), mkt[m]]; beta[nm] = float(np.linalg.lstsq(X, wml[m], rcond=None)[0][1])
    worst = wml.nsmallest(10)
    out = {
        "sample": {"first_month": str(wml.index[0].date()), "last_month": str(wml.index[-1].date()), "months": int(len(wml)),
                   "stocks_ever_in_universe": int(uni.any().sum())},
        "deciles": {c: r3(A.stats(dec[c])) for c in ["1", "5", "10", "WML", "MKT"]},
        "wml_raw_same_months": r3(A.stats(wml[keep])), "wml_vol_scaled": r3(A.stats(vs[keep])),
        "split": {"A": {"raw": r3(A.stats(wml[keep][:split])), "scaled": r3(A.stats(vs[keep][:split]))},
                  "B": {"raw": r3(A.stats(wml[keep][B:])), "scaled": r3(A.stats(vs[keep][B:]))}},
        "states": {"bear_up": [float(wml[bear & up].mean()), int((bear & up).sum())], "bear_down": [float(wml[bear & ~up].mean()), int((bear & ~up).sum())],
                   "bull_up": [float(wml[~bear & up].mean()), int((~bear & up).sum())], "bull_down": [float(wml[~bear & ~up].mean()), int((~bear & ~up).sum())]},
        "beta": beta, "avg_weight": float(w[keep].mean()), "latest_weight": float(w.dropna().iloc[-1]),
        "worst_months": {str(k.date()): {"wml": float(v), "mkt": float(mkt[k]), "bear": bool(bear[k]), "scaled": float(vs[k]) if pd.notna(vs[k]) else None,
                                         "weight": float(w[k]) if pd.notna(w[k]) else None} for k, v in worst.items()},
        "recent": {str(k.date()): [float(wml[k]), float(w[k])] for k in wml.index[-14:]},
        "usd_100m": {"raw_worst_month": float(wml.min() * cfg["notional_usd"]), "scaled_worst_month": float(vs[keep].min() * cfg["notional_usd"])},
    }
    os.makedirs(FIG, exist_ok=True)
    json.dump(out, open("reports/metrics.json", "w"), indent=1)
    pd.DataFrame({"WML": wml, "VS": vs, "w": w, "MKT": mkt, "bear": bear}).to_csv("reports/monthly_series.csv")

    # 1. cumulative
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for s, c, lab in [(wml[keep], RED, "Raw momentum (winners minus losers)"), (vs[keep], NAVY, "Volatility-scaled to 12% a year")]:
        ax.plot((1 + s).cumprod(), color=c, lw=1.8, label=lab)
    ax.set_yscale("log"); ax.axhline(1, color=GREY, lw=0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.2, 0.5, 1, 2, 4])
    ax.set_title("$1 in US stock momentum, 1999–2026 (log scale)", loc="left"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG + "cumulative.png"); plt.close(fig)

    # 2. states
    st = out["states"]; lab = ["Bear market,\nmarket up", "Bear market,\nmarket down", "Normal,\nmarket up", "Normal,\nmarket down"]
    vals = [st[k][0] * 100 for k in ["bear_up", "bear_down", "bull_up", "bull_down"]]; ns = [st[k][1] for k in ["bear_up", "bear_down", "bull_up", "bull_down"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    b = ax.bar(lab, vals, color=[RED, GREY, NAVY, GREY])
    for r_, v, n in zip(b, vals, ns):
        ax.text(r_.get_x() + r_.get_width() / 2, v + 0.3 if v >= 0 else v / 2, f"{v:+.1f}%\n({n} months)", ha="center", va="bottom" if v >= 0 else "center", fontsize=10, color="black" if v >= 0 else "white")
    ax.axhline(0, color=GREY, lw=0.8); ax.set_ylabel("Average monthly return, %"); ax.set_ylim(min(vals) - 1, max(vals) + 2)
    ax.set_title("Momentum's average month by market state", loc="left")
    fig.tight_layout(); fig.savefig(FIG + "states.png"); plt.close(fig)

    # 3. exposure
    rv = dwml.rolling(cfg["vol_lookback_days"], min_periods=100).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(9, 4.4)); ax2 = ax.twinx()
    ax.plot(rv * 100, color=GREY, lw=1.2, label="Trailing 6-month volatility of momentum (left)")
    ax2.step(w.index, w, where="post", color=NAVY, lw=1.6, label="Exposure set by the 12% target (right)")
    ax.set_ylabel("Volatility, % a year"); ax2.set_ylabel("Exposure, × capital"); ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=9)
    ax.set_title("When momentum gets volatile, the scaled book shrinks", loc="left")
    fig.tight_layout(); fig.savefig(FIG + "exposure.png"); plt.close(fig)

    # 4. split comparison
    S = out["split"]; cats = ["Sharpe ratio", "Max drawdown", "Worst month"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, key, sc in zip(axs, ["sharpe", "max_dd", "worst_month"], [1, 100, 100]):
        x = np.arange(2); ax.bar(x - 0.18, [S["A"]["raw"][key] * sc, S["B"]["raw"][key] * sc], 0.36, color=RED, label="Raw")
        ax.bar(x + 0.18, [S["A"]["scaled"][key] * sc, S["B"]["scaled"][key] * sc], 0.36, color=NAVY, label="Scaled")
        ax.set_xticks(x); ax.set_xticklabels(["1999–2015", "2016–2026"]); ax.set_title(cats[["sharpe", "max_dd", "worst_month"].index(key)] + ("" if key == "sharpe" else ", %"), fontsize=11)
        ax.axhline(0, color=GREY, lw=0.6)
    axs[1].legend(frameon=False, fontsize=9, loc="lower right"); fig.suptitle("Both halves of the sample, raw vs volatility-scaled", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(FIG + "split.png"); plt.close(fig)
    print(json.dumps({k: out[k] for k in ["wml_raw_same_months", "wml_vol_scaled", "states", "beta", "latest_weight"]}, indent=1))


if __name__ == "__main__":
    main()
