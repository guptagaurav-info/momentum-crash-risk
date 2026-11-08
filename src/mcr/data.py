"""Monthly US stock panel from the defeatbeta Yahoo Finance mirror: prices, dollar volume and the liquid universe."""
import numpy as np
import pandas as pd

RAW = "data/"


def monthly_panel():
    m = pd.read_parquet(RAW + "monthly_long.parquet")
    m["date"] = m.m.dt.to_timestamp("M")
    px = m.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ndays = m.pivot_table(index="date", columns="symbol", values="n").sort_index()
    return px, ndays


def clean_returns(px, ndays, max_up=3.0):
    """Monthly simple returns; a month needs 10+ trading days of prices, and moves above +300% are treated as bad prints."""
    r = px / px.shift(1) - 1
    r = r.where(ndays >= 10)
    return r.mask(r > max_up)


def dollar_volume_12m():
    """Trailing 12-month average of mean daily dollar volume, known at each month-end."""
    m = pd.read_parquet(RAW + "monthly_long.parquet")
    m["date"] = m.m.dt.to_timestamp("M")
    dv = m.pivot_table(index="date", columns="symbol", values="dv").sort_index()
    return dv.rolling(12, min_periods=10).mean()
