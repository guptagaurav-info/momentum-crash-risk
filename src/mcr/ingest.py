"""Download the two Hugging Face files, check pinned checksums and build the monthly panel."""
import hashlib, os, urllib.request, yaml
import pandas as pd, pyarrow.parquet as pq
BASE = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US/"


def monthly_long(path, syms):
    f = pq.ParquetFile(path); parts = []
    for i in range(f.num_row_groups):  # row groups keep memory under ~2 GB
        d = f.read_row_group(i, columns=["symbol", "report_date", "close", "volume"]).to_pandas()
        d = d[d.symbol.isin(syms)]
        if d.empty:
            continue
        d["close"] = d.close.astype("float64"); d["d"] = pd.to_datetime(d.report_date); d["m"] = d.d.dt.to_period("M")
        d["dv"] = d.close * d.volume
        parts.append(d.sort_values("d").groupby(["symbol", "m"]).agg(close=("close", "last"), dv=("dv", "mean"), n=("close", "size")).reset_index())
    m = pd.concat(parts)
    return m.groupby(["symbol", "m"]).agg(close=("close", "last"), dv=("dv", "mean"), n=("n", "sum")).reset_index()


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml")); os.makedirs("data/raw", exist_ok=True)
    for f in ["stock_prices", "stock_profile"]:
        p = f"data/raw/{f}.parquet"
        if not os.path.exists(p):
            urllib.request.urlretrieve(BASE + f + ".parquet", p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f, h, "OK" if h == cfg["sha256"][f] else "MISMATCH (mirror updated; results may differ)")
    prof = pd.read_parquet("data/raw/stock_profile.parquet", columns=["symbol", "country", "sector", "industry"])
    u = prof[(prof.country == "United States") & prof.sector.notna() & (prof.sector != "")]
    u = u[~u.industry.fillna("").str.contains("Shell Companies")]
    u.to_parquet("data/universe.parquet")
    monthly_long("data/raw/stock_prices.parquet", set(u.symbol)).to_parquet("data/monthly_long.parquet")
