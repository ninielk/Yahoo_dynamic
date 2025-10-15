from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf

TRADING_DAYS = 252


def _download_one(ticker: str, start: str = "2016-01-01", end: str | None = None) -> pd.DataFrame:
    # yfinance traite 'end' comme EXCLUSIF -> +1 jour pour inclure la date choisie
    end_param = None
    if end is not None:
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
        end_param = end_dt.strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=start,
        end=end_param,
        auto_adjust=True,
        interval="1d",
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"Aucune donnée pour le ticker: {ticker}")
    return df[["Close"]].rename(columns={"Close": ticker}).dropna()


def download_prices_two(asset: str, benchmark: str, start: str = "2016-01-01", end: str | None = None) -> pd.DataFrame:
    """Télécharge deux séries de prix et garde l'intersection des dates (end inclusif)."""
    a = _download_one(asset, start=start, end=end)
    b = _download_one(benchmark, start=start, end=end)
    out = pd.concat([a, b], axis=1).dropna(how="any").reset_index().rename(columns={"Date": "Date"})
    if end is not None:
        mask = (out["Date"] >= pd.to_datetime(start)) & (out["Date"] <= pd.to_datetime(end))
        out = out.loc[mask].reset_index(drop=True)
    return out  # colonnes: Date, <asset>, <benchmark>


def compute_returns_two(
    prices: pd.DataFrame,
    asset_col: str,
    bench_col: str,
    asset_label: str = "Asset",
    bench_label: str = "Benchmark",
) -> pd.DataFrame:
    # Garde-fous colonnes
    for col in (asset_col, bench_col):
        if col not in prices.columns:
            raise KeyError(f"Colonne manquante: {col}. Colonnes={prices.columns.tolist()}")

    out = pd.DataFrame({"Date": prices["Date"]})
    pa = pd.to_numeric(prices.loc[:, asset_col].squeeze(), errors="coerce").astype(float)
    pb = pd.to_numeric(prices.loc[:, bench_col].squeeze(), errors="coerce").astype(float)

    out[f"price_{asset_label}"] = pa
    out[f"price_{bench_label}"] = pb
    out[f"ret_{asset_label}"] = pa.pct_change()
    out[f"ret_{bench_label}"] = pb.pct_change()
    return out.dropna()


def base100_from_price(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").astype(float).dropna()
    if p.empty:
        return p
    return (p / p.iloc[0]) * 100.0


def base100_from_returns(r: pd.Series) -> pd.Series:
    r = pd.to_numeric(r, errors="coerce").astype(float).fillna(0.0)
    return (1.0 + r).cumprod() * 100.0


def drawdown_from_returns(r: pd.Series) -> pd.Series:
    eq = (1.0 + pd.to_numeric(r, errors="coerce").fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0
