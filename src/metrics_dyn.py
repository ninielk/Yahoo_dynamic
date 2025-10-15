from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

TRADING_DAYS = 252
Z_95 = 1.96
_DDOF = 1
_EPS = 1e-12


@dataclass
class AnnParams:
    mu_ann: float
    sigma_ann: float
    sigma_sample: float


def _clean_series(r: pd.Series) -> pd.Series:
    return pd.to_numeric(r, errors="coerce").astype(float).dropna()


def daily_to_annual_mean(mu_daily: float) -> float:
    return float(mu_daily) * TRADING_DAYS


def annual_to_daily_rate(rate_ann: float) -> float:
    return float(rate_ann) / TRADING_DAYS


def daily_to_annual_vol(sig_daily: float) -> float:
    return float(sig_daily) * np.sqrt(TRADING_DAYS)


def z_from_conf(conf: float, two_sided: bool = True) -> float:
    conf = float(np.clip(conf, _EPS, 1 - _EPS))
    if two_sided:
        return float(st.norm.ppf(1 - (1 - conf) / 2))
    return float(st.norm.ppf(conf))


def annualize_mean_vol(r: pd.Series) -> AnnParams:
    r = _clean_series(r)
    if r.empty:
        return AnnParams(np.nan, np.nan, np.nan)
    mu_d = r.mean()
    sig_d = r.std(ddof=_DDOF)
    return AnnParams(
        mu_ann=float(daily_to_annual_mean(mu_d)),
        sigma_ann=float(daily_to_annual_vol(sig_d)),
        sigma_sample=float(sig_d),
    )


def sample_vol(r: pd.Series, annualized: bool = False) -> float:
    r = _clean_series(r)
    if r.empty:
        return np.nan
    s = r.std(ddof=_DDOF)
    return float(daily_to_annual_vol(s) if annualized else s)


def capm_beta(asset_r: pd.Series, mkt_r: pd.Series, rf_daily: float = 0.0) -> float:
    x = _clean_series(mkt_r)
    y = _clean_series(asset_r)
    idx = x.index.intersection(y.index)
    if len(idx) < 30:
        return np.nan
    x_ex = x.loc[idx] - rf_daily
    y_ex = y.loc[idx] - rf_daily
    X = sm.add_constant(x_ex.values)
    model = sm.OLS(y_ex.values, X, missing="drop").fit()
    return float(model.params[1]) if model.params.shape[0] >= 2 else np.nan


def capm_mu_ann_from_series(
    asset_r: pd.Series, mkt_r: pd.Series, rf_ann: float, mu_mkt_ann: float | None = None
):
    rf_daily = annual_to_daily_rate(rf_ann)
    beta = capm_beta(asset_r, mkt_r, rf_daily=rf_daily)
    if mu_mkt_ann is None:
        mu_mkt_ann = daily_to_annual_mean(_clean_series(mkt_r).mean())
    mu_capm = float(rf_ann + beta * (mu_mkt_ann - rf_ann)) if np.isfinite(beta) else np.nan
    return mu_capm, float(beta)


def mu_ann_from_premium(rf_ann: float, risk_premium_ann: float) -> float:
    return float(rf_ann + risk_premium_ann)


def dt_critical(mu_ann: float, sigma_ann: float, rf_ann: float, z: float = Z_95) -> float:
    if not np.isfinite(mu_ann) or not np.isfinite(sigma_ann) or sigma_ann <= 0:
        return np.nan
    excess = float(mu_ann) - float(rf_ann)
    if not np.isfinite(excess) or excess <= 0:
        return np.inf
    return float((z * sigma_ann / excess) ** 2)


def fisher_variance_test(x: pd.Series, y: pd.Series):
    x = _clean_series(x)
    y = _clean_series(y)
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan, np.nan, np.nan
    s1 = x.var(ddof=_DDOF)
    s2 = y.var(ddof=_DDOF)
    if s1 >= s2:
        F, df1, df2 = s1 / (s2 + _EPS), n1 - 1, n2 - 1
    else:
        F, df1, df2 = s2 / (s1 + _EPS), n2 - 1, n1 - 1
    p_upper = 1 - st.f.cdf(F, df1, df2)
    p_two = min(1.0, 2 * min(p_upper, st.f.cdf(1 / F, df1, df2)))
    return float(F), float(p_two), int(df1), int(df2)
