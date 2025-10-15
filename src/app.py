from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import statsmodels.api as sm
import streamlit as st

from dynamic_data import (
    download_prices_two,
    compute_returns_two,
    base100_from_price,
    base100_from_returns,
    drawdown_from_returns,
    TRADING_DAYS,
)
from metrics_dyn import (
    AnnParams,
    annualize_mean_vol,
    sample_vol,
    capm_beta,
    capm_mu_ann_from_series,
    mu_ann_from_premium,
    dt_critical,
    z_from_conf,
    TRADING_DAYS as TDYN,
    Z_95,
    fisher_variance_test,
)

assert TRADING_DAYS == TDYN

# ---------- Couleurs ----------
COL_ASSET = "#B10967"  # magenta (BTC / actif)
COL_BENCH = "#412761"  # indigo (S&P / bench)
GOLD_HEX = "#F8AF00"   # or


def color_for_label(label: str, default: str) -> str:
    if label.strip().lower() in {"gold", "or", "xau", "xauusd", "gld", "gc=f"}:
        return GOLD_HEX
    return default


# ---------- Style Matplotlib ----------
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "grid.color": "#DDDDDD",
        "grid.alpha": 0.8,
        "axes.grid": True,
        "axes.grid.which": "both",
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    }
)


def _fig_ax(figsize=(10, 4.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
    return fig, ax


def _fmt_compact(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{x:.0f}"


# ===================== UI =====================
st.set_page_config(page_title="2-Tickers Risk Dashboard (Yahoo)", layout="wide")
st.title("2-Tickers Risk & Correlation Dashboard (Yahoo Finance)")

today = pd.Timestamp.today().date()
with st.sidebar:
    st.header("Configuration")
    asset_ticker = st.text_input("Ticker ACTIF", value="BTC-USD")
    bench_ticker = st.text_input("Ticker BENCHMARK", value="^GSPC")
    asset_label = st.text_input("Label ACTIF ", value="BTC")
    bench_label = st.text_input("Label BENCHMARK ", value="S&P500")

    # Dates inclusives
    start_date = st.date_input("Date de début", value=pd.to_datetime("2016-01-01").date(), max_value=today)
    end_date = st.date_input("Date de fin (incluse)", value=today, min_value=start_date, max_value=today)

    win_vol = st.number_input("Fenêtre Vol (jours)", min_value=10, max_value=252, value=30, step=5)
    win_corr = st.number_input("Fenêtre Corr (jours)", min_value=20, max_value=252, value=90, step=5)

    rf_choice = st.selectbox("Taux sans risque (annuel)", ["0.00%", "2.00%", "Custom"], index=1)
    rf_ann = (
        st.number_input("rf annuel (%)", 0.0, 20.0, 2.00, 0.25) / 100.0
        if rf_choice == "Custom"
        else float(rf_choice.strip("%")) / 100.0
    )

    mu_method = st.selectbox(
        "Méthode μ (pour t* et CAPM)",
        ["Réel (moyenne empirique)", "CAPM — μ_mkt empirique", "CAPM — μ_mkt = rf + prime"],
        index=0,
    )
    prime_ann = (
        st.number_input("Prime de risque du marché (annuelle, %)", 0.0, 20.0, 5.00, 0.25) / 100.0
        if mu_method.endswith("prime")
        else None
    )

    conf_choice = st.selectbox("Confiance pour z", ["80%", "90%", "95%", "97.5%", "99%"], index=2)
    z_value = z_from_conf({"80%": 0.80, "90%": 0.90, "95%": 0.95, "97.5%": 0.975, "99%": 0.99}[conf_choice], True)

# ===================== Données =====================
@st.cache_data(ttl=86400, show_spinner=True)
def get_prices(asset: str, bench: str, start: str, end: str) -> pd.DataFrame:
    return download_prices_two(asset, bench, start=start, end=end)


try:
    prices = get_prices(
        asset_ticker,
        bench_ticker,
        start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
    )
except Exception as e:
    st.error(f"Erreur de téléchargement : {e}")
    st.stop()

# --- Sécurisation du choix de colonnes (fix du bug) ---
cols = [c for c in prices.columns if c != "Date"]
if len(cols) < 2:
    st.error(f"Je n'ai pas deux colonnes de prix dans le DataFrame: {prices.columns.tolist()}")
    st.stop()

asset_col_name = asset_ticker if asset_ticker in prices.columns else cols[0]
bench_col_name = bench_ticker if bench_ticker in prices.columns else cols[1]

d = compute_returns_two(
    prices,
    asset_col=asset_col_name,
    bench_col=bench_col_name,
    asset_label=asset_label,
    bench_label=bench_label,
)

# features rolling
d[f"vol_{asset_label}"] = d[f"ret_{asset_label}"].rolling(int(win_vol)).std(ddof=1) * np.sqrt(TRADING_DAYS)
d[f"vol_{bench_label}"] = d[f"ret_{bench_label}"].rolling(int(win_vol)).std(ddof=1) * np.sqrt(TRADING_DAYS)
d[f"corr_{asset_label}_{bench_label}"] = d[f"ret_{asset_label}"].rolling(int(win_corr)).corr(d[f"ret_{bench_label}"])

# ===================== KPIs =====================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    v = d[f"vol_{asset_label}"].dropna()
    st.metric(f"Vol {asset_label} (ann.)", "N/A" if v.empty else f"{v.iloc[-1]:.1%}")
with c2:
    v = d[f"vol_{bench_label}"].dropna()
    st.metric(f"Vol {bench_label} (ann.)", "N/A" if v.empty else f"{v.iloc[-1]:.1%}")
with c3:
    s = sample_vol(d[f"ret_{asset_label}"], annualized=True)
    st.metric(f"Vol sample {asset_label} (ann.)", "N/A" if not np.isfinite(s) else f"{s:.2%}")
with c4:
    s = sample_vol(d[f"ret_{bench_label}"], annualized=True)
    st.metric(f"Vol sample {bench_label} (ann.)", "N/A" if not np.isfinite(s) else f"{s:.2%}")
with c5:
    c = d[f"corr_{asset_label}_{bench_label}"].dropna()
    st.metric("Corr. rolling (dernière)", "N/A" if c.empty else f"{c.iloc[-1]:.2f}")

st.divider()

# ===================== Onglets =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Volatilité", "Corrélation", "Base 100", "Drawdown", "OLS", "Temps > r_f & CAPM", "Fisher"]
)

# -------- Volatilité --------
with tab1:
    st.subheader("Volatilité annualisée (rolling)")
    fig, ax = _fig_ax((10, 4.6))
    ax.plot(d["Date"], d[f"vol_{asset_label}"], label=asset_label, color=color_for_label(asset_label, COL_ASSET))
    ax.plot(d["Date"], d[f"vol_{bench_label}"], label=bench_label, color=color_for_label(bench_label, COL_BENCH))
    ax.set_ylabel("Vol annualisée")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# -------- Corrélation --------
with tab2:
    st.subheader(f"Corrélation roulante — {asset_label} ~ {bench_label}")
    fig, ax = _fig_ax((10, 4.6))
    ax.plot(
        d["Date"],
        d[f"corr_{asset_label}_{bench_label}"],
        label=f"{asset_label} ~ {bench_label}",
        color="#007078",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1, color="#666", alpha=0.8)
    ax.set_ylabel("Corrélation (fenêtre)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# -------- Base 100 --------
with tab3:
    st.subheader("Évolution comparée — Base 100")
    mode = st.radio("Méthode", ["Base 100 (log)", "Base 100 (linéaire)", "Base 100 (min-max 0–100)"], index=0, horizontal=True)

    idx = pd.DataFrame({"Date": d["Date"]})
    if f"price_{asset_label}" in d.columns:
        idx[asset_label] = base100_from_price(d[f"price_{asset_label}"])
        idx[bench_label] = base100_from_price(d[f"price_{bench_label}"])
    else:
        idx[asset_label] = base100_from_returns(d[f"ret_{asset_label}"])
        idx[bench_label] = base100_from_returns(d[f"ret_{bench_label}"])
    idx = idx.dropna()

    def _style(ax, ylab):
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_compact))
        ax.set_ylabel(ylab)

    if mode == "Base 100 (log)":
        fig, ax = _fig_ax((10, 4.8))
        ax.plot(idx["Date"], idx[asset_label], label=asset_label, color=color_for_label(asset_label, COL_ASSET))
        ax.plot(idx["Date"], idx[bench_label], label=bench_label, color=color_for_label(bench_label, COL_BENCH))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylabel("Indice base 100 (log)")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    elif mode == "Base 100 (linéaire)":
        fig, ax = _fig_ax((10, 4.8))
        ax.plot(idx["Date"], idx[asset_label], label=asset_label, color=color_for_label(asset_label, COL_ASSET))
        ax.plot(idx["Date"], idx[bench_label], label=bench_label, color=color_for_label(bench_label, COL_BENCH))
        _style(ax, "Indice base 100")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    else:
        mm = idx.copy()
        for col in [asset_label, bench_label]:
            x = mm[col].values
            lo, hi = np.nanmin(x), np.nanmax(x)
            mm[col] = 100.0 * (x - lo) / (hi - lo) if hi > lo else 0.0
        fig, ax = _fig_ax((10, 4.8))
        ax.plot(mm["Date"], mm[asset_label], label=asset_label, color=color_for_label(asset_label, COL_ASSET))
        ax.plot(mm["Date"], mm[bench_label], label=bench_label, color=color_for_label(bench_label, COL_BENCH))
        ax.set_ylabel("Échelle normalisée (0–100)")
        ax.set_ylim(-3, 103)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

# -------- Drawdown --------
with tab4:
    st.subheader("Drawdown cumulatif")
    dd = pd.DataFrame({"Date": d["Date"]})
    dd[asset_label] = drawdown_from_returns(d[f"ret_{asset_label}"])
    dd[bench_label] = drawdown_from_returns(d[f"ret_{bench_label}"])
    fig, ax = _fig_ax((10, 4.6))
    ax.plot(dd["Date"], dd[asset_label], label=asset_label, color=color_for_label(asset_label, COL_ASSET))
    ax.plot(dd["Date"], dd[bench_label], label=bench_label, color=color_for_label(bench_label, COL_BENCH))
    ax.set_ylabel("Drawdown")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# -------- OLS --------
with tab5:
    st.subheader(f"Scatter & OLS — {asset_label} ~ {bench_label}")
    fig, ax = _fig_ax((6.5, 5))
    ax.scatter(
        d[f"ret_{bench_label}"],
        d[f"ret_{asset_label}"],
        s=9,
        alpha=0.6,
        color=color_for_label(asset_label, COL_ASSET),
        edgecolors="none",
    )
    ax.set_xlabel(f"r_{bench_label}")
    ax.set_ylabel(f"r_{asset_label}")
    st.pyplot(fig, clear_figure=True)

    sub = d[[f"ret_{asset_label}", f"ret_{bench_label}"]].dropna()
    if sub.shape[0] > 50:
        X = sm.add_constant(sub[f"ret_{bench_label}"])
        y = sub[f"ret_{asset_label}"]
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        st.code(m.summary().as_text())
    else:
        st.code("Pas assez de points pour une régression fiable.")

# -------- Temps > r_f & CAPM --------
with tab6:
    st.subheader("Temps minimal pour battre le taux sans risque (t*)")

    p_asset: AnnParams = annualize_mean_vol(d[f"ret_{asset_label}"])
    p_bench: AnnParams = annualize_mean_vol(d[f"ret_{bench_label}"])

    if mu_method == "Réel (moyenne empirique)":
        mu_asset, mu_bench = p_asset.mu_ann, p_bench.mu_ann
    elif mu_method == "CAPM — μ_mkt empirique":
        mu_bench = p_bench.mu_ann
        mu_asset, beta = capm_mu_ann_from_series(d[f"ret_{asset_label}"], d[f"ret_{bench_label}"], rf_ann, mu_bench)
    else:
        mu_mkt_capm = mu_ann_from_premium(rf_ann, prime_ann or 0.0)
        beta = capm_beta(d[f"ret_{asset_label}"], d[f"ret_{bench_label}"], rf_daily=rf_ann / TRADING_DAYS)
        mu_asset = mu_ann_from_premium(rf_ann, (beta if np.isfinite(beta) else np.nan) * (mu_mkt_capm - rf_ann))
        mu_bench = mu_mkt_capm

    t_asset = dt_critical(mu_asset, p_asset.sigma_ann, rf_ann, z=z_value)
    t_bench = dt_critical(mu_bench, p_bench.sigma_ann, rf_ann, z=z_value)

    st.caption(f"Période: **{start_date} → {end_date}** — r_f: **{rf_ann:.2%}** — z: **{z_value:.2f}** — μ: **{mu_method}**.")
    st.markdown(
        f"""
        • **{asset_label}**: μ={mu_asset:.2%}, σ={p_asset.sigma_ann:.2%} → **t\\*** = {('∞' if not np.isfinite(t_asset) else f'{t_asset:.2f} ans')}  
        • **{bench_label}**: μ={mu_bench:.2%}, σ={p_bench.sigma_ann:.2%} → **t\\*** = {('∞' if not np.isfinite(t_bench) else f'{t_bench:.2f} ans')}
        """
    )

    params_df = pd.DataFrame(
        {"Actif": [asset_label, bench_label], "μ_ann": [mu_asset, mu_bench], "σ_ann (sample)": [p_asset.sigma_ann, p_bench.sigma_ann], "t* (ans)": [t_asset, t_bench]}
    )
    st.dataframe(params_df.style.format({"μ_ann": "{:.2%}", "σ_ann (sample)": "{:.2%}", "t* (ans)": "{:.2f}"}), use_container_width=True)

# -------- Fisher --------
with tab7:
    st.subheader("Test d’égalité des variances (Fisher) — rendements quotidiens")
    F, p, df1, df2 = fisher_variance_test(d[f"ret_{asset_label}"], d[f"ret_{bench_label}"])
    out = pd.DataFrame([(f"{asset_label} vs {bench_label}", F, p, df1, df2)], columns=["Paire", "F", "p-value", "df1", "df2"])
    st.dataframe(out.style.format({"F": "{:.4f}", "p-value": "{:.2e}"}), use_container_width=True)

# ========= Note =========
st.markdown(
    "<small>*Vol sample (quotidienne) = écart-type des rendements journaliers ; "
    "Vol sample (annualisée) = vol quotidienne × √252. "
    "Les vols affichées en tête sont des **volatilités roulantes** annualisées.*</small>",
    unsafe_allow_html=True,
)
