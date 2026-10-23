#!/usr/bin/env python3
"""
Cost-Sensitive AI Thresholds for Decision Intelligence:
Optimizing Data-Driven Cutoffs Under Asymmetric Business Loss
=============================================================

Single-file, fully reproducible experiment for the Short Communication of the
same title (target journal: Knowledge-Based Systems, Elsevier).

Research questions
  RQ1  How much decision cost is lost when an AI credit-default model is run at
       an accuracy-oriented cutoff instead of a cost-sensitive one, and how does
       the gap grow with the business-loss ratio r = C_FN / C_FP?
  RQ2  Does the closed-form decision-theoretic threshold t*(r) = 1 / (1 + r),
       which turns explicit managerial cost knowledge into a decision rule
       without retraining, recover the attainable savings, and under which
       condition (probability calibration)?
  RQ3  How robust is the knowledge-based threshold when managers only know an
       interval [r_lo, r_hi] for r, and does a minimax-regret threshold limit
       the worst-case loss?

Usage
  python main.py                # full protocol: 20 repeated stratified splits
  python main.py --repeats 5    # faster check run
All tables (CSV) and figures (PNG at 600 dpi + vector PDF) are written to ./results.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "uci_credit_default.csv.gz"
UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
CONTENT_SHA256 = "debf24d75b240e4f9e7faecd964ae5859976878bb9a19917962ed5670992c83b"

COST_RATIOS = (1, 2, 3, 5, 10, 20)            # business-loss scenarios r = C_FN / C_FP
CURVE_RATIOS = (1, 3, 5, 10, 20)              # cost curves drawn in Figure 1a
ABLATION_RATIOS = (2, 5, 10)                  # calibration ablation (Table 2A)
INTERVALS = {"A": (3.0, 10.0), "B": (2.0, 20.0)}  # expert knowledge on r (Table 2B)
T_GRID = np.round(np.linspace(0.001, 0.999, 999), 3)  # threshold search grid
R_FINE = np.geomspace(1.0, 20.0, 61)          # fine grid of cost ratios (figures)
PROTECTED = ("SEX", "MARRIAGE", "AGE")        # excluded from model inputs (fair lending)
MONEY = ["LIMIT_BAL"] + [f"BILL_AMT{k}" for k in range(1, 7)] + [f"PAY_AMT{k}" for k in range(1, 7)]
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PRIMARY = ("LR-cal", "GBDT-cal")              # calibrated models used in Table 1
ABLATION_VARIANTS = ("GBDT", "GBDT-cal", "GBDT-bal", "GBDT-bal-pc", "GBDT-bal-cal", "LR", "LR-cal")
POLICIES = ("default_0.5", "accuracy_opt", "f1_opt", "cost_weighted_retrain",
            "empirical_cost_opt", "kb_threshold")

# Figure styling. Categorical slots and the ordinal ramp were checked for
# colour-vision-deficiency separation (adjacent CVD dE >= 8; ramp steps dL >= 0.06).
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
RAMP5 = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
SEQ = ("#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.solid_capstyle": "round",
    })


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, str]:
    """Load the UCI 'Default of Credit Card Clients' data (Yeh, 2009; CC BY 4.0).

    The repository ships a gzip copy of the original file. If it is missing, the
    official UCI archive is downloaded and converted (reading .xls needs xlrd).
    """
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] downloading {UCI_URL}")
        with urllib.request.urlopen(UCI_URL, timeout=120) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        xls = next(n for n in archive.namelist() if n.lower().endswith(".xls"))
        try:
            raw = pd.read_excel(archive.open(xls), header=1)
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Reading the original .xls requires: pip install xlrd") from exc
        raw = raw.rename(columns={"default payment next month": "DEFAULT"})
        raw.to_csv(DATA_FILE, index=False, compression={"method": "gzip", "mtime": 0})
    df = pd.read_csv(DATA_FILE)
    digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    if df.shape != (30000, 25) or int(df["DEFAULT"].sum()) != 6636:
        raise ValueError("Integrity check failed: expected 30,000 clients and 6,636 defaults.")
    if digest != CONTENT_SHA256:
        warnings.warn("Data content hash differs from the reference copy used in the paper.")
    return df, digest


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Behavioural credit features; protected demographics are deliberately excluded."""
    X = df.drop(columns=["ID", "DEFAULT", *PROTECTED]).copy()
    edu = X.pop("EDUCATION").replace({0: 4, 5: 4, 6: 4})  # undocumented codes -> 'others'
    for code, name in ((1, "EDU_GRADUATE"), (2, "EDU_UNIVERSITY"), (3, "EDU_HIGHSCHOOL")):
        X[name] = (edu == code).astype(float)
    bills = X[[f"BILL_AMT{k}" for k in range(1, 7)]].astype(float)
    pays = X[[f"PAY_AMT{k}" for k in range(1, 7)]].astype(float)
    limit = X["LIMIT_BAL"].astype(float)
    owed = bills.clip(lower=0).sum(axis=1)
    X["UTIL_LAST"] = bills["BILL_AMT1"] / limit
    X["UTIL_MEAN"] = bills.mean(axis=1) / limit
    X["REPAY_RATIO"] = (pays.sum(axis=1) / owed.replace(0, np.nan)).fillna(1.0).clip(0, 5)
    X["N_DELAYED"] = (X[PAY] > 0).sum(axis=1)
    X["MAX_DELAY"] = X[PAY].max(axis=1).clip(lower=0)
    return X.astype(float), df["DEFAULT"].to_numpy(dtype=int)


# --------------------------------------------------------------------------- #
# Predictive layer (models) and knowledge-based decision layer (thresholds)
# --------------------------------------------------------------------------- #
def make_lr(class_weight=None) -> Pipeline:
    money = Pipeline([("slog", FunctionTransformer(signed_log1p)), ("z", StandardScaler())])
    pre = ColumnTransformer([("money", money, MONEY)], remainder=StandardScaler())
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=1.0, max_iter=5000, class_weight=class_weight))])


def make_gbdt(seed: int, class_weight=None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, class_weight=class_weight, random_state=seed)


def calibrated(estimator) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(estimator, method="isotonic", cv=5)


def kb_threshold(r):
    """Bayes-optimal cutoff for 'decline if p >= t' when C_FN = r * C_FP (Elkan, 2001)."""
    return 1.0 / (1.0 + np.asarray(r, dtype=float))


def prior_correct(p_w: np.ndarray, k: float) -> np.ndarray:
    """Undo class re-weighting: odds = k * weighted odds, with k = w0 / w1."""
    return k * p_w / (k * p_w + (1.0 - p_w))


def error_counts(p: np.ndarray, y: np.ndarray, thresholds) -> tuple[np.ndarray, np.ndarray, int, int]:
    """FN and FP counts of the rule 'decline if p >= t' for every threshold t."""
    t = np.atleast_1d(np.asarray(thresholds, dtype=float))
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    fn = np.searchsorted(pos, t, side="left")
    fp = neg.size - np.searchsorted(neg, t, side="left")
    return fn, fp, pos.size, neg.size


def cost_per_1k(fn, fp, r, n):
    """Expected misclassification cost per 1,000 applicants in units of C_FP."""
    return 1000.0 * (np.asarray(r) * fn + fp) / n


def decision_metrics(p, y, t, r) -> dict:
    fn, fp, n1, n0 = error_counts(p, y, t)
    fn, fp, n = int(fn[0]), int(fp[0]), n1 + n0
    tp = n1 - fn
    return {"threshold": float(t), "cost_per_1k": float(cost_per_1k(fn, fp, r, n)),
            "accuracy": (n - fn - fp) / n, "decline_rate": (tp + fp) / n,
            "default_capture": tp / n1, "good_declined": fp / n0}


def ece(p, y, bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if np.any(idx == b)))


def reliability(p, y, bins: int = 20):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, bins - 1)
    keep = [b for b in range(bins) if np.any(idx == b)]
    return (np.array([p[idx == b].mean() for b in keep]), np.array([y[idx == b].mean() for b in keep]))


# --------------------------------------------------------------------------- #
# Statistics: Nadeau-Bengio corrected resampled t (repeated random splits)
# --------------------------------------------------------------------------- #
def corrected_t(d, rho: float) -> tuple[float, float, float, float]:
    d = np.asarray(d, dtype=float)
    R, mean, var = d.size, d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / R + rho) * var)
    tcrit = stats.t.ppf(0.975, R - 1)
    if se == 0:
        p = 1.0 if mean == 0 else 0.0
    else:
        p = float(2 * stats.t.sf(abs(mean / se), R - 1))
    return float(mean), float(mean - tcrit * se), float(mean + tcrit * se), p


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- #
# One repeated split: fit every model once, then evaluate all decision rules
# --------------------------------------------------------------------------- #
def fit_split(j: int, X: pd.DataFrame, y: np.ndarray, seed0: int) -> dict:
    seed = seed0 + j
    idx = np.arange(y.size)
    i_rest, i_te = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    i_tr, i_va = train_test_split(i_rest, test_size=0.25, stratify=y[i_rest], random_state=seed)
    Xtr, Xva, Xte, ytr = X.iloc[i_tr], X.iloc[i_va], X.iloc[i_te], y[i_tr]
    preds, fit_s = {}, {}

    def fit(name, est, **kw):
        t0 = time.perf_counter()
        est.fit(Xtr, ytr, **kw)
        fit_s[name] = time.perf_counter() - t0
        preds[name] = (est.predict_proba(Xva)[:, 1], est.predict_proba(Xte)[:, 1])

    fit("LR", make_lr())
    fit("LR-cal", calibrated(make_lr()))
    fit("GBDT", make_gbdt(seed))
    fit("GBDT-cal", calibrated(make_gbdt(seed)))
    fit("GBDT-bal", make_gbdt(seed, "balanced"))
    fit("GBDT-bal-cal", calibrated(make_gbdt(seed, "balanced")))
    k = ytr.mean() / (1.0 - ytr.mean())  # w0/w1 under 'balanced' weights
    preds["GBDT-bal-pc"] = tuple(prior_correct(p, k) for p in preds["GBDT-bal"])
    for r in COST_RATIOS:  # training-time cost sensitivity needs one refit per ratio
        fit(f"LR-cw{r}", make_lr(class_weight={0: 1.0, 1: float(r)}))
        fit(f"GBDT-cw{r}", make_gbdt(seed), sample_weight=np.where(ytr == 1, float(r), 1.0))
    return {"split": j, "seed": seed, "yva": y[i_va], "yte": y[i_te], "preds": preds,
            "fit_s": fit_s, "n_train": i_tr.size, "n_test": i_te.size}


def analyse_split(s: dict) -> dict:
    j, yva, yte, P = s["split"], s["yva"], s["yte"], s["preds"]
    out = {"main": [], "quality": [], "ablation": [], "interval": [], "curves": {}}
    nv, nt = yva.size, yte.size

    for m in PRIMARY:
        fam = m.split("-")[0]
        pva, pte = P[m]
        fn_v, fp_v, n1v, _ = error_counts(pva, yva, T_GRID)
        tp_v = n1v - fn_v
        t_acc = T_GRID[np.argmax(nv - fn_v - fp_v)]
        t_f1 = T_GRID[np.argmax(2 * tp_v / np.maximum(2 * tp_v + fp_v + fn_v, 1))]
        fn_t, fp_t, n1t, n0t = error_counts(pte, yte, T_GRID)

        for r in COST_RATIOS:  # ---- Table 1
            t_emp = T_GRID[np.argmin(r * fn_v + fp_v)]
            rules = {"default_0.5": (pte, 0.5), "accuracy_opt": (pte, t_acc), "f1_opt": (pte, t_f1),
                     "cost_weighted_retrain": (P[f"{fam}-cw{r}"][1], 0.5),
                     "empirical_cost_opt": (pte, t_emp), "kb_threshold": (pte, float(kb_threshold(r)))}
            oracle = float(cost_per_1k(fn_t, fp_t, r, nt).min())
            trivial = 1000.0 * min(r * n1t, n0t) / nt  # best no-model policy: approve all or decline all
            for pol, (p, t) in rules.items():
                out["main"].append({"split": j, "model": m, "r": r, "policy": pol,
                                    "oracle_cost_per_1k": oracle, "trivial_cost_per_1k": trivial,
                                    **decision_metrics(p, yte, t, r)})

        # ---- assumed-vs-true cost ratio regret surface and reduction curves (figures)
        thr_hat = kb_threshold(R_FINE)
        fn_h, fp_h, _, _ = error_counts(pte, yte, thr_hat)
        c_hat = cost_per_1k(fn_h[None, :], fp_h[None, :], R_FINE[:, None], nt)    # true x assumed
        c_grid = cost_per_1k(fn_t[None, :], fp_t[None, :], R_FINE[:, None], nt)
        fn_a, fp_a, _, _ = error_counts(pte, yte, t_acc)
        c_acc = cost_per_1k(fn_a, fp_a, R_FINE, nt)
        oracle = np.minimum(np.minimum(c_grid.min(axis=1), c_hat.min(axis=1)), c_acc)
        out["curves"][m] = {
            "regret": 100 * (c_hat - oracle[:, None]) / oracle[:, None],
            "regret_acc": 100 * (c_acc - oracle) / oracle,
            "reduction": 100 * (c_acc - np.diag(c_hat)) / c_acc,
            "t_acc": t_acc,
        }
        if m == "GBDT-cal":  # normalised expected-cost curves (Figure 1a)
            out["curves"]["nec"] = {r: cost_per_1k(fn_t, fp_t, r, nt) / (1000 * min(r * n1t, n0t) / nt)
                                    for r in CURVE_RATIOS}

        # ---- knowledge given as an interval of r (Table 2B)
        for key, (lo, hi) in INTERVALS.items():
            rg = np.geomspace(lo, hi, 25)
            c_val = cost_per_1k(fn_v[:, None], fp_v[:, None], rg[None, :], nv)
            reg_val = (c_val - c_val.min(axis=0)) / c_val.min(axis=0)
            t_rob = T_GRID[np.argmin(reg_val.max(axis=1))]       # minimax regret (validation)
            r_mean = (hi - lo) / np.log(hi / lo)                  # E[r] under log-uniform belief
            cands = {"accuracy_opt": t_acc, "kb_r_lo": float(kb_threshold(lo)),
                     "kb_r_hi": float(kb_threshold(hi)), "kb_expected_r": float(kb_threshold(r_mean)),
                     "kb_minimax_regret": t_rob}
            thr = np.array(list(cands.values()))
            fn_c, fp_c, _, _ = error_counts(pte, yte, thr)
            c_c = cost_per_1k(fn_c[None, :], fp_c[None, :], rg[:, None], nt)
            orc = np.minimum(cost_per_1k(fn_t[None, :], fp_t[None, :], rg[:, None], nt).min(axis=1),
                             c_c.min(axis=1))
            reg = 100 * (c_c - orc[:, None]) / orc[:, None]
            for i, name in enumerate(cands):
                out["interval"].append({"split": j, "model": m, "interval": key, "r_lo": lo, "r_hi": hi,
                                        "policy": name, "threshold": thr[i],
                                        "worst_regret_pct": reg[:, i].max(),
                                        "mean_regret_pct": reg[:, i].mean()})

    # ---- model quality and calibration ablation (Table 2A)
    for v in ABLATION_VARIANTS:
        pte = P[v][1]
        out["quality"].append({"split": j, "variant": v, "auc": roc_auc_score(yte, pte),
                               "brier": brier_score_loss(yte, pte), "ece": ece(pte, yte)})
        fn_t, fp_t, _, _ = error_counts(pte, yte, T_GRID)
        for r in COST_RATIOS:
            t_star = float(kb_threshold(r))
            c_star = decision_metrics(pte, yte, t_star, r)["cost_per_1k"]
            c_grid = cost_per_1k(fn_t, fp_t, r, nt)
            orc = min(float(c_grid.min()), c_star)
            out["ablation"].append({"split": j, "variant": v, "r": r, "kb_threshold": t_star,
                                    "oracle_threshold": float(T_GRID[np.argmin(c_grid)]),
                                    "regret_pct": 100 * (c_star - orc) / orc})
    return out


# --------------------------------------------------------------------------- #
# Aggregation into the two paper tables
# --------------------------------------------------------------------------- #
def table1(main: pd.DataFrame, rho: float) -> pd.DataFrame:
    rows = []
    for m in PRIMARY:
        block = []
        for r in COST_RATIOS:
            g = main[(main.model == m) & (main.r == r)]
            w = g.pivot(index="split", columns="policy", values="cost_per_1k")
            acc = g.pivot(index="split", columns="policy", values="accuracy")
            dec = g.pivot(index="split", columns="policy", values="decline_rate")
            thr = g.pivot(index="split", columns="policy", values="threshold")
            red = 100 * (w["accuracy_opt"] - w["kb_threshold"]) / w["accuracy_opt"]
            red_m, red_lo, red_hi, _ = corrected_t(red, rho)
            _, _, _, p_acc = corrected_t(w["accuracy_opt"] - w["kb_threshold"], rho)
            _, _, _, p_emp = corrected_t(w["empirical_cost_opt"] - w["kb_threshold"], rho)
            _, _, _, p_cw = corrected_t(w["cost_weighted_retrain"] - w["kb_threshold"], rho)
            row = {"model": m, "r": r, "kb_threshold": float(kb_threshold(r)),
                   "accuracy_opt_threshold": thr["accuracy_opt"].mean(),
                   "empirical_threshold": thr["empirical_cost_opt"].mean(),
                   **{f"cost_{pol}": w[pol].mean() for pol in POLICIES},
                   "cost_oracle": g.groupby("split").oracle_cost_per_1k.first().mean(),
                   "cost_best_trivial": g.groupby("split").trivial_cost_per_1k.first().mean(),
                   "nec_kb": (w["kb_threshold"] / g.groupby("split").trivial_cost_per_1k.first()).mean(),
                   "nec_accuracy_opt": (w["accuracy_opt"] /
                                        g.groupby("split").trivial_cost_per_1k.first()).mean(),
                   "reduction_vs_accuracy_pct": red_m, "reduction_ci_lo": red_lo, "reduction_ci_hi": red_hi,
                   "p_vs_accuracy": p_acc, "p_vs_empirical": p_emp, "p_vs_retrain": p_cw,
                   "accuracy_at_accuracy_opt": acc["accuracy_opt"].mean(),
                   "accuracy_at_kb": acc["kb_threshold"].mean(),
                   "decline_rate_accuracy_opt": dec["accuracy_opt"].mean(),
                   "decline_rate_kb": dec["kb_threshold"].mean()}
            block.append(row)
        adj = holm([b["p_vs_accuracy"] for b in block])
        for b, a in zip(block, adj):
            b["p_vs_accuracy_holm"] = a
        rows += block
    return pd.DataFrame(rows)


def table2(quality: pd.DataFrame, ablation: pd.DataFrame, interval: pd.DataFrame, rho: float):
    q = quality.groupby("variant")[["auc", "brier", "ece"]].mean()
    a = ablation[ablation.r.isin(ABLATION_RATIOS)].pivot_table(
        index="variant", columns="r", values="regret_pct", aggfunc="mean")
    a.columns = [f"kb_regret_pct_r{c}" for c in a.columns]
    a["kb_regret_pct_mean_all_r"] = ablation.groupby("variant").regret_pct.mean()
    panel_a = q.join(a).reindex(list(ABLATION_VARIANTS)).reset_index()
    panel_a.insert(0, "panel", "A_calibration")

    rows = []
    for (m, key, pol), g in interval.groupby(["model", "interval", "policy"], sort=False):
        mean, lo, hi, _ = corrected_t(g.worst_regret_pct.to_numpy(), rho)
        rows.append({"panel": "B_interval_knowledge", "model": m, "interval": key,
                     "r_lo": g.r_lo.iloc[0], "r_hi": g.r_hi.iloc[0], "policy": pol,
                     "threshold_mean": g.threshold.mean(), "worst_regret_pct": mean,
                     "worst_regret_ci_lo": lo, "worst_regret_ci_hi": hi,
                     "mean_regret_pct": g.mean_regret_pct.mean()})
    return panel_a, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def panel_label(ax, text):
    ax.text(-0.13, 1.04, text, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, va="bottom", ha="left")


def log_ratio_axis(ax, which="x"):
    ticks = [1, 2, 3, 5, 10, 20]
    if which == "x":
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_yscale("log"); ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def figure1(curves: dict, t1: pd.DataFrame, rho: float, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 3.05), constrained_layout=True)
    nec = {r: np.mean([c["nec"][r] for c in curves], axis=0) for r in CURVE_RATIOS}
    t_acc = float(np.mean([c["GBDT-cal"]["t_acc"] for c in curves]))
    ax1.axvline(t_acc, color=MUTED, lw=0.9, zorder=1)
    ax1.text(t_acc - 0.008, 0.37, "accuracy-optimal\ncutoff", color=INK2, fontsize=7, va="bottom",
             ha="right")
    for r, col in zip(CURVE_RATIOS, RAMP5):
        ax1.plot(T_GRID, nec[r], color=col, lw=1.6, label=f"r = {r}", zorder=2)
        ts = float(kb_threshold(r))
        ax1.plot(ts, np.interp(ts, T_GRID, nec[r]), "o", ms=5, color=col, mec="white", mew=1.0, zorder=3)
    ax1.plot([], [], "o", ms=5, color=INK2, mec="white", label="t*(r)")
    ax1.set_xlim(0, 0.62); ax1.set_ylim(0.35, 1.3)
    ax1.set_xlabel("Decision threshold t (decline if p ≥ t)")
    ax1.set_ylabel("Normalized expected cost (best no-model policy = 1)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=6, handlelength=1.2,
               columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
    panel_label(ax1, "(a)")

    for m, col, lab in (("GBDT-cal", SLOT[0], "GBDT"), ("LR-cal", SLOT[1], "LR")):
        red = np.array([c[m]["reduction"] for c in curves])
        stats_ = np.array([corrected_t(red[:, i], rho) for i in range(R_FINE.size)])
        ax2.fill_between(R_FINE, stats_[:, 1], stats_[:, 2], color=col, alpha=0.14, lw=0)
        ax2.plot(R_FINE, stats_[:, 0], color=col, lw=1.8, label=f"{lab}: knowledge-based t*(r)")
        g = t1[t1.model == m]
        cw = 100 * (g.cost_accuracy_opt - g.cost_cost_weighted_retrain) / g.cost_accuracy_opt
        ax2.plot(g.r, cw, "s", ms=4.5, mfc="white", mec=col, mew=1.2,
                 label=f"{lab}: cost-weighted retraining")
    ax2.axhline(0, color=AXIS, lw=0.8)
    log_ratio_axis(ax2, "x")
    ax2.set_xlim(1, 20)
    ax2.set_xlabel("Business loss ratio r = C$_{FN}$/C$_{FP}$ (log scale)")
    ax2.set_ylabel("Cost reduction vs. accuracy-optimal cutoff (%)")
    ax2.legend(loc="upper left", handlelength=1.8)
    panel_label(ax2, "(b)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Figure_1.{ext}", bbox_inches="tight")
    plt.close(fig)


def figure2(pool: dict, quality: pd.DataFrame, curves: dict, interval: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 3.2), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.12]})
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=0.9, zorder=1)
    ece_m = quality.groupby("variant").ece.mean()
    series = (("GBDT", "GBDT, log-loss", "o"),
              ("GBDT-bal", "GBDT, class-balanced", "s"),
              ("GBDT-bal-pc", "  + prior correction", "^"),
              ("GBDT-bal-cal", "  + isotonic recalibration", "D"))
    for (v, lab, mk), col in zip(series, SLOT):
        x, yb = reliability(pool[v], pool["y"])
        ax1.plot(x, yb, "-", color=col, lw=1.4, zorder=2)
        ax1.plot(x, yb, mk, color=col, ms=3.8, mec="white", mew=0.7, zorder=3)
        ax1.plot([], [], "-" + mk, color=col, ms=3.8, mec="white", mew=0.7, lw=1.4,
                 label=f"{lab} (ECE {ece_m[v]:.3f})")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("Mean predicted default probability")
    ax1.set_ylabel("Observed default rate")
    leg = ax1.legend(loc="upper left", handlelength=1.8, fontsize=7, borderaxespad=0.2, frameon=True,
                     facecolor="white", edgecolor="none", framealpha=0.92)
    leg.set_zorder(5)
    ax1.text(0.97, 0.03, "pooled test predictions,\n20 equal-mass bins", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=7, color=INK2)
    panel_label(ax1, "(a)")

    reg = np.mean([c["GBDT-cal"]["regret"] for c in curves], axis=0)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    edges = np.geomspace(R_FINE[0] / np.sqrt(R_FINE[1] / R_FINE[0]),
                         R_FINE[-1] * np.sqrt(R_FINE[1] / R_FINE[0]), R_FINE.size + 1)
    mesh = ax2.pcolormesh(edges, edges, np.clip(reg, 0, 60), cmap=cmap, norm=Normalize(0, 60),
                          shading="flat", rasterized=True)
    cs = ax2.contour(R_FINE, R_FINE, reg, levels=[5, 10, 25], colors=[INK], linewidths=0.6)
    ax2.clabel(cs, fmt=lambda v: f"{v:.0f}%", fontsize=7, inline_spacing=2)
    log_ratio_axis(ax2, "x"); log_ratio_axis(ax2, "y")
    ax2.set_xlim(1, 20); ax2.set_ylim(1, 20); ax2.grid(False)
    lo, hi = INTERVALS["A"]
    rob = interval[(interval.model == "GBDT-cal") & (interval.interval == "A") &
                   (interval.policy == "kb_minimax_regret")].threshold.mean()
    r_rob = (1 - rob) / rob
    ax2.plot([r_rob, r_rob], [lo, hi], color=SLOT[1], lw=2.4, solid_capstyle="butt", zorder=4)
    for yy in (lo, hi):
        ax2.plot([1, 20], [yy, yy], color=INK, lw=0.6, zorder=3)
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    ax2.text(1.07, np.sqrt(lo * hi), f"expert interval\n$r \\in [{lo:.0f}, {hi:.0f}]$", fontsize=7,
             color=INK, va="center", bbox=box, zorder=5)
    ax2.annotate(f"minimax-regret cutoff t = {rob:.3f}\n(equivalent to $\\hat{{r}}$ = {r_rob:.1f})",
                 xy=(r_rob, hi), xytext=(1.07, 15.0), fontsize=7, color=INK, va="center",
                 bbox=box, zorder=5, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    ax2.set_xlabel("Assumed cost ratio $\\hat{r}$ used to set $t^*(\\hat{r})$")
    ax2.set_ylabel("True cost ratio r")
    cb = fig.colorbar(mesh, ax=ax2, fraction=0.05, pad=0.02, extend="max")
    cb.set_label("Regret vs. ex-post optimal cutoff (%)", fontsize=7)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=7)
    panel_label(ax2, "(b)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Figure_2.{ext}", bbox_inches="tight")
    plt.close(fig)


def graphical_abstract(t1: pd.DataFrame, out: Path) -> None:
    """Elsevier graphical abstract: readable at 13 x 5 cm, generated from the results."""
    fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 50); ax.axis("off")
    ax.text(1.5, 45, "Separate the AI prediction from the business decision", fontsize=8,
            fontweight="bold", color=INK, va="center")
    boxes = [(1.5, "Data", "30,000 credit\nclients"),
             (21.5, "AI model", "calibrated\np(default)"),
             (41.5, "Knowledge", "loss ratio r\nt* = 1/(1+r)"),
             (61.5, "Decision", "decline if\np ≥ t*")]
    for x0, head, body in boxes:
        ax.add_patch(FancyBboxPatch((x0, 12), 17.5, 24, boxstyle="round,pad=0.3,rounding_size=1.8",
                                    fc="#f4f8fe", ec=SLOT[0], lw=0.9))
        ax.text(x0 + 8.75, 30.5, head, ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
        ax.text(x0 + 8.75, 20.5, body, ha="center", va="center", fontsize=6.4, color=INK2, linespacing=1.25)
    for x0 in (19.3, 39.3, 59.3):
        ax.add_patch(FancyArrowPatch((x0, 24), (x0 + 2.4, 24), arrowstyle="-|>", mutation_scale=7,
                                     color=INK2, lw=0.9))
    ax.text(1.5, 5, "No retraining when business costs change", fontsize=6.6, color=INK, va="center")
    g = t1[t1.model == "GBDT-cal"].set_index("r")
    sub = fig.add_axes([0.655, 0.2, 0.33, 0.56])
    rs = [3, 5, 10]
    vals = [g.loc[r, "reduction_vs_accuracy_pct"] for r in rs]
    sub.bar(range(len(rs)), vals, width=0.55, color=SLOT[0])
    for i, v in enumerate(vals):
        sub.text(i, v + max(vals) * 0.05, f"−{v:.0f}%", ha="center", fontsize=7, color=INK)
    sub.set_xticks(range(len(rs))); sub.set_xticklabels([f"{r}:1" for r in rs], fontsize=6.6)
    sub.set_ylim(0, max(vals) * 1.25); sub.set_yticks([])
    sub.spines["left"].set_visible(False); sub.grid(False)
    sub.tick_params(axis="x", length=0, pad=1.5)
    sub.set_title("Cost vs. accuracy cutoff", fontsize=7, color=INK, pad=3)
    sub.set_xlabel("loss ratio r", fontsize=6.6, labelpad=1.5, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Graphical_Abstract.{ext}", dpi=600)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=20, help="repeated stratified 60/20/20 splits")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--cache", type=Path, default=None,
                    help="optional pickle of fitted-split analyses, reused to re-render outputs quickly")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_style()
    t_start = time.perf_counter()

    df, digest = load_data()
    X, y = build_features(df)
    print(f"[data] {len(y):,} clients, default rate {y.mean():.4f}, {X.shape[1]} features, sha256 {digest[:12]}")

    if args.cache is not None and args.cache.exists():
        import pickle
        with open(args.cache, "rb") as fh:
            results, pool, fit_times, rho = pickle.load(fh)
        print(f"[cache] reusing {len(results)} analysed splits from {args.cache}")
    else:
        results, pool, fit_times = [], {v: [] for v in ABLATION_VARIANTS}, []
        pool["y"] = []
        rho = None
        for j in range(args.repeats):
            s = fit_split(j, X, y, args.seed)
            rho = s["n_test"] / s["n_train"]
            results.append(analyse_split(s))
            for v in ABLATION_VARIANTS:
                pool[v].append(s["preds"][v][1])
            pool["y"].append(s["yte"])
            t0 = time.perf_counter()
            _ = [s["preds"]["GBDT-cal"][1] >= kb_threshold(r) for r in COST_RATIOS]
            fit_times.append({**s["fit_s"], "kb_decisions_all_ratios": time.perf_counter() - t0})
            print(f"[split {j + 1:>2}/{args.repeats}] done ({time.perf_counter() - t_start:,.0f}s elapsed)")
        pool = {k: np.concatenate(v) for k, v in pool.items()}
        if args.cache is not None:
            import pickle
            with open(args.cache, "wb") as fh:
                pickle.dump((results, pool, fit_times, rho), fh)
    args.repeats = len(results)

    main_df = pd.DataFrame([r for res in results for r in res["main"]])
    quality = pd.DataFrame([r for res in results for r in res["quality"]])
    ablation = pd.DataFrame([r for res in results for r in res["ablation"]])
    interval = pd.DataFrame([r for res in results for r in res["interval"]])
    curves = [res["curves"] for res in results]

    t1 = table1(main_df, rho)
    t2a, t2b = table2(quality, ablation, interval, rho)
    regret_surface = pd.DataFrame(np.mean([c["GBDT-cal"]["regret"] for c in curves], axis=0),
                                  index=np.round(R_FINE, 4), columns=np.round(R_FINE, 4))
    regret_surface.index.name = "true_r \\ assumed_r"
    regret_surface["accuracy_opt_cutoff"] = np.mean([c["GBDT-cal"]["regret_acc"] for c in curves], axis=0)

    t1.to_csv(args.out / "table1_cost_results.csv", index=False, float_format="%.6g")
    pd.concat([t2a, t2b], ignore_index=True).to_csv(args.out / "table2_ablation_robustness.csv",
                                                    index=False, float_format="%.6g")
    main_df.to_csv(args.out / "per_split_decisions.csv", index=False, float_format="%.6g")
    regret_surface.to_csv(args.out / "regret_surface_gbdt.csv", float_format="%.4f")

    figure1(curves, t1, rho, args.out)
    figure2(pool, quality, curves, interval, args.out)
    graphical_abstract(t1, args.out)

    ft = pd.DataFrame(fit_times)
    meta = {
        "paper": "Cost-Sensitive AI Thresholds for Decision Intelligence: Optimizing Data-Driven "
                 "Cutoffs Under Asymmetric Business Loss",
        "data": {"source": "UCI ML Repository, Default of Credit Card Clients (Yeh, 2009), "
                           "doi:10.24432/C55S3H, CC BY 4.0", "rows": int(len(y)),
                 "defaults": int(y.sum()), "content_sha256": digest,
                 "features_used": list(X.columns), "excluded_protected": list(PROTECTED)},
        "protocol": {"repeats": args.repeats, "seed": args.seed, "split": "stratified 60/20/20",
                     "calibration": "isotonic, 5-fold CalibratedClassifierCV on the training split",
                     "inference": "Nadeau-Bengio corrected resampled t (rho = n_test/n_train), Holm",
                     "cost_ratios": list(COST_RATIOS), "intervals": INTERVALS},
        "runtime_seconds": {"total": round(time.perf_counter() - t_start, 1),
                            "fit_gbdt_calibrated_mean": round(ft["GBDT-cal"].mean(), 3),
                            "fit_six_cost_weighted_gbdt_mean": round(
                                ft[[f"GBDT-cw{r}" for r in COST_RATIOS]].sum(axis=1).mean(), 3),
                            "kb_decisions_six_ratios_mean": float(ft["kb_decisions_all_ratios"].mean())},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
                        "cpu_count": os.cpu_count()},
    }
    (args.out / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    q = quality.groupby("variant")[["auc", "brier", "ece"]].mean().round(4)
    print("\nModel quality (test, mean over splits)\n", q.to_string())
    show = t1[["model", "r", "kb_threshold", "cost_accuracy_opt", "cost_kb_threshold",
               "reduction_vs_accuracy_pct", "reduction_ci_lo", "reduction_ci_hi", "p_vs_accuracy_holm"]]
    print("\nTable 1 (excerpt)\n", show.round(4).to_string(index=False))
    print(f"\nOutputs written to {args.out} in {meta['runtime_seconds']['total']} s")


if __name__ == "__main__":
    main()
