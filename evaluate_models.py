"""November 19 evaluation: calibration, horizon-eligible metrics, top-N outreach targeting, and a recalibration study.

Usage: python build_snapshots.py && python evaluate_models.py   (pandas, numpy, matplotlib; offline only)

Outputs:
  processed/model_evaluation.md         report; do not edit by hand
  processed/reliability_eligible.png    reliability plot for horizon-eligible customers
  data/models/evaluation.json           eligible-fit coefficients and Platt calibrators (gitignored)
  data/eval_predictions.csv.gz          per-row validation and test predictions (gitignored)

Chronological roles, per horizon h (periods from build_snapshots.split; labels resolved before the next role starts):
  fit        train        s + h <= 2010-12-31                    coefficients and feature scaling
  calibrate  validation   s > 2010-12-31, s + h <= 2011-06-30    Platt calibrators (eligible rows only)
  compare    test         s > 2011-06-30, s + h <= 2011-12-08    candidate comparison. EXPLORATORY: test was already
                                                                 inspected on Nov 12, and no later data exist.

Candidates and decision rule, fixed before any candidate was scored on test:
  recency, rfm             the Nov 12 logistic models, fitted on all training rows (recency was selected)
  recency+platt, rfm+platt the same models, recalibrated on validation eligible rows: p' = sigmoid(a + b*logit(p));
                           probability 0 outside eligibility (those rows cannot have an event)
  recency-elig, rfm-elig   the same specifications refitted on eligible training rows only; 0 outside eligibility
  A candidate counts as an improvement over `recency` only if its eligible-group test Brier score is lower and the
  customer-bootstrap 95% interval of the difference excludes zero. Because test is not untouched, any such result is
  provisional and needs a later holdout.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from build_snapshots import D, HORIZONS, LAST_COMPLETE, OUT as SNAPSHOTS, TRAIN_END, VAL_END, build, load
from train_baselines import EXPLORATORY, MODELS as BASELINES, SPECS, auc, brier, design, eligible, fit_logit, periods, predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAPACITIES = (50, 100, 200)       # contacts per weekly scoring date
BOOT, SEED = 1000, 0              # customer-level bootstrap
CANDIDATES = ["recency", "rfm", "recency+platt", "rfm+platt", "recency-elig", "rfm-elig"]
CALIBRATED = {"recency+platt", "rfm+platt"}
COLORS = dict(zip(CANDIDATES, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]))
REPORT, FIGURE = Path("processed/model_evaluation.md"), Path("processed/reliability_eligible.png")
PARAMS, PREDICTIONS = Path("data/models/evaluation.json"), Path("data/eval_predictions.csv.gz")


def is_eligible(d, h):
    return (d["days_since_last"] >= D - h).to_numpy()


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_spec(rows, cols, y):
    x, scale = design(rows, cols)
    return fit_logit(x, rows[y].to_numpy(float)), scale


def fit_candidates(df, h):
    """All predictors for horizon h. Coefficients come from train rows, calibrators from validation rows; test rows are
    never read here (test_evaluation.py checks this by changing test labels). Returns {name: f(rows) -> p}, params."""
    y, P = f"y_{h}", periods(df, h)
    tr, va = P["train"], eligible(P["validation"], h)
    etr = eligible(tr, h)
    rate, erate = tr[y].mean(), etr[y].mean()
    preds = {"constant": lambda d: np.full(len(d), rate),
             "elig-constant": lambda d: np.where(is_eligible(d, h), erate, 0.0)}
    params = {"train_rate": rate, "train_eligible_rate": erate}
    for name, cols in SPECS.items():
        b, sc = fit_spec(tr, cols, y)
        base = lambda d, b=b, sc=sc, cols=cols: predict(design(d, cols, sc)[0], b)
        ab = fit_logit(np.column_stack([np.ones(len(va)), logit(base(va))]), va[y].to_numpy(float))
        be, sce = fit_spec(etr, cols, y)
        preds[name] = base
        preds[f"{name}+platt"] = lambda d, base=base, ab=ab: np.where(
            is_eligible(d, h), 1 / (1 + np.exp(-(ab[0] + ab[1] * logit(base(d))))), 0.0)
        preds[f"{name}-elig"] = lambda d, be=be, sce=sce, cols=cols: np.where(
            is_eligible(d, h), predict(design(d, cols, sce)[0], be), 0.0)
        params[name] = {"coef": b.tolist(), "platt": ab.tolist(),
                        "elig_coef": be.tolist(), "elig_scale": {k: list(v) for k, v in sce.items()}}
    return preds, params


def calibration(y, p):
    """Calibration-in-the-large (mean predicted vs observed) and the calibration intercept/slope from y ~ logit(p)."""
    a, b = fit_logit(np.column_stack([np.ones(len(p)), logit(p)]), y) if np.ptp(p) > 0 else (np.nan, np.nan)
    return p.mean(), y.mean(), a, b


def reliability(y, p, k):
    """k equal-count groups of predicted probability (ties split by row order), with mean prediction and event rate."""
    b = pd.qcut(pd.Series(p).rank(method="first"), k, labels=False).to_numpy()
    return (pd.DataFrame({"y": y, "p": p, "bin": b + 1}).groupby("bin")
            .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean")))


def captured(week, key, n_contacts, y):
    """Expected events among the top `n_contacts` of one week, ranked by the `key` columns (descending). Rows tied on
    every key at the cut-off share the remaining slots pro rata, i.e. the expected value under random tie-breaking."""
    g = week.groupby(key)[y].agg(["size", "sum"]).sort_index(ascending=False)
    before = g["size"].cumsum() - g["size"]
    take = np.clip(n_contacts - before, 0, g["size"])
    return float((take / g["size"] * g["sum"]).sum())


def top_n(rows, h, rankers, capacities=CAPACITIES):
    """Weekly top-N among eligible rows, pooled over weeks. Precision = captured / contacted;
    capture = captured / all events that week (every event is in the eligible group)."""
    y, out = f"y_{h}", []
    for s, w in rows.groupby("scoring_date"):
        n, ev = len(w), float(w[y].sum())
        for N in capacities:
            k = min(N, n)
            out.append(("random", N, s, k, k / n * ev, ev))
            out += [(name, N, s, k, captured(w, key, N, y), ev) for name, key in rankers.items()]
    t = pd.DataFrame(out, columns=["ranker", "N", "scoring_date", "contacted", "captured", "events"])
    t["weekly_capture"] = t["captured"] / t["events"]
    g = t.groupby(["ranker", "N"], sort=False)
    res = g[["contacted", "captured", "events"]].sum()
    res["precision"], res["capture"] = res["captured"] / res["contacted"], res["captured"] / res["events"]
    res["weekly_min"], res["weekly_max"] = g["weekly_capture"].min(), g["weekly_capture"].max()
    return res


def boot_diff(cust, y, pa, pb, boot=BOOT, seed=SEED):
    """Brier(pa) - Brier(pb) with a 95% percentile interval, resampling customers (all their weekly rows together)."""
    codes, uniq = pd.factorize(cust)
    n = np.bincount(codes, minlength=len(uniq))
    d = np.bincount(codes, (pa - y) ** 2 - (pb - y) ** 2, minlength=len(uniq))
    w = np.random.default_rng(seed).multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=boot)
    lo, hi = np.percentile((w @ d) / (w @ n), [2.5, 97.5])
    return d.sum() / n.sum(), lo, hi


def audit(df, inv, n=1000, seed=1):
    """Independent day-by-day recomputation of features and labels for a random sample of rows, plus a rebuild from
    invoices truncated at TRAIN_END (features must not change). Returns (sample mismatches, truncation differences)."""
    by = dict(tuple(inv.groupby("customer")))
    bad = 0
    for r in df.sample(n, random_state=seed).itertuples():
        g, s = by[r.customer], r.scoring_date
        past = g[g["date"] < s + pd.Timedelta(days=1)]
        days, last = set(g["date"].dt.normalize()), past["date"].max().normalize()
        ok = ((s - last).days, (s - past["date"].min().normalize()).days, len(past), round(past["amount"].sum(), 2)) == \
             (r.days_since_last, r.tenure_days, r.frequency, round(r.monetary, 2))
        for h in HORIZONS:
            event = None
            for k in range(1, h + 1):
                t = s + pd.Timedelta(days=k)
                last = t if t in days else last
                if (t - last).days >= D:
                    event = t
                    break
            last = past["date"].max().normalize()
            want = 1 if event is not None and event <= LAST_COMPLETE else \
                0 if event is None and s + pd.Timedelta(days=h) <= LAST_COMPLETE else None
            got = getattr(r, f"y_{h}")
            ok &= want == (None if pd.isna(got) else int(got))
        bad += not ok
    dates = pd.DatetimeIndex(sorted(df.loc[df["scoring_date"] <= TRAIN_END, "scoring_date"].unique()))
    cut = build(inv[inv["date"] < TRAIN_END + pd.Timedelta(days=1)], dates)
    cols = ["customer", "scoring_date", "days_since_last", "tenure_days", "frequency", "monetary"]
    m = df.loc[df["scoring_date"] <= TRAIN_END, cols].merge(cut[cols], on=cols[:2], how="outer", suffixes=("", "_cut"))
    diffs = int(m.isna().any(axis=1).sum() + sum((m[c] != m[f"{c}_cut"]).sum() for c in cols[2:]))
    return bad, diffs


def run(df):
    out = {}
    for h in HORIZONS:
        y = f"y_{h}"
        preds, params = fit_candidates(df, h)
        P = periods(df, h)
        # Role boundaries: each role's labels are resolved before the next role's first scoring date.
        assert (P["train"]["scoring_date"] + pd.Timedelta(days=h)).max() < P["validation"]["scoring_date"].min()
        assert (P["validation"]["scoring_date"] + pd.Timedelta(days=h)).max() < P["test"]["scoring_date"].min()
        assert all(params[n]["platt"][1] > 0 and params[n]["coef"][1] > 0 for n in SPECS)  # rank-preserving
        scored = {}
        for period in ("validation", "test"):
            g = P[period].copy()
            for name, f in preds.items():
                if not (period == "validation" and name in CALIBRATED):   # never score calibrators in-sample
                    g[name] = f(g)
            scored[period] = g
        out[h] = (P, preds, params, scored)
    return out


def metrics_rows(h, scored):
    y, rows = f"y_{h}", []
    for period, g in scored.items():
        m = is_eligible(g, h)
        for name in ["constant", "elig-constant", *CANDIDATES]:
            if name in g:
                yy, p = g[y].to_numpy(float), g[name].to_numpy()
                rows.append((period, name, auc(yy, p), brier(yy, p), auc(yy[m], p[m]), brier(yy[m], p[m]),
                             *calibration(yy[m], p[m])))
    return pd.DataFrame(rows, columns=["period", "model", "auc_all", "brier_all", "auc_elig", "brier_elig",
                                       "mean_p", "observed", "cal_intercept", "cal_slope"])


def plot(results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    for j, (h, (_, _, _, scored)) in enumerate(results.items()):
        for i, period in enumerate(("validation", "test")):
            ax, g = axes[i, j], scored[period]
            e = g[is_eligible(g, h)]
            yy = e[f"y_{h}"].to_numpy(float)
            ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
            for name in CANDIDATES:
                if name in e and not (h == D and name.endswith("-elig")):   # identical to the originals at h = 90
                    r = reliability(yy, e[name].to_numpy(), 10)
                    ax.plot(r["mean_predicted"], r["observed"], marker="o", ms=3, lw=1.2, color=COLORS[name], label=name)
            ax.plot(e["elig-constant"].iloc[0], yy.mean(), marker="X", ms=10, color="k", ls="", label="elig-constant")
            tag = " (exploratory)" if h in EXPLORATORY or period == "test" else ""
            ax.set_title(f"h = {h}, {period}{tag}\n{len(e):,} eligible rows, event rate {yy.mean():.1%}", fontsize=9)
            ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("mean predicted probability (decile groups)")
            if j == 0:
                ax.set_ylabel("observed event rate")
    axes[1, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Reliability for horizon-eligible customers (days_since_last ≥ 90 − h).\nCalibrated candidates are not shown "
                 "on validation, where they were fitted; at h = 90 the -elig models equal the originals.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=110)
    plt.close(fig)


def drift(df, h=60):
    """Cumulative features by period for eligible rows, and the event rate by frequency band (shows non-stationarity)."""
    rows = []
    for period, g in periods(df, h).items():
        e = eligible(g, h)
        band = pd.cut(e["frequency"], [0, 1, 3, 10, np.inf], labels=["1", "2–3", "4–10", ">10"])
        r = e.groupby(band, observed=False)[f"y_{h}"].mean()
        rows.append((period, e["tenure_days"].median(), e["frequency"].median(), e["monetary"].median(),
                     e[f"y_{h}"].mean(), *r.tolist(), *(band.value_counts(normalize=True).reindex(r.index).tolist())))
    return rows


def report(df, results, audit_result):
    f = lambda v, k=3: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{k}f}"
    pct = lambda v: f"{v:.1%}"
    tag = lambda h: f"{h} (exploratory)" if h in EXPLORATORY else f"{h}"
    bad, diffs = audit_result
    L = ["# Model evaluation: calibration, eligible customers and top-N targeting (Nov 19)", "",
         "Generated by `evaluate_models.py` from `data/weekly_snapshots.csv.gz`. Do not edit by hand. Offline only: "
         "the website still shows the rule-based watchlist.", "",
         "> **The test period is not an untouched holdout.** Its results for the selected recency model were inspected on "
         "Nov 12, and descriptive test event rates by recency band were viewed during this review. Every test number below "
         "is **exploratory**. The data end on 9 Dec 2011, and test already runs to the last complete day, so no later "
         "holdout exists in this dataset (Section 7).", "",
         "## 1. Review of the current implementation", "",
         "Confirmed, with checks run by this script on the real data:", "",
         f"- **Outcome, censoring and features.** An independent day-by-day recomputation of the features and all three "
         f"labels for 1,000 randomly sampled snapshot rows found **{bad} mismatches**. Rebuilding the snapshots from invoices "
         f"truncated at {TRAIN_END:%Y-%m-%d} left every feature on every scoring date up to that day unchanged "
         f"(**{diffs} differences**), so features use no later purchases.",
         f"- **Eligibility.** At horizon h only `days_since_last` ≥ {D} − h can have an event; `eligible()` asserts this on "
         "every period.",
         "- **Split.** Each role's labels are resolved before the next role's first scoring date (asserted for every horizon). "
         "Predictors come from an explicit allowlist; the scaling is fitted on training rows.",
         "- **No leakage found.** The only future-derived columns (`crossing_date`, `y_*`) never reach `design()`.", "",
         "Issues found and handled:", "",
         "- *Wording error, fixed:* `baseline_results.md` called 2010-12-31 the \"earliest\" training cutoff that keeps 13 "
         "validation and 10 test scoring dates at h = 90. It is the **latest** such cutoff (2010-11-30 also qualifies). "
         "The split and every number are unchanged.",
         "- *Feature drift, not leakage:* `tenure_days`, `frequency` and `monetary` are cumulative since 1 Dec 2009, so they "
         "grow with calendar time and a given value means something different in each period (Section 6, drift table). "
         "This makes `rfm` under-predict in later periods. Left unchanged here; the fix is proposed in Section 7.",
         "- *Model misspecification, not a bug:* the Nov 12 models are fitted on all active rows. At h = 30 a single logistic "
         "curve in recency must be near 0 below 60 days and high above it, so it becomes very steep and is badly calibrated "
         "inside the eligible group. The Nov 12 selection used overall validation Brier, which mostly rewards that "
         "eligibility separation. Both points motivate Sections 3 and 6.", "",
         "## 2. Evaluation setup", "",
         "| Role | Period (scoring date s) | Used for |", "|---|---|---|",
         f"| fit | train: s + h ≤ {TRAIN_END:%Y-%m-%d} | coefficients and scaling |",
         f"| calibrate | validation: s > {TRAIN_END:%Y-%m-%d}, s + h ≤ {VAL_END:%Y-%m-%d} | Platt calibrators (eligible rows) |",
         f"| compare | test: s > {VAL_END:%Y-%m-%d}, s + h ≤ {LAST_COMPLETE:%Y-%m-%d} | candidate comparison (exploratory) |",
         "| final holdout | none available | – |", "",
         "- **Populations.** *All active*: every snapshot row in the period. *Eligible*: rows with `days_since_last` ≥ 90 − h "
         "(at h = 90 this is everyone, so the two coincide).",
         "- **Constant baselines** (fitted on train): `constant` = training event rate for every row; `elig-constant` = 0 for "
         "ineligible rows and the training **eligible-group** event rate for eligible rows. Within the eligible group, "
         "`elig-constant` is the reference constant.",
         "- **Candidates** (fixed before scoring, see the script docstring): `recency` and `rfm` (Nov 12, fitted on all "
         "training rows; `recency` was selected); `+platt` = recalibrated on validation eligible rows, "
         "p′ = sigmoid(a + b·logit p), and 0 outside eligibility; `-elig` = the same specification refitted on eligible "
         "training rows only, and 0 outside eligibility. At h = 90 the `-elig` models equal the originals.",
         "- **Calibration summary** (eligible rows): mean predicted vs observed rate, and the intercept and slope from a "
         "logistic regression of the outcome on logit(p). A slope of 1 and matching means indicate good calibration; "
         "slope < 1 means predictions are too extreme.",
         f"- **Uncertainty.** Brier differences have 95% intervals from a customer-level bootstrap ({BOOT} resamples, all of a "
         "customer's weekly rows drawn together), because the same customer appears in many overlapping weekly snapshots.", ""]

    # 3. Existing models
    L += ["## 3. Existing models: discrimination and Brier score", "",
          "Nov 12 models as fitted then (coefficients reproduced exactly). `rfm` is scored on test here for the first time; "
          "this is exploratory and is not used to re-select the Nov 12 model.", "",
          "| h | Period | Model | AUC all | Brier all | AUC eligible | Brier eligible | Mean p (elig) | Observed (elig) | Cal. slope (elig) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    tables = {}
    for h, (P, preds, params, scored) in results.items():
        t = metrics_rows(h, scored)
        tables[h] = t
        for r in t[t["model"].isin(["constant", "elig-constant", "recency", "rfm"])].itertuples():
            L.append(f"| {tag(h)} | {r.period} | {r.model} | {f(r.auc_all)} | {f(r.brier_all, 4)} | {f(r.auc_elig)} | "
                     f"{f(r.brier_elig, 4)} | {f(r.mean_p)} | {f(r.observed)} | {f(r.cal_slope, 2)} |")
    L += ["", "`elig-constant` has no AUC within the eligible group (one value). Its *overall* AUC shows how much ranking "
          "eligibility alone buys.", ""]

    # 4. Reliability
    L += ["## 4. Calibration (reliability)", "", f"![Reliability plot]({FIGURE.name})", "",
          "Decile groups of predicted probability among eligible customers. Points on the dashed diagonal are well calibrated; "
          "the black cross is the eligible-group constant. Quintile tables with counts follow.", ""]
    for h, (P, preds, params, scored) in results.items():
        L += [f"<details><summary>h = {tag(h)}: quintile tables (eligible rows)</summary>", ""]
        for period, g in scored.items():
            e = g[is_eligible(g, h)]
            for name in [n for n in CANDIDATES if n in e]:
                r = reliability(e[f"y_{h}"].to_numpy(float), e[name].to_numpy(), 5)
                L += [f"**{period}, {name}**", "", "| Group | n | Mean predicted | Observed |", "|---:|---:|---:|---:|",
                      *[f"| {i} | {x.n:,} | {x.mean_predicted:.3f} | {x.observed:.3f} |" for i, x in r.iterrows()], ""]
        L += ["</details>", ""]

    # 5. Top-N
    L += ["## 5. Outreach targeting: top N per week", "",
          f"At every weekly scoring date, eligible customers are ranked and the top N are \"contacted\", for N = "
          f"{', '.join(map(str, CAPACITIES))} (median eligible customers per week: about 380 at h = 30, 1,000 at h = 60, "
          "1,950 at h = 90).", "",
          "- **Precision@N** = events among the contacted ÷ contacted. **Capture@N** = events among the contacted ÷ all events "
          "that week (every event is in the eligible group, so this is the share of that week's future dormancy caught).",
          "- **Pooling:** contacts, captured events and events are summed over all weeks in the period, then divided. "
          "Weekly capture min–max is shown for spread. If a week has fewer than N eligible customers, all are contacted.",
          "- **Ties:** customers tied on every ranking key at the cut-off share the remaining slots pro rata (the expected result "
          "of random tie-breaking), so no hidden ordering such as customer ID affects the numbers.",
          "- **Rankers:** `random` (expected value: precision = the week's eligible event rate); `website rule` = the deployed "
          "watchlist order (recency descending, then higher spend); `recency` model (monotone in recency, so it is recency "
          "order with random tie-breaking); `rfm`; `rfm-elig`. Platt recalibration is monotone (slope > 0, asserted), so the "
          "`+platt` rankings equal their base models and are not repeated; `recency-elig` likewise equals `recency`.",
          "- **This measures predictive targeting, not the effect of contacting anyone.** A high capture means the list contains "
          "customers who went on to become dormant, not that an intervention would have changed that; many of the highest-risk "
          "customers may be beyond reach. Estimating outreach impact needs a randomized or otherwise causal design.", ""]
    for h, (P, preds, params, scored) in results.items():
        L += [f"**h = {tag(h)}**", "",
              "| Period | Ranker | " + " | ".join(f"P@{N} | C@{N}" for N in CAPACITIES) + " | C@100 weekly range |",
              "|---|---|" + "---:|---:|" * len(CAPACITIES) + "---|"]
        for period, g in scored.items():
            e = g[is_eligible(g, h)]
            rankers = {"website rule": ["days_since_last", "monetary"], "recency": ["recency"],
                       "rfm": ["rfm"], "rfm-elig": ["rfm-elig"]}
            t = top_n(e, h, rankers)
            for name in ["random", *rankers]:
                cells = " | ".join(f"{pct(t.loc[(name, N), 'precision'])} | {pct(t.loc[(name, N), 'capture'])}"
                                   for N in CAPACITIES)
                rng = f"{pct(t.loc[(name, 100), 'weekly_min'])}–{pct(t.loc[(name, 100), 'weekly_max'])}"
                L.append(f"| {period}{' (exploratory)' if period == 'test' else ''} | {name} | {cells} | {rng} |")
        L.append("")

    # 6. Calibration and selection study
    L += ["## 6. Recalibration and model comparison (exploratory on test)", "",
          "Eligible customers only. Δ Brier = candidate − reference, with a customer-bootstrap 95% interval; negative is better. "
          "Validation rows are shown only for candidates that were not calibrated on validation.", "",
          "| h | Period | Model | AUC | Brier | Mean p | Observed | Cal. slope | Δ Brier vs recency [95% CI] | Δ Brier vs elig-constant [95% CI] |",
          "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    verdicts = {}
    for h, (P, preds, params, scored) in results.items():
        t = tables[h].set_index(["period", "model"])
        for period, g in scored.items():
            e = g[is_eligible(g, h)]
            yy = e[f"y_{h}"].to_numpy(float)
            for name in [n for n in CANDIDATES if n in e]:
                r = t.loc[(period, name)]
                dr = boot_diff(e["customer"], yy, e[name].to_numpy(), e["recency"].to_numpy())
                dc = boot_diff(e["customer"], yy, e[name].to_numpy(), e["elig-constant"].to_numpy())
                fmt = lambda d: f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
                L.append(f"| {tag(h)} | {period} | {name} | {f(r.auc_elig)} | {f(r.brier_elig, 4)} | {f(r.mean_p)} | "
                         f"{f(r.observed)} | {f(r.cal_slope, 2)} | {'–' if name == 'recency' else fmt(dr)} | {fmt(dc)} |")
                if period == "test" and name != "recency" and dr[2] < 0:
                    verdicts.setdefault(h, []).append(name)
    L += ["", "Decision rule (fixed in advance): a candidate improves on `recency` if its test Δ Brier interval lies "
          "entirely below zero. Candidates meeting it, per horizon: "
          + "; ".join(f"h = {h}: {', '.join(verdicts.get(h, [])) or 'none'}" for h in HORIZONS)
          + ". Because test was already inspected, this is a provisional ranking, not a confirmed selection.", "",
          "Fitted calibrators (validation eligible rows), p′ = sigmoid(a + b·logit p):", "",
          "| h | Base model | a | b |", "|---|---|---:|---:|"]
    for h, (P, preds, params, scored) in results.items():
        for name in SPECS:
            a, b = params[name]["platt"]
            L.append(f"| {tag(h)} | {name} | {a:.3f} | {b:.3f} |")
    L += ["", "Feature drift (eligible rows, h = 60): medians of the cumulative features and the event rate by frequency band "
          "(share of rows in brackets).", "",
          "| Period | Median tenure (days) | Median frequency | Median monetary (£) | Event rate | 1 invoice | 2–3 | 4–10 | >10 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in drift(df):
        L.append(f"| {r[0]} | {r[1]:.0f} | {r[2]:.0f} | {r[3]:,.0f} | {pct(r[4])} | "
                 + " | ".join(f"{pct(a)} ({pct(b)})" for a, b in zip(r[5:9], r[9:13])) + " |")
    L += ["", FINDINGS, ""]
    return "\n".join(L)


FINDINGS = """## 7. Findings, limitations and next step

Findings (all test numbers exploratory):

1. **Eligibility explains most of the headline performance.** The two-level `elig-constant` (0 if the customer cannot
   reach 90 days within h, else the training eligible rate) has an overall AUC of 0.98 at h = 30 and 0.88–0.89 at h = 60, and
   at h = 30 a *lower* overall Brier score than the selected recency model (test 0.0238 vs 0.0294; validation 0.0224 vs
   0.0317). At h = 60 the two are about equal. Overall metrics should not be used to judge these models.
2. **Inside the eligible group, the Nov 12 models are miscalibrated at h = 30 and h = 60.** Calibration slopes are
   0.32–0.69 (predictions far too spread out) and mean predictions sit below the observed rate. This is why `recency`
   loses to the eligible constant on Brier at h = 30 (test +0.023, interval excludes 0). At h = 90, where eligibility is
   everyone, `recency` is reasonably calibrated (test mean 0.544 vs 0.533 observed, slope 1.13) and beats the constant
   (−0.050).
3. **RFM ranks better but is badly calibrated, because its cumulative features drift.** Within the eligible group, `rfm`
   has a higher AUC than `recency` at every horizon on both validation and test (+0.03 to +0.06). But its mean prediction
   is well below the observed rate in both later periods (h = 90 test: 0.372 vs 0.533), so its Brier score is worse than
   `recency` at every horizon (clearly at h = 30 and h = 90; within noise at h = 60). The drift table shows why: tenure, frequency and spend accumulate from the start of the data, and
   the same invoice count carries a higher dormancy rate later on.
4. **Two fixes, with different effects.**
   - *Refitting on eligible rows* (`-elig`) fixes the slope (0.86–1.21). `recency-elig` beats both `recency` and the
     eligible constant on test at h = 30 and h = 60, and on validation too, which was not used to fit it. It does not fix
     `rfm`'s level, because drift is not a slope problem (`rfm-elig` test mean 0.756 vs 0.850 at h = 30, 0.549 vs 0.700 at
     h = 60).
   - *Platt recalibration on validation* helps `rfm` at every horizon. `rfm+platt` has the lowest eligible-group test
     Brier score at all three horizons (0.1210, 0.1719, 0.1806), with `recency-elig` within 0.001 at h = 30. Validation is
     later than train, so its calibrator partly absorbs the drift. But it overshoots the level (test mean 0.895 vs 0.850,
     0.745 vs 0.700, 0.564 vs 0.533) because validation (Jan–Jun 2011) had more dormancy than test. For `recency`, Platt
     helps at h = 30 and h = 60 and slightly *hurts* at h = 90 (+0.0035, interval excludes 0). Recalibration is not
     automatically an improvement.
5. **No model meaningfully improves top-N targeting over the deployed rule.** Ranking eligible customers by recency (the
   website order) already gives 91–98% precision for N ≤ 200 at every horizon, against 53–89% for random selection. The
   best model adds at most about 2 points (N = 200 at h = 30 and h = 60; at N = 50 the models are sometimes below the
   rule), and at h = 90 `rfm` (and so `rfm+platt`, which
   ranks identically) is *worse* than the rule (test P@200 89.3% vs 92.5%). RFM's AUC gain comes from the middle of the
   ranking, not from the very top, where customers about to reach 90 days are near-certain events. These top-N lists
   are mostly customers already days from the threshold; whether contacting them changes anything is a causal question this
   evaluation cannot answer.

Provisional reading: for **probabilities**, a model fitted on eligible customers and calibrated on a period like the one it
will be used in is clearly better than the Nov 12 models. `rfm+platt` and `recency-elig` lead. For **ranking a small
contact list**, the rule-based order is as good as any model here. Neither conclusion has an untouched confirmation.

Limitations:

- **The test period had already been inspected**, and the candidates were compared on it. The best test candidate's score
  is optimistic: six candidates were compared and the best was picked after the fact.
- **Repeated weekly snapshots.** A customer contributes many overlapping rows. The customer bootstrap intervals cover
  Brier differences; AUC and top-N values have no intervals and the weeks are not independent, so top-N differences of
  1–2 points should not be read as real.
- **h = 90 is exploratory**: 10 test scoring dates (Jul–Sep 2011) with heavily overlapping windows, and a validation period
  whose event rate (64%) differs from train and test (53%).
- **Seasonality and drift.** One year of training covers each season once. Calibrators inherit the season they were
  fitted in. Cumulative features drift (Section 6).
- **Unchanged carry-overs:** tenure is left-truncated at 1 Dec 2009; later-day cancellations still count as purchases.
- **Capture is predictive, not causal** (Section 5).

Holdout: the data end on 9 Dec 2011 and test already reaches the last complete day, so **no genuinely later holdout can be
built from this dataset**. The only clean confirmation would be prospective: freeze one model and calibrator, score new
weeks as they arrive, and evaluate after the h-day outcomes mature, without changing anything in between.

Recommended next step: replace the cumulative features with window-stable ones (for example invoices and spend in the last
365 days, and tenure capped at 365 days). Fit on eligible training rows and compare the uncalibrated `-elig` variants on
**validation**, which they do not touch. Then freeze the choice before looking at test again. Keep the website on the rule-based
watchlist until then."""


if __name__ == "__main__":
    df = pd.read_csv(SNAPSHOTS, dtype={"customer": str}, parse_dates=["scoring_date", "last_purchase", "crossing_date"])
    results = run(df)
    if BASELINES.exists():   # the Nov 12 coefficients must be reproduced exactly
        old = json.loads(BASELINES.read_text())
        for h, (_, _, params, _) in results.items():
            for name in SPECS:
                assert np.allclose(list(old[str(h)][name]["coef"].values()), params[name]["coef"], atol=1e-5), (h, name)
    audit_result = audit(df, load())
    assert audit_result == (0, 0), f"audit failed: {audit_result}"
    plot(results)
    REPORT.write_text(report(df, results, audit_result))
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps({str(h): r[2] for h, r in results.items()}, indent=2))
    keep = ["customer", "scoring_date", "days_since_last"]
    pd.concat([g[keep + ["constant", "elig-constant", *[c for c in CANDIDATES if c in g]]]
               .assign(h=h, period=period, y=g[f"y_{h}"], eligible=is_eligible(g, h))
               for h, (_, _, _, scored) in results.items() for period, g in scored.items()]
              ).to_csv(PREDICTIONS, index=False)
    print(REPORT.read_text())
