"""Initial probabilistic baselines for dormancy within 30/60/90 days, on the weekly snapshots.

Usage: python build_snapshots.py && python train_baselines.py   (pandas + numpy only)

Outputs:
  data/models/baselines.json        fitted coefficients and train-only scaling (gitignored)
  processed/baseline_results.md     split comparison and results; do not edit by hand

Protocol: calendar split from build_snapshots.split (no random rows); censored labels dropped; transforms and
coefficients fitted on train only; the specification is chosen per horizon on validation Brier score; only the
chosen specification and the constant event-rate baseline are scored on test, once.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_snapshots import HORIZONS, LAST_COMPLETE, OUT as SNAPSHOTS, TRAIN_END, VAL_END, split

# Explicit allowlist. Everything else in the snapshot table (customer, scoring_date, last_purchase,
# crossing_date, y_*) is an identifier, a date or derived from the future, and must never become a predictor.
SPECS = {"recency": ["days_since_last"],
         "rfm": ["days_since_last", "tenure_days", "frequency", "monetary"]}
LOG = {"tenure_days", "frequency", "monetary"}   # right-skewed: log1p before standardising
FORBIDDEN = {"customer", "scoring_date", "last_purchase", "crossing_date", *(f"y_{h}" for h in HORIZONS)}
CANDIDATES = [("2011-03-31", "2011-07-31"), ("2011-01-31", "2011-06-30"), ("2010-12-31", "2011-06-30"),
              ("2010-12-31", "2011-05-31"), ("2010-11-30", "2011-05-31")]
EXPLORATORY = {90}   # 10 test scoring dates with overlapping 90-day windows: too few for a firm conclusion
MODELS, REPORT = Path("data/models/baselines.json"), Path("processed/baseline_results.md")


def design(df, cols, scale=None):
    """Allowlisted columns -> standardised matrix. `scale` comes from train; pass None only for train."""
    assert set(cols) <= set(SPECS["rfm"]) and not set(cols) & FORBIDDEN, f"illegal predictors: {cols}"
    x = df[cols].astype(float).copy()
    for c in cols:
        if c in LOG:
            x[c] = np.log1p(x[c])
    if scale is None:
        scale = {c: (x[c].mean(), x[c].std()) for c in cols}
    x = np.column_stack([np.ones(len(x))] + [(x[c] - scale[c][0]) / scale[c][1] for c in cols])
    return x, scale


def fit_logit(x, y, iters=50, ridge=1e-6):
    """Newton-Raphson logistic regression (ponytail: no sklearn; fine for <=5 features, swap in sklearn for more)."""
    b = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-x @ b))
        h = x.T @ (x * (p * (1 - p))[:, None]) + ridge * np.eye(len(b))
        step = np.linalg.solve(h, x.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def predict(x, b):
    return 1 / (1 + np.exp(-x @ b))


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    return np.nan if n1 in (0, len(y)) else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def periods(df, h, train_end=TRAIN_END, val_end=VAL_END):
    """Labelled rows per period. Censored rows are dropped; the split itself never lets a label pass a cutoff."""
    d = df[df[f"y_{h}"].notna()]
    sp = split(d["scoring_date"], h, pd.Timestamp(train_end), pd.Timestamp(val_end))
    return {n: d[sp == n] for n in ("train", "validation", "test")}


def describe(g, h):
    return (g["scoring_date"].nunique(), g["customer"].nunique(), len(g), int(g[f"y_{h}"].sum()), g[f"y_{h}"].mean())


def eligible(df, h, d=90):
    """Rows that can reach the d-day threshold within h days. Everyone else is a structural negative."""
    m = df["days_since_last"] >= d - h
    assert (df.loc[~m, f"y_{h}"] == 0).all(), "event outside the eligible group"
    return df[m]


def eligible_eval(P, fits, chosen, h):
    """Diagnostics only (reported, never used for selection): selected model on validation and test,
    other specs on validation only, constant = train event rate within the training eligible group."""
    E = {n: eligible(g, h) for n, g in P.items()}
    rate = E["train"][f"y_{h}"].mean()
    rows = {}
    for period in ("validation", "test"):
        g = E[period]
        y = g[f"y_{h}"].to_numpy(float)
        rows[(period, "constant")] = (np.nan, brier(y, np.full(len(y), rate)))
        for name, (b, scale, cols) in fits.items():
            if period == "validation" or name == chosen:
                p = predict(design(g, cols, scale)[0], b)
                rows[(period, name)] = (auc(y, p), brier(y, p))
    return E, rate, rows


def run(df):
    results, models = {}, {}
    for h in HORIZONS:
        P = periods(df, h)
        ytr = P["train"][f"y_{h}"].to_numpy(float)
        rate = ytr.mean()
        res = {"constant": {}}
        fits = {}
        for name, cols in SPECS.items():
            xtr, scale = design(P["train"], cols)
            fits[name] = (fit_logit(xtr, ytr), scale, cols)
            res[name] = {}
        for period in ("validation", "test"):
            y = P[period][f"y_{h}"].to_numpy(float)
            res["constant"][period] = (np.nan, brier(y, np.full(len(y), rate)))
            if period == "validation":
                for name, (b, scale, cols) in fits.items():
                    p = predict(design(P[period], cols, scale)[0], b)
                    res[name][period] = (auc(y, p), brier(y, p))
                chosen = min(SPECS, key=lambda n: res[n]["validation"][1])
            else:  # test: chosen specification only, evaluated once
                b, scale, cols = fits[chosen]
                p = predict(design(P[period], cols, scale)[0], b)
                res[chosen][period] = (auc(y, p), brier(y, p))
        results[h] = (P, res, chosen, rate, eligible_eval(P, fits, chosen, h))
        models[h] = {"chosen": chosen, "train_event_rate": rate,
                     **{n: {"features": c, "coef": dict(zip(["intercept", *c], b.round(6).tolist())),
                            "scale": {k: list(v) for k, v in s.items()}, "log1p": [x for x in c if x in LOG]}
                        for n, (b, s, c) in fits.items()}}
    return results, models


def report(df, results):
    f = lambda v, k=3: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{k}f}"
    L = ["# Baseline results: dormancy within 30/60/90 days (initial)", "",
         "Generated by `train_baselines.py` from `data/weekly_snapshots.csv.gz`. Do not edit by hand. "
         "These are **initial baseline results**; calibration and top-N evaluation (Nov 19) are in `model_evaluation.md`.", "",
         "## 1. Split choice", "",
         "Each candidate splits on the scoring date s, separately for each horizon h: train needs s+h ≤ train end; "
         "validation needs s > train end and s+h ≤ validation end; test needs s > validation end and s+h ≤ "
         f"{LAST_COMPLETE:%Y-%m-%d}. Every training label is therefore fully observed before validation begins, and every "
         "validation label before test begins. Cells: scoring dates / customers / labelled rows / event rate.", "",
         "| Train end | Val end | h | Train | Validation | Test |", "|---|---|---|---|---|---|"]
    for te, ve in CANDIDATES:
        for h in HORIZONS:
            cells = [f"{a} / {b:,} / {c:,} / {e:.1%}" for a, b, c, _, e in (describe(g, h) for g in periods(df, h, te, ve).values())]
            mark = " **(chosen)**" if (pd.Timestamp(te), pd.Timestamp(ve)) == (TRAIN_END, VAL_END) else ""
            L.append(f"| {te}{mark} | {ve} | {h} | " + " | ".join(cells) + " |")
    L += ["",
          f"**Chosen: train to {TRAIN_END:%Y-%m-%d}, validation to {VAL_END:%Y-%m-%d}, test to {LAST_COMPLETE:%Y-%m-%d}.** "
          "It is the latest train cutoff that keeps at least 13 validation and 10 test scoring dates at h = 90 "
          "(the previous split had 4 and 6). Trade-off: the training set is smaller (31–39 scoring dates, Mar–Nov 2010), and the "
          "validation period covers Jan–Jun 2011, when dormancy is more common (the post-Christmas lull: event rates 23% / 45% / 64% "
          "vs 15% / 34% / 53% in train). A model selected on validation is thus selected under a shifted base rate. "
          "Test (Jul–Nov 2011) has a base rate close to train. Rows whose outcome window straddles a cutoff are unused.", "",
          "## 2. Models", "",
          "- **constant**: the train event rate for every row (AUC undefined).",
          "- **recency**: logistic regression on `days_since_last`.",
          "- **rfm**: logistic regression on `days_since_last`, log1p(`tenure_days`), log1p(`frequency`), log1p(`monetary`).",
          "- Predictors come from an explicit allowlist. Standardisation means/SDs and coefficients are fit on train only. "
          "Censored labels are dropped. Selection: lower validation Brier score per horizon. Only the chosen model is scored on test.", "",
          "## 3. Results", "",
          "| h | Period | Scoring dates | Customers | Rows | Events | Event rate | Model | AUC | Brier |",
          "|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for h, (P, res, chosen, rate, _) in results.items():
        tag = " (exploratory)" if h in EXPLORATORY else ""
        a, b, c, e, r = describe(P["train"], h)
        L.append(f"| {h}{tag} | train | {a} | {b:,} | {c:,} | {e:,} | {r:.1%} | – | – | – |")
        for period in ("validation", "test"):
            a, b, c, e, r = describe(P[period], h)
            for name in ("constant", *SPECS):
                if period in res[name]:
                    au, br = res[name][period]
                    star = " ✓ chosen" if name == chosen else ""
                    L.append(f"| {h}{tag} | {period} | {a} | {b:,} | {c:,} | {e:,} | {r:.1%} | {name}{star} | {f(au)} | {f(br, 4)} |")
    L += ["", "Test rows show only the constant baseline and the model selected on validation; the other specification "
          "was never scored on test.", "",
          "## 4. Customers eligible to become dormant within the horizon", "",
          "At horizon h, only customers with `days_since_last` ≥ 90 − h can reach 90 days of inactivity by s + h; everyone "
          "else is a guaranteed negative (checked in code). The table restricts every period to that group. The constant is the "
          "event rate of the **training** eligible group. These are diagnostics: model selection is unchanged (Section 3), "
          "and rfm is shown on validation only.", "",
          "| h | Period | Eligible rows | Share of rows | Events | Event rate | Model | AUC | Brier |",
          "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for h, (P, res, chosen, rate, (E, erate, rows)) in results.items():
        tag = " (exploratory)" if h in EXPLORATORY else ""
        L.append(f"| {h}{tag} | train | {len(E['train']):,} | {len(E['train']) / len(P['train']):.1%} | "
                 f"{int(E['train'][f'y_{h}'].sum()):,} | {erate:.1%} | – | – | – |")
        for (period, name), (au, br) in rows.items():
            g = E[period]
            star = " ✓ selected" if name == chosen else ""
            L.append(f"| {h}{tag} | {period} | {len(g):,} | {len(g) / len(P[period]):.1%} | {int(g[f'y_{h}'].sum()):,} | "
                     f"{g[f'y_{h}'].mean():.1%} | {name}{star} | {f(au)} | {f(br, 4)} |")
    L += ["", "Why the overall AUC is so much higher: AUC is the chance that a random event row scores above a random non-event "
          "row. Overall, most non-events are customers with low recency who *cannot* become dormant within h days (at h = 30, "
          "anyone under 60 days); a recency model ranks them below every eligible customer for free. Those easy pairs dominate, "
          "so the overall AUC mostly measures that eligibility rule. Inside the eligible group, the model has to separate customers "
          "who will stay silent from those who will buy again, which is the real prediction problem, and the AUC falls accordingly. "
          "Brier is affected the same way: the many near-zero predictions for ineligible rows are almost free.", "",
          "Findings within the eligible group:", "",
          "- h = 30: the selected recency model ranks modestly (AUC 0.69 validation, 0.67 test) and its Brier score is **worse** "
          "than the eligible-group constant on both validation and test. Its overall Brier advantage comes almost entirely from "
          "the ineligible rows.",
          "- h = 60: the recency model is slightly better than the eligible-group constant on Brier (validation and test) and "
          "ranks moderately (AUC 0.75 / 0.71).",
          "- h = 90: every active customer is eligible (recency < 90 always satisfies ≥ 0), so this equals the overall result.",
          "- rfm has a higher validation AUC than recency within the eligible group at every horizon. This is reported, not acted on; "
          "the selection stays as in Section 3 and no test number was used to choose anything.", "",
          "Reading the numbers:", "",
          f"- The overall h = 30 AUC near 0.99 is structural, not leakage: a customer can only reach 90 days of inactivity within 30 days "
          "if their recency is already at least 60. Section 4 gives the results within that eligible group.",
          "- On validation, rfm ranks better (higher AUC) at every horizon but has the worse Brier score, consistent with the "
          "base-rate shift between train and validation. Selection by Brier therefore favours recency; the Nov 19 calibration "
          "work should revisit this choice (for example, rfm with a recalibrated intercept).", "",
          "## 5. Limitations", "",
          "- Rows are weekly snapshots of the same customers with overlapping outcome windows. They are not independent, so the "
          "effective sample size is far smaller than the row count; no confidence intervals are reported yet (a customer-level "
          "bootstrap is the planned fix).",
          "- The 90-day horizon is **exploratory**: 10 test scoring dates (Jul–Sep 2011) whose windows overlap heavily.",
          "- Base rates shift between periods (seasonality), so the Brier score mixes ranking quality with calibration-in-the-large. "
          "Calibration is not yet assessed.",
          "- Tenure is left-truncated at 1 Dec 2009. Later-day cancellations are still counted as purchases (cleaning rule).",
          "- The models are not used by the website, which still shows the rule-based watchlist.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    df = pd.read_csv(SNAPSHOTS, dtype={"customer": str},
                     parse_dates=["scoring_date", "last_purchase", "crossing_date"])
    results, models = run(df)
    MODELS.parent.mkdir(parents=True, exist_ok=True)
    MODELS.write_text(json.dumps({str(h): m for h, m in models.items()}, indent=2))
    REPORT.write_text(report(df, results))
    print(REPORT.read_text())
