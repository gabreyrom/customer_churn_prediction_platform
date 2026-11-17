"""Model sensitivity (Nov 26): how one model's predicted dormancy risk moves when its inputs are changed.

This is a property of the model's equation. It is NOT a prediction of what happens to a customer if someone
intervenes, and the model is not validated for use (see processed/model_evaluation.md).

Usage: python build_snapshots.py && python sensitivity.py   (offline; needs matplotlib through evaluate_models.py)
Outputs:
  processed/sensitivity_model.json   coefficients, scaling, calibrator and training ranges; plain numbers, committed,
                                     and the only thing the website's "Model sensitivity" page reads
  processed/model_sensitivity.md     report; do not edit by hand

Model: `rfm+platt` from evaluate_models.py, dormancy threshold 90 days, horizons 30 / 60 / 90 days.
  inputs   days_since_last (recency), tenure_days, frequency, monetary: the snapshot features of build_snapshots.py,
           all cumulative since the first observed purchase (NOT the 365-day values of the segments page)
  p        = sigmoid(coef . standardised(inputs)), log1p first for tenure, frequency and monetary; fitted on train
  risk     = sigmoid(a + b * logit(p)) when recency >= 90 - horizon, else exactly 0: a customer with fewer days of
             inactivity cannot reach 90 days within the horizon. (a, b) fitted on validation eligible rows.
Everything below the `__main__` guard is offline; the functions above it need numpy and the JSON file only.
"""
import json
from pathlib import Path

import numpy as np

PARAMS, REPORT = Path("processed/sensitivity_model.json"), Path("processed/model_sensitivity.md")
MODEL = "rfm+platt"
FEATURES = ["days_since_last", "tenure_days", "frequency", "monetary"]
LABELS = {"days_since_last": "Recency (days since last purchase)", "tenure_days": "Tenure (days since first observed purchase)",
          "frequency": "Frequency (invoices so far)", "monetary": "Monetary (gross spend so far, £)"}


def load(path=PARAMS):
    return json.loads(Path(path).read_text())


def risk(params, h, profile):
    """Predicted probability of reaching the dormancy threshold within h days. `profile` maps each feature name to a
    number or an array. Returns 0 where recency < threshold - h (not eligible)."""
    m = params["horizons"][str(h)]
    z = m["coef"][0]
    for f, c in zip(FEATURES, m["coef"][1:]):
        v = np.asarray(profile[f], dtype=float)
        v = np.log1p(v) if f in params["log1p"] else v
        z = z + c * (v - m["scale"][f][0]) / m["scale"][f][1]
    p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
    a, b = m["platt"]
    out = 1 / (1 + np.exp(-(a + b * np.log(p / (1 - p)))))
    return np.where(np.asarray(profile["days_since_last"], dtype=float) >= params["threshold"] - h, out, 0.0)


def problems(params, profile):
    """Why a profile cannot be a real snapshot row of an active customer (empty list = consistent)."""
    r, t, f, m = (profile[k] for k in FEATURES)
    out = []
    if not 0 <= r < params["threshold"]:
        out.append(f"Recency must be between 0 and {params['threshold'] - 1} days: at {params['threshold']} days the "
                   "customer is already dormant and is no longer scored.")
    if t < r:
        out.append("Tenure cannot be shorter than recency: the first purchase is never after the last one.")
    if f < 1 or f != int(f):
        out.append("Frequency must be a whole number of at least 1.")
    elif f == 1 and t != r:
        out.append("With a single invoice, the first and last purchase are the same day, so tenure must equal recency.")
    if m <= 0:
        out.append("Monetary must be positive.")
    return out


def outside_training(params, h, profile):
    """Features whose value lies outside the 1st–99th percentile of the eligible training rows (extrapolation)."""
    rng = params["horizons"][str(h)]["range"]
    return [f for f in FEATURES if not rng[f]["p01"] <= profile[f] <= rng[f]["p99"]]


def sweep(params, h, profile, feature, values):
    """Risk along `values` of one feature, the others held at `profile`. Values that would make the profile
    inconsistent (see `problems`) are dropped, so every point is a possible customer."""
    values = [v for v in values if not problems(params, {**profile, feature: v})]
    return values, risk(params, h, {**profile, feature: np.array(values, dtype=float)})


if __name__ == "__main__":
    import pandas as pd

    from build_snapshots import D, LAST_COMPLETE, OUT as SNAPSHOTS, TRAIN_END, VAL_END
    from evaluate_models import fit_candidates, is_eligible
    from train_baselines import EXPLORATORY, HORIZONS, LOG, SPECS, auc, brier, design, eligible, periods

    assert SPECS["rfm"] == FEATURES
    snap = pd.read_csv(SNAPSHOTS, dtype={"customer": str}, parse_dates=["scoring_date", "last_purchase", "crossing_date"])
    params = {"model": MODEL, "threshold": D, "features": FEATURES, "log1p": sorted(LOG),
              "fitted_on": f"scoring dates with outcome window ending by {TRAIN_END:%Y-%m-%d}",
              "calibrated_on": f"eligible rows, scoring dates after {TRAIN_END:%Y-%m-%d} with window ending by {VAL_END:%Y-%m-%d}",
              "horizons": {}}
    for h in HORIZONS:
        preds, fitted = fit_candidates(snap, h)
        P = periods(snap, h)
        etr, ete = eligible(P["train"], h), eligible(P["test"], h)
        q = etr[FEATURES].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        y, p = ete[f"y_{h}"].to_numpy(float), preds[MODEL](ete)
        params["horizons"][str(h)] = {
            "coef": fitted["rfm"]["coef"], "platt": fitted["rfm"]["platt"],
            "scale": {f: list(v) for f, v in design(P["train"], FEATURES)[1].items()},
            "range": {f: dict(zip(["p01", "p25", "p50", "p75", "p99"], q[f].round(2).tolist())) for f in FEATURES},
            "train_eligible_rows": len(etr), "exploratory": h in EXPLORATORY,
            "test": {"rows": len(ete), "auc": round(auc(y, p), 3), "brier": round(brier(y, p), 4),
                     "mean_predicted": round(float(p.mean()), 3), "observed": round(float(y.mean()), 3)}}
        # The stored numbers must reproduce the evaluation pipeline's own predictions on every snapshot row.
        assert np.allclose(risk(params, h, {f: snap[f].to_numpy() for f in FEATURES}), preds[MODEL](snap), atol=1e-9)
        assert (risk(params, h, snap[~is_eligible(snap, h)]) == 0).all()
    PARAMS.write_text(json.dumps(params, indent=1))

    pp = lambda v: f"{100 * v:+.1f}"
    L = ["# Model sensitivity: how predicted dormancy risk responds to its inputs (Nov 26)", "",
         "Generated by `sensitivity.py` from `data/weekly_snapshots.csv.gz`. Do not edit by hand.", "",
         "> **Model sensitivity, not intervention effect.** Each number is what the model's equation returns for a "
         "different set of input values. It does not say what would happen to a customer if their behaviour were changed "
         "or if they were contacted, and the model is **not validated**: its test period had already been inspected and "
         "no later independent holdout exists (`model_evaluation.md`, Section 7).", "",
         "## 1. Model", "",
         f"- **Model:** `{MODEL}` from `evaluate_models.py`: a logistic regression on recency, tenure, frequency and "
         "monetary value (log1p for the last three, standardised with training means), then a Platt recalibration.",
         f"- **Dormancy threshold:** {D} days without a valid purchase. **Forecast horizons:** 30, 60 and 90 days "
         "(90 is exploratory).",
         "- **Inputs** are the snapshot features of `build_snapshots.py` on a scoring date: cumulative since the first "
         "observed purchase. They are not the trailing 365-day frequency and spend used for the RFM segments.",
         f"- **Fitted on** scoring dates whose outcome window ends by {TRAIN_END:%d %b %Y}; **recalibrated on** eligible "
         f"rows with windows ending by {VAL_END:%d %b %Y}. Outcomes are observed through {LAST_COMPLETE:%d %b %Y}.",
         f"- **Eligibility:** at horizon h a customer needs recency ≥ {D} − h to be able to reach {D} days in time. Below "
         "that the risk is exactly 0 by arithmetic, not by estimation.",
         f"- **Reproducible:** `processed/sensitivity_model.json` holds every number needed. This script asserts that it "
         f"reproduces the evaluation pipeline's `{MODEL}` prediction on all {len(snap):,} snapshot rows.", "",
         "| Horizon | Eligible when recency ≥ | Coefficients per standard deviation: recency / log tenure / log frequency / "
         "log monetary | Calibrator (a, b) | Test, eligible rows (exploratory): mean predicted vs observed | AUC | Brier |",
         "|---|---:|---|---|---|---:|---:|"]
    for h in HORIZONS:
        m = params["horizons"][str(h)]
        L.append(f"| {h} days{' (exploratory)' if m['exploratory'] else ''} | {D - h} days | "
                 + " / ".join(f"{c:+.2f}" for c in m["coef"][1:]) + f" | {m['platt'][0]:.2f}, {m['platt'][1]:.2f} | "
                 f"{m['test']['mean_predicted']:.3f} vs {m['test']['observed']:.3f} | {m['test']['auc']:.3f} | {m['test']['brier']:.4f} |")
    L += ["", "The calibrator slope b is below 1 at 30 and 60 days: it shrinks the raw model's differences, so the "
          "sensitivities below are smaller than the raw coefficients suggest.", "",
          "## 2. Recency", "",
          "Typical profile = the median tenure, frequency and spend of the eligible training rows at that horizon. "
          "Recency moves alone, which is realistic: it is what happens to a customer who does not buy as days pass "
          "(tenure also grows by the same days; that is included).", "",
          "| Horizon | Typical profile (tenure / invoices / spend) | " + " | ".join(f"Recency {r}" for r in (0, 29, 30, 45, 59, 60, 75, 89)) + " |",
          "|---|---|" + "---:|" * 8]
    typical = {}
    for h in HORIZONS:
        rng = params["horizons"][str(h)]["range"]
        base = {f: rng[f]["p50"] for f in FEATURES}
        typical[h] = base
        cells = []
        for r in (0, 29, 30, 45, 59, 60, 75, 89):
            prof = {**base, "days_since_last": r, "tenure_days": base["tenure_days"] - base["days_since_last"] + r}
            assert not problems(params, prof)
            cells.append(f"{float(risk(params, h, prof)):.3f}")
        L.append(f"| {h} days | first purchase {base['tenure_days'] - base['days_since_last']:.0f} days before the last / "
                 f"{base['frequency']:.0f} / £{base['monetary']:,.0f} | " + " | ".join(cells) + " |")
    L += ["", "A risk of 0.000 marks the eligibility boundary: recency 59 at the 30-day horizon and 29 at the 60-day "
          "horizon cannot reach 90 days in time; one more day of inactivity makes the customer eligible and the risk jumps. "
          "The jump is a property of the outcome definition.", "",
          "## 3. One input at a time", "",
          "From the typical profile, one input moves from the 25th to the 75th percentile of the eligible training rows. "
          "Change in predicted risk, in percentage points. Frequency starts at 2 because a single invoice is not possible "
          "when the first and last purchase days differ.", "",
          "| Horizon | Typical risk | Recency p25 → p75 | Tenure p25 → p75 | Frequency 2 (or p25) → p75 | Monetary p25 → p75 |",
          "|---|---:|---|---|---|---|"]
    for h in HORIZONS:
        rng, base = params["horizons"][str(h)]["range"], typical[h]
        cells = []
        for f in FEATURES:
            a, b = rng[f]["p25"], rng[f]["p75"]
            if f == "frequency":   # one invoice is impossible when the first and last purchase days differ
                a = max(a, 2)
            lo, hi = {**base, f: a}, {**base, f: b}
            assert not problems(params, lo) and not problems(params, hi)
            unit = "£" if f == "monetary" else ""
            cells.append(f"{unit}{a:,.0f} → {unit}{b:,.0f}: **{pp(float(risk(params, h, hi)) - float(risk(params, h, lo)))}**")
        L.append(f"| {h} days | {float(risk(params, h, base)):.3f} | " + " | ".join(cells) + " |")
    L += ["", "- Longer inactivity raises the predicted risk; more invoices and higher spend lower it. Tenure has little "
          "weight once the others are known.",
          "- **These are not independent levers.** In real customers frequency, spend and tenure move together; a "
          "one-at-a-time change describes the equation, not a customer who could exist with only that difference.",
          "- **No causal reading.** A lower predicted risk for a customer with more invoices does not mean that getting "
          "a customer to place one more order lowers their risk by that amount.", "",
          "## 4. Limitations", "",
          "- **Not validated.** The test period was inspected on Nov 12 and Nov 19 and six candidates were compared on it; "
          f"`{MODEL}` had the lowest eligible-group test Brier score after the fact. No later holdout exists.",
          "- **Cumulative inputs drift.** Tenure, frequency and spend grow with the amount of history observed, so the same "
          "values mean something different later on. The calibrator absorbed part of this for early 2011 only.",
          "- **Level is off in the test period** (table in Section 1: mean predicted vs observed).",
          "- **Outside the training range** (below the 1st or above the 99th percentile of eligible training rows) the "
          "equation extrapolates. The website page flags such inputs.",
          "- One retailer, one dataset, a 90-day threshold only. The model must not be applied to another dataset "
          "without retraining or a transfer validation.", ""]
    REPORT.write_text("\n".join(L))
    print("\n".join(L))
