import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import sensitivity as sv
from shared import dataset

P = sv.load()
D = P["threshold"]

st.title("Model Sensitivity")
st.warning(
    "**Model sensitivity, not the effect of an intervention.** This page shows what one model's equation returns when "
    "its input values are changed. It does not predict what would happen to a customer if their behaviour changed or if "
    "they were contacted. **The model is not validated:** its test period had already been inspected, and no later "
    "independent holdout exists. No customer on this website is scored with it.",
    icon="⚠️",
)
if not dataset()[1]["bundled"]:
    st.info("You are using uploaded data on the other pages. This model was trained on Online Retail II and must not be "
            "read as describing your customers; that would need retraining or a transfer validation.")

c1, c2 = st.columns([1, 2])
h = c1.radio("Forecast horizon", [30, 60, 90], index=1, horizontal=True,
             format_func=lambda v: f"{v} days" + (" (exploratory)" if P["horizons"][str(v)]["exploratory"] else ""))
M = P["horizons"][str(h)]
c2.markdown(
    f"**Model:** `{P['model']}` (logistic regression on recency, tenure, frequency and spend, then Platt recalibration), "
    f"Online Retail II  \n**Outcome:** reaching **{D} days** without a valid purchase within the next **{h} days**  \n"
    f"**Dormancy threshold:** {D} days, fixed. The model exists for this threshold only, so the shared threshold "
    "control does not apply here."
)
t = M["test"]
st.caption(
    f"Fitted on {P['fitted_on']}; recalibrated on {P['calibrated_on']}. In the already-inspected test period "
    f"(eligible customers, {t['rows']:,} rows) it predicted {t['mean_predicted']:.3f} on average against {t['observed']:.3f} "
    f"observed, AUC {t['auc']:.3f}. Inputs are cumulative since the first observed purchase and drift over time."
)

st.subheader("Two profiles")
st.caption(f"Inputs are the model's own features on a scoring date. Defaults are the median eligible training customer. "
           f"At this horizon a customer needs a recency of at least **{D - h} days** to be able to reach {D} days in "
           f"time; below that the risk is 0 by arithmetic.")
rng = M["range"]
base = {f: rng[f]["p50"] for f in sv.FEATURES}


def profile(col, title, key, recency):
    with col.container(border=True):
        st.markdown(f"**{title}**")
        r = st.number_input(sv.LABELS["days_since_last"], 0, D - 1, int(recency), key=f"{key}-r-{h}")
        gap = st.number_input("Days between first and last purchase", 0, 2000, int(base["tenure_days"] - base["days_since_last"]),
                              key=f"{key}-g-{h}", help="Tenure = this + recency, so tenure can never be shorter than recency.")
        f = st.number_input(sv.LABELS["frequency"], 1, 500, int(max(base["frequency"], 2)), key=f"{key}-f-{h}")
        m = st.number_input(sv.LABELS["monetary"], 0.01, 1e6, float(base["monetary"]), step=50.0, key=f"{key}-m-{h}")
        return {"days_since_last": r, "tenure_days": r + gap, "frequency": f, "monetary": m}


a, b, out = st.columns([2, 2, 2])
pa = profile(a, "Profile A", "a", base["days_since_last"])
pb = profile(b, "Profile B", "b", min(base["days_since_last"] + 14, D - 1))
bad = {name: sv.problems(P, p) for name, p in (("A", pa), ("B", pb))}
with out:
    for name, p in (("A", pa), ("B", pb)):
        for msg in bad[name]:
            st.error(f"Profile {name}: {msg}")
        far = sv.outside_training(P, h, p)
        if far and not bad[name] and p["days_since_last"] >= D - h:
            st.warning(f"Profile {name} is outside the training range for: {', '.join(sv.LABELS[f].split(' (')[0].lower() for f in far)}. "
                       "The model is extrapolating.")
    if not any(bad.values()):
        ra, rb = float(sv.risk(P, h, pa)), float(sv.risk(P, h, pb))
        st.metric(f"Model output for A (dormant within {h} days)", f"{ra:.1%}")
        st.metric("Model output for B", f"{rb:.1%}", f"{100 * (rb - ra):+.1f} points vs A", delta_color="off")
        for name, p in (("A", pa), ("B", pb)):
            if p["days_since_last"] < D - h:
                st.caption(f"Profile {name} is not eligible: {p['days_since_last']} + {h} days is less than {D}, so "
                           "the outcome cannot happen within the horizon.")
        st.caption("The difference is between two sets of input values. It is not the effect of moving a customer "
                   "from A to B.")
if any(bad.values()):
    st.stop()

st.subheader("One input at a time, from profile A")
st.caption("Each chart changes one input and holds the others at profile A. Only values that can belong to a real "
           "customer are drawn. Dashed line: profile A. Shaded: 1st–99th percentile of the eligible training customers; "
           "outside it the model extrapolates.")
gap = pa["tenure_days"] - pa["days_since_last"]
grids = {"days_since_last": range(0, D), "tenure_days": range(int(pa["days_since_last"]), 801, 5),
         "frequency": range(1, 61), "monetary": np.unique(np.geomspace(10, 20000, 60).round(0))}
cols = st.columns(2)
for i, f in enumerate(sv.FEATURES):
    if f == "days_since_last":   # as days pass without a purchase, tenure grows by the same number of days
        xs = list(grids[f])
        ys = sv.risk(P, h, {**pa, f: np.array(xs, float), "tenure_days": np.array(xs, float) + gap})
    else:
        xs, ys = sv.sweep(P, h, pa, f, grids[f])
    if not len(xs):
        cols[i % 2].caption(f"{sv.LABELS[f]}: no other value is possible with profile A's other inputs.")
        continue
    d = pd.DataFrame({"x": xs, "risk": ys})
    scale = alt.Scale(type="log" if f == "monetary" else "linear")
    x = alt.X("x:Q", title=sv.LABELS[f], scale=scale)
    band = alt.Chart(pd.DataFrame({"lo": [max(rng[f]["p01"], min(xs))], "hi": [min(rng[f]["p99"], max(xs))]})).mark_rect(
        color="#1f4e79", opacity=0.07).encode(x=alt.X("lo:Q", scale=scale), x2="hi:Q")
    line = alt.Chart(d).mark_line(color="#1f4e79", strokeWidth=2, interpolate="step-after" if f == "days_since_last" else "linear").encode(
        x, alt.Y("risk:Q", title=f"Model output ({h}-day)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("x:Q", title=sv.LABELS[f]), alt.Tooltip("risk:Q", title="Model output", format=".1%")])
    mark = alt.Chart(pd.DataFrame({"x": [pa[f]]})).mark_rule(strokeDash=[4, 4], color="#52514e").encode(x="x:Q")
    cols[i % 2].altair_chart((band + line + mark).properties(height=230), width="stretch")
st.caption(
    f"Recency chart: below {D - h} days the output is exactly 0 (not eligible), then it jumps. Tenure moves with recency "
    "there, as it does for a customer who simply does not buy. In the other charts the inputs move alone, which real "
    "customers rarely do: frequency, spend and tenure rise together. Read them as a description of the equation."
)

with st.expander("Model details and limitations"):
    st.markdown(
        f"""
- **Reproducible:** every number comes from `processed/sensitivity_model.json`, written by `sensitivity.py`, which
  checks that it reproduces the offline evaluation's `{P['model']}` predictions on every weekly snapshot.
- **Inputs:** recency, tenure, invoice count and gross spend on a scoring date, cumulative since the first observed
  purchase. These differ from the trailing 365-day frequency and spend on the Customer segments page.
- **Coefficients per standard deviation** (recency, log tenure, log frequency, log spend):
  {", ".join(f"{c:+.2f}" for c in M["coef"][1:])}; calibrator a = {M["platt"][0]:.2f}, b = {M["platt"][1]:.2f}.
- **Not validated.** Six candidate models were compared on a test period that had already been looked at; this one had
  the lowest eligible-group Brier score after the fact. The data end in December 2011, so no later holdout exists.
- **Drift.** Cumulative inputs grow with the amount of history observed, so the same values mean something different later.
- **Not causal.** A lower output for a profile with more invoices does not mean that one more order lowers a customer's risk.
- Full offline report: `processed/model_sensitivity.md`; evaluation: `processed/model_evaluation.md`.
"""
    )
