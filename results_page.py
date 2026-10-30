import random
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

MOCK = "Illustrative mock data"
DATA_START, DATA_END = date(2010, 1, 4), date(2011, 12, 9)
WARNING_WINDOW = 30  # days before the threshold that count as "approaching"
COLORS = {"Engaged": "#4a7fb0", "Approaching dormancy": "#d9a441", "Dormant": "#9a9a9a"}


@st.cache_data
def mock_transactions(n_customers=60, seed=627):
    """Deterministic fake invoices; NOT derived from Online Retail II."""
    rng = random.Random(seed)
    rows = []
    for i in range(1, n_customers + 1):
        day = DATA_START + timedelta(days=rng.randint(0, 300))
        stop = DATA_END - timedelta(days=rng.choice([0, 0, 0, 20, 60, 100, 200, 350]))
        gap, basket = rng.uniform(10, 70), rng.uniform(40, 400)
        while day <= stop:
            rows.append((f"MOCK-{i:03d}", day, round(basket * rng.uniform(0.5, 1.6), 2)))
            day += timedelta(days=max(1, int(rng.expovariate(1 / gap))))
    df = pd.DataFrame(rows, columns=["customer", "date", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def classify(recency, threshold):
    if recency >= threshold:
        return "Dormant"
    if recency >= threshold - WARNING_WINDOW:
        return "Approaching dormancy"
    return "Engaged"


st.title("Customer Status")
st.warning(f"**{MOCK}.** Demo view. Customers and values are generated for illustration and are not real results.", icon="⚠️")

c1, c2, c3 = st.columns(3)
as_of = c1.date_input("“As of” observation date", DATA_END, min_value=date(2010, 6, 1), max_value=DATA_END)
threshold = c2.selectbox("Dormancy threshold (days without a valid purchase)", [60, 90, 120], index=1)
top_n = c3.slider("Contact-list size N", 5, 30, 10)

tx = mock_transactions()
tx = tx[tx["date"] <= pd.Timestamp(as_of)]
rfm = tx.groupby("customer").agg(last=("date", "max"), frequency=("date", "nunique"), monetary=("amount", "sum"))
rfm["recency"] = (pd.Timestamp(as_of) - rfm["last"]).dt.days
rfm["status"] = rfm["recency"].apply(classify, threshold=threshold)
rfm["monetary"] = rfm["monetary"].round(2)

with st.expander("Definitions and provisional segment rules", expanded=True):
    d1, d2 = st.columns(2)
    d1.markdown(
        """
**Recency** – days from the last valid purchase to the “as of” date.
**Frequency** – number of distinct valid invoices up to the “as of” date.
**Monetary** – total spend on valid invoices up to the “as of” date.
"""
    )
    d2.markdown(
        f"""
| Segment | Rule (threshold T = {threshold}) |
|---|---|
| Engaged | recency < {threshold - WARNING_WINDOW} days |
| Approaching dormancy | {threshold - WARNING_WINDOW} ≤ recency < {threshold} days (still active) |
| Dormant | recency ≥ {threshold} days (threshold already crossed) |
"""
    )

st.subheader(f"Current status · {MOCK}")
st.caption("Status describes where each customer stands today, based on recency. It is not a forecast.")
counts = rfm["status"].value_counts()
m = st.columns(4)
m[0].metric("Customers observed", len(rfm))
for col, s in zip(m[1:], COLORS):
    col.metric(s, int(counts.get(s, 0)))

st.subheader(f"Recency distribution · {MOCK}")
color = alt.Color("status:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), title="Status")
hist = alt.Chart(rfm.reset_index()).mark_bar().encode(
    alt.X("recency:Q", bin=alt.Bin(step=15), title="Recency (days since last purchase)"),
    alt.Y("count():Q", title="Customers"),
    color,
)
rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(strokeDash=[4, 4], color="#b03a2e").encode(x="x:Q")
st.altair_chart((hist + rule).properties(height=260), width="stretch")
st.caption(f"Dashed line = dormancy threshold ({threshold} days). {MOCK}.")

left, right = st.columns(2)
cols = ["recency", "frequency", "monetary", "status"]
with left:
    st.subheader(f"RFM-based watchlist (top {top_n})")
    st.caption(
        f"Active customers approaching the threshold, ranked by spend. A rule-based list, not a trained prediction. {MOCK}."
    )
    priority = rfm[rfm["status"] == "Approaching dormancy"].sort_values("monetary", ascending=False).head(top_n)
    st.dataframe(priority[cols], width="stretch")
    if len(priority) < top_n:
        st.caption(f"Only {len(priority)} customers are currently approaching dormancy.")
with right:
    st.subheader("Already dormant (separate re-activation list)")
    st.caption(f"Customers who have already crossed the {threshold}-day threshold. {MOCK}.")
    st.dataframe(rfm[rfm["status"] == "Dormant"].sort_values("monetary", ascending=False)[cols], width="stretch")

st.subheader(f"Selected customer purchase timeline · {MOCK}")
default = priority.index[0] if len(priority) else rfm.index[0]
cust = st.selectbox("Customer", rfm.index, index=list(rfm.index).index(default))
ct = tx[tx["customer"] == cust]
bars = alt.Chart(ct).mark_bar(width=3, color="#1f4e79").encode(
    alt.X("date:T", title="Invoice date", scale=alt.Scale(domain=[str(DATA_START), str(as_of)])),
    alt.Y("amount:Q", title="Invoice amount"),
    tooltip=["date:T", "amount:Q"],
)
marks = pd.DataFrame({"date": [pd.Timestamp(as_of), ct["date"].max() + pd.Timedelta(days=threshold)],
                      "label": ["As of", "Dormancy point"]})
lines = alt.Chart(marks).mark_rule(strokeDash=[4, 4]).encode(
    x="date:T", color=alt.Color("label:N", scale=alt.Scale(range=["#1c2430", "#b03a2e"]), title=None)
)
st.altair_chart((bars + lines).properties(height=240), width="stretch")
r = rfm.loc[cust]
st.caption(f"{cust}: recency {r.recency} days · frequency {r.frequency} · monetary {r.monetary:,.2f} · **{r.status}**")

st.subheader("Future dormancy risk")
st.caption("Predicted risk for currently active customers will appear here once a model has been trained and evaluated. Not available yet.")
p = st.columns(4)
for col, (t, d) in zip(p, [
    ("Dormancy prediction", "Forecast which active customers will cross the threshold in the next period."),
    ("Model comparison", "Compare simple RFM rules against statistical / ML models."),
    ("Calibration", "Check whether predicted risks match observed dormancy rates."),
    ("Alert-timing evaluation", "Measure how early customers are flagged before crossing the threshold."),
]):
    with col.container(border=True):
        st.markdown(f"**{t}** · :orange-badge[Planned]")
        st.caption(d)
