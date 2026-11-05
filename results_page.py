from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

SOURCE = "Online Retail II (UCI), cleaned: valid purchases only"
WARNING_WINDOW = 30  # days before the threshold that count as "approaching"
COLORS = {"Engaged": "#4a7fb0", "Approaching dormancy": "#d9a441", "Dormant": "#9a9a9a"}


@st.cache_data
def load_invoices():
    """One row per valid purchase invoice, built by prepare_data.py from the UCI workbook."""
    return pd.read_csv("processed/invoices.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])


def classify(recency, threshold):
    if recency >= threshold:
        return "Dormant"
    if recency >= threshold - WARNING_WINDOW:
        return "Approaching dormancy"
    return "Engaged"


all_tx = load_invoices()
DATA_START, DATA_END = all_tx["date"].min().date(), all_tx["date"].max().date()

st.title("Customer Status")
st.caption(f"{SOURCE}. Status is relative to the chosen “as of” date inside the dataset, not to today.")

c1, c2, c3 = st.columns(3)
threshold = c2.selectbox("Dormancy threshold (days without a valid purchase)", [60, 90, 120], index=1)
as_of = c1.date_input("“As of” observation date", DATA_END,
                      min_value=DATA_START + timedelta(days=threshold), max_value=DATA_END)
top_n = c3.slider("Contact-list size N", 5, 30, 10)

st.info(
    f"**Dataset history:** {DATA_START:%d %b %Y} – {DATA_END:%d %b %Y} · **As of:** {as_of:%d %b %Y}. "
    f"Only purchases on or before the as-of date are used. The final dataset day ({DATA_END:%d %b %Y}) is partial: "
    "purchases stop at midday.",
    icon="📅",
)

tx = all_tx[all_tx["date"] < pd.Timestamp(as_of) + pd.Timedelta(days=1)]
rfm = tx.groupby("customer").agg(last=("date", "max"), frequency=("invoice", "nunique"), monetary=("amount", "sum"))
rfm["recency"] = (pd.Timestamp(as_of) - rfm["last"].dt.normalize()).dt.days
rfm["status"] = rfm["recency"].apply(classify, threshold=threshold)
rfm["monetary"] = rfm["monetary"].round(2)

with st.expander("Definitions and provisional segment rules", expanded=True):
    d1, d2 = st.columns(2)
    d1.markdown(
        """
**Recency** – days from the last valid purchase to the “as of” date.
**Frequency** – number of distinct valid invoices up to the “as of” date.
**Monetary** – gross spend on valid invoices up to the “as of” date (£, returns not netted).
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

st.subheader(f"Customer status as of {as_of:%d %b %Y}")
st.caption("Status describes where each customer stands on the as-of date, based on recency. It is not a forecast.")
counts = rfm["status"].value_counts()
m = st.columns(4)
m[0].metric("Customers with a purchase so far", f"{len(rfm):,}")
for col, s in zip(m[1:], COLORS):
    col.metric(s, f"{int(counts.get(s, 0)):,}")

st.subheader("Recency distribution")
color = alt.Color("status:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), title="Status")
hist = alt.Chart(rfm.reset_index()).mark_bar().encode(
    alt.X("recency:Q", bin=alt.Bin(step=15), title="Recency (days since last purchase)"),
    alt.Y("count():Q", title="Customers"),
    color,
)
rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(strokeDash=[4, 4], color="#b03a2e").encode(x="x:Q")
st.altair_chart((hist + rule).properties(height=260), width="stretch")
st.caption(f"Dashed line = dormancy threshold ({threshold} days).")

left, right = st.columns(2)
cols = ["recency", "frequency", "monetary", "status"]
with left:
    st.subheader(f"Rule-based watchlist (top {top_n})")
    st.caption(
        f"Active customers approaching the {threshold}-day threshold. Ranked by recency, closest to the threshold "
        "first; ties broken by higher gross spend. A rule-based list, not predicted dormancy risk."
    )
    priority = (rfm[rfm["status"] == "Approaching dormancy"]
                .sort_values(["recency", "monetary"], ascending=False).head(top_n))
    st.dataframe(priority[cols], width="stretch")
    if len(priority) < top_n:
        st.caption(f"Only {len(priority)} customers are currently approaching dormancy.")
with right:
    st.subheader("Already dormant (separate re-activation list)")
    dormant = rfm[rfm["status"] == "Dormant"]
    st.caption(f"{len(dormant):,} customers have already crossed the {threshold}-day threshold.")
    st.dataframe(dormant.sort_values("monetary", ascending=False)[cols], width="stretch")

st.subheader("Selected customer purchase timeline")
default = priority.index[0] if len(priority) else rfm.index[0]
cust = st.selectbox("Customer", rfm.index, index=list(rfm.index).index(default))
ct = tx[tx["customer"] == cust]
bars = alt.Chart(ct).mark_bar(width=3, color="#1f4e79").encode(
    alt.X("date:T", title="Invoice date", scale=alt.Scale(domain=[str(DATA_START), str(as_of)])),
    alt.Y("amount:Q", title="Invoice amount (£)"),
    tooltip=["invoice:N", "date:T", "amount:Q"],
)
marks = pd.DataFrame({"date": [pd.Timestamp(as_of), ct["date"].max().normalize() + pd.Timedelta(days=threshold)],
                      "label": ["As of", "Dormancy point"]})
lines = alt.Chart(marks).mark_rule(strokeDash=[4, 4]).encode(
    x="date:T", color=alt.Color("label:N", scale=alt.Scale(range=["#1c2430", "#b03a2e"]), title=None)
)
st.altair_chart((bars + lines).properties(height=240), width="stretch")
r = rfm.loc[cust]
st.caption(f"Customer {cust}: recency {r.recency} days · frequency {r.frequency} · monetary £{r.monetary:,.2f} · **{r.status}**")

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
