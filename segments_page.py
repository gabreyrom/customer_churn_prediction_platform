import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import segments as sg
from shared import SOURCE, controls, load_invoices, show_timeline

# Position on the chart shows recency and frequency, so color only has to carry the spend level.
GROUPS = {"High spend (M high)": "#2a78d6", "Lower spend (M low)": "#1baf7a"}

all_tx = load_invoices()
DATA_START, DATA_END = all_tx["date"].min().date(), all_tx["date"].max().date()
REF_START, REF_END = sg.REFERENCE_DATES.min(), sg.REFERENCE_DATES.max()

st.title("Customer Segments")
st.caption(f"{SOURCE}. RFM lifecycle segments for customers who are active on the chosen observation date.")
st.warning(
    "**Descriptive behavioral groups.** Segments summarize past purchases up to the observation date. They are not "
    "model-generated dormancy probabilities, a segment is not a permanent label, and membership says nothing about "
    "the effect of contacting a customer.",
    icon="ℹ️",
)

c1, c2 = st.columns(2)
as_of, threshold = controls(c1, c2, DATA_START, DATA_END)

if pd.Timestamp(as_of) < REF_END:
    st.error(
        f"**No segments for {as_of:%d %b %Y}.** The cutoffs are learned from purchases up to {REF_END:%d %b %Y} "
        f"(reference period {REF_START:%d %b} – {REF_END:%d %b %Y}). Segmenting an earlier date would compare customers "
        "against purchases made after that date, so segments start on "
        f"{REF_END:%d %b %Y}.\n\nThe observation date has **not** been changed: it is shared with the Customer status "
        f"page, which still works for this date. Choose {REF_END:%d %b %Y} or later above to see segments.",
        icon="🚫",
    )
    st.stop()
cuts = sg.cutoffs(all_tx, as_of, threshold)
if cuts is None:
    st.error(f"Segments are not available: fewer than {sg.MIN_REFERENCE} reference customers to set cutoffs.")
    st.stop()
table = sg.rfm_table(all_tx, as_of)
act = sg.assign(table, cuts, threshold)
summ = sg.summary(act)

st.info(
    f"**As of:** {as_of:%d %b %Y} (purchases after this date are not used) · **Dormancy threshold:** {threshold} days · "
    f"**Reference period:** {REF_START:%d %b} – {REF_END:%d %b %Y} ({len(sg.REFERENCE_DATES)} weekly dates, "
    f"{', '.join(f'{d:%d}' for d in sg.REFERENCE_DATES)} Dec; {cuts['n']:,} customer rows)\n\n"
    f"**Cutoffs (reference medians, fixed for every later date):** high R = last purchase ≤ **{cuts['recency']:g} days** "
    f"ago · high F = ≥ **{cuts['frequency']:g} invoices** in the last {sg.WINDOW} days · high M = ≥ "
    f"**£{cuts['monetary']:,.2f}** in the last {sg.WINDOW} days. A value equal to a cutoff counts as high.\n\n"
    "**Read with care:** the reference dates are all in December, a seasonal peak followed by a quiet week, so the share "
    f"of high-R customers shifts through the year. Customers first observed less than {sg.WINDOW} days ago are compared "
    "on less than a year of invoices and lean towards low F and low M.",
    icon="📅",
)

m = st.columns(4)
m[0].metric("Customers with a purchase so far", f"{len(table):,}")
m[1].metric("Active (segmented)", f"{len(act):,}")
m[2].metric("Dormant (not segmented)", f"{len(table) - len(act):,}")
m[3].metric(f"{sg.NEW} (flag)", f"{int(act['new'].sum()):,}",
            help=f"Active customers whose first observed purchase was less than {sg.NEW_DAYS} days before the "
                 "observation date. A flag on top of the RFM segment, not a segment.")
st.caption(f"Dormant customers (recency ≥ {threshold} days) keep their status on the Customer status page and get no "
           "RFM segment. Every active customer has exactly one of the eight segments. "
           f"**{sg.NEW}** is a separate flag: first observed purchase less than {sg.NEW_DAYS} days ago. The data start on "
           f"{DATA_START:%d %b %Y}, so a first observed purchase may not be the customer's true first purchase.")

st.subheader("Customers by segment")
left, right = st.columns([2, 3])
codes = {v: k for k, v in sg.NAMES.items()}
show = summ.reset_index().assign(code=lambda d: d["segment"].map(codes), pct=lambda d: d["share"] * 100)
with left:
    base = alt.Chart(show).encode(alt.Y("segment:N", sort=sg.ORDER, title=None, axis=alt.Axis(labelLimit=260)),
                                  alt.X("customers:Q", title="Active customers"))
    st.altair_chart((base.mark_bar(color="#1f4e79", cornerRadiusEnd=4, height=16)
                     .encode(tooltip=["segment", "code", "customers", alt.Tooltip("share:Q", format=".1%")])
                     + base.mark_text(align="left", dx=4, color="#1c2430").encode(text=alt.Text("customers:Q", format=",")))
                    .properties(height=300), width="stretch")
with right:
    st.dataframe(
        show[["segment", "code", "customers", "pct", "new", "recency", "frequency", "monetary", "tenure"]],
        hide_index=True, width="stretch",
        column_config={
            "segment": "Segment", "code": st.column_config.TextColumn("R F M", help="H = high, L = low, in the order R, F, M"),
            "customers": "Customers", "pct": st.column_config.NumberColumn("Share", format="%.1f%%"),
            "new": st.column_config.NumberColumn(sg.NEW, help="Customers in the segment carrying the flag"),
            "recency": st.column_config.NumberColumn("Median recency (days)", format="%.0f"),
            "frequency": st.column_config.NumberColumn(f"Median invoices ({sg.WINDOW} d)", format="%.1f"),
            "monetary": st.column_config.NumberColumn(f"Median spend ({sg.WINDOW} d, £)", format="%.0f"),
            "tenure": st.column_config.NumberColumn("Median observed tenure (days)", format="%.0f"),
        },
    )
    small = summ[(summ["customers"] > 0) & (summ["customers"] < 30)].index.tolist()
    st.caption("Typical values are medians over the segment's customers."
               + (f" Fewer than 30 customers, so medians are unstable: {', '.join(small)}." if small else ""))

st.subheader("Recency versus frequency")
rng = np.random.default_rng(0)  # fixed jitter so points with the same whole-number values do not hide each other
pts = act.reset_index().assign(
    group=lambda d: np.where(d["code"].str[2] == "H", *GROUPS),
    flag=lambda d: np.where(d["new"], "Yes", "No"),
    x=lambda d: d["recency"] + rng.uniform(-0.4, 0.4, len(d)),
    y=lambda d: d["frequency"] * np.exp(rng.uniform(-0.1, 0.1, len(d))),
)
scatter = alt.Chart(pts).mark_circle(size=28, opacity=0.55).encode(
    alt.X("x:Q", title="Recency (days since last purchase; left = more recent)", scale=alt.Scale(domain=[-1, threshold], nice=False)),
    alt.Y("y:Q", title=f"Frequency (invoices, last {sg.WINDOW} days; log scale)", scale=alt.Scale(type="log", domain=[0.8, float(pts["y"].max()) * 1.2], nice=False),
          axis=alt.Axis(values=[1, 2, 5, 10, 20, 50, 100, 200])),
    alt.Color("group:N", scale=alt.Scale(domain=list(GROUPS), range=list(GROUPS.values())), title="Spend level"),
    tooltip=[alt.Tooltip("customer:N", title="Customer"), alt.Tooltip("segment:N", title="Segment"),
             alt.Tooltip("code:N", title="R F M"), alt.Tooltip("flag:N", title=sg.NEW), alt.Tooltip("recency:Q", title="Recency (days)"),
             alt.Tooltip("frequency:Q", title=f"Invoices ({sg.WINDOW} d)"),
             alt.Tooltip("monetary:Q", title=f"Spend ({sg.WINDOW} d, £)", format=",.2f"),
             alt.Tooltip("tenure:Q", title="Observed tenure (days)")],
)
rule = lambda enc, v: alt.Chart(pd.DataFrame({"v": [v]})).mark_rule(strokeDash=[4, 4], color="#52514e").encode(**{enc: "v:Q"})
# Lines sit between whole values (wider than the jitter), so every high point is left of / above its line.
f_cut = np.ceil(cuts["frequency"])
st.altair_chart((scatter + rule("x", cuts["recency"] + 0.5) + rule("y", np.sqrt(f_cut * max(f_cut - 1, 0.7))))
                .properties(height=420).interactive(), width="stretch")
st.caption(
    f"One point per active customer ({len(act):,}), slightly jittered. Dashed lines are the R and F cutoffs: "
    "**left** of the vertical line is high R, **above** the horizontal line is high F. Color shows high or low spend (M), so position and "
    f"color together give the full segment. Hover for the customer, segment, {sg.NEW.lower()} flag and values; scroll to zoom."
)

st.subheader("Inspect a segment")
seg = st.selectbox("Segment", [s for s in sg.ORDER if summ.loc[s, "customers"] > 0],
                   format_func=lambda s: f"{s} ({summ.loc[s, 'customers']:,})")
st.caption(f"**{seg}** – {sg.DESCRIPTIONS[codes[seg]]} "
           "Sorted by spend. Click a row to see that customer's purchases.")
members = act[act["segment"] == seg].sort_values("monetary", ascending=False)
picked = st.dataframe(members[["code", "new", "recency", "frequency", "monetary", "tenure"]], width="stretch", height=280,
                      column_config={"new": st.column_config.CheckboxColumn(sg.NEW)},
                      on_select="rerun", selection_mode="single-row", key=f"members-{seg}-{as_of}-{threshold}")
cust = members.index[(picked.selection.rows or [0])[0]]
r = members.loc[cust]
st.markdown(f"**Customer {cust} purchase timeline** (up to {as_of:%d %b %Y})")
ct = all_tx[(all_tx["customer"] == cust) & (all_tx["date"] < pd.Timestamp(as_of) + pd.Timedelta(days=1))]
show_timeline(ct, DATA_START, as_of, threshold)
st.caption(f"Customer {cust}: recency {r.recency} days · {r.frequency} invoices and £{r.monetary:,.2f} in the last "
           f"{sg.WINDOW} days · first observed purchase {r.tenure} days before the as-of date · **{r.segment}** ({r.code})"
           + (f" · {sg.NEW} (flag)" if r.new else ""))

with st.expander("Segment definitions and limitations"):
    st.markdown(
        f"""
**Recency** – days from the last valid purchase to the observation date. **Frequency** – distinct valid invoices in the
{sg.WINDOW} days up to the observation date. **Monetary** – gross spend (£) on those invoices; returns are not netted.
The Customer status page shows frequency and spend over the *whole* observed history; here a fixed {sg.WINDOW}-day
window is used so the same cutoff means the same thing on every date.

| Segment | R (recent) | F (frequent) | M (spend) | Description |
|---|---|---|---|---|
"""
        + "\n".join(f"| {name} | {' | '.join('High' if c == 'H' else 'Low' for c in code)} | {sg.DESCRIPTIONS[code]} |"
                    for code, name in sg.NAMES.items())
        + f"""

- **Cutoffs** are medians among customers who were active, and first observed at least {sg.REFERENCE_MIN_TENURE} days
  earlier, on {len(sg.REFERENCE_DATES)} weekly dates in December 2010, the first dates with a full {sg.WINDOW}-day window.
  They do not change with the observation date.
- **{sg.NEW}** is a flag, not a segment: first observed purchase less than {sg.NEW_DAYS} days before the observation
  date, with at least {sg.MIN_PRIOR} days of data before that purchase. Flagged customers keep their RFM segment. The
  first *observed* purchase may not be the customer's true first purchase: the data start on {DATA_START:%d %b %Y}.
- **Shorter history.** Customers first seen less than {sg.WINDOW} days ago are compared on less than a year of invoices,
  so they lean towards low F and low M.
- **Seasonality.** The reference dates are all in December: heavy buying early in the month, very little in the last
  week. The recency cutoff reflects that, and the share of high-R customers changes with the time of year.
- Small segments have unstable medians.
- Later dormancy rates by segment are reported offline in `processed/segment_evaluation.md`. They are not shown here
  because they would not be known on the observation date.
"""
    )
