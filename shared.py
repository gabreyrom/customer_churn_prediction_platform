"""Pieces used by more than one page: the dataset in use, the shared controls and the purchase timeline."""
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from segments import dormancy_date

BUNDLED = {"name": "Online Retail II", "source": "Online Retail II (UCI), cleaned: valid purchases only",
           "currency": "£", "bundled": True}


@st.cache_data
def load_invoices():
    """One row per valid purchase invoice, built by prepare_data.py from the UCI workbook."""
    return pd.read_csv("processed/invoices.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])


def dataset():
    """(invoices, info) for the dataset in use: the file uploaded in this browser session, else the bundled data.
    Uploaded data live only in st.session_state: never cached across sessions, written to disk or sent anywhere."""
    up = st.session_state.get("uploaded")
    return (up["invoices"], up["info"]) if up else (load_invoices(), BUNDLED)


def controls(date_col, threshold_col, data_start, data_end):
    """Observation date and dormancy threshold, shared by every page through st.session_state.
    Returns (as_of, threshold). The earliest date is `threshold` days after the data start, as before."""
    ss = st.session_state
    # Streamlit drops a widget's state when its page is not shown; writing the keys back keeps them across pages.
    ss["threshold"] = ss.get("threshold", 90)
    earliest = data_start + timedelta(days=ss["threshold"])
    ss["as_of"] = ss.get("as_of", data_end)
    if ss["as_of"] < earliest:  # only possible after raising the threshold; say so instead of failing
        st.warning(f"The observation date was moved from {ss['as_of']:%d %b %Y} to {earliest:%d %b %Y}: a "
                   f"{ss['threshold']}-day threshold needs {ss['threshold']} days of history before the date.")
        ss["as_of"] = earliest
    ss["as_of"] = min(ss["as_of"], data_end)  # the Data page resets the date when the dataset changes; this is a guard
    threshold = threshold_col.selectbox("Dormancy threshold (days without a valid purchase)", [60, 90, 120], key="threshold")
    as_of = date_col.date_input("Observation (“as of”) date", min_value=earliest, max_value=data_end, key="as_of",
                                help="Shared with the other page. Only purchases on or before this date are used.")
    return as_of, threshold


def show_timeline(ct, data_start, as_of, threshold, currency="£"):
    """One customer's observed invoices up to `as_of`, plus the conditional dormancy date as text.
    The dormancy date is drawn on the chart only when it is already in the past (on or before `as_of`)."""
    last = ct["date"].max().normalize()
    dormant_on = dormancy_date(last, threshold)
    reached = dormant_on <= pd.Timestamp(as_of)
    bars = alt.Chart(ct).mark_bar(width=3, color="#1f4e79").encode(
        alt.X("date:T", title="Invoice date", scale=alt.Scale(domain=[str(data_start), str(as_of)])),
        alt.Y("amount:Q", title=f"Invoice amount ({currency})" if currency else "Invoice amount"),
        tooltip=["invoice:N", "date:T", "amount:Q"],
    )
    labels = ["As of", "Dormancy threshold reached"]
    marks = pd.DataFrame({"date": [pd.Timestamp(as_of), dormant_on], "label": labels}).iloc[: 2 if reached else 1]
    lines = alt.Chart(marks).mark_rule(strokeDash=[4, 4]).encode(
        x="date:T", color=alt.Color("label:N", scale=alt.Scale(domain=labels[: len(marks)], range=["#1c2430", "#b03a2e"][: len(marks)]), title=None)
    )
    st.altair_chart((bars + lines).properties(height=240), width="stretch")
    rule = f"latest purchase on or before the as-of date, {last:%d %b %Y}, plus the {threshold}-day threshold"
    if reached:
        st.caption(f"**Dormancy threshold reached on {dormant_on:%d %b %Y}** ({rule}), with no purchase since, "
                   "up to the as-of date. Marked on the chart.")
    else:
        st.caption(f"**Dormancy date if there is no further purchase: {dormant_on:%d %b %Y}** ({rule}). "
                   "A calendar rule, not a model prediction; a purchase before then moves the date.")
