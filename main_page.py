import streamlit as st

st.title("Protect Your Customer Lifecycle")
st.markdown(
    "#### Turn purchase history into a current view of your customers. See who is engaged or dormant, "
    "spot active customers who may be drifting away, and prioritize the customers who need attention."
)
st.write("")

left, right = st.columns([3, 2], gap="large")
with left, st.container(border=True):
    st.subheader("Upload transaction data")
    st.file_uploader("Transaction file (CSV or Excel)", type=["csv", "xlsx"], disabled=True)
    st.caption(
        "**Not yet available.** Uploaded files are not processed in this version. "
        "Use the demo to preview the results view with illustrative mock data."
    )
with right, st.container(border=True):
    st.subheader("Try the demo")
    st.write("Preview the customer status view with illustrative mock data.")
    if st.button("Open results with demo data", type="primary", width="stretch"):
        st.switch_page("results_page.py")
    with st.popover("Explore continuous monitoring", width="stretch"):
        st.markdown(
            "**Continuous monitoring (future concept, not implemented)**\n\n"
            "A future integration could connect to a sales system, refresh transaction data on a schedule, "
            "and update customer status and predictions automatically. "
            "This integration is outside the current scope and is not available."
        )

st.write("")
st.subheader("What you'll see")
items = [
    ("Current customer status", "Engaged, approaching dormancy, or dormant as of a chosen date.", ""),
    ("Activity summaries", "Recency, frequency, and spend for each customer.", ""),
    ("Segment counts", "How many customers fall into each status.", ""),
    ("Purchase histories", "A timeline of each customer's purchases.", ""),
    ("Dormancy risk for active customers", "Likelihood that a currently active customer becomes dormant.", "Planned"),
    ("Ranked list for your capacity N", "Top N customers to contact, limited to your outreach capacity.", ""),
]
for row in (items[:3], items[3:]):
    for col, (title, text, tag) in zip(st.columns(3), row):
        with col.container(border=True):
            st.markdown(f"**{title}**" + (f" · :orange-badge[{tag}]" if tag else ""))
            st.caption(text)

st.write("")
st.info(
    "**About your data.** Results reflect only the customers and time period represented in your data. "
    "For a useful view, provide recent transaction records and enough purchase history to observe repeat "
    "behavior. Refresh the data regularly.",
    icon="ℹ️",
)
