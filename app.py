import streamlit as st

st.set_page_config(page_title="Customer Lifecycle", layout="wide")
st.sidebar.info("Data: **Online Retail II** (UCI), Dec 2009 – Dec 2011. Rule-based status and descriptive RFM segments only; upload and predicted risk are not yet available.")
st.navigation([
    st.Page("main_page.py", title="Home", default=True),
    st.Page("results_page.py", title="Customer status"),
    st.Page("segments_page.py", title="Customer segments"),
]).run()
