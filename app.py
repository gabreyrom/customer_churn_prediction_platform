import streamlit as st

st.set_page_config(page_title="Customer Lifecycle", layout="wide")
st.sidebar.warning("**Illustrative mock data** in the demo. Upload processing and predictive risk are not yet available.")
st.navigation([
    st.Page("main_page.py", title="Main page for now", default=True),
    st.Page("results_page.py", title="Customer status (demo)"),
]).run()
