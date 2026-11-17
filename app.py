import streamlit as st

st.set_page_config(page_title="Customer Lifecycle", layout="wide")
# Streamlit forgets a widget's value when a page without that widget is shown. Writing the shared observation date
# and threshold back on every run keeps them while the user visits the Data, Alert timing or Model sensitivity pages.
for key in ("as_of", "threshold"):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
up = st.session_state.get("uploaded")
st.sidebar.info(
    (f"Data: **{up['info']['name']}** (uploaded, this session only)." if up
     else "Data: **Online Retail II** (UCI), Dec 2009 – Dec 2011.")
    + " Status and RFM segments are descriptive. Alert timing and model sensitivity are offline, exploratory analyses "
      "of Online Retail II; no customer is scored."
)
st.navigation([
    st.Page("main_page.py", title="Home", default=True),
    st.Page("results_page.py", title="Customer status"),
    st.Page("segments_page.py", title="Customer segments"),
    st.Page("alerts_page.py", title="Alert timing"),
    st.Page("sensitivity_page.py", title="Model sensitivity"),
    st.Page("data_page.py", title="Data"),
]).run()
