import pandas as pd
import streamlit as st

import upload
from shared import BUNDLED, dataset

ss = st.session_state
inv, info = dataset()

st.title("Data")
st.caption("Choose the purchase data behind the Customer status and Customer segments pages: the bundled "
           "Online Retail II dataset, or your own file for descriptive analysis.")

st.subheader("Dataset in use")
with st.container(border=True):
    st.markdown(f"**{info['name']}** · {inv['customer'].nunique():,} customers · {len(inv):,} invoices · "
                f"{inv['date'].min():%d %b %Y} – {inv['date'].max():%d %b %Y}")
    if info["bundled"]:
        st.caption(BUNDLED["source"] + ". Always available.")
    else:
        st.caption("Uploaded in this browser session. It is held in the app's memory for this session only and is "
                   "gone when you close or reload the page.")
        if st.button("Switch back to Online Retail II"):
            ss.pop("uploaded", None), ss.pop("as_of", None)
            st.rerun()

st.subheader("Upload your own transactions")
st.markdown(
    f"""
- **File:** CSV (comma, semicolon, tab or pipe separated), up to {upload.MAX_UPLOAD_MB} MB.
- **What you get:** customer status, the rule-based watchlist, purchase timelines and RFM segments, computed from your
  file only. Segment cutoffs come from an early reference period **inside your file**, never from Online Retail II.
- **Privacy:** the file is processed in this app's memory for your session. It is not saved to disk, stored in a
  database or sent to another service. On the hosted site that memory is on the hosting provider's server.
"""
)
buttons, guide = st.columns([3, 7], gap="large")
with buttons:
    st.download_button("Download minimum CSV template", upload.template_csv(), "minimum_template.csv", "text/csv",
                       type="primary", width="stretch",
                       help="Synthetic data in the simplest layout: four columns, one row per invoice. "
                            "It passes every check and has enough history for RFM segments.")
    st.caption("Replace the rows with your own, keep the header, and upload.")
    st.download_button("Download a sample file with problems", upload.sample_csv(), "sample_transactions.csv", "text/csv",
                       width="stretch",
                       help="For trying the quality checks. Synthetic data in a different layout: semicolons, day-first "
                            "dates, invoice lines with quantity and unit price, extra columns and a few unusable rows.")
with guide, st.container(border=True):
    st.markdown(
        f"""
**Upload guidelines**

| Column | Meaning | Example |
|---|---|---|
| `customer` | Who bought. Any stable identifier; kept as text | `C0042` |
| `invoice` | One purchase. Unique for each invoice | `SO-50017` |
| `date` | When the purchase was made. One format for the whole column | `2023-06-14` |
| `amount` | **Total purchase amount of that invoice.** Decimal point, no thousands separator, no negative values | `1250.50` |

- **Other column names are fine:** you choose which column is which after uploading. Extra columns are ignored.
- **Invoice lines instead of whole invoices?** Choose *One row per invoice line* under “Each row is”: rows with the same
  invoice identifier are added up into one invoice. If the file has a unit price and a quantity, map both.
  Choose *One row per invoice* only when each invoice appears once; repeated identifiers are then treated as duplicates.
- **History needed:** at least **{upload.MIN_HISTORY_DAYS} days** between the first and last purchase for customer status,
  and at least **{upload.SEGMENT_HISTORY_DAYS} days** (about 13 months) for RFM segments. With less than that, status
  still works and the segments page explains why it cannot.
- **Invalid rows are left out and counted:** missing customer or invoice, unreadable or future date, amount that is not
  a number. If more than {upload.MAX_EXCLUDED_SHARE:.0%} of the rows are left out, the file cannot be used.
- **Cancellations and refunds:** rows with a zero or negative amount are left out. They are **not subtracted** from the
  original purchase, which still counts in full.
- **Descriptive analysis only.** The Online Retail II prediction model is not applied to uploaded data: it was trained on
  one retailer and is not validated even there.
"""
    )

file = st.file_uploader("Transaction file (CSV)", type=["csv"])
if file is None:
    st.stop()
data = file.getvalue()
try:
    preview = upload.read_table(data, nrows=200)
except Exception as e:  # unreadable file: show the reason instead of a traceback
    st.error(f"The file could not be read as CSV: {e}")
    st.stop()
st.caption(f"First rows of **{file.name}** ({len(data) / 1e6:.1f} MB, {len(preview.columns)} columns):")
st.dataframe(preview.head(5), hide_index=True, width="stretch")

st.markdown("**Map the columns**")
guess = upload.guess_mapping(list(preview.columns))
FIELDS = {"customer": "Customer identifier", "invoice": "Invoice / transaction identifier", "date": "Transaction date",
          "amount": "Purchase amount (or unit price)", "quantity": "Quantity (optional)"}
options = ["(none)", *preview.columns]
mapping = {}
for col, (field, label) in zip(st.columns(5), FIELDS.items()):
    pick = col.selectbox(label, options, index=options.index(guess[field]) if guess[field] else 0, key=f"map-{field}-{file.file_id}")
    mapping[field] = None if pick == "(none)" else pick
GRAINS = {"One row per invoice line (rows of an invoice are added up)": "line",
          "One row per invoice (repeated identifiers are duplicates)": "invoice"}
repeats = preview[mapping["invoice"]].duplicated().any() if mapping["invoice"] else True
grain = st.radio("Each row is", list(GRAINS), index=0 if repeats else 1, key=f"grain-{file.file_id}")
c1, c2, c3 = st.columns(3)
ORDERS = {"Detect automatically": None, "Day first (31/12/2024)": True, "Month first (12/31/2024)": False}
order = c1.selectbox("Date order", list(ORDERS), help="Only matters for dates written like 03/04/2024.")
name = c2.text_input("Dataset name", file.name.rsplit(".", 1)[0][:40])
currency = c3.text_input("Currency symbol (optional)", "", max_chars=3)

key = (file.file_id, tuple(mapping.items()), order, grain)
if st.button("Check this file"):
    with st.spinner("Reading and checking the file…"):
        try:
            raw = upload.read_table(data, usecols=[c for c in mapping.values() if c])
            ss["checked"] = {"key": key, "res": upload.to_invoices(raw, mapping, ORDERS[order], grain=GRAINS[grain])}
        except Exception as e:
            ss["checked"] = {"key": key, "res": {"invoices": None, "summary": {}, "warnings": [],
                                                 "errors": [f"The file could not be processed: {e}"]}}
checked = ss.get("checked")
if not checked or checked["key"] != key:
    st.caption("Choose the columns, then check the file. Nothing is used until the checks pass and you confirm.")
    st.stop()

res, s = checked["res"], checked["res"]["summary"]
st.subheader("Quality summary")
for e in res["errors"]:
    st.error(e, icon="🚫")
for w in res["warnings"]:
    st.warning(w, icon="⚠️")
if "invoices" in s:
    m = st.columns(4)
    m[0].metric("Rows read", f"{s['rows']:,}")
    m[1].metric("Rows used", f"{s['rows_used']:,}", f"-{s['rows_excluded']:,} excluded" if s["rows_excluded"] else None, delta_color="off")
    m[2].metric("Invoices", f"{s['invoices']:,}")
    m[3].metric("Customers", f"{s['customers']:,}")
    facts = {
        "Purchase history": f"{s['start']:%d %b %Y} – {s['end']:%d %b %Y} ({s['history_days']:,} days)",
        "Total purchase amount": f"{currency}{s['total_amount']:,.2f}",
        "Rows per invoice (average)": f"{s['lines_per_invoice']:.1f}",
        "Rows identical to another row on the mapped columns": f"{s['identical_rows']:,}"
        + (" (dropped as copies)" if GRAINS[grain] == "invoice" else " (kept: different products can look the same)"),
        "Customers with more than one invoice": f"{s['repeat_customers']:.0%}",
        "RFM segments available from": f"{s['segments_from']:%d %b %Y}" if s["segments_from"] is not None else "not available (history too short)",
        **{f"Excluded: {k}": f"{v:,} rows" for k, v in s["excluded"].items()},
    }
    st.dataframe(pd.DataFrame({"Check": list(facts), "Result": list(facts.values())}), hide_index=True, width="stretch")
elif s:
    st.dataframe(pd.DataFrame({"Check": [f"Excluded: {k}" for k in s["excluded"]], "Result": [f"{v:,} rows" for v in s["excluded"].values()]}),
                 hide_index=True, width="stretch")

if res["errors"]:
    st.info("This file cannot be used yet. Fix the points above (or change the mapping) and check again. "
            f"The dataset in use is still **{info['name']}**.")
    st.stop()
st.success("The checks passed. Only the valid rows will be used.")
if st.button("Use this dataset", type="primary"):
    ss["uploaded"] = {"invoices": res["invoices"],
                      "info": {"name": name or "Uploaded data", "source": f"Uploaded file “{name or file.name}”: valid purchases only",
                               "currency": currency, "bundled": False}}
    ss.pop("as_of", None)       # the shared date restarts at the end of the new dataset; the threshold is kept
    ss.pop("checked", None)
    st.switch_page("results_page.py")
