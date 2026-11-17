"""Checks for upload.py and for the pages running on an uploaded dataset.
Run: python test_upload.py (or pytest). The page checks need streamlit."""
import pandas as pd

import segments as sg
import upload

MAP = {"customer": "cust", "invoice": "inv", "date": "when", "amount": "amt", "quantity": None}
TODAY = "2025-01-01"


def table(rows, cols=("cust", "inv", "when", "amt")):
    return pd.DataFrame(rows, columns=list(cols), dtype=str)


def good(n=40, days=200):
    """n customers, each with an invoice on day k and day k + days (two lines each)."""
    d = lambda k: (pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
    return table([(f"{c:03d}", f"i{c}-{j}", d(c % 30 + j * days), "5.5") for c in range(n) for j in (0, 1) for _ in (0, 1)])


def run(raw, **kw):
    return upload.to_invoices(raw, kw.pop("mapping", MAP), today=TODAY, **kw)


def test_lines_are_summed_into_invoices():
    r = run(good())
    inv = r["invoices"]
    assert not r["errors"] and list(inv.columns) == ["customer", "invoice", "date", "amount"]
    assert len(inv) == 80 and inv["invoice"].is_unique and (inv["amount"] == 11.0).all()      # two 5.5 lines each
    assert inv["customer"].iloc[0] == "000" and r["summary"]["lines_per_invoice"] == 2          # leading zeros kept
    assert r["summary"]["history_days"] == 229 and r["summary"]["segments_from"] is None


def test_quantity_times_price_and_earliest_date():
    raw = table([("a", "1", "2024-01-01 10:00", "2.5", "4"), ("a", "1", "2024-01-01 09:00", "1", "3"), ("b", "2", "2024-06-01 08:00", "7", "1")],
                ("cust", "inv", "when", "amt", "qty"))
    inv = run(raw, mapping={**MAP, "quantity": "qty"})["invoices"].set_index("invoice")
    assert inv.loc["1", "amount"] == 13.0 and inv.loc["1", "date"] == pd.Timestamp("2024-01-01 09:00")


def test_row_rules_and_counts():
    raw = pd.concat([good(), table([(None, "x1", "2024-02-01", "5"), ("c", None, "2024-02-01", "5"), ("c", "x3", "not a date", "5"),
                                    ("c", "x4", "2031-01-01", "5"), ("c", "x5", "2024-02-01", "abc"), ("c", "x6", "2024-02-01", "-5"),
                                    ("c", "x7", "2024-02-01", "0"), ("c", "i1-0", "2024-02-01", "5"), ("  ", "x9", "2024-02-01", "5")])])
    r = run(raw)
    ex = r["summary"]["excluded"]
    assert ex == {"Customer identifier missing": 2, "Invoice identifier missing": 1, "Date missing or not readable": 1,
                  "Date after today (01 Jan 2025)": 1, "Amount missing or not a number": 1,
                  "Amount zero or negative (return, cancellation or free item)": 2, "Invoice lists more than one customer": 3}
    assert not r["errors"] and "i1-0" not in set(r["invoices"]["invoice"]) and r["summary"]["conflicting_invoices"] == 1
    assert r["summary"]["rows_used"] + r["summary"]["rows_excluded"] == len(raw) and len(r["invoices"]) == 79


def test_duplicate_invoice_rows():
    raw = pd.concat([good(), good().head(4)])                      # invoices i0-0 and i0-1 appear twice more
    line = run(raw)["invoices"].set_index("invoice")
    assert line.loc["i0-0", "amount"] == 22.0                      # as lines, copies are added up (and counted)
    assert run(raw)["summary"]["identical_rows"] == 84
    one = good().drop_duplicates()                                 # an invoice-level file: one row per invoice
    dup = pd.concat([one, one.head(3), one.tail(1).assign(amt="99")])
    r = run(dup, grain="invoice")
    assert r["summary"]["excluded"] == {"Exact copy of an earlier invoice row": 3, "Invoice identifier repeated with different values": 2}
    assert len(r["invoices"]) == 79 and (r["invoices"]["amount"] == 5.5).all() and any("several rows" in w for w in r["warnings"])


def test_blocking_errors():
    assert "Choose a column for: date" in run(good(), mapping={**MAP, "date": None})["errors"][0]
    assert "more than once" in run(good(), mapping={**MAP, "amount": "inv"})["errors"][0]
    assert "not found" in run(good(), mapping={**MAP, "amount": "nope"})["errors"][0]
    assert "no data rows" in run(good().head(0))["errors"][0]
    short = run(good(days=60))
    assert short["invoices"] is None and "At least 120 days" in short["errors"][0]
    mostly_bad = pd.concat([good(), table([("c", f"z{i}", "2024-02-01", "-1") for i in range(200)])])
    assert "cannot be used" in run(mostly_bad)["errors"][0] and run(mostly_bad)["invoices"] is None
    assert "No valid purchase rows" in run(good().assign(amt="x"))["errors"][0]
    assert "Only one customer" in run(good().assign(cust="same"))["errors"][0]


def test_date_order_and_file_reading():
    rows = [("a", "1", "03/04/2024", "5"), ("a", "2", "25/12/2024", "5"), ("b", "3", "13/01/2024", "5")]
    auto = run(table(rows))["invoices"].set_index("invoice")["date"]
    assert auto["1"] == pd.Timestamp("2024-04-03") and auto["2"] == pd.Timestamp("2024-12-25")       # day first detected
    forced = run(table(rows), dayfirst=False)
    assert forced["summary"]["excluded"]["Date missing or not readable"] == 2                          # never a silent mix
    raw = upload.read_table("Kunde;Beleg;Datum;Betrag\n007;A1;2024-01-01;3,50\n007;A2;2024-07-01;4\n".encode("latin-1"))
    assert list(raw.columns) == ["Kunde", "Beleg", "Datum", "Betrag"] and raw["Kunde"][0] == "007"   # ; detected, text kept
    g = upload.guess_mapping(["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"])
    assert g == {"quantity": "Quantity", "customer": "Customer ID", "invoice": "Invoice", "date": "InvoiceDate", "amount": "Price"}


def test_bundled_table_round_trips():
    b = pd.read_csv("processed/invoices.csv.gz", dtype=str)
    r = upload.to_invoices(b, upload.guess_mapping(b.columns), grain="invoice")
    want = pd.read_csv("processed/invoices.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])
    assert not r["errors"] and not r["warnings"] and len(r["invoices"]) == len(want)
    assert (r["invoices"].to_numpy() == want.astype({"date": "datetime64[ns]"}).to_numpy()).all()
    assert (sg.reference_dates(want["date"].min()) == sg.REFERENCE_DATES).all()      # the general rule gives 2–30 Dec 2010


def sample():
    raw = upload.read_table(upload.sample_csv().encode())
    r = upload.to_invoices(raw, upload.guess_mapping(raw.columns))
    assert not r["errors"] and r["summary"]["excluded"] == {"Customer identifier missing": 25, "Date missing or not readable": 5,
                                                            "Amount zero or negative (return, cancellation or free item)": 30}
    return r


def test_sample_segments_use_its_own_reference_period():
    r = sample()
    inv = r["invoices"]
    refs = sg.reference_dates(inv["date"].min())
    assert refs.max() == r["summary"]["segments_from"] and refs.min() > inv["date"].min() + pd.Timedelta(days=sg.WINDOW)
    as_of = inv["date"].max().normalize()
    cuts = sg.cutoffs(inv, as_of, 90, refs)
    bundled = pd.read_csv("processed/invoices.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])
    assert cuts != sg.cutoffs(bundled, "2011-12-09", 90)                               # not the Online Retail II cutoffs
    assert sg.cutoffs(inv, refs.max() - pd.Timedelta(days=1), 90, refs) is None        # no segments before its reference end
    past = inv[inv["date"] < refs.max() + pd.Timedelta(days=1)]                        # cutoffs need nothing after the reference
    assert sg.cutoffs(past, as_of, 90, refs) == cuts
    mid = refs.max() + pd.Timedelta(days=60)
    cut = inv[inv["date"] < mid + pd.Timedelta(days=1)]
    pd.testing.assert_frame_equal(sg.assign(sg.rfm_table(inv, mid), cuts, 90), sg.assign(sg.rfm_table(cut, mid), cuts, 90))


def test_minimum_template_passes_and_supports_segments():
    data = upload.template_csv().encode()
    raw = upload.read_table(data)
    assert list(raw.columns) == upload.REQUIRED                                   # only the four required columns
    assert raw["invoice"].is_unique and raw["customer"].nunique() > 300
    assert raw["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all() and (raw["amount"].astype(float) > 0).all()
    mapping = upload.guess_mapping(list(raw.columns))
    assert mapping == {"quantity": None, **{f: f for f in upload.REQUIRED}}       # recognised without help
    r = upload.to_invoices(raw, mapping, grain="invoice")
    s = r["summary"]
    assert not r["errors"] and not r["warnings"] and s["rows_excluded"] == 0 and s["invoices"] == len(raw)
    assert s["history_days"] >= upload.SEGMENT_HISTORY_DAYS and s["segments_from"] is not None
    inv = r["invoices"]
    as_of, refs = inv["date"].max(), sg.reference_dates(inv["date"].min())
    for threshold in (60, 90, 120):
        cuts = sg.cutoffs(inv, as_of, threshold, refs)
        assert cuts is not None and cuts["n"] >= sg.MIN_REFERENCE
        assert (sg.summary(sg.assign(sg.rfm_table(inv, as_of), cuts, threshold))["customers"] > 0).all()   # all eight groups
    from streamlit.testing.v1 import AppTest
    info = {"name": "template", "source": "Uploaded file “template”: valid purchases only", "currency": "", "bundled": False}
    for page in ("results_page.py", "segments_page.py"):
        at = AppTest.from_file(page, default_timeout=60)
        at.session_state["uploaded"] = {"invoices": inv, "info": info}
        at.run()
        assert not at.exception and not at.error and at.metric[0].value == f"{inv['customer'].nunique():,}"
    page = AppTest.from_file("data_page.py", default_timeout=60).run()
    text = " ".join(m.value for m in page.markdown)
    for must in ("Upload guidelines", "Total purchase amount of that invoice", "no thousands separator", "One row per invoice line",
                 "120 days", "394 days", "not subtracted", "prediction model is not applied"):
        assert must in text, must


def test_pages_run_on_uploaded_data_and_keep_bundled_default():
    from streamlit.testing.v1 import AppTest
    inv = sample()["invoices"]
    info = {"name": "sample", "source": "Uploaded file “sample”: valid purchases only", "currency": "", "bundled": False}
    for page in ("results_page.py", "segments_page.py"):
        at = AppTest.from_file(page, default_timeout=60)
        at.session_state["uploaded"] = {"invoices": inv, "info": info}
        at.run()
        assert not at.exception and at.metric[0].value == f"{inv['customer'].nunique():,}" and "Uploaded file" in at.caption[0].value
        assert at.date_input[0].value == inv["date"].max().date()
        assert not any("£" in e.value for e in [*at.caption, *at.info])
        default = AppTest.from_file(page, default_timeout=60).run()                    # nothing uploaded: bundled data
        assert default.metric[0].value == "5,851" and "Online Retail II" in default.caption[0].value
    at = AppTest.from_file("segments_page.py", default_timeout=60)
    at.session_state["uploaded"] = {"invoices": inv, "info": info}
    at.run()
    refs = sg.reference_dates(inv["date"].min())
    assert all(f"{d:%d %b %Y}" in at.info[0].value for d in refs) and "Dec 2010" not in at.info[0].value
    for page, words in (("sensitivity_page.py", "trained on Online Retail II"), ("alerts_page.py", "Online Retail II only")):
        at = AppTest.from_file(page, default_timeout=60)
        at.session_state["uploaded"] = {"invoices": inv, "info": info}
        at.run()
        assert not at.exception and any(words in i.value for i in at.info)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all upload checks passed")
