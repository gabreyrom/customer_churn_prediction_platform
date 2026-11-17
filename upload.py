"""Turn an uploaded transaction table into the canonical invoice table used by the status and segment pages.

Pandas only (no Streamlit); the website's Data page calls these functions and keeps the result in the session.

Canonical invoice table (the format of processed/invoices.csv.gz): one row per valid purchase invoice
  customer  text   customer identifier (kept as text, so leading zeros survive)
  invoice   text   transaction / invoice identifier, unique in the table
  date      time   earliest timestamp of the invoice
  amount    float  gross amount of the invoice, rounded to 2 decimals once

Input: a CSV. The user maps four columns (customer, invoice, date, amount) and optionally a quantity column, in which
case amount = quantity x mapped (unit price) column, and says what a row is (`grain`):
  "line"     one row per invoice line. Rows sharing an invoice identifier are lines of one invoice: amounts are summed
             and the earliest date is kept. Identical rows are kept (two products can look the same on the mapped
             columns, the rule prepare_data.py also follows) and only counted.
  "invoice"  one row per invoice. A repeated identifier is then a duplicate: exact copies are dropped, and identifiers
             repeated with different values are excluded as a whole, because the true amount is unknown.

Rows are excluded, and counted, when (first matching rule): customer missing; invoice missing; date missing or not
readable (the column must use one date format throughout); date after today; amount (or quantity) missing or not a number; amount <= 0. Non-positive lines are returns,
cancellations or free items; as in prepare_data.py they are dropped, never netted against purchases.
Invoices whose rows name more than one customer are excluded as a whole.

The Online Retail II-specific rules of prepare_data.py (sheet overlap, same-day full cancellations, TEST/ADJUST
codes, postage-only invoices) are NOT applied: they need columns and conventions a generic file does not have.
"""
import csv
import io
import warnings

import numpy as np
import pandas as pd

import segments as sg

REQUIRED = ["customer", "invoice", "date", "amount"]
OPTIONAL = ["quantity"]
HINTS = {"customer": ["customer", "client", "cust", "member", "buyer", "account"],
         "invoice": ["invoice", "order", "transaction", "receipt", "basket", "ticket"],
         "date": ["date", "time", "timestamp"],
         "amount": ["amount", "total", "price", "revenue", "sales", "value"],
         "quantity": ["quantity", "qty", "units"]}
MIN_HISTORY_DAYS = 120       # the largest dormancy threshold: every threshold needs that much history before a date
MAX_EXCLUDED_SHARE = 0.5     # more unusable rows than this usually means a wrong column mapping
MAX_UPLOAD_MB = 100          # also set as server.maxUploadSize in .streamlit/config.toml
SEGMENT_HISTORY_DAYS = sg.WINDOW + 1 + 7 * (sg.REFERENCE_COUNT - 1)   # first date on which segments exist


def read_table(data, usecols=None, nrows=None):
    """CSV bytes -> DataFrame of text columns. Delimiter (, ; tab |) and encoding (UTF-8, else Latin-1) are detected."""
    head = data[:65536]
    try:
        text, encoding = head.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        text, encoding = head.decode("latin-1"), "latin-1"
    try:
        sep = csv.Sniffer().sniff("\n".join(text.splitlines()[:20]), delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        return pd.read_csv(io.BytesIO(data), sep=sep, encoding=encoding, dtype=str, usecols=usecols, nrows=nrows)
    except UnicodeDecodeError:   # a non-UTF-8 byte after the sampled head
        return pd.read_csv(io.BytesIO(data), sep=sep, encoding="latin-1", dtype=str, usecols=usecols, nrows=nrows)


def guess_mapping(columns):
    """A first guess for each field from the column names; None where nothing matches. Each column is used once."""
    out, used = {}, set()
    for field in ["quantity", *REQUIRED]:        # quantity first, so "Qty" is not taken as the amount
        norm = {c: "".join(ch for ch in c.lower() if ch.isalnum()) for c in columns if c not in used}
        hit = next((c for hint in HINTS[field] for c, n in norm.items() if hint in n), None)
        out[field] = hit
        used.add(hit)
    return out


def _number(s):
    return pd.to_numeric(s.str.replace(r"[£$€\s]", "", regex=True), errors="coerce")


def _dates(s, dayfirst=None):
    """Parse a text column with one consistent format. dayfirst=None tries month-first and day-first and keeps the
    reading that leaves fewer unreadable dates (a wrong order fails on every day above 12). Returns (dates, dayfirst)."""
    def parse(first):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(s, errors="coerce", dayfirst=first).astype("datetime64[ns]")
    if dayfirst is not None:
        return parse(dayfirst), dayfirst
    month, day = parse(False), parse(True)
    return (day, True) if day.isna().sum() < month.isna().sum() else (month, False)


def to_invoices(raw, mapping, dayfirst=None, today=None, grain="line"):
    """Validate and aggregate. Returns {"invoices", "summary", "errors", "warnings"}.
    `errors` non-empty means the data must not be used; `invoices` is then None."""
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    res = {"invoices": None, "summary": {}, "errors": [], "warnings": []}
    cols = {f: mapping.get(f) for f in REQUIRED + OPTIONAL}
    missing = [f for f in REQUIRED if not cols[f]]
    if missing:
        res["errors"].append(f"Choose a column for: {', '.join(missing)}.")
        return res
    chosen = [c for c in cols.values() if c]
    if len(set(chosen)) < len(chosen):
        res["errors"].append("Each field needs its own column; one column is selected more than once.")
        return res
    absent = [c for c in chosen if c not in raw.columns]
    if absent:
        res["errors"].append(f"Column(s) not found in the file: {', '.join(absent)}.")
        return res
    if raw.empty:
        res["errors"].append("The file has no data rows.")
        return res

    text = lambda c: raw[c].astype("string").str.strip().replace("", pd.NA)
    dates, dayfirst = _dates(text(cols["date"]), dayfirst)
    df = pd.DataFrame({"customer": text(cols["customer"]).str.removesuffix(".0"), "invoice": text(cols["invoice"]),
                       "date": dates, "amount": _number(text(cols["amount"])).astype(float)})
    if cols["quantity"]:
        df["amount"] = df["amount"] * _number(text(cols["quantity"]))
    rules = [("Customer identifier missing", df["customer"].isna()),
             ("Invoice identifier missing", df["invoice"].isna()),
             ("Date missing or not readable", df["date"].isna()),
             (f"Date after today ({today:%d %b %Y})", df["date"] >= today + pd.Timedelta(days=1)),
             ("Amount missing or not a number", df["amount"].isna()),
             ("Amount zero or negative (return, cancellation or free item)", df["amount"] <= 0)]
    drop, excluded = pd.Series(False, index=df.index), {}
    for name, mask in rules:
        excluded[name] = int((mask & ~drop).sum())
        drop |= mask.fillna(False)
    valid = df[~drop]
    identical = valid.duplicated()
    if grain == "invoice":
        excluded["Exact copy of an earlier invoice row"] = int(identical.sum())
        valid = valid[~identical]
        repeated = valid["invoice"].duplicated(keep=False)
        excluded["Invoice identifier repeated with different values"] = int(repeated.sum())
        valid = valid[~repeated]
    owners = valid.groupby("invoice")["customer"].nunique()
    conflict = owners.index[owners > 1]
    excluded["Invoice lists more than one customer"] = int(valid["invoice"].isin(conflict).sum())
    valid = valid[~valid["invoice"].isin(conflict)]
    n_excluded = len(df) - len(valid)

    summary = {"rows": len(df), "rows_used": len(valid), "rows_excluded": n_excluded,
               "excluded": {k: v for k, v in excluded.items() if v}, "conflicting_invoices": len(conflict),
               "dayfirst": dayfirst}
    res["summary"] = summary
    if valid.empty:
        res["errors"].append("No valid purchase rows are left after the checks. Check the column mapping"
                             + (" and the date order." if excluded["Date missing or not readable"] else "."))
        return res
    g = valid.groupby("invoice")
    inv = (g.agg(customer=("customer", "first"), date=("date", "min"), amount=("amount", "sum"), lines=("amount", "size"),
                 days=("date", lambda d: d.dt.normalize().nunique()))
           .reset_index().sort_values(["customer", "date"], ignore_index=True))
    inv["amount"] = inv["amount"].round(2)
    start, end = inv["date"].min().normalize(), inv["date"].max().normalize()
    per_customer = inv.groupby("customer").size()
    summary.update(
        invoices=len(inv), customers=int(inv["customer"].nunique()), start=start, end=end, history_days=(end - start).days,
        total_amount=float(inv["amount"].sum()), lines_per_invoice=float(inv["lines"].mean()),
        identical_rows=int(identical.sum()), multi_day_invoices=int((inv["days"] > 1).sum()),
        repeat_customers=float((per_customer > 1).mean()),
        segments_from=start + pd.Timedelta(days=SEGMENT_HISTORY_DAYS) if (end - start).days >= SEGMENT_HISTORY_DAYS else None)

    if n_excluded / len(df) > MAX_EXCLUDED_SHARE:
        res["errors"].append(f"{n_excluded / len(df):.0%} of the rows cannot be used (limit {MAX_EXCLUDED_SHARE:.0%}). "
                             "This usually means a wrong column mapping or date format.")
    if summary["history_days"] < MIN_HISTORY_DAYS:
        res["errors"].append(f"The purchases span {summary['history_days']} days. At least {MIN_HISTORY_DAYS} days are "
                             "needed to tell active from dormant customers at every dormancy threshold.")
    if summary["customers"] < 2:
        res["errors"].append("Only one customer was found. Check the customer column.")
    if n_excluded:
        res["warnings"].append(f"{n_excluded:,} of {len(df):,} rows were excluded (details in the quality summary).")
    if len(conflict):
        res["warnings"].append(f"{len(conflict):,} invoice identifiers appear under more than one customer and were "
                               "excluded. Invoice identifiers must be unique across customers.")
    if summary["multi_day_invoices"]:
        res["warnings"].append(f"{summary['multi_day_invoices']:,} invoices have rows on more than one day; the earliest "
                               "date is used. If identifiers are reused over time, purchases are being merged.")
    if excluded.get("Invoice identifier repeated with different values"):
        res["warnings"].append("Some invoice identifiers appear on several rows with different values and were excluded. "
                               "If the file has one row per invoice line, choose that option instead.")
    if grain == "line" and summary["lines_per_invoice"] == 1 and summary["identical_rows"] == 0:
        pass  # an invoice-level file read as lines: nothing to merge, nothing lost
    elif grain == "line" and summary["identical_rows"] and summary["lines_per_invoice"] < 1.5:
        res["warnings"].append(f"{summary['identical_rows']:,} rows are exact copies of another row, and most invoices "
                               "have a single row. If the file has one row per invoice, choose that option so copies "
                               "are not added up.")
    if summary["segments_from"] is None:
        res["warnings"].append(f"RFM segments need {SEGMENT_HISTORY_DAYS} days of history ({sg.WINDOW} days for frequency "
                               f"and spend, plus the reference period); this file has {summary['history_days']}. "
                               "Customer status will work; the segments page will not.")
    if res["errors"]:
        return res
    res["invoices"] = inv[REQUIRED].astype({"customer": object, "invoice": object, "amount": "float64"})   # same types as the bundled table
    return res


def sample_csv(seed=0, customers=400):
    """A small synthetic line-level file in a different layout from the bundled data (semicolon-separated, day-first
    dates, quantity and unit price, a few unusable rows), for trying the upload. Not real customers."""
    rng = np.random.default_rng(seed)
    start, days = pd.Timestamp("2022-01-03"), 640
    rows, order = [], 50000
    for c in range(customers):
        first, gap = rng.integers(0, days - 3), rng.gamma(2.0, rng.choice([12, 25, 60, 150]))
        t = float(first)
        while t < days and rng.uniform() > 0.04:           # each order has a small chance of being the last
            order += 1
            when = start + pd.Timedelta(days=int(t), hours=int(rng.integers(8, 19)), minutes=int(rng.integers(0, 60)))
            for _ in range(rng.integers(1, 6)):
                rows.append((f"SO-{order}", f"C{c:04d}", when.strftime("%d/%m/%Y %H:%M"), f"SKU{rng.integers(1, 300):03d}",
                             int(rng.integers(1, 12)), round(float(rng.lognormal(1.5, 0.8)), 2)))
            t += rng.exponential(gap) + 1
    df = pd.DataFrame(rows, columns=["Order No", "Client", "Order Date", "Item", "Qty", "Unit Price"])
    bad = rng.choice(len(df), 60, replace=False)
    df.loc[bad[:25], "Client"] = None                      # anonymous sales
    df.loc[bad[25:50], "Qty"] *= -1                        # returns
    df["Order Date"] = df["Order Date"].astype(object)
    df.loc[bad[50:55], "Order Date"] = "unknown"
    df.loc[bad[55:], "Unit Price"] = 0.0                   # free items
    return df.to_csv(index=False, sep=";")


def template_csv():
    """The minimum layout: the four canonical columns, one row per invoice, YYYY-MM-DD dates, positive amounts.
    Synthetic: the valid invoices of `sample_csv`, so it passes every check and has enough history for RFM segments."""
    raw = read_table(sample_csv().encode())
    inv = to_invoices(raw, guess_mapping(list(raw.columns)))["invoices"]
    return inv.assign(date=inv["date"].dt.strftime("%Y-%m-%d")).to_csv(index=False)
