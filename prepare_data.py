"""Build the compact invoice-level files used by the website from the raw UCI workbook.

Usage: python prepare_data.py   (needs openpyxl; reads data/online_retail_II.xlsx)

Outputs (processed/):
  invoices.csv.gz       one row per valid purchase invoice: customer, invoice, date, amount
  cancellations.csv.gz  one row per cancellation invoice, kept for later time-aware use
  data_quality.md       counts for every rule below

No rule looks past the end of a purchase's calendar day. The only rule that looks beyond the row itself
removes a purchase that was fully cancelled on the same day (the website uses end-of-day as-of dates).
"""
from collections import Counter
from pathlib import Path

import pandas as pd

RAW = Path("data/online_retail_II.xlsx")
OUT = Path("processed")
SHEET_1, SHEET_2 = "Year 2009-2010", "Year 2010-2011"
EXCLUDED = r"^(?:TEST\d+|ADJUST\d*)$"  # test products and manual adjustments: never purchase evidence
NON_PRODUCT = {"POST", "DOT", "C2", "M", "BANK CHARGES", "D"}  # count toward amount, not toward activity
LINE = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

report = []


def log(rule, df):
    report.append((rule, len(df), df["customer"].nunique(), df["Invoice"].nunique()))


sheets = pd.read_excel(RAW, sheet_name=None, dtype={"Invoice": str, "StockCode": str, "Customer ID": str})
s1, s2 = sheets[SHEET_1], sheets[SHEET_2]

# 1. The first sheet repeats 1-9 Dec 2010, which the second sheet also holds. Keep the second copy.
overlap = s1["InvoiceDate"] >= s2["InvoiceDate"].min().normalize()
key = lambda d: d[LINE].astype(str).sort_values(LINE).reset_index(drop=True)
assert key(s1[overlap]).equals(key(s2[s2["InvoiceDate"] <= s1["InvoiceDate"].max()])), "overlap is not an exact copy"
df = pd.concat([s1[~overlap], s2], ignore_index=True)
df["customer"] = df["Customer ID"].str.removesuffix(".0")
report.append(("Raw rows, both sheets", len(s1) + len(s2), None, None))
report.append(("Removed: sheet-1 copy of 1-9 Dec 2010", int(overlap.sum()), None, None))
log("After overlap removal", df)

cancel = df["Invoice"].str.startswith("C")
rules = [
    ("Missing customer ID", df["customer"].isna()),
    ("Missing invoice date", df["InvoiceDate"].isna()),
    ("Cancellation invoice (C prefix)", cancel),
    ("Quantity <= 0", df["Quantity"] <= 0),
    ("Price <= 0", df["Price"] <= 0),
    ("TEST* / ADJUST* line", df["StockCode"].str.match(EXCLUDED, na=False)),
]
drop = pd.Series(False, index=df.index)
for name, mask in rules:
    log(f"Excluded: {name} (first matching rule)", df[mask & ~drop])
    drop |= mask
valid = df[~drop].assign(amount=lambda d: d["Quantity"] * d["Price"])

# 3. An invoice is activity only if it has at least one product line.
has_product = valid.assign(p=~valid["StockCode"].isin(NON_PRODUCT)).groupby("Invoice")["p"].transform("any")
log("Excluded: invoice with only postage/carriage/manual/fees", valid[~has_product])
valid = valid[has_product]

# 4. Same-day full cancellations. For each customer and day, an invoice is a candidate when every one of its
# (StockCode, Quantity, Price) lines is matched by a same-day cancellation line issued at or after it. Candidates
# are removed only if the day's cancellations cover all of them together; otherwise the day is ambiguous and kept.
items = lambda d: Counter(zip(d["StockCode"], d["Quantity"], d["Price"].round(4)))
canc = df[cancel & df["customer"].notna()].assign(day=lambda d: d["InvoiceDate"].dt.normalize(), Quantity=lambda d: -d["Quantity"])
same_day_removed, ambiguous = [], []
for (cust, day), cd in canc.groupby(["customer", "day"]):
    pd_ = valid[(valid["customer"] == cust) & (valid["InvoiceDate"].dt.normalize() == day)]
    cand = {i: items(g) for i, g in pd_.groupby("Invoice")
            if items(g) <= items(cd[cd["InvoiceDate"] >= g["InvoiceDate"].min()])}
    if cand and sum(cand.values(), Counter()) <= items(cd):
        same_day_removed += list(cand)
    elif cand:
        ambiguous.append((cust, day, sorted(cand)))
reversed_lines = valid[valid["Invoice"].isin(same_day_removed)]
log("Excluded: purchase fully cancelled on the same day", reversed_lines)
valid = valid[~valid["Invoice"].isin(same_day_removed)]
log("Valid purchase lines", valid)


def to_invoices(lines):
    """One row per invoice. Rounding convention: line amounts are summed unrounded, each invoice
    total is rounded to pence once, and every reported total is a sum of these rounded invoice amounts."""
    out = (lines.groupby(["customer", "Invoice"], as_index=False)
           .agg(date=("InvoiceDate", "min"), amount=("amount", "sum"))
           .rename(columns={"Invoice": "invoice"}).sort_values(["customer", "date"]))
    out["amount"] = out["amount"].round(2)
    return out


invoices = to_invoices(valid)
assert invoices["invoice"].is_unique, "an invoice maps to more than one customer"
cancellations = to_invoices(df[cancel & df["customer"].notna()].assign(amount=lambda d: d["Quantity"] * d["Price"]))

OUT.mkdir(exist_ok=True)
invoices.to_csv(OUT / "invoices.csv.gz", index=False)
cancellations.to_csv(OUT / "cancellations.csv.gz", index=False)

# Reported only, never applied: identical lines within a sheet, and same-day full reversals.
dup = valid.duplicated(LINE)
with_dup, without_dup = invoices["amount"].sum(), to_invoices(valid[~dup])["amount"].sum()
rev = to_invoices(reversed_lines)
reversed_big = [f"| {r.invoice} | {r.customer} | {r.date:%Y-%m-%d %H:%M} | {r.amount:,.2f} |" for r in rev.nlargest(5, "amount").itertuples()]

fmt = lambda x: "" if x is None else f"{x:,}"
lines = [
    "# Data quality report: Online Retail II",
    "",
    "Generated by `prepare_data.py` from `data/online_retail_II.xlsx` (UCI). Do not edit by hand.",
    "",
    "## Cleaning rules (applied in order; counts are rows removed by that rule only)",
    "",
    "| Step | Rows | Customers | Invoices |",
    "|---|---:|---:|---:|",
    *[f"| {n} | {fmt(r)} | {fmt(cu)} | {fmt(i)} |" for n, r, cu, i in report],
    "",
    "Overlap check: the removed sheet-1 rows are an exact row-for-row copy of the sheet-2 rows for the same dates.",
    "",
    "## Output",
    "",
    f"- `invoices.csv.gz`: {len(invoices):,} invoices, {invoices['customer'].nunique():,} customers, "
    f"{invoices['date'].min():%Y-%m-%d %H:%M} to {invoices['date'].max():%Y-%m-%d %H:%M}, "
    f"gross amount {invoices['amount'].sum():,.2f}.",
    f"- `cancellations.csv.gz`: {len(cancellations):,} cancellation invoices from {cancellations['customer'].nunique():,} customers, "
    f"total {cancellations['amount'].sum():,.2f}. Not subtracted from anything.",
    f"- Customers with exactly one valid invoice: {(invoices.groupby('customer').size() == 1).sum():,}.",
    "",
    "## Definitions",
    "",
    "- **Valid purchase invoice**: not a C-prefix invoice; customer ID and date present; every counted line has quantity > 0 and price > 0; "
    "at least one product line. TEST*/ADJUST* lines are dropped. Postage, carriage, manual (M), bank charges and discount lines "
    "count toward the amount but cannot make an invoice count on their own.",
    "- **amount**: gross sum of quantity × price over the invoice's counted lines. Returns are not netted off.",
    "- **date**: earliest timestamp on the invoice.",
    "- **Rounding**: line amounts are summed unrounded; each invoice amount is rounded to pence once; "
    "every total in this report is the sum of those rounded invoice amounts.",
    "",
    "## Kept as-is (reported, not removed)",
    "",
    f"- Identical lines within a sheet: {dup.sum():,} rows across {valid.loc[dup, 'Invoice'].nunique():,} invoices. "
    f"Gross amount with them {with_dup:,.2f}, without them {without_dup:,.2f} "
    f"(difference {with_dup - without_dup:,.2f}). They cannot be told apart from genuine repeat lines.",
    "- Purchases matched by a cancellation on a **later** day are **kept**. Removing them would leak the future into earlier "
    "as-of views. Cancellations are stored separately for later time-aware treatment.",
    "",
    "## Same-day full cancellations (removed)",
    "",
    f"{len(rev):,} purchase invoices from {rev['customer'].nunique():,} customers were fully cancelled on the same calendar day "
    f"(line-for-line: StockCode, quantity, price; cancellation at or after the purchase). Gross amount removed: "
    f"{rev['amount'].sum():,.2f}. Customers left with no valid purchase: "
    f"{len(set(rev['customer']) - set(invoices['customer'])):,}. Largest:",
    "",
    "| Invoice | Customer | Date | Amount |",
    "|---|---|---|---:|",
    *reversed_big,
    "",
    f"Ambiguous days kept unchanged ({len(ambiguous)}): every candidate invoice is fully matched on its own, "
    "but the day's cancellations cannot cover all of them, so no single one is removed.",
    "",
    *[f"- customer {c}, {d:%Y-%m-%d}: invoices {', '.join(i)}" for c, d, i in ambiguous],
    "",
]
(OUT / "data_quality.md").write_text("\n".join(lines))
print("\n".join(lines))
