"""Weekly customer snapshots with dormancy outcome labels, built from processed/invoices.csv.gz.

Usage: python build_snapshots.py   (pandas only; run prepare_data.py first if processed/ is missing)

Outputs:
  data/weekly_snapshots.csv.gz   one row per (customer, scoring date); gitignored, rebuild with this script
  processed/snapshot_audit.md    methodology and audit counts; do not edit by hand

Definitions (all dates are calendar days; a scoring date means end of that day):
  - Recency on day t = t - (day of last valid purchase on or before t), as in results_page.py.
  - Active at s: at least one purchase on or before s and recency < D.
  - Crossing: the first day c > s with recency == D, i.e. c = last purchase day + D with no purchase in between.
    A purchase on day c itself resets the clock (recency is 0 that day). Later returns do not undo a crossing.
  - Label for horizon h: 1 if the crossing falls in (s, s + h], else 0.
    Censored (empty) when min(crossing, s + h) > LAST_COMPLETE: the outcome needs days we do not fully observe.
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = 90                                    # dormancy threshold, days
HORIZONS = (30, 60, 90)
LAST_COMPLETE = pd.Timestamp("2011-12-08")  # data stop at 12:50 on 9 Dec 2011; 8 Dec is the last full day
STEP = 7                                  # weekly scoring dates, anchored on LAST_COMPLETE (Thursdays)
# Calendar split on the scoring date s, applied per horizon h so every label is fully known before the next period:
#   train: s + h <= TRAIN_END,  validation: s > TRAIN_END and s + h <= VAL_END,  test: s > VAL_END and s + h <= LAST_COMPLETE
# Chosen in train_baselines.py (see processed/baseline_results.md for the comparison of candidates).
TRAIN_END, VAL_END = pd.Timestamp("2010-12-31"), pd.Timestamp("2011-06-30")
OUT = Path("data/weekly_snapshots.csv.gz")
REPORT = Path("processed/snapshot_audit.md")


def load():
    return pd.read_csv("processed/invoices.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])


def scoring_dates(inv, d=D):
    """Weekly dates ending on LAST_COMPLETE; the first is at least d days after the data start (as in the app)."""
    first = inv["date"].min().normalize() + pd.Timedelta(days=d)
    n = (LAST_COMPLETE - first).days // STEP + 1
    return pd.date_range(end=LAST_COMPLETE, periods=n, freq=f"{STEP}D")


def next_crossing(days, d=D):
    """For sorted unique purchase days, the crossing day reached after each purchase if no gap was bridged:
    walk forward while the next purchase lands on or before current + d."""
    out = np.empty(len(days), dtype="datetime64[ns]")
    nxt = None
    for k in range(len(days) - 1, -1, -1):
        if k == len(days) - 1 or days[k + 1] - days[k] > np.timedelta64(d, "D"):
            nxt = days[k] + np.timedelta64(d, "D")
        out[k] = nxt
    return out


def build(inv, dates, d=D, horizons=HORIZONS, last_complete=LAST_COMPLETE):
    inv = inv.sort_values(["customer", "date"])
    rows = []
    for cust, g in inv.groupby("customer", sort=True):
        t = g["date"].to_numpy()
        day = g["date"].dt.normalize().to_numpy()
        cum_amt = g["amount"].cumsum().round(2).to_numpy()
        pdays = np.unique(day)
        cross = next_crossing(pdays, d)
        s = dates.to_numpy()
        n_inv = np.searchsorted(t, s + np.timedelta64(1, "D"), side="left")  # invoices before end of day s
        k = np.searchsorted(pdays, s, side="right") - 1                      # last purchase day index <= s
        ok = k >= 0
        rec = np.where(ok, (s - pdays[np.maximum(k, 0)]) // np.timedelta64(1, "D"), -1)
        ok &= rec < d
        for i in np.flatnonzero(ok):
            rows.append((cust, s[i], rec[i], (s[i] - pdays[0]) // np.timedelta64(1, "D"),
                         n_inv[i], cum_amt[n_inv[i] - 1], pdays[k[i]], cross[k[i]]))
    df = pd.DataFrame(rows, columns=["customer", "scoring_date", "days_since_last", "tenure_days",
                                     "frequency", "monetary", "last_purchase", "crossing_date"])
    for h in horizons:
        end = df["scoring_date"] + pd.Timedelta(days=h)
        lab = (df["crossing_date"] <= end).astype("Int8")
        lab[np.minimum(df["crossing_date"], end) > last_complete] = pd.NA
        df[f"y_{h}"] = lab
    return df


def split(scoring_date, h, train_end=TRAIN_END, val_end=VAL_END, last_complete=LAST_COMPLETE):
    s, end = scoring_date, scoring_date + pd.Timedelta(days=h)
    return np.select([end <= train_end, (s > train_end) & (end <= val_end), (s > val_end) & (end <= last_complete)],
                     ["train", "validation", "test"], "unused")


def check_output(df, inv):
    assert not df.duplicated(["customer", "scoring_date"]).any(), "duplicate customer-date rows"
    assert (df["last_purchase"] <= df["scoring_date"]).all() and (df["crossing_date"] > df["scoring_date"]).all()
    assert (df["days_since_last"].between(0, D - 1)).all() and (df["tenure_days"] >= df["days_since_last"]).all()
    assert (df["scoring_date"] <= LAST_COMPLETE).all() and (df["frequency"] >= 1).all() and (df["monetary"] > 0).all()
    assert df["customer"].map(type).eq(str).all()
    first = inv.groupby("customer")["date"].min().dt.normalize()
    assert (df["last_purchase"] >= df["customer"].map(first)).all()


def report(df, inv):
    L = ["# Weekly snapshot and outcome-label audit", "",
         "Generated by `build_snapshots.py` from `processed/invoices.csv.gz`. Do not edit by hand.", "",
         "## Method", "",
         f"- Dormancy threshold D = {D} days. Scoring dates are end-of-day, every {STEP} days, ending on "
         f"{LAST_COMPLETE:%Y-%m-%d} (Thursdays), starting {df['scoring_date'].min():%Y-%m-%d} (first date with D days of history, as in the dashboard).",
         f"- Population at s: customers with a valid purchase on or before s and recency < {D} (still active).",
         "- Features use only invoices dated on or before the end of s: `days_since_last` (calendar days, as the dashboard), "
         "`tenure_days` (s minus first purchase day; left-truncated at 1 Dec 2009), `frequency` (distinct invoices), "
         "`monetary` (gross £, sum of rounded invoice amounts).",
         f"- `crossing_date` = the first day after s on which recency reaches {D}: last purchase day + {D}, moved forward by every "
         f"purchase on or before the pending crossing day. A purchase on the crossing day resets the clock. A later return does not undo the event.",
         f"- `y_h` = 1 if the crossing falls in (s, s+h]. Empty (censored) when min(crossing, s+h) is after {LAST_COMPLETE:%Y-%m-%d}: "
         f"9 Dec 2011 is partial (purchases stop at 12:50); 8 Dec has a normal volume and closes at 20:01.",
         f"- Split on s, per horizon h: train s+h ≤ {TRAIN_END:%Y-%m-%d}; validation s > {TRAIN_END:%Y-%m-%d} and s+h ≤ {VAL_END:%Y-%m-%d}; "
         f"test s > {VAL_END:%Y-%m-%d} and s+h ≤ {LAST_COMPLETE:%Y-%m-%d}. A label's outcome window therefore never passes its period's cutoff, "
         "and scoring dates whose window straddles a cutoff are left unused (the gap grows with h).", "",
         "## Totals", "",
         f"- {len(df):,} snapshot rows, {df['customer'].nunique():,} customers, {df['scoring_date'].nunique()} scoring dates "
         f"({df['scoring_date'].min():%Y-%m-%d} to {df['scoring_date'].max():%Y-%m-%d}).",
         f"- Customers in invoices but never active on a scoring date: {inv['customer'].nunique() - df['customer'].nunique():,}.",
         f"- Active customers per scoring date: min {df.groupby('scoring_date').size().min():,}, "
         f"median {df.groupby('scoring_date').size().median():,.0f}, max {df.groupby('scoring_date').size().max():,}.", "",
         "## By horizon and split", "",
         "| Horizon | Split | Scoring dates | Snapshots | Events | Event rate | Censored |", "|---|---|---|---:|---:|---:|---:|"]
    for h in HORIZONS:
        sp = split(df["scoring_date"], h)
        for name in ["train", "validation", "test", "unused"]:
            g = df[sp == name]
            y = g[f"y_{h}"]
            rng = f"{g['scoring_date'].min():%Y-%m-%d} – {g['scoring_date'].max():%Y-%m-%d}" if len(g) else "–"
            rate = f"{y.mean():.1%}" if y.notna().any() else "–"
            L.append(f"| {h} | {name} | {rng} | {len(g):,} | {int(y.sum()):,} | {rate} | {int(y.isna().sum()):,} |")
        L.append(f"| {h} | **all** | | {len(df):,} | {int(df[f'y_{h}'].sum()):,} | {df[f'y_{h}'].mean():.1%} | {int(df[f'y_{h}'].isna().sum()):,} |")
    L += ["", "Censored rows fall only in `unused` (the last scoring dates). Unused labelled rows are the price of the "
          "cutoff gaps; they are not leaked into any period.", "", "## Traceable examples", ""]
    for cust, why in EXAMPLES:
        g = df[df["customer"] == cust]
        if g.empty:
            continue
        p = inv[inv["customer"] == cust]["date"].dt.strftime("%Y-%m-%d")
        fmt = {c: g[c].dt.strftime("%Y-%m-%d") for c in ["scoring_date", "last_purchase", "crossing_date"]}
        sample = g.drop(columns="customer").assign(**fmt).iloc[::max(1, len(g) // 8)]
        L += [f"### Customer {cust}: {why}", "", f"Purchase days ({p.nunique()}): {', '.join(sorted(set(p))[:30])}", "",
              "```", sample.to_string(index=False), "```", ""]
    return "\n".join(L)


EXAMPLES = []  # filled in __main__ from the data so the picks are reproducible

if __name__ == "__main__":
    inv = load()
    df = build(inv, scoring_dates(inv))
    check_output(df, inv)
    # Reproducible example picks: a dormancy-then-return customer, a frequent buyer, and a last-purchase customer.
    per = inv.assign(day=inv["date"].dt.normalize()).groupby("customer")["day"]
    gaps = per.apply(lambda s: s.drop_duplicates().diff().dt.days.max())
    EXAMPLES[:] = [(gaps[(gaps > D) & (per.nunique() >= 4)].index[0], "crossed the threshold, then returned"),
                   (gaps[gaps == D].index[0], f"bought exactly on the day recency would reach {D}: clock reset, no event"),
                   ]
    # A customer whose 90-day crossing would fall after LAST_COMPLETE: censored, not negative.
    late = df[df["y_90"].isna() & (df["crossing_date"] > LAST_COMPLETE) & ~df["customer"].isin([c for c, _ in EXAMPLES])]
    EXAMPLES.append((late["customer"].iloc[0], "outcome window runs past the last complete day: censored"))
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    REPORT.write_text(report(df, inv))
    print(REPORT.read_text())
