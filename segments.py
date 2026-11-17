"""RFM lifecycle segments for customers who are active on an observation date. Pandas only; used by segments_page.py.

Usage: python segments.py   (offline; also needs numpy via build_snapshots.py) -> processed/segment_evaluation.md

Definitions (an observation date means the end of that calendar day; nothing later is read):
  - recency    days from the last valid purchase day to the observation date (whole history, as on the status page).
  - frequency  distinct valid invoices in the trailing WINDOW days (as_of - WINDOW, as_of].
  - monetary   gross amount of those invoices (returns not netted, as in prepare_data.py).
  - tenure     days from the first *observed* purchase day to the observation date. The data start on 1 Dec 2009,
               so a "first" purchase may be a returning customer's.
  Frequency and monetary use a fixed trailing window, not the whole history, because cumulative counts grow with the
  amount of history observed (processed/model_evaluation.md, section 6), so one cutoff would mean different things
  on different dates.
  - Cutoffs: the median recency, frequency and monetary value of the reference customers: active customers first
    observed at least REFERENCE_MIN_TENURE days earlier (December's many first-time buyers would otherwise pull the
    frequency and spend medians down), pooled over REFERENCE_DATES (the first dates with a complete WINDOW of history, all inside the offline
    training period). They are fixed numbers afterwards; an observation date before the last reference date gets none.
  - High/low: R high = recency <= cutoff (more recent). F, M high = value >= cutoff. A value equal to the cutoff is High.
  - Segment: the name of the R/F/M combination in NAMES, for every active customer. Dormant customers get none.
  - Recently acquired (`new`): a separate flag, never a segment. True when tenure < NEW_DAYS and the data cover at
    least MIN_PRIOR days before the first observed purchase.
  - Dormancy date if there is no further purchase: last purchase day + threshold (the `crossing_date` rule of
    build_snapshots.py). A calendar rule, not a prediction.
"""
import pandas as pd

WINDOW = 365        # days of history behind frequency and monetary
NEW_DAYS = 30       # tenure below this = recently acquired (flag)
MIN_PRIOR = 270     # days of data needed before a first observed purchase to call it an acquisition
REFERENCE_MIN_TENURE = 90  # reference customers need this much observed tenure; unrelated to the NEW_DAYS flag
MIN_REFERENCE = 100  # fewer pooled reference rows than this: no cutoffs, no segments
REFERENCE_COUNT = 5  # weekly reference dates


def reference_dates(data_start):
    """The first REFERENCE_COUNT weekly dates on which a full WINDOW of history exists behind every purchase counted:
    data start + WINDOW + 1 days, then weekly. The same rule for any dataset; nothing later than these dates is used."""
    return pd.date_range(pd.Timestamp(data_start).normalize() + pd.Timedelta(days=WINDOW + 1), periods=REFERENCE_COUNT, freq="7D")


REFERENCE_DATES = reference_dates("2009-12-01")  # Online Retail II: Thursdays 2 – 30 Dec 2010
NEW = "Recently acquired"
NAMES = {  # (R, F, M) high/low -> name
    "HHH": "Core customers",
    "HHL": "Frequent, lower spend",
    "HLH": "Infrequent, high spend",
    "HLL": "Recent light buyers",
    "LHH": "Core, less recent",
    "LHL": "Frequent lower spend, less recent",
    "LLH": "Infrequent high spend, less recent",
    "LLL": "Low activity",
}
DESCRIPTIONS = {
    "HHH": "Bought recently, often, and spent a lot over the last year.",
    "HHL": "Bought recently and often, with below-median spend.",
    "HLH": "Bought recently; few invoices but above-median spend (large orders).",
    "HLL": "Bought recently, but few invoices and low spend over the last year.",
    "LHH": "Frequent, high-spend customers whose last purchase is older than the reference median.",
    "LHL": "Frequent, lower-spend customers whose last purchase is older than the reference median.",
    "LLH": "Few, large orders; last purchase older than the reference median.",
    "LLL": "Still active, but below the reference median on recency, frequency and spend.",
}
ORDER = list(NAMES.values())


def rfm_table(inv, as_of):
    """One row per customer with a valid purchase on or before the end of `as_of`."""
    as_of = pd.Timestamp(as_of).normalize()
    tx = inv[inv["date"] < as_of + pd.Timedelta(days=1)]
    out = tx.assign(day=tx["date"].dt.normalize()).groupby("customer").agg(first=("day", "min"), last=("day", "max"))
    win = tx[tx["date"] >= as_of + pd.Timedelta(days=1 - WINDOW)].groupby("customer")
    out["recency"] = (as_of - out["last"]).dt.days
    out["tenure"] = (as_of - out["first"]).dt.days
    out["frequency"] = win["invoice"].nunique().reindex(out.index, fill_value=0)
    out["monetary"] = win["amount"].sum().reindex(out.index, fill_value=0.0).round(2)
    out["new"] = (out["tenure"] < NEW_DAYS) & ((out["first"] - inv["date"].min().normalize()).dt.days >= MIN_PRIOR)
    return out


def cutoffs(inv, as_of, threshold, reference_dates=REFERENCE_DATES):
    """Reference medians, or None when `as_of` is before the last reference date or the reference is too small."""
    if pd.Timestamp(as_of) < reference_dates.max():
        return None  # would need purchases after the observation date
    ref = pd.concat([rfm_table(inv, d) for d in reference_dates])
    ref = ref[(ref["recency"] < threshold) & (ref["tenure"] >= REFERENCE_MIN_TENURE)]
    if len(ref) < MIN_REFERENCE:
        return None
    return {**ref[["recency", "frequency", "monetary"]].median().to_dict(), "n": len(ref)}


def assign(table, cuts, threshold):
    """Active customers (recency < threshold) with their R/F/M code and exactly one of the eight segments.
    The `new` (recently acquired) flag is carried along unchanged; it never alters the segment."""
    act = table[table["recency"] < threshold].copy()
    hl = lambda high: high.map({True: "H", False: "L"})
    act["code"] = (hl(act["recency"] <= cuts["recency"]) + hl(act["frequency"] >= cuts["frequency"])
                   + hl(act["monetary"] >= cuts["monetary"]))
    act["segment"] = act["code"].map(NAMES)
    assert act["segment"].isin(ORDER).all() and act.index.is_unique, "a customer has no segment or more than one"
    return act


def dormancy_date(last_purchase, threshold):
    """The day recency reaches `threshold` if the customer does not buy again after `last_purchase`."""
    return pd.Timestamp(last_purchase).normalize() + pd.Timedelta(days=threshold)


def summary(act):
    """Count and typical (median) values per segment, in ORDER; empty segments are kept with a zero count."""
    g = act.groupby("segment")
    out = g.agg(customers=("recency", "size"), new=("new", "sum"), recency=("recency", "median"),
                frequency=("frequency", "median"), monetary=("monetary", "median"), tenure=("tenure", "median")).reindex(ORDER)
    out[["customers", "new"]] = out[["customers", "new"]].fillna(0).astype(int)
    out.insert(1, "share", out["customers"] / max(len(act), 1))
    return out


if __name__ == "__main__":  # offline evaluation: later observed dormancy by segment. Not shown on the website.
    from pathlib import Path

    from build_snapshots import LAST_COMPLETE, build, load

    D, H = 90, 90
    DATES = pd.to_datetime(["2011-03-31", "2011-06-30", "2011-09-08"])  # after the reference; s + H <= LAST_COMPLETE
    assert (DATES > REFERENCE_DATES.max()).all() and (DATES + pd.Timedelta(days=H) <= LAST_COMPLETE).all()
    inv = load()
    cuts = cutoffs(inv, DATES[0], D)
    labels = build(inv, DATES, d=D, horizons=(H,)).set_index(["scoring_date", "customer"])[f"y_{H}"]
    rate = lambda y: f"{y.mean():.1%}" if len(y) else "–"
    L = ["# RFM lifecycle segments: counts and later observed dormancy", "",
         "Generated by `segments.py` from `processed/invoices.csv.gz`. Do not edit by hand. Offline only: "
         "the website shows segments but no later outcomes.", "",
         "## Method", "",
         f"- Segment definitions are in the `segments.py` docstring and the README. Dormancy threshold {D} days.",
         f"- **Cutoffs** (medians of {cuts['n']:,} reference rows: active customers first observed at least {REFERENCE_MIN_TENURE} days earlier, pooled over "
         f"{len(REFERENCE_DATES)} weekly dates, {REFERENCE_DATES.min():%d %b} – {REFERENCE_DATES.max():%d %b %Y}): "
         f"recency ≤ {cuts['recency']:g} days = high R; frequency ≥ {cuts['frequency']:g} invoices = high F; "
         f"monetary ≥ £{cuts['monetary']:,.2f} = high M (last {WINDOW} days). Outcomes were not used to set them.",
         f"- **Scoring dates:** {', '.join(f'{d:%d %b %Y}' for d in DATES)}; all later than the reference dates. "
         "Segments use only purchases on or before the scoring date.",
         f"- **Outcome:** the customer first reaches {D} days without a valid purchase within {H} days after the scoring "
         f"date (label `y_{H}` from `build_snapshots.py`). A later return does not undo it.",
         f"- **Follow-up:** every scoring date + {H} days is on or before {LAST_COMPLETE:%d %b %Y}, the last complete day, "
         "so every active customer has a fully observed outcome and none is excluded. Later scoring dates would have "
         "incomplete follow-up and are left out of the outcome table (they appear in the count table only).",
         "- The three outcome windows do not overlap, but the same customer can appear on more than one date. "
         "These dates fall in periods already inspected in the model evaluation; the rates are descriptive.", "",
         "## Later observed dormancy by segment", "",
         "n = active customers in the segment on the scoring date. Rates from fewer than 30 customers are marked *.", "",
         "| Segment | R F M | " + " | ".join(f"{d:%d %b %Y}: n | rate" for d in DATES) + " | All dates: n | rate |",
         "|---|---|" + "---:|" * (2 * len(DATES) + 2)]
    acts = []
    for d in DATES:
        a = assign(rfm_table(inv, d), cuts, D)
        y = labels.loc[d]
        assert set(a.index) == set(y.index) and y.notna().all(), "segment population and labelled population differ"
        acts.append(a.assign(y=y.astype(int), date=d))
    allacts = pd.concat(acts)
    codes = {v: " ".join(k) for k, v in NAMES.items()}
    cell = lambda y: f"{len(y):,} | {rate(y)}{'*' if 0 < len(y) < 30 else ''}"
    for seg in ORDER:
        L.append(f"| {seg} | {codes[seg]} | " + " | ".join(cell(a.loc[a['segment'] == seg, 'y']) for a in acts)
                 + f" | {cell(allacts.loc[allacts['segment'] == seg, 'y'])} |")
    L.append("| **All active** | | " + " | ".join(cell(a["y"]) for a in acts) + f" | {cell(allacts['y'])} |")
    L += ["", f"### {NEW} flag (separate from the segment)", "",
          f"Flagged = first observed purchase less than {NEW_DAYS} days before the scoring date, with at least {MIN_PRIOR} "
          "days of data before it. A flagged customer is also counted in their R/F/M segment above. The first observed "
          "purchase may not be the customer's true first purchase (the data start on 1 Dec 2009).", "",
          "| Flag | " + " | ".join(f"{d:%d %b %Y}: n | rate" for d in DATES) + " | All dates: n | rate |",
          "|---|" + "---:|" * (2 * len(DATES) + 2)]
    for name, flag in [(NEW, True), (f"Not flagged", False)]:
        L.append(f"| {name} | " + " | ".join(cell(a.loc[a["new"] == flag, "y"]) for a in acts)
                 + f" | {cell(allacts.loc[allacts['new'] == flag, 'y'])} |")
    L += ["", "These are observed frequencies for groups, not predicted probabilities for individuals, and they say "
          "nothing about what contacting a customer would change. Low-R segments are closer to the threshold by "
          "construction, so their higher rates largely restate recency. The flag's rate is likewise descriptive: "
          "many first-time buyers in this data are one-off buyers.", "",
          "## Customers per segment by observation date", "",
          "Each active customer receives exactly one of the eight segments (asserted in `assign`). The last date has "
          f"no complete follow-up. The final row counts customers carrying the {NEW} flag; they are already included "
          "in the segment rows.", "",
          "| Segment | " + " | ".join(f"{d:%d %b %Y}" for d in [*DATES, inv['date'].max()]) + " |",
          "|---|" + "---:|" * (len(DATES) + 1)]
    count_dates = [*DATES, inv["date"].max().normalize()]
    sums = [summary(assign(rfm_table(inv, d), cuts, D)) for d in count_dates]
    for seg in ORDER:
        L.append(f"| {seg} | " + " | ".join(f"{s.loc[seg, 'customers']:,} ({s.loc[seg, 'share']:.0%})" for s in sums) + " |")
    L.append("| **All active** | " + " | ".join(f"{s['customers'].sum():,}" for s in sums) + " |")
    L.append(f"| *of which {NEW} (flag)* | " + " | ".join(f"{s['new'].sum():,} ({s['new'].sum() / s['customers'].sum():.0%})" for s in sums) + " |")
    L += ["", "## Why a trailing window: cumulative measures shift with observed history", "",
          f"Medians among active customers first observed at least {REFERENCE_MIN_TENURE} days earlier. A cutoff learned in December 2010 on cumulative "
          "values would label more and more customers High simply because more history is visible.", "",
          f"| Observation date | Recency | Frequency, last {WINDOW} d | Frequency, cumulative | "
          f"Monetary, last {WINDOW} d | Monetary, cumulative |", "|---|---:|---:|---:|---:|---:|"]
    for d in [REFERENCE_DATES.max(), *count_dates]:
        t = rfm_table(inv, d)
        t = t[(t["recency"] < D) & (t["tenure"] >= REFERENCE_MIN_TENURE)]
        cum = (inv[inv["date"] < d + pd.Timedelta(days=1)].groupby("customer")
               .agg(f=("invoice", "nunique"), m=("amount", "sum")).reindex(t.index))
        L.append(f"| {d:%d %b %Y} | {t['recency'].median():g} | {t['frequency'].median():g} | {cum['f'].median():g} | "
                 f"£{t['monetary'].median():,.0f} | £{cum['m'].median():,.0f} |")
    L += ["", "## Limitations", "",
          "- **Reference period.** All reference dates are in December 2010, the only training-period dates with a "
          f"complete {WINDOW}-day window. Recency in particular is seasonal (the retailer is quiet in late December), "
          "so the share of High-R customers moves with the time of year.",
          f"- **Unequal history.** A customer first seen less than {WINDOW} days ago is compared on less than a year of "
          "invoices and is more likely to be Low F / Low M; this includes every flagged customer. The reference customers "
          f"themselves have between {REFERENCE_MIN_TENURE} and about 390 days of observed history, because the data start "
          "only a year before the reference dates. Tenure is "
          "left-truncated at 1 Dec 2009: a first observed purchase may belong to a returning customer, including "
          f"some flagged {NEW}.",
          "- **Window shift is reduced, not removed.** Trailing-window medians still move with season and with the "
          "customer mix (see the last observation date above).",
          "- **Small segments.** Mixed combinations (for example frequent but low spend) hold few customers; their "
          "medians and rates are unstable. Frequency and spend are strongly correlated, so the eight groups are uneven.",
          "- **Carried over:** purchases cancelled on a later day still count; amounts are gross.", ""]
    Path("processed/segment_evaluation.md").write_text("\n".join(L))
    print("\n".join(L))
