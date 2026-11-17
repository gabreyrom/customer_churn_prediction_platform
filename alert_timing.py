"""Alert timing (Nov 26): what was knowable about 60, 30 and 7 days before an observed dormancy event. Offline only.

Usage: python build_snapshots.py && python alert_timing.py   (pandas, numpy; evaluate_models.py also needs matplotlib)
Output: processed/alert_timing.md   (do not edit by hand; shown as-is on the website's "Alert timing" page)

Definitions (dormancy threshold D = 90 days, weekly scoring dates and features from build_snapshots.py):
  - Event: a customer's recency first reaches D on day c (`crossing_date`), with c on or before LAST_COMPLETE.
    One event per (customer, c); a customer who returns and lapses again has several.
  - Lead-time match: for a requested lead L, the alert date is the latest weekly scoring date s with s <= c - L.
    Scoring is weekly, so the actual lead c - s is between L and L + 6 days. The row used is the ordinary snapshot row
    of that customer on s: its features and scores use purchases on or before s only. The event date is used only
    afterwards, to pick which row to look at. That is what makes the event-aligned tables retrospective.
  - Recency on the alert date is fixed by the lead: recency = D - (c - s). An event 7 days away means 83 days
    without a purchase; 60 days away means 30. Recency-based alerts therefore cannot separate customers at a given lead.
  - Prospective counterpart ("lead band"): every active customer on s whose recency is in [D - L - 6, D - L], i.e. who
    WOULD cross in L..L+6 days if they do not buy again. Outcome: they did cross on that pending day (last purchase + D).
    A row is used only if the pending day is on or before LAST_COMPLETE, whatever happened; otherwise follow-up is
    incomplete and the row is excluded (keeping only the already-known returns would bias the rate down).
  - Models: fitted on the training period only (evaluate_models.fit_candidates). Everything here uses scoring dates
    after TRAIN_END, so no row was used to fit the coefficients. HORIZON maps each lead to the model horizon that
    covers it. These periods were already inspected in earlier milestones: all numbers are exploratory.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_snapshots import D, LAST_COMPLETE, OUT as SNAPSHOTS, STEP, TRAIN_END

LEADS = (60, 30, 7)
HORIZON = {60: 90, 30: 60, 7: 30}     # smallest model horizon that contains leads L..L+6
CAPACITIES = (50, 100, 200)
WARNING_WINDOW = 30                   # website rule: "Approaching dormancy" when recency >= D - 30
RANKERS = {"website rule (recency, then spend)": ["days_since_last", "monetary"], "rfm model": ["rfm-elig"]}
REPORT = Path("processed/alert_timing.md")


def events(snap, last_complete=LAST_COMPLETE):
    """Observed dormancy events: one row per (customer, crossing_date) with the crossing on or before last_complete."""
    e = snap.loc[snap["crossing_date"] <= last_complete, ["customer", "crossing_date"]]
    return e.drop_duplicates().reset_index(drop=True)


def match(ev, dates, lead):
    """Alert date per event: the latest scoring date at least `lead` days before the crossing (NaT if none)."""
    dates = np.sort(pd.DatetimeIndex(dates).to_numpy())
    k = np.searchsorted(dates, (ev["crossing_date"] - pd.Timedelta(days=lead)).to_numpy(), side="right") - 1
    s = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns]")
    s[k >= 0] = dates[k[k >= 0]]
    return ev.assign(lead=lead, scoring_date=s, actual_lead=(ev["crossing_date"] - s).dt.days.astype("Int64"))


def inclusion(week, key, n):
    """Chance that each row of one scoring date is in the top n when ranked by `key` (descending); rows tied on
    every key at the cut-off share the remaining places equally (the same convention as evaluate_models.captured)."""
    g = week.groupby(key).size().sort_index(ascending=False)
    share = (np.clip(n - (g.cumsum() - g), 0, g) / g).rename("p")
    return week.join(share, on=key)["p"]


def band(snap, lead, d=D, last_complete=LAST_COMPLETE):
    """Prospective rows for a lead: recency in [d - lead - STEP + 1, d - lead], with `pending`, `complete`, `event`."""
    b = snap[snap["days_since_last"].between(d - lead - STEP + 1, d - lead)].copy()
    b["pending"] = b["last_purchase"] + pd.Timedelta(days=d)
    b["complete"] = b["pending"] <= last_complete
    b["event"] = (b["crossing_date"] == b["pending"]).astype(int)
    return b


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    return np.nan if n1 in (0, len(y)) else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def score(snap):
    """Add train-only model scores for each horizon: `recency-elig_h` and `rfm-elig_h` (0 where not eligible)."""
    from evaluate_models import fit_candidates
    for h in sorted(set(HORIZON.values())):
        preds, _ = fit_candidates(snap, h)
        for name in ("recency-elig", "rfm-elig"):
            snap[f"{name}_{h}"] = preds[name](snap)
    return snap


def report(snap):
    pct = lambda v: "–" if pd.isna(v) else f"{v:.1%}"
    ev = events(snap)
    dates = snap["scoring_date"].unique()
    after = snap[snap["scoring_date"] > TRAIN_END]          # out of sample for the coefficients
    weeks = dict(tuple(after.groupby("scoring_date")))
    L = ["# Alert timing: 60, 30 and 7 days before an observed dormancy event (Nov 26)", "",
         "Generated by `alert_timing.py` from `data/weekly_snapshots.csv.gz`. Do not edit by hand. **Offline, exploratory "
         "analysis** of the Online Retail II data. No customer was alerted or contacted; nothing here measures the effect "
         "of an alert.", "",
         "> **Retrospective by construction.** The event date is known only afterwards. On the alert date nobody knew "
         "which customers were 60, 30 or 7 days from dormancy. Section 3 gives the forward-looking counterpart: among "
         "customers who looked the same on that date, how many went on to become dormant.", "",
         "## 1. Method", "",
         f"- **Event:** a customer's recency first reaches {D} days (the existing definition; `crossing_date` in "
         f"`build_snapshots.py`), on or before {LAST_COMPLETE:%d %b %Y}, the last complete day. A later return does not undo it.",
         f"- **Scoring cadence:** the existing weekly scoring dates (Thursdays). **Lead match:** for a requested lead L, the "
         f"alert date is the latest scoring date at least L days before the event, so the **actual lead is L to L + {STEP - 1} days**.",
         "- **No future information in a score.** The row used is the customer's ordinary snapshot on the alert date; "
         "features, rule status and model scores use purchases on or before that date. Weekly top-N lists are ranked "
         "among **all** customers active on that date, not among customers known to have an event.",
         f"- **Recency is fixed by the lead:** recency on the alert date = {D} − actual lead. So at a given lead every "
         "event customer has (almost) the same recency, and any recency-only alert treats them alike.",
         f"- **Period:** alert dates after {TRAIN_END:%d %b %Y}. The models were fitted on earlier dates only "
         f"(train-only coefficients from `evaluate_models.fit_candidates`, no recalibration). Model horizons: "
         + ", ".join(f"lead {l} → {h}-day model" for l, h in HORIZON.items()) + ".",
         "- **Repeated customers:** each event is counted once per lead. A customer with several events contributes "
         "several. The same event appears under all three leads (on three different alert dates).",
         "- **Overlapping windows:** within one lead, an event has exactly one alert date, and in Section 3 a customer's "
         "purchase spell falls in a 7-day recency band on exactly one weekly date, so no row is repeated within a table. "
         "Rows are still not independent: the same customers recur across leads and spells, and no intervals are reported.",
         f"- **Incomplete follow-up:** an event after {LAST_COMPLETE:%d %b %Y} is not observed and cannot be aligned. In "
         "Section 3 a row is used only when its pending dormancy day is on or before that date, whether or not the customer "
         "returned; the rest are excluded and counted.",
         "- **Exploratory.** These dates overlap the validation and test periods inspected on Nov 12 and Nov 19. "
         "No later independent holdout exists in this dataset.", "",
         "## 2. Event-aligned view (retrospective)", "",
         f"Observed events in the data: **{len(ev):,}** from {ev['customer'].nunique():,} customers "
         f"({(ev.groupby('customer').size() > 1).sum():,} customers have more than one).", "",
         "| Requested lead | Events matched | Alert dates | Actual lead (min–max, mean) | Recency on alert date | "
         "Flagged “Approaching dormancy” by the website rule |", "|---|---:|---|---|---|---:|"]
    matched = {}
    for lead in LEADS:
        m = match(ev, dates, lead)
        m = m[m["scoring_date"] > TRAIN_END].merge(after, on=["customer", "scoring_date"], suffixes=("", "_row"))
        assert (m["crossing_date"] == m["crossing_date_row"]).all() and (m["days_since_last"] == D - m["actual_lead"]).all()
        assert m["actual_lead"].between(lead, lead + STEP - 1).all() and not m.duplicated(["customer", "crossing_date"]).any()
        matched[lead] = m
        r = m["days_since_last"]
        L.append(f"| {lead} days | {len(m):,} | {m['scoring_date'].min():%d %b %Y} – {m['scoring_date'].max():%d %b %Y} | "
                 f"{m['actual_lead'].min()}–{m['actual_lead'].max()}, {m['actual_lead'].mean():.1f} | {r.min()}–{r.max()} days | "
                 f"{pct((r >= D - WARNING_WINDOW).mean())} |")
    L += ["", f"Denominator: events whose alert date is after {TRAIN_END:%d %b %Y}. The website rule flags recency ≥ "
          f"{D - WARNING_WINDOW} days, so it is silent 60 days ahead, flags only the events exactly 30 days ahead, and flags "
          "every event 7 days ahead. This follows from the arithmetic, not from the data. (Some leads in a range never "
          "occur because this retailer has no Saturday sales.)", "",
          "### Model risk on the alert date", "",
          "Mean predicted probability of dormancy within the model horizon. “All active” = every customer active on the "
          "same alert dates (the population a real alert would be drawn from), event or not.", "",
          "| Lead | Model horizon | Model | Event customers: mean risk | All active: mean risk | Event customers: median "
          "percentile among that date's active customers |", "|---|---|---|---:|---:|---:|"]
    for lead, m in matched.items():
        h = HORIZON[lead]
        pool = after[after["scoring_date"].isin(m["scoring_date"])]
        for name in ("recency-elig", "rfm-elig"):
            c = f"{name}_{h}"
            rank = pool.groupby("scoring_date")[c].rank(pct=True)
            mr = rank[pool.set_index(["customer", "scoring_date"]).index.isin(m.set_index(["customer", "scoring_date"]).index)]
            L.append(f"| {lead} days | {h} days | {'recency' if name == 'recency-elig' else 'rfm'} | {m[c].mean():.3f} | "
                     f"{pool[c].mean():.3f} | {mr.median():.0%} |")
    L += ["", "Risk is 0 for customers who cannot reach the threshold within the horizon (recency < 90 − horizon), which "
          "lowers the “all active” mean at the 30- and 60-day horizons.", "",
          "### Were event customers on that week's top-N list?", "",
          "For each event, the weekly list is built on its alert date from all active customers, and the event counts as "
          "captured if its customer is in the top N. **Capture = captured events ÷ matched events at that lead.** Ties at "
          "the cut-off share the remaining places. `random` = N ÷ active customers that week.", "",
          "| Lead | Events | Ranker | " + " | ".join(f"Capture @{n}" for n in CAPACITIES) + " |",
          "|---|---:|---|" + "---:|" * len(CAPACITIES)]
    for lead, m in matched.items():
        h = HORIZON[lead]
        idx = m.set_index(["customer", "scoring_date"]).index
        cap = {name: [] for name in ["random", *RANKERS]}
        for n in CAPACITIES:
            got = {name: 0.0 for name in cap}
            for s in m["scoring_date"].unique():
                w = weeks[s]
                hit = w.set_index(["customer", "scoring_date"]).index.isin(idx)
                got["random"] += min(n / len(w), 1) * hit.sum()
                for name, key in RANKERS.items():
                    key = [f"{k}_{h}" if k.endswith("-elig") else k for k in key]
                    got[name] += inclusion(w, key, n)[hit].sum()
            for name in cap:
                cap[name].append(got[name] / len(m))
        for name, vals in cap.items():
            L.append(f"| {lead} days | {len(m):,} | {name} | " + " | ".join(pct(v) for v in vals) + " |")
    L += ["", "- The website rule ranks by recency, so it reaches events only shortly before they happen. Seven days "
          "ahead, event customers have 77–83 days of recency; the list fills first with customers at 84–89 days, which is "
          "why small lists capture little. A recency-only model ranks exactly like the rule and is not repeated.",
          "- The `rfm` model also uses tenure, frequency and spend. It captures more than the rule on short lists 7 days "
          "ahead and at 30 days, but **60 days ahead both rankers capture fewer events than a random list**: their lists "
          "are still filled with customers who are much closer to the threshold.",
          "- A list of the N highest risks is a different target from “who is L days from dormancy”. Low capture at a long "
          "lead does not mean the list is wrong; it means the list is about nearer events.", "",
          "## 3. Forward-looking counterpart: customers in the same recency band", "",
          "On each alert date, every active customer whose recency says they *would* become dormant in L to L + 6 days "
          "if they do not buy again. This population is known on the alert date. **Denominator: customer-dates in the "
          "band with complete follow-up.** “Became dormant” = no purchase before the pending day.", "",
          "| Lead band | Recency band | Period | Rows (complete follow-up) | Excluded (incomplete) | Became dormant | "
          "recency model: mean risk | rfm model: mean risk | AUC within band: lower spend first | "
          "AUC within band: rfm model |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    mid = pd.Timestamp("2011-06-30")
    for lead in LEADS:
        h = HORIZON[lead]
        b = band(after, lead)
        for label, part in [("Jan–Jun 2011", b[b["scoring_date"] <= mid]), ("Jul–Dec 2011", b[b["scoring_date"] > mid]), ("all", b)]:
            ok = part[part["complete"]]
            assert not ok.duplicated(["customer", "last_purchase"]).any()
            y = ok["event"].to_numpy()
            L.append(f"| {lead}–{lead + STEP - 1} days | {D - lead - STEP + 1}–{D - lead} days | {label} | {len(ok):,} | "
                     f"{(~part['complete']).sum():,} | {pct(y.mean())} | {ok[f'recency-elig_{h}'].mean():.3f} | "
                     f"{ok[f'rfm-elig_{h}'].mean():.3f} | {auc(y, -ok['monetary'].to_numpy()):.3f} | "
                     f"{auc(y, ok[f'rfm-elig_{h}'].to_numpy()):.3f} |")
    L += ["", "- **Calibration:** compare each model's mean risk with the “became dormant” share in the same row. In a "
          "band the model's event (dormant within the horizon) and “became dormant on the pending day” are the same "
          "event. Complete follow-up here is slightly wider than the label rule in `build_snapshots.py`.",
          "- **Ranking within a band:** recency is nearly constant, so recency cannot rank customers here. Lower "
          "spend alone separates future-dormant customers about as well as the rfm model (0.5 would mean no "
          "information). The website's tie-break is the opposite order, higher spend first: it favours customer value, "
          "not likelihood of dormancy.",
          "- **Reading the base rates:** even 60 days ahead, a large share of customers in the band go on to become "
          "dormant, and the share rises as the lead shortens. An alert at a longer lead reaches more customers who "
          "would have returned anyway.", "",
          "## 4. What this does and does not show", "",
          "- Recency alone fixes *when* a customer can be flagged. Earlier alerts need information beyond recency, and the "
          "only such model here (`rfm`) relies on cumulative features that drift over time and is miscalibrated in later "
          "periods (`model_evaluation.md`, Section 6).",
          "- **Not shown:** whether alerting or contacting a customer changes anything. Capture and base rates describe "
          "who became dormant, not who could have been kept. That needs a randomized or otherwise causal design.",
          "- **Still needed for model-based alerts on the website:** window-stable features in place of cumulative ones, a "
          "model frozen before a genuinely later (prospective) evaluation, calibration checked in that later period, and "
          "a decision on the alert threshold or list size. Until then the website keeps the rule-based watchlist.",
          "- Carried over: one dataset, one retailer, a 90-day threshold only, seasonality (one year of training), "
          "later-day cancellations counted as purchases.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    snap = pd.read_csv(SNAPSHOTS, dtype={"customer": str}, parse_dates=["scoring_date", "last_purchase", "crossing_date"])
    REPORT.write_text(report(score(snap)))
    print(REPORT.read_text())
