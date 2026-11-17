"""Checks for alert_timing.py: lead-time matching, information boundaries, incomplete follow-up, top-N ties.
Run: python test_alerts.py (or pytest). Needs pandas and numpy only."""
import numpy as np
import pandas as pd

from alert_timing import band, events, inclusion, match
from build_snapshots import build

T0 = pd.Timestamp("2010-01-07")                      # a Thursday
day = lambda n, hour=12: T0 + pd.Timedelta(days=n, hours=hour)
WEEKS = pd.date_range(T0, periods=60, freq="7D")     # weekly scoring dates: days 0, 7, 14, ...
LC = day(400, 0)


def invoices(rows):
    return pd.DataFrame([(c, f"{i:06d}", day(d), 10.0 * (i + 1)) for i, (c, d) in enumerate(rows)],
                        columns=["customer", "invoice", "date", "amount"])


def test_lead_matching():
    ev = pd.DataFrame({"customer": list("abcd"), "crossing_date": [day(100, 0), day(105, 0), day(98, 0), day(20, 0)]})
    m = match(ev, WEEKS, 30)
    # latest weekly date at least 30 days before: day 100 -> day 70 (lead 30); day 105 -> day 70 (35); day 98 -> day 63 (35)
    assert m["scoring_date"].tolist()[:3] == [day(70, 0), day(70, 0), day(63, 0)] and m["actual_lead"].tolist()[:3] == [30, 35, 35]
    assert pd.isna(m["scoring_date"][3]) and pd.isna(m["actual_lead"][3])      # no scoring date early enough
    for lead in (60, 30, 7):
        got = match(pd.DataFrame({"customer": "x", "crossing_date": [day(n, 0) for n in range(120, 200)]}), WEEKS, lead)
        assert got["actual_lead"].between(lead, lead + 6).all() and set(got["actual_lead"]) == set(range(lead, lead + 7))


def test_events_repeats_and_unobserved():
    inv = invoices([("a", 0), ("a", 200), ("b", 0), ("b", 60), ("c", 350)])
    snap = build(inv, WEEKS, last_complete=LC)
    ev = events(snap, LC)
    # a: lapses on day 90, returns, lapses again on day 290; b: day 150; c would cross on day 440 > LC: not an event
    assert sorted(zip(ev["customer"], ev["crossing_date"])) == [("a", day(90, 0)), ("a", day(290, 0)), ("b", day(150, 0))]


def test_alert_row_uses_no_later_purchases():
    inv = invoices([("a", 0), ("a", 20), ("a", 130), ("b", 5), ("b", 30), ("b", 125)])
    snap = build(inv, WEEKS, last_complete=LC)
    feats = ["days_since_last", "tenure_days", "frequency", "monetary"]
    for lead in (60, 30, 7):
        m = match(events(snap, LC), WEEKS, lead).dropna(subset=["scoring_date"])
        rows = m.merge(snap, on=["customer", "scoring_date"], suffixes=("", "_row"))
        assert len(rows) == len(m) and (rows["crossing_date"] == rows["crossing_date_row"]).all()
        assert (rows["days_since_last"] == 90 - rows["actual_lead"]).all()       # recency is fixed by the lead
        for r in rows.itertuples():                                              # same features with the future removed
            cut = build(inv[inv["date"] < r.scoring_date + pd.Timedelta(days=1)], pd.DatetimeIndex([r.scoring_date]), last_complete=LC)
            assert cut.loc[cut["customer"] == r.customer, feats].iloc[0].tolist() == [getattr(r, f) for f in feats]


def test_band_and_incomplete_follow_up():
    lc = day(120, 0)
    inv = invoices([("lapse", 0), ("back", 0), ("back", 85), ("late", 40), ("lateback", 40), ("lateback", 125)])
    snap = build(inv, WEEKS, last_complete=lc)
    b = band(snap, 7, last_complete=lc)                 # recency 77..83, i.e. 7..13 days before the pending day
    assert band(snap, 7, last_complete=lc).query("customer == 'back'")["complete"].tolist() == [True, False]   # two spells
    b = b[b["scoring_date"] <= day(125, 0)]             # each customer's first spell
    assert b["days_since_last"].between(77, 83).all() and not b.duplicated(["customer", "last_purchase"]).any()
    got = b.set_index("customer")[["complete", "event"]].to_dict("index")
    assert got["lapse"] == {"complete": True, "event": 1}        # pending day 90 <= day 120, no purchase before it
    assert got["back"] == {"complete": True, "event": 0}         # bought on day 85
    # pending day 130 is after the last complete day: excluded whatever happened, even though "lateback" returned on day 125
    assert not got["late"]["complete"] and not got["lateback"]["complete"]


def test_inclusion_ties():
    week = pd.DataFrame({"score": [3, 2, 2, 2, 1], "spend": [0, 9, 5, 5, 0]})
    assert np.allclose(inclusion(week, ["score"], 2), [1, 1 / 3, 1 / 3, 1 / 3, 0])       # one place shared by three
    assert np.allclose(inclusion(week, ["score", "spend"], 3), [1, 1, 0.5, 0.5, 0])      # spend breaks the first tie
    assert inclusion(week, ["score"], 10).eq(1).all() and np.isclose(inclusion(week, ["score"], 2).sum(), 2)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all alert-timing checks passed")
