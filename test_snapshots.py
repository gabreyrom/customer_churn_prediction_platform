"""Checks for build_snapshots.py on hand-made purchase histories. Run: python test_snapshots.py (or pytest)."""
import pandas as pd

from build_snapshots import build

T0 = pd.Timestamp("2010-01-01")
day = lambda n, hour=12: T0 + pd.Timedelta(days=n, hours=hour)


def snap(purchase_days, s, last_complete=day(1000, 0).normalize(), **kw):
    inv = pd.DataFrame({"customer": "00042", "invoice": [f"{i:06d}" for i in range(len(purchase_days))],
                        "date": [d if isinstance(d, pd.Timestamp) else day(d) for d in purchase_days], "amount": 10.0})
    out = build(inv, pd.DatetimeIndex([day(s, 0)]), last_complete=last_complete, **kw)
    return out.iloc[0] if len(out) else None


def test_feature_freezing_and_end_of_day():
    a, b = snap([0, 20, day(30, 23)], 30), snap([0, 20, day(30, 23), 31, 40], 30)
    cols = ["days_since_last", "tenure_days", "frequency", "monetary"]
    assert a[cols].tolist() == b[cols].tolist() == [0, 30, 3, 30.0]   # 23:00 purchase on s counts, later ones do not
    assert a["customer"] == "00042"                                    # IDs stay strings with leading zeros


def test_purchase_on_crossing_day_resets():
    r = snap([0, 90], 10)          # recency would hit 90 on day 90, but the purchase that day resets it
    assert r["crossing_date"] == day(180, 0) and r["y_90"] == 0
    r = snap([0, 91], 10)          # one day late: crossed on day 90
    assert r["crossing_date"] == day(90, 0) and r["y_90"] == 1 and r["y_60"] == 0


def test_repeated_purchases_reset():
    r = snap([0, 60, 120, 180], 10)
    assert r["crossing_date"] == day(270, 0) and (r[["y_30", "y_60", "y_90"]] == 0).all()


def test_event_then_reactivation_is_event():
    r = snap([0, 120], 50)         # crosses on day 90, returns on day 120
    assert r["crossing_date"] == day(90, 0) and r["y_30"] == 0 and r["y_60"] == 1 and r["y_90"] == 1


def test_inactive_and_new_customers_excluded():
    assert snap([0], 90) is None   # recency 90 at s: already dormant
    assert snap([50], 10) is None  # no purchase yet


def test_incomplete_follow_up():
    lc = day(70, 0).normalize()
    r = snap([0], 10, last_complete=lc)               # crossing day 90 > last complete day 70
    assert r["y_30"] == r["y_60"] == 0 and pd.isna(r["y_90"])     # s+60 = day 70 is still observed
    r = snap([0], 10, last_complete=day(95, 0))       # crossing observed even though s+90 is beyond the data
    assert r["y_90"] == 1
    r = snap([0, 85], 10, last_complete=day(95, 0))   # a reset whose new crossing is unobserved: censored, not 0
    assert pd.isna(r["y_90"]) and r["y_60"] == 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all snapshot checks passed")
