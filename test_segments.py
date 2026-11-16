"""Checks for segments.py on hand-made invoices and the real data. Run: python test_segments.py (or pytest)."""
import itertools

import pandas as pd

import segments as sg

T0 = pd.Timestamp("2010-01-01")
day = lambda n, hour=12: T0 + pd.Timedelta(days=n, hours=hour)
CUTS = {"recency": 30, "frequency": 3, "monetary": 100.0}


def invoices(rows):
    """rows: (customer, day or timestamp, amount)"""
    return pd.DataFrame([(c, f"{i:06d}", d if isinstance(d, pd.Timestamp) else day(d), a) for i, (c, d, a) in enumerate(rows)],
                        columns=["customer", "invoice", "date", "amount"])


def test_rfm_and_date_boundary():
    inv = invoices([("00042", 0, 10.0), ("00042", 400, 20.0), ("00042", day(500, 23), 30.0), ("00042", 501, 999.0),
                    ("00007", 501, 5.0)])
    t = sg.rfm_table(inv, day(500, 0))
    assert list(t.index) == ["00042"]                      # IDs stay strings; a customer first seen later is absent
    r = t.loc["00042"]
    # 23:00 on the as-of day counts, day 501 does not; day 0 is outside the 365-day window but still sets tenure
    assert (r.recency, r.tenure, r.frequency, r.monetary) == (0, 500, 2, 50.0)
    assert sg.rfm_table(inv, day(136, 0)).loc["00042", "frequency"] == 1   # window (as_of - 365, as_of] ...
    assert sg.rfm_table(inv, day(365, 0)).loc["00042", "frequency"] == 0   # ... drops a purchase exactly 365 days old


def test_ties_and_mapping():
    t = pd.DataFrame({"recency": [30, 31, 30, 0, 89, 90], "frequency": [3, 3, 2, 9, 1, 9], "monetary": [100.0, 99.99, 100.0, 5.0, 1.0, 9e9],
                      "tenure": 400, "new": [False, True, False, True, False, True]}, index=list("abcdef"))
    a = sg.assign(t, CUTS, 90)
    assert a["code"].to_dict() == {"a": "HHH", "b": "LHL", "c": "HLH", "d": "HHL", "e": "LLL"}   # equal to cutoff = high
    assert "f" not in a.index                                                               # dormant: no segment
    assert a.loc["e", "segment"] == "Low activity" and a.loc["a", "segment"] == "Core customers"
    assert a["new"].to_dict() == {"a": False, "b": True, "c": False, "d": True, "e": False}   # flag passes through
    assert a.loc["b", "segment"] == sg.NAMES["LHL"] and a.loc["d", "segment"] == sg.NAMES["HHL"]   # ... and never renames
    assert set(sg.NAMES) == {"".join(p) for p in itertools.product("HL", repeat=3)}
    assert len(set(sg.ORDER)) == 8 and sg.NEW not in sg.ORDER and sg.NAMES["LLL"] == "Low activity"


def test_recently_acquired_flag():
    inv = invoices([("old", 0, 1.0), ("old", 390, 1.0), ("new", 390, 1.0), ("early", 100, 1.0)])
    flag = lambda cust, d: bool(sg.rfm_table(inv, day(d, 0)).loc[cust, "new"])
    assert sg.NEW_DAYS == 30 and sg.MIN_PRIOR == 270
    assert flag("new", 390) and flag("new", 419) and not flag("new", 420)   # tenure 0 and 29 flagged, 30 is not
    assert not flag("old", 400)                                             # a later purchase does not make a customer new
    assert not flag("early", 110)                                           # only 100 days of data before it
    inv2 = invoices([("x", 0, 1.0), ("edge", 270, 1.0), ("short", 269, 1.0)])
    t = sg.rfm_table(inv2, day(275, 0))
    assert t.loc["edge", "new"] and not t.loc["short", "new"]               # exactly 270 days of prior data is enough
    a = sg.assign(sg.rfm_table(inv, day(400, 0)), CUTS, 90)
    assert a.loc["new", "new"] and a.loc["new", "segment"] == sg.NAMES["HLL"]   # flagged, still in an RFM segment
    assert sg.summary(a).loc[sg.NAMES["HLL"], "new"] == 1


def test_dormancy_date():
    assert sg.dormancy_date(day(10, 15), 90) == day(100, 0)                 # purchase day + threshold, time dropped
    assert sg.dormancy_date(day(10), 60) == day(70, 0) and sg.dormancy_date(day(10), 120) == day(130, 0)
    inv = invoices([("a", 0, 1.0), ("a", 50, 1.0), ("a", 200, 1.0)])
    last = lambda d: sg.rfm_table(inv, day(d, 0)).loc["a", "last"]
    assert sg.dormancy_date(last(60), 90) == day(140, 0)                    # the day-200 purchase is not known on day 60
    assert sg.dormancy_date(last(200), 90) == day(290, 0)


def test_insufficient_data():
    inv = invoices([("a", 0, 1.0), ("a", 300, 1.0)])
    ref = pd.DatetimeIndex([day(310, 0)])
    assert sg.cutoffs(inv, day(309, 0), 90, ref) is None            # as-of before the reference date
    assert sg.cutoffs(inv, day(320, 0), 90, ref) is None            # too few reference customers
    s = sg.summary(sg.assign(sg.rfm_table(inv, day(320, 0)), CUTS, 90))
    assert s["customers"].sum() == 1 and (s["customers"] == 0).sum() == 7   # missing groups stay, with zero


def test_real_data_no_future_and_one_segment():
    inv = pd.read_csv("processed/invoices.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])
    assert sg.cutoffs(inv, "2010-12-29", 90) is None
    for d in ["2011-03-31", "2011-09-08"]:
        cut = inv[inv["date"] < pd.Timestamp(d) + pd.Timedelta(days=1)]   # everything after the date removed
        full, trunc = sg.assign(sg.rfm_table(inv, d), sg.cutoffs(inv, d, 90), 90), sg.assign(sg.rfm_table(cut, d), sg.cutoffs(cut, d, 90), 90)
        pd.testing.assert_frame_equal(full, trunc)
        assert sg.cutoffs(inv, d, 90) == sg.cutoffs(inv, "2011-12-09", 90)   # cutoffs do not move with the date
        assert full["segment"].isin(sg.ORDER).all() and full.index.is_unique and sg.summary(full)["customers"].sum() == len(full)
        assert 0 < full["new"].sum() < len(full) and (full.loc[full["new"], "tenure"] < 30).all()
        assert (full["last"] <= pd.Timestamp(d)).all() and (full["frequency"] >= 1).all()


def test_pages_share_controls_and_respect_reference_period():
    import datetime as dt

    from streamlit.testing.v1 import AppTest
    texts = lambda at: " ".join(e.value for e in [*at.caption, *at.error, *at.info, *at.warning])
    status = AppTest.from_file("results_page.py", default_timeout=60).run()
    status.date_input[0].set_value(dt.date(2011, 3, 31)).run()
    status.selectbox[0].set_value(60).run()
    assert not status.exception and ("Dormancy date if there is no further purchase" in texts(status)
                                     or "Dormancy threshold reached on" in texts(status))
    seg = AppTest.from_file("segments_page.py", default_timeout=60)
    seg.session_state["as_of"], seg.session_state["threshold"] = status.session_state["as_of"], status.session_state["threshold"]
    seg.run()                                    # the other page picks up both settings from the shared state
    assert not seg.exception and seg.date_input[0].value == dt.date(2011, 3, 31) and seg.selectbox[0].value == 60
    assert "31 Mar 2011" in seg.info[0].value and "60 days" in seg.info[0].value
    inv = pd.read_csv("processed/invoices.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])
    members = lambda t: sg.assign(sg.rfm_table(inv, "2011-03-31"), sg.cutoffs(inv, "2011-03-31", t), t)
    top = lambda a: a[a["segment"] == sg.ORDER[0]].sort_values("monetary", ascending=False).iloc[0]   # default timeline
    act = members(60)
    assert seg.metric[1].value == f"{len(act):,}" and seg.metric[3].value == f"{act['new'].sum():,}"
    assert f"Dormancy date if there is no further purchase: {sg.dormancy_date(top(act)['last'], 60):%d %b %Y}" in texts(seg)
    seg.selectbox[0].set_value(120).run()        # the date follows the threshold
    assert f"Dormancy date if there is no further purchase: {sg.dormancy_date(top(members(120))['last'], 120):%d %b %Y}" in texts(seg)
    seg.date_input[0].set_value(dt.date(2010, 6, 30)).run()   # before the reference period: explained, date untouched
    assert not seg.exception and "No segments for 30 Jun 2010" in seg.error[0].value and not seg.metric
    assert seg.session_state["as_of"] == dt.date(2010, 6, 30)
    status.session_state["as_of"], status.session_state["threshold"] = seg.session_state["as_of"], seg.session_state["threshold"]
    status.run()                                 # the status page still works for that early date
    assert not status.exception and status.date_input[0].value == dt.date(2010, 6, 30) and status.metric


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all segment checks passed")
