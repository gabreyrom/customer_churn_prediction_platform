"""Checks for fixed_window.py: feature windows, information cut-offs, customer histories, censoring, and the
separation of fitting and evaluation rows. Run: python test_fixed_window.py (or pytest). Pandas and numpy only,
except the last check, which also reads processed/ and needs matplotlib through evaluate_models.py."""
import numpy as np
import pandas as pd

import fixed_window as fw
from build_snapshots import build, load, scoring_dates
from train_baselines import periods

T0 = pd.Timestamp("2010-01-01")
day = lambda n, hour=12: T0 + pd.Timedelta(days=n, hours=hour)
NO_CANC = pd.DataFrame({"customer": pd.Series(dtype=str), "invoice": pd.Series(dtype=str),
                        "date": pd.Series(dtype="datetime64[ns]"), "amount": pd.Series(dtype=float)})


def table(rows):
    return pd.DataFrame([(c, f"{i:06d}", d if isinstance(d, pd.Timestamp) else day(d), a) for i, (c, d, a) in enumerate(rows)],
                        columns=["customer", "invoice", "date", "amount"])


def feats(inv, s, canc=NO_CANC):
    snap = build(inv, pd.DatetimeIndex([day(s, 0)]), last_complete=day(5000, 0))
    return fw.window_features(inv, canc, snap).set_index("customer")


def test_window_edges():
    inv = table([("a", 0, 1.0), ("a", 20, 2.0), ("a", day(21, 0), 4.0), ("a", 199, 8.0), ("a", day(200, 23), 16.0), ("a", 201, 32.0),
                 ("a", 380, 64.0)])
    r = feats(inv, 200).loc["a"]
    # 180-day window = days 21..200: day 20 is out, midnight of day 21 is in, 23:00 on the scoring day is in, day 201 is not
    assert (r.inv_180, r.spend_180) == (3, 28.0) and (r.inv_365, r.spend_365) == (5, 31.0)
    assert (r.frequency, r.monetary) == (5, 31.0)                                  # cumulative, unchanged
    r = feats(inv, 385).loc["a"]                                                   # 365-day window = days 21..385
    assert (r.inv_365, r.spend_365, r.inv_180, r.spend_180) == (5, 124.0, 1, 64.0)  # day 20 dropped; 180 days = 206..385


def test_purchase_days_tenure_cap_and_cancellations():
    inv = table([("a", 0, 10.0), ("a", day(390, 9), 5.0), ("a", day(390, 15), 7.0), ("b", 395, 3.0)])
    canc = table([("a", 385, -4.0), ("a", 100, -1.0), ("a", 401, -50.0), ("b", 396, -99.0)])
    f = feats(inv, 400, canc)
    a, b = f.loc["a"], f.loc["b"]
    assert (a.inv_180, a.days_180) == (2, 1)                                       # two invoices, one purchase day
    assert (a.tenure_days, a.tenure_365, a.tenure_180) == (400, 365, 180) and (b.tenure_365, b.tenure_180) == (5, 5)
    assert (a.cancel_n_180, a.net_180) == (1, 8.0)                                 # day 385 only: day 100 is too old, day 401 is the future
    assert (a.cancel_n_365, a.net_365) == (2, 7.0)
    assert (b.cancel_n_180, b.net_180, b.spend_180) == (1, 0.0, 3.0)               # net spend never goes below zero
    assert fw.complete_from(T0, 180) == day(179, 0) and fw.spec_window("fw180-net") == 180 and fw.spec_window("fw365") == 365


def test_population_is_untouched_and_future_is_never_read():
    inv = table([("a", 0, 1.0), ("a", 60, 2.0), ("a", 300, 4.0), ("b", 10, 8.0), ("b", 250, 16.0), ("c", 240, 32.0)])
    canc = table([("a", 50, -1.0), ("b", 260, -2.0)])
    dates = pd.DatetimeIndex([day(d, 0) for d in (63, 126, 252)])
    snap = build(inv, dates, last_complete=day(5000, 0))
    full = fw.window_features(inv, canc, snap)
    assert len(full) == len(snap) and (full[["customer", "scoring_date"]].to_numpy() == snap[["customer", "scoring_date"]].to_numpy()).all()
    cols = [c for c in full if c.split("_")[0] in ("inv", "days", "spend", "cancel", "net", "tenure") and c != "tenure_days"]
    for s in dates:                                                                # rebuild with everything after s deleted
        cut = lambda d: d[d["date"] < s + pd.Timedelta(days=1)]
        part = fw.window_features(cut(inv), cut(canc), build(cut(inv), pd.DatetimeIndex([s]), last_complete=day(5000, 0)))
        want = full[full["scoring_date"] == s].set_index("customer")[cols]
        pd.testing.assert_frame_equal(part.set_index("customer")[cols], want, check_dtype=False)


def synthetic(seed=0, customers=80, h=60):
    """Weekly rows over the real calendar with eligibility-respecting labels and the fixed-window columns."""
    rng = np.random.default_rng(seed)
    s = pd.date_range("2010-03-04", "2011-12-08", freq="7D")
    df = pd.DataFrame({"customer": np.repeat([f"{i:05d}" for i in range(customers)], len(s)), "scoring_date": np.tile(s, customers)})
    n = len(df)
    df["days_since_last"] = rng.integers(0, 90, n)
    df["tenure_days"] = df["days_since_last"] + rng.integers(0, 500, n)
    df["frequency"], df["monetary"] = rng.integers(1, 30, n), rng.uniform(10, 5000, n).round(2)
    for w in fw.WINDOWS:
        df[f"inv_{w}"] = rng.integers(0, 20, n)
        df[f"days_{w}"], df[f"spend_{w}"] = df[f"inv_{w}"], rng.uniform(0, 3000, n).round(2)
        df[f"net_{w}"], df[f"cancel_n_{w}"], df[f"tenure_{w}"] = df[f"spend_{w}"], rng.integers(0, 3, n), df["tenure_days"].clip(upper=w)
    p = 1 / (1 + np.exp(-(df["days_since_last"] - 60) / 10 + 0.1 * df["inv_180"] - 1))
    df[f"y_{h}"] = pd.array(np.where(df["days_since_last"] >= 90 - h, rng.uniform(size=n) < p, 0), "Int8")
    return df


def test_fitting_reads_training_rows_only():
    h, start = 60, pd.Timestamp("2009-12-01")
    df = synthetic()
    coef = lambda d, **kw: {n: p["coef"] for n, p in fw.fit_models(d, h, start, **kw)[1].items()}
    base, P = coef(df), periods(df, h)
    flip = lambda idx: df.assign(y_60=df["y_60"].where(~df.index.isin(idx) | (df["days_since_last"] < 90 - h), 1 - df["y_60"]))
    assert coef(flip(P["validation"].index)) == base and coef(flip(P["test"].index)) == base   # no calibration, no peeking
    assert coef(flip(P["train"].index)) != base
    later = df["scoring_date"] > pd.Timestamp("2010-12-31")
    assert coef(df.assign(inv_180=df["inv_180"].where(~later, 999), spend_365=df["spend_365"].where(~later, 9e9))) == base
    # 180-day models are fitted only where the window is complete: changing earlier training rows moves fw365 alone
    early = df["scoring_date"] < fw.complete_from(start, 180)
    moved = coef(df.assign(days_since_last=df["days_since_last"].where(~early | (df["days_since_last"] < 30), 89)))
    assert moved["fw365"] != base["fw365"] and all(moved[n] == base[n] for n in ("fw180", "fw180-days", "fw180-net"))
    params = fw.fit_models(df, h, start)[1]
    assert params["fw180"]["from"] >= fw.complete_from(start, 180) > params["fw365"]["from"]
    assert coef(df, fit_end="2011-06-30") != base                                  # the exploratory refit does use more rows


def test_censored_rows_and_ineligible_rows():
    h, start = 60, pd.Timestamp("2009-12-01")
    df = synthetic()
    cens = df.copy()
    late = cens["scoring_date"] > pd.Timestamp("2011-11-01")
    cens.loc[late, "y_60"] = pd.NA                                                 # unknown outcomes near the end of the data
    assert not periods(cens, h)["test"]["y_60"].isna().any() and len(periods(cens, h)["test"]) <= len(periods(df, h)["test"])
    base = {n: p["coef"] for n, p in fw.fit_models(df, h, start)[1].items()}
    assert {n: p["coef"] for n, p in fw.fit_models(cens, h, start)[1].items()} == base
    preds = fw.fit_models(df, h, start)[0]
    rows = df.head(200)
    for f in preds.values():
        p = f(rows)
        assert (p[rows["days_since_last"] < 30] == 0).all() and ((p > 0) & (p < 1))[rows["days_since_last"] >= 30].all()


def test_only_allowlisted_predictors():
    df = synthetic().head(50)
    for bad in (["crossing_date"], ["days_since_last", "y_60"], ["customer"], ["frequency"], ["scoring_date"]):
        try:
            fw.design(df.assign(crossing_date=1), bad)
        except AssertionError:
            continue
        raise AssertionError(f"{bad} was accepted as a predictor")
    assert fw.design(df, fw.SPECS["fw180-net"])[0].shape == (50, 6)


def test_selection_rule():
    m = pd.DataFrame({"brier_elig": [0.20, 0.19, 0.18, 0.17, 0.16], "d_inc_hi": [0.0, -0.001, 0.002, -0.005, -0.004]},
                     index=["recency-elig", "fw365", "fw180", "fw180-days", "fw180-net"])
    cap = lambda low: pd.DataFrame({"capture": [0.30, 0.30, 0.30, 0.30, 0.25 if low else 0.295]}, index=pd.MultiIndex.from_tuples(
        [(n, 100) for n in ("website rule", "fw365", "fw180", "fw180-days", "fw180-net")]))
    assert fw.select(m, cap(low=False)) == "fw180-net"          # lowest Brier among those with an interval below zero
    assert fw.select(m, cap(low=True)) == "fw180-days"          # fw180-net loses too much top-100 capture
    assert fw.select(m.assign(d_inc_hi=0.001), cap(low=False)) == "recency-elig"   # nobody qualifies: the incumbent stays


def test_real_data_windows_and_report():
    inv = load()
    canc = pd.read_csv("processed/cancellations.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])
    dates = scoring_dates(inv)[[0, 30, 38, 45, 70, 92]]
    df = fw.window_features(inv, canc, build(inv, dates))
    start = inv["date"].min().normalize()
    young = df["scoring_date"] < fw.complete_from(start, 365)                      # before 30 Nov 2010: window = whole history
    assert young.any() and (~young).any()
    assert (df.loc[young, "inv_365"] == df.loc[young, "frequency"]).all() and np.allclose(df.loc[young, "spend_365"], df.loc[young, "monetary"])
    assert (df["inv_365"] <= df["frequency"]).all() and (df["inv_180"] <= df["inv_365"]).all() and (df["days_180"] <= df["inv_180"]).all()
    assert (df["inv_180"] >= 1).all() and (df["net_180"] <= df["spend_180"]).all() and (df["tenure_180"] <= 180).all()   # active => a purchase within 90 days
    assert (df.loc[~young, "inv_365"] < df.loc[~young, "frequency"]).any()         # later on the two definitions part ways
    s = dates[4]
    cut = lambda d: d[d["date"] < s + pd.Timedelta(days=1)]
    part = fw.window_features(cut(inv), cut(canc), build(cut(inv), pd.DatetimeIndex([s])))
    cols = [c for c in part if c.split("_")[0] in ("inv", "days", "spend", "cancel", "net")]
    pd.testing.assert_frame_equal(part[cols], df.loc[df["scoring_date"] == s, cols].reset_index(drop=True), check_dtype=False)
    text = fw.REPORT.read_text()
    for must in ("Neither later period is an independent holdout", "Selection rule, written before the first run", "not confirmation",
                 "## 9. Confirmation plan", "Exploratory course demonstration"):
        assert must in text, must
    assert "Warning: the selection changed" not in text and "| customer |" not in text      # no individual predictions


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all fixed-window checks passed")
