"""Checks for train_baselines.py: predictor allowlist, split boundaries, train-only fitting. Run: python test_baselines.py"""
import numpy as np
import pandas as pd

from build_snapshots import split
from train_baselines import FORBIDDEN, SPECS, design, eligible, fit_logit, periods

TE, VE, LC = pd.Timestamp("2010-12-31"), pd.Timestamp("2011-06-30"), pd.Timestamp("2011-12-08")


def test_allowlist_rejects_future_and_ids():
    for spec in SPECS.values():
        assert not set(spec) & FORBIDDEN
    df = pd.DataFrame({"days_since_last": [1, 2], "crossing_date": [3, 4], "customer": ["a", "b"]})
    for bad in (["crossing_date"], ["customer"], ["days_since_last", "y_30"], ["scoring_date"]):
        try:
            design(df, bad)
        except AssertionError:
            continue
        raise AssertionError(f"{bad} was accepted as a predictor")
    x, _ = design(pd.DataFrame({"days_since_last": [1.0, 3.0], "crossing_date": [0, 99]}), ["days_since_last"])
    assert x.shape == (2, 2)  # intercept + recency only; the extra column never enters


def test_split_boundaries():
    s = pd.Series(pd.to_datetime(["2010-12-01", "2010-12-02", "2011-01-01", "2011-05-31", "2011-06-01",
                                  "2011-07-01", "2011-11-08", "2011-11-09"]))
    got = list(split(s, 30, TE, VE, LC))
    # train only if s+30 <= Dec 31; val needs s > Dec 31 and s+30 <= Jun 30; test needs s > Jun 30 and s+30 <= Dec 8
    assert got == ["train", "unused", "validation", "validation", "unused", "test", "test", "unused"], got


def test_periods_drop_censored_and_never_cross_cutoffs():
    s = pd.date_range("2010-03-04", "2011-12-08", freq="7D")
    df = pd.DataFrame({"customer": "1", "scoring_date": s, "y_90": pd.array([1] * (len(s) - 5) + [pd.NA] * 5, "Int8")})
    P = periods(df, 90, TE, VE)
    assert (P["train"]["scoring_date"] + pd.Timedelta(days=90)).max() <= TE
    assert P["validation"]["scoring_date"].min() > TE and (P["validation"]["scoring_date"] + pd.Timedelta(days=90)).max() <= VE
    assert P["test"]["scoring_date"].min() > VE and all(g["y_90"].notna().all() for g in P.values())


def test_scaling_is_train_only_and_fit_recovers_signal():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"days_since_last": rng.uniform(0, 89, 5000)})
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp(-(tr["days_since_last"] - 45) / 10))).astype(float)
    x, scale = design(tr, ["days_since_last"])
    va = pd.DataFrame({"days_since_last": [1000.0, 2000.0]})  # wildly different distribution
    assert design(va, ["days_since_last"], scale)[1] == scale
    b = fit_logit(x, y)
    assert b[1] > 0 and abs(b[0]) < 0.2  # positive slope; balanced around the mean


def test_eligible_group_boundary():
    df = pd.DataFrame({"days_since_last": [59, 60, 89], "y_30": pd.array([0, 1, 0], "Int8")})
    assert eligible(df, 30)["days_since_last"].tolist() == [60, 89]   # 90 - 30 = 60 is the first eligible recency
    try:
        eligible(df.assign(y_30=pd.array([1, 1, 0], "Int8")), 30)       # an event at recency 59 is impossible
    except AssertionError:
        return
    raise AssertionError("impossible event outside the eligible group was not caught")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all baseline checks passed")
