"""Checks for evaluate_models.py: eligibility, role boundaries, calibration-data separation, top-N arithmetic.
Run: python test_evaluation.py (or pytest)."""
import numpy as np
import pandas as pd

from evaluate_models import boot_diff, captured, fit_candidates, reliability, top_n
from train_baselines import periods

H = 30


def synthetic(seed=0, customers=60):
    """Weekly snapshots over the real calendar with labels that respect eligibility (no event below 90 - H days)."""
    rng = np.random.default_rng(seed)
    s = pd.date_range("2010-03-04", "2011-12-08", freq="7D")
    df = pd.DataFrame({"customer": np.repeat([f"{i:05d}" for i in range(customers)], len(s)),
                       "scoring_date": np.tile(s, customers)})
    n = len(df)
    df["days_since_last"] = rng.integers(0, 90, n)
    df["tenure_days"] = df["days_since_last"] + rng.integers(0, 400, n)
    df["frequency"], df["monetary"] = rng.integers(1, 30, n), rng.uniform(10, 5000, n).round(2)
    p = 1 / (1 + np.exp(-(df["days_since_last"] - 75) / 6))
    df[f"y_{H}"] = pd.array(np.where(df["days_since_last"] >= 90 - H, rng.uniform(size=n) < p, 0), "Int8")
    return df


def params_of(df):
    return fit_candidates(df, H)[1]


def test_role_boundaries():
    for h in (30, 60, 90):
        df = synthetic().assign(**{f"y_{h}": synthetic()[f"y_{H}"]}) if h != H else synthetic()
        df[f"y_{h}"] = pd.array(np.where(df["days_since_last"] >= 90 - h, df[f"y_{H}"], 0), "Int8")
        P = periods(df, h)
        end = lambda g: (g["scoring_date"] + pd.Timedelta(days=h)).max()
        assert end(P["train"]) <= pd.Timestamp("2010-12-31") < P["validation"]["scoring_date"].min()
        assert end(P["validation"]) <= pd.Timestamp("2011-06-30") < P["test"]["scoring_date"].min()
        assert end(P["test"]) <= pd.Timestamp("2011-12-08")


def test_calibration_uses_validation_only_and_never_test():
    df = synthetic()
    base = params_of(df)
    P = periods(df, H)
    flip = lambda d, idx: d.assign(**{f"y_{H}": d[f"y_{H}"].where(~d.index.isin(idx) | (d["days_since_last"] < 90 - H),
                                                                    1 - d[f"y_{H}"])})
    # Changing every test label changes nothing that was fitted.
    assert params_of(flip(df, P["test"].index)) == base
    # Changing validation labels moves the calibrators but not the coefficients.
    v = params_of(flip(df, P["validation"].index))
    for spec in ("recency", "rfm"):
        assert v[spec]["coef"] == base[spec]["coef"] and v[spec]["elig_coef"] == base[spec]["elig_coef"]
        assert v[spec]["platt"] != base[spec]["platt"]
    # Changing training labels moves the coefficients.
    assert params_of(flip(df, P["train"].index))["recency"]["coef"] != base["recency"]["coef"]


def test_eligibility_structural_zero_and_eligible_only_fit():
    df = synthetic()
    preds, base = fit_candidates(df, H)
    rows = pd.DataFrame({"days_since_last": [0, 59, 60, 89], "tenure_days": [100] * 4, "frequency": [3] * 4,
                         "monetary": [200.0] * 4})
    for name in ("elig-constant", "recency+platt", "rfm+platt", "recency-elig", "rfm-elig"):
        p = preds[name](rows)
        assert (p[:2] == 0).all() and (p[2:] > 0).all(), name      # 59 days cannot cross 90 within 30 days; 60 can
    assert (preds["recency"](rows) > 0).all()                       # the Nov 12 models have no structural zero
    # -elig models see eligible training rows only: altering ineligible training features changes rfm, not rfm-elig.
    tr = periods(df, H)["train"]
    idx = tr.index[tr["days_since_last"] < 90 - H]
    moved = fit_candidates(df.assign(frequency=df["frequency"].where(~df.index.isin(idx), 1000)), H)[1]
    assert moved["rfm"]["coef"] != base["rfm"]["coef"] and moved["rfm"]["elig_coef"] == base["rfm"]["elig_coef"]


def test_top_n_ties_capacity_and_pooling():
    week = pd.DataFrame({"score": [3, 2, 2, 2, 1], "spend": [0, 9, 5, 5, 0], "y": [1, 0, 1, 1, 0]})
    assert captured(week, ["score"], 2, "y") == 1 + 2 / 3            # one slot shared by three tied rows, 2 events
    assert captured(week, ["score", "spend"], 2, "y") == 1             # spend breaks the tie: row with y = 0 wins
    assert captured(week, ["score", "spend"], 3, "y") == 2             # one slot left for the tied (2, 5) pair, both events
    assert captured(week, ["score"], 10, "y") == 3                     # capacity above the week's size: everyone
    w = pd.concat([week.assign(scoring_date="a"), week.assign(scoring_date="b", y=[1, 1, 1, 0, 0])])
    t = top_n(w.rename(columns={"y": "y_30"}), 30, {"m": ["score"]}, capacities=(2,))
    assert np.isclose(t.loc[("m", 2), "precision"], 2 * (1 + 2 / 3) / 4)        # each week captures 1 + 2/3; summed, then divided
    assert np.isclose(t.loc[("m", 2), "capture"], 2 * (1 + 2 / 3) / 6)
    assert np.isclose(t.loc[("random", 2), "precision"], 6 / 10)              # random = pooled event rate


def test_bootstrap_and_reliability_basics():
    y, p = np.array([0, 1, 1, 0.0]), np.array([0.2, 0.7, 0.9, 0.4])
    assert boot_diff(pd.Series(list("aabb")), y, p, p) == (0, 0, 0)
    d = boot_diff(pd.Series(list("aabb")), y, p, np.full(4, 0.5))
    assert np.isclose(d[0], np.mean((p - y) ** 2) - 0.25) and d[1] <= d[0] <= d[2]
    r = reliability(np.r_[y, y], np.r_[p, p], 4)
    assert r["n"].sum() == 8 and np.isclose((r["n"] * r["observed"]).sum(), 4)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all evaluation checks passed")
