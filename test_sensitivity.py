"""Checks for sensitivity.py and its page, using the committed processed/sensitivity_model.json.
Run: python test_sensitivity.py (or pytest). The page check needs streamlit."""
import numpy as np

import sensitivity as sv

P = sv.load()
D = P["threshold"]
BASE = {"days_since_last": 70, "tenure_days": 200, "frequency": 3, "monetary": 500.0}


def test_stored_model_is_complete():
    assert P["model"] == "rfm+platt" and D == 90 and P["features"] == sv.FEATURES and set(P["horizons"]) == {"30", "60", "90"}
    for m in P["horizons"].values():
        assert len(m["coef"]) == 5 and len(m["platt"]) == 2 and m["platt"][1] > 0 and set(m["scale"]) == set(sv.FEATURES)
    assert P["horizons"]["90"]["exploratory"] and not P["horizons"]["30"]["exploratory"]


def test_risk_matches_hand_calculation():
    for h in (30, 60, 90):
        m = P["horizons"][str(h)]
        x = [70, np.log1p(200), np.log1p(3), np.log1p(500.0)]
        z = m["coef"][0] + sum(c * (v - m["scale"][f][0]) / m["scale"][f][1] for c, v, f in zip(m["coef"][1:], x, sv.FEATURES))
        want = 1 / (1 + np.exp(-(m["platt"][0] + m["platt"][1] * z)))          # logit(sigmoid(z)) = z
        assert np.isclose(float(sv.risk(P, h, BASE)), want)


def test_eligibility_boundary():
    for h in (30, 60, 90):
        r = np.arange(0, D)
        p = sv.risk(P, h, {**BASE, "days_since_last": r, "tenure_days": r + 130})
        assert (p[r < D - h] == 0).all() and (p[r >= D - h] > 0).all() and (p < 1).all()   # 0 exactly when recency + h < 90
        assert (np.diff(p[r >= D - h]) > 0).all()                                           # more inactivity, more risk
    assert float(sv.risk(P, 30, {**BASE, "days_since_last": 59})) == 0 < float(sv.risk(P, 30, {**BASE, "days_since_last": 60}))


def test_direction_of_other_inputs():
    for h in (30, 60, 90):
        base = float(sv.risk(P, h, BASE))
        assert float(sv.risk(P, h, {**BASE, "frequency": 10})) < base and float(sv.risk(P, h, {**BASE, "monetary": 5000.0})) < base


def test_impossible_profiles_are_rejected():
    assert sv.problems(P, BASE) == []
    bad = [{"days_since_last": 90}, {"days_since_last": -1}, {"tenure_days": 69}, {"frequency": 0}, {"frequency": 2.5},
           {"frequency": 1}, {"monetary": 0}]                      # frequency 1 needs tenure == recency
    assert all(sv.problems(P, {**BASE, **b}) for b in bad)
    assert sv.problems(P, {**BASE, "frequency": 1, "tenure_days": 70}) == []
    xs, ys = sv.sweep(P, 60, BASE, "tenure_days", range(0, 400, 10))
    assert min(xs) == 70 and len(xs) == len(ys)                    # tenure below recency is never drawn
    xs, _ = sv.sweep(P, 60, BASE, "frequency", range(1, 6))
    assert xs == [2, 3, 4, 5]
    assert sv.outside_training(P, 60, {**BASE, "monetary": 1e6}) == ["monetary"] and sv.outside_training(P, 60, BASE) == []


def test_page_labels_and_no_customer_scores():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("sensitivity_page.py", default_timeout=60).run()
    assert not at.exception
    text = " ".join(e.value for e in [*at.warning, *at.markdown, *at.caption])
    for must in ("Model sensitivity, not the effect of an intervention", "not validated", "rfm+platt", "within the next **60 days**",
                 "reaching **90 days**", "no later independent holdout"):
        assert must in text, must
    m = P["horizons"]["60"]["range"]
    a = {"days_since_last": m["days_since_last"]["p50"], "tenure_days": m["tenure_days"]["p50"], "frequency": 2, "monetary": m["monetary"]["p50"]}
    assert at.metric[0].value == f"{float(sv.risk(P, 60, a)):.1%}"
    at.number_input[0].set_value(10).run()                         # recency 10 at a 60-day horizon: not eligible
    assert at.metric[0].value == "0.0%" and any("not eligible" in c.value for c in at.caption)
    at.number_input[2].set_value(1).run()                          # one invoice but first and last purchase 44 days apart
    assert any("single invoice" in e.value for e in at.error) and not at.metric


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all sensitivity checks passed")
