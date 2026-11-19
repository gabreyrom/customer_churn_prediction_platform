"""Fixed-window features and a pre-specified model comparison (modeling iteration after Nov 26). Offline only.

Usage: python fixed_window.py   (pandas, numpy; matplotlib through evaluate_models.py; about a minute)
Output: processed/fixed_window_evaluation.md   (do not edit by hand)
Nothing here is read by the website, and no individual prediction is written anywhere.

FEATURES, all as of the end of a weekly scoring date s (population and labels from build_snapshots.py, unchanged):
  days_since_last      recency, as before
  inv_W, spend_W       distinct valid invoices and their gross amount in the W calendar days s-W+1 .. s
  days_W               distinct purchase days in the window (an alternative to inv_W: 18% of invoices share a
                       customer-day with another invoice, i.e. one visit split over several invoices)
  cancel_n_W, net_W    cancellation invoices dated in the window, and max(spend_W - their amount, 0). Only cancellations
                       dated on or before s are used, so nothing from the future enters. Floored at 0 because same-day
                       fully cancelled purchases were already removed from the invoices but their cancellation was not.
  tenure_W             min(tenure_days, W): observed tenure, capped
  W is 365 (requested) or 180.

CONCERN that shapes the design: the data start on 1 Dec 2009 and training ends in 2010, so NO training scoring date
has a full 365-day window. In training, inv_365 / spend_365 / tenure_365 equal the old cumulative features exactly;
`fw365` therefore has the same coefficients as the old `rfm-elig` and differs only in the values it is given later.
A 180-day window is complete from 29 May 2010, so `fw180*` models are fitted on those training dates only: the
feature then means the same thing when fitted and when applied.

CANDIDATES (fixed before any of them was scored on any period):
  baselines   constant, elig-constant, recency-elig (logistic on recency), rfm-elig (old cumulative features),
              and the website rule (recency, then spend) as a ranker. All from evaluate_models.py, unchanged.
  fw365       recency + log1p(inv_365) + log1p(spend_365) + log1p(tenure_365)
  fw180       the same with W = 180
  fw180-days  fw180 with purchase days instead of invoices
  fw180-net   fw180 with net spend instead of gross, plus log1p(cancel_n_180)
  Every model is a logistic regression fitted on eligible training rows (recency >= 90 - h) and returns 0 for
  ineligible rows, the specification the Nov 19 evaluation recommended. No recalibration is done in this iteration:
  there is no period left to fit a calibrator on without reusing validation.

ROLES, per horizon h (build_snapshots.split; censored labels dropped, never counted as negative):
  fit       train       s + h <= 2010-12-31
  compare   validation  s > 2010-12-31 and s + h <= 2011-06-30   candidates are selected here and only here
  explore   test        s > 2011-06-30 and s + h <= 2011-12-08   reported after selection, never used to choose
  Neither later period is fresh. Validation was examined for the old models on Nov 12 and used to fit calibrators on
  Nov 19; test was examined on Nov 12 and used to compare six candidates on Nov 19. The fixed-window candidates had
  not been scored on either before this script first ran, but every design choice here was made knowing those results.

SELECTION RULE (written before the first run; implemented in `select`):
  Per horizon, a fixed-window candidate qualifies if, on validation eligible rows,
    (a) its Brier score is lower than recency-elig's and the customer-bootstrap 95% interval of the difference
        excludes 0, and
    (b) its top-100 capture is not more than 1 percentage point below the website rule's.
  Among qualifying candidates the lowest eligible Brier score is selected; if none qualifies, recency-elig stays.
  Primary horizons are 30 and 60 days; 90 days is exploratory, as before. AUC, calibration and the other list sizes
  are reported but do not enter the rule.
  Pre-specified exploratory extras, reported after selection and not used for it: test-period results for every
  model, and a refit on all rows with s + h <= 2011-06-30 scored on test (a full year of seasons in training).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_snapshots import D, HORIZONS, LAST_COMPLETE, TRAIN_END, VAL_END, build, load, scoring_dates
from train_baselines import EXPLORATORY, FORBIDDEN, auc, brier, eligible, fit_logit, periods, predict

WINDOWS = (365, 180)
SPECS = {"fw365": ["days_since_last", "inv_365", "spend_365", "tenure_365"],
         "fw180": ["days_since_last", "inv_180", "spend_180", "tenure_180"],
         "fw180-days": ["days_since_last", "days_180", "spend_180", "tenure_180"],
         "fw180-net": ["days_since_last", "inv_180", "net_180", "cancel_n_180", "tenure_180"]}
ALLOWED = {c for cols in SPECS.values() for c in cols}
BASELINES = ["constant", "elig-constant", "recency-elig", "rfm-elig"]
INCUMBENT = "recency-elig"
CAPACITIES = (50, 100, 200)
TOLERANCE = 0.01   # allowed shortfall in top-100 capture against the website rule
REPORT = Path("processed/fixed_window_evaluation.md")


def spec_window(name):
    return int(SPECS[name][-1].split("_")[1])


def complete_from(data_start, w):
    """First scoring day whose w-day window lies entirely inside the data."""
    return pd.Timestamp(data_start).normalize() + pd.Timedelta(days=w - 1)


def window_features(inv, canc, snap, windows=WINDOWS):
    """Add the fixed-window columns to the snapshot rows. Row set and order are those of `snap`: nobody is added or
    dropped, and each value uses invoices and cancellations dated before the end of the row's scoring date only."""
    dates = np.sort(pd.DatetimeIndex(snap["scoring_date"].unique()).to_numpy())
    end = dates + np.timedelta64(1, "D")                                   # exclusive upper edge: end of day s
    by_c = {c: g for c, g in canc.groupby("customer")}
    out = []
    for cust, g in inv.sort_values("date").groupby("customer"):
        t, cum = g["date"].to_numpy(), np.r_[0.0, g["amount"].cumsum().to_numpy()]
        days = np.unique(g["date"].dt.normalize().to_numpy())
        c = by_c.get(cust)
        tc = np.sort(c["date"].to_numpy()) if c is not None else np.array([], dtype="datetime64[ns]")
        cumc = np.r_[0.0, -c.sort_values("date")["amount"].cumsum().to_numpy()] if c is not None else np.zeros(1)
        cols = {"customer": cust, "scoring_date": dates}
        hi, hid, hic = (np.searchsorted(a, end, side="left") for a in (t, days, tc))
        for w in windows:
            start = end - np.timedelta64(w, "D")                           # inclusive lower edge: start of day s-w+1
            lo, lod, loc = (np.searchsorted(a, start, side="left") for a in (t, days, tc))
            spend, cancelled = cum[hi] - cum[lo], cumc[hic] - cumc[loc]
            cols.update({f"inv_{w}": hi - lo, f"days_{w}": hid - lod, f"spend_{w}": spend.round(2),
                         f"cancel_n_{w}": hic - loc, f"net_{w}": np.maximum(spend - cancelled, 0).round(2)})
        out.append(pd.DataFrame(cols))
    df = snap.merge(pd.concat(out, ignore_index=True), on=["customer", "scoring_date"], how="left", validate="one_to_one")
    for w in windows:
        df[f"tenure_{w}"] = df["tenure_days"].clip(upper=w)
    assert len(df) == len(snap) and df[[f"inv_{w}" for w in windows]].notna().all().all()
    return df


def design(df, cols, scale=None):
    """Allowlisted fixed-window columns -> standardised matrix (log1p for everything except recency)."""
    assert set(cols) <= ALLOWED and not set(cols) & FORBIDDEN, f"illegal predictors: {cols}"
    x = df[cols].astype(float).copy()
    for c in cols:
        if c != "days_since_last":
            x[c] = np.log1p(x[c])
    if scale is None:
        scale = {c: (x[c].mean(), x[c].std()) for c in cols}
    return np.column_stack([np.ones(len(x))] + [(x[c] - scale[c][0]) / scale[c][1] for c in cols]), scale


def fit_rows(rows, h, name, data_start):
    """Eligible rows a spec is fitted on: for a window shorter than the available history, only dates on which the
    whole window lies inside the data."""
    e = eligible(rows, h)
    w = spec_window(name)
    return e if w >= 365 else e[e["scoring_date"] >= complete_from(data_start, w)]


def fit_models(df, h, data_start, fit_end=TRAIN_END):
    """{name: f(rows) -> probability} and the fitted parameters. Reads only labelled rows with s + h <= fit_end."""
    y = f"y_{h}"
    lab = df[df[y].notna()]
    tr = lab[lab["scoring_date"] + pd.Timedelta(days=h) <= pd.Timestamp(fit_end)]
    preds, params = {}, {}
    for name, cols in SPECS.items():
        rows = fit_rows(tr, h, name, data_start)
        x, scale = design(rows, cols)
        b = fit_logit(x, rows[y].to_numpy(float))
        preds[name] = lambda d, b=b, scale=scale, cols=cols: np.where(
            (d["days_since_last"] >= D - h).to_numpy(), predict(design(d, cols, scale)[0], b), 0.0)
        params[name] = {"coef": dict(zip(["intercept", *cols], b.tolist())), "rows": len(rows),
                        "dates": rows["scoring_date"].nunique(), "from": rows["scoring_date"].min()}
    return preds, params


def baselines(df, h, fit_end=TRAIN_END):
    """The earlier models, refitted by the earlier code. With the default cut-off they are the Nov 19 models."""
    from evaluate_models import fit_candidates
    from train_baselines import SPECS as OLD, design as old_design
    if pd.Timestamp(fit_end) == TRAIN_END:
        preds = fit_candidates(df, h)[0]
        return {n: preds[n] for n in BASELINES}
    y = f"y_{h}"
    lab = df[df[y].notna()]
    tr = lab[lab["scoring_date"] + pd.Timedelta(days=h) <= pd.Timestamp(fit_end)]
    etr = eligible(tr, h)
    elig = lambda d: (d["days_since_last"] >= D - h).to_numpy()
    out = {"constant": lambda d: np.full(len(d), tr[y].mean()), "elig-constant": lambda d: np.where(elig(d), etr[y].mean(), 0.0)}
    for name, cols in OLD.items():
        x, sc = old_design(etr, cols)
        b = fit_logit(x, etr[y].to_numpy(float))
        out[f"{name}-elig"] = lambda d, b=b, sc=sc, cols=cols: np.where(elig(d), predict(old_design(d, cols, sc)[0], b), 0.0)
    return out


def evaluate(rows, h, preds):
    """Metrics per model on one period (all rows and eligible rows), Brier differences against the incumbent and the
    eligible constant with customer-bootstrap intervals, and weekly top-N among eligible rows."""
    from evaluate_models import boot_diff, calibration, top_n
    y = f"y_{h}"
    g = rows.copy()
    for name, f in preds.items():
        g[name] = f(g)
    e = g[g["days_since_last"] >= D - h]
    yy, ye = g[y].to_numpy(float), e[y].to_numpy(float)
    out = []
    for name in preds:
        p, pe = g[name].to_numpy(), e[name].to_numpy()
        mean_p, obs, _, slope = calibration(ye, pe)
        d_inc = boot_diff(e["customer"], ye, pe, e[INCUMBENT].to_numpy()) if name != INCUMBENT else (0.0, 0.0, 0.0)
        d_const = boot_diff(e["customer"], ye, pe, e["elig-constant"].to_numpy()) if name != "elig-constant" else (0.0, 0.0, 0.0)
        out.append((name, auc(yy, p), brier(yy, p), auc(ye, pe), brier(ye, pe), mean_p, obs, slope, *d_inc, *d_const))
    m = pd.DataFrame(out, columns=["model", "auc_all", "brier_all", "auc_elig", "brier_elig", "mean_p", "observed", "slope",
                                   "d_inc", "d_inc_lo", "d_inc_hi", "d_const", "d_const_lo", "d_const_hi"]).set_index("model")
    rankers = {"website rule": ["days_since_last", "monetary"],
               **{n: [n] for n in preds if n not in ("constant", "elig-constant")}}
    return m, top_n(e, h, rankers, CAPACITIES), g


def select(metrics, topn):
    """The selection rule of the module docstring, on one horizon's validation results."""
    rule = topn.loc[("website rule", 100), "capture"]
    ok = [n for n in SPECS if metrics.loc[n, "d_inc_hi"] < 0 and topn.loc[(n, 100), "capture"] >= rule - TOLERANCE]
    return min(ok, key=lambda n: metrics.loc[n, "brier_elig"]) if ok else INCUMBENT


def prepare():
    inv = load()
    canc = pd.read_csv("processed/cancellations.csv.gz", dtype={"customer": str, "invoice": str}, parse_dates=["date"])
    return inv, canc, window_features(inv, canc, build(inv, scoring_dates(inv)))


def report(inv, canc, df):
    start = inv["date"].min().normalize()
    f3, f4, pct = (lambda v: "–" if pd.isna(v) else f"{v:.3f}"), (lambda v: f"{v:.4f}"), (lambda v: f"{v:.1%}")
    ci = lambda r, k: "–" if r[k] == 0 and r[f"{k}_lo"] == 0 else f"{r[k]:+.4f} [{r[f'{k}_lo']:+.4f}, {r[f'{k}_hi']:+.4f}]"
    tag = lambda h: f"{h} (exploratory)" if h in EXPLORATORY else f"{h}"
    res = {}
    for h in HORIZONS:
        P = periods(df, h)
        assert (P["train"]["scoring_date"] + pd.Timedelta(days=h)).max() < P["validation"]["scoring_date"].min()
        assert (P["validation"]["scoring_date"] + pd.Timedelta(days=h)).max() < P["test"]["scoring_date"].min()
        fw, params = fit_models(df, h, start)
        preds = {**baselines(df, h), **fw}
        val = evaluate(P["validation"], h, preds)
        chosen = select(val[0], val[1])                                   # decided on validation, before test is scored
        fw2, params2 = fit_models(df, h, start, VAL_END)
        res[h] = dict(P=P, params=params, val=val, chosen=chosen, test=evaluate(P["test"], h, preds),
                      refit=evaluate(P["test"], h, {**baselines(df, h, VAL_END), **fw2}), params2=params2)
        old = baselines(df, h)["rfm-elig"](P["validation"])
        res[h]["same_as_old"] = float(np.abs(fw["fw365"](P["validation"].assign(
            inv_365=P["validation"]["frequency"], spend_365=P["validation"]["monetary"],
            tenure_365=P["validation"]["tenure_days"])) - old).max())

    L = ["# Fixed-window features: model comparison", "",
         "Generated by `fixed_window.py` from `processed/invoices.csv.gz` and `processed/cancellations.csv.gz`. Do not edit "
         "by hand. Offline only: the website and the sensitivity page are unchanged, and no individual prediction is published.", "",
         "> **Exploratory course analysis.** Neither later period is an independent holdout. Validation (Jan–Jun 2011) was "
         "examined for the earlier models and used to fit calibrators; test (Jul–Dec 2011) was examined twice and used to "
         "compare candidates. The fixed-window candidates were first scored by this script, under a rule written beforehand, "
         "but the design was informed by everything seen earlier. Nothing here confirms a model (Section 8).", "",
         "## 1. Methodological concerns found", "",
         f"1. **No training date has a full 365-day window.** The data start on {start:%d %b %Y} and training labels end by "
         f"{TRAIN_END:%d %b %Y}. During training, the 365-day features *are* the cumulative features. `fw365` therefore has "
         "the same coefficients as the old `rfm-elig` (largest difference in predicted probability when it is given the "
         f"cumulative values: {max(r['same_as_old'] for r in res.values()):.1e}). What changes is only the input later: a "
         "year of activity instead of everything since December 2009. A 180-day window is complete from "
         f"{complete_from(start, 180):%d %b %Y}, so the `fw180` models are fitted on later training dates only and mean the "
         "same thing when fitted and when used.",
         "2. **Training covers March–November 2010 only.** No December–February scoring date is in training, and validation "
         "is the post-Christmas period with a higher dormancy rate. Level errors on validation are partly seasonal.",
         "3. **Both later periods have been looked at before** (banner above). Differences of a few thousandths in Brier "
         "score or a point in capture are within what repeated looking can produce.",
         "4. **Weekly snapshots repeat customers.** Intervals come from resampling customers, for Brier differences only. "
         "AUC and top-N have none.",
         "5. **Carried over:** tenure is left-truncated at the data start; cancellations on a later day leave the purchase "
         "in the gross amounts; the 90-day horizon has few, heavily overlapping scoring dates.", "",
         "## 2. Features", "",
         "All values use invoices and cancellations dated on or before the end of the scoring date. The population is the "
         "existing one (every customer active on the date); no customer is added or removed, and nothing is filtered on "
         "what happened later.", "",
         "| Feature | Definition |", "|---|---|",
         "| `days_since_last` | Days since the last valid purchase (unchanged) |",
         "| `inv_W`, `spend_W` | Distinct valid invoices and their gross amount in the W days up to and including the scoring date |",
         "| `days_W` | Distinct purchase days in the window |",
         "| `cancel_n_W`, `net_W` | Cancellation invoices dated in the window; gross spend minus their amount, not below 0 |",
         "| `tenure_W` | Observed tenure, capped at W days |", "",
         "W = 365 or 180. An invoice exactly W days before the scoring date is outside the window; one at 23:00 on the "
         "scoring date is inside.", "",
         "### What needed inspecting", ""]
    e60 = {n: eligible(g, 60) for n, g in res[60]["P"].items()}
    per = inv.assign(day=inv["date"].dt.normalize()).groupby(["customer", "day"]).size()
    L += [f"- **Shorter history.** A customer first seen fewer than W days ago has a partly empty window. Share of eligible "
          f"rows (h = 60) with tenure under 365 days: " + ", ".join(f"{n} {pct((g['tenure_days'] < 365).mean())}" for n, g in e60.items())
          + "; under 180 days: " + ", ".join(f"{n} {pct((g['tenure_days'] < 180).mean())}" for n, g in e60.items())
          + ". The capped tenure tells the model how much of the window could be filled. The alternative, a rate per "
          "observed day, is unstable for customers a few days old and was not used.",
          "- **Recent customers** are kept. Dropping them would remove the group with the highest dormancy rate and "
          "would make the evaluated population differ from the one a list is drawn from. Section 6 reports them separately.",
          f"- **Invoice boundaries.** {pct(per[per > 1].sum() / len(inv))} of invoices share a customer-day with another "
          f"invoice ({pct((per > 1).mean())} of purchase days hold more than one). Counting invoices treats a split order as "
          "several purchases; `fw180-days` counts purchase days instead. Invoices remain the definition on the website.",
          f"- **Cancellations.** {canc['customer'].nunique():,} of {inv['customer'].nunique():,} customers have at least one "
          f"cancellation invoice. Eligible rows (h = 60) with a cancellation in the last 180 days: "
          + ", ".join(f"{n} {pct((g['cancel_n_180'] > 0).mean())}" for n, g in e60.items())
          + f"; cancelled amount as a share of gross 180-day spend: "
          + ", ".join(f"{n} {pct(1 - g['net_180'].sum() / g['spend_180'].sum())}" for n, g in e60.items())
          + ". `fw180-net` uses net spend and the cancellation count. Trade-off: net spend is closer to revenue, but a "
          "cancellation can only be matched to a customer and a date, not to its purchase, and same-day fully cancelled "
          "purchases are already absent from the invoices, hence the floor at 0.",
          "- **Window completeness** (scoring dates with the whole window inside the data):", "",
          "| Horizon | Period | Scoring dates | Full 365-day window | Full 180-day window |", "|---|---|---:|---:|---:|"]
    for h in HORIZONS:
        for n, g in res[h]["P"].items():
            d = pd.Series(g["scoring_date"].unique())
            L.append(f"| {h} | {n} | {len(d)} | {(d >= complete_from(start, 365)).sum()} | {(d >= complete_from(start, 180)).sum()} |")
    L += ["", "### Do the features still shift between periods?", "",
          "Medians among eligible rows (h = 60). Cumulative features grow with the calendar; the 180-day ones should not.", "",
          "| Period | Event rate | Invoices: cumulative | last 365 d | last 180 d | Spend (£): cumulative | last 365 d | last 180 d | "
          "Tenure: uncapped | capped 365 | capped 180 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, g in e60.items():
        md = lambda c: f"{g[c].median():,.0f}"
        L.append(f"| {n} | {pct(g['y_60'].mean())} | {md('frequency')} | {md('inv_365')} | {md('inv_180')} | {md('monetary')} | "
                 f"{md('spend_365')} | {md('spend_180')} | {md('tenure_days')} | {md('tenure_365')} | {md('tenure_180')} |")
    L += ["", "## 3. Protocol", "",
          f"- **Outcome:** recency first reaches {D} days within h days after the scoring date (h = 30, 60, 90; 90 exploratory).",
          "- **Populations:** *all active* = every customer active on a weekly scoring date; *eligible* = recency ≥ 90 − h, "
          "the only customers who can reach the threshold in time. At h = 90 the two coincide.",
          f"- **Incomplete follow-up:** a label is missing when neither the event nor the end of the window is observed by "
          f"{LAST_COMPLETE:%d %b %Y}. Those rows are dropped from every period, never counted as negative.",
          f"- **Roles:** fit on train (s + h ≤ {TRAIN_END:%d %b %Y}); compare and select on validation (s after that, "
          f"s + h ≤ {VAL_END:%d %b %Y}); test (later) is reported afterwards as exploratory. No calibrator is fitted.",
          "- **Models:** logistic regressions fitted on eligible training rows; probability 0 for ineligible rows. "
          "`constant` = training event rate; `elig-constant` = training eligible event rate, 0 if ineligible.",
          "- **Top-N:** each week, the N highest-ranked **eligible** customers. Precision = events among them ÷ N. "
          "Capture = events among them ÷ all events of that week (every event is in the eligible group). Pooled over weeks; "
          "ties at the cut-off share the remaining places. `website rule` = recency, then higher spend.",
          f"- **Selection rule, written before the first run:** per horizon a fixed-window candidate qualifies if its eligible "
          f"validation Brier score is below `{INCUMBENT}`'s with a customer-bootstrap 95% interval excluding 0, and its "
          f"top-100 capture is within {TOLERANCE:.0%} of the website rule's or better. The qualifying candidate with the "
          f"lowest Brier score is selected; otherwise `{INCUMBENT}` stays. AUC and calibration are reported, not used.", "",
          "Rows the models were fitted on:", "", "| Horizon | Model | Training rows (eligible) | Scoring dates | From |", "|---|---|---:|---:|---|"]
    for h in HORIZONS:
        for n, p in res[h]["params"].items():
            L.append(f"| {tag(h)} | {n} | {p['rows']:,} | {p['dates']} | {p['from']:%d %b %Y} |")

    def table(m, order):
        rows = ["| Model | AUC all | Brier all | AUC eligible | Brier eligible | Mean predicted (elig.) | Observed (elig.) | "
                f"Cal. slope | Δ Brier vs {INCUMBENT} [95% CI] | Δ Brier vs elig-constant [95% CI] |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for n in order:
            r = m.loc[n]
            rows.append(f"| {n} | {f3(r.auc_all)} | {f4(r.brier_all)} | {f3(r.auc_elig)} | {f4(r.brier_elig)} | {f3(r.mean_p)} | "
                        f"{f3(r.observed)} | {'–' if pd.isna(r.slope) else f'{r.slope:.2f}'} | {ci(r, 'd_inc')} | {ci(r, 'd_const')} |")
        return rows

    def toptable(t, order):
        rows = ["| Ranker | " + " | ".join(f"Precision @{n} | Capture @{n}" for n in CAPACITIES) + " | Capture @100, weekly range |",
                "|---|" + "---:|" * (2 * len(CAPACITIES)) + "---|"]
        for n in order:
            rows.append(f"| {n} | " + " | ".join(f"{pct(t.loc[(n, N), 'precision'])} | {pct(t.loc[(n, N), 'capture'])}" for N in CAPACITIES)
                        + f" | {pct(t.loc[(n, 100), 'weekly_min'])}–{pct(t.loc[(n, 100), 'weekly_max'])} |")
        return rows

    order = [*BASELINES, *SPECS]
    rank_order = ["random", "website rule", "recency-elig", "rfm-elig", *SPECS]
    L += ["", "## 4. Validation comparison (the selection period)", ""]
    for h in HORIZONS:
        m, t, g = res[h]["val"]
        e = g[g["days_since_last"] >= D - h]
        L += [f"### h = {tag(h)}", "",
              f"{g['scoring_date'].nunique()} scoring dates, {len(g):,} rows from {g['customer'].nunique():,} customers; "
              f"{len(e):,} eligible rows, {int(e[f'y_{h}'].sum()):,} events.", "", *table(m, order), "",
              *toptable(t, rank_order), "", f"**Selected by the rule: `{res[h]['chosen']}`.**", ""]
    L += ["`recency-elig` ranks by recency alone, so its list is the website's order with ties broken at random.", "",
          "## 5. Selection outcome", "",
          "| Horizon | Selected | Qualified on Brier (interval below 0) | Passed the top-100 check |", "|---|---|---|---|"]
    for h in HORIZONS:
        m, t, _ = res[h]["val"]
        a = [n for n in SPECS if m.loc[n, "d_inc_hi"] < 0]
        b = [n for n in SPECS if t.loc[(n, 100), "capture"] >= t.loc[("website rule", 100), "capture"] - TOLERANCE]
        L.append(f"| {tag(h)} | `{res[h]['chosen']}` | {', '.join(a) or 'none'} | {', '.join(b) or 'none'} |")
    L += ["", "## 6. Short histories and calibration by tenure (validation, eligible rows)", "",
          "Observed tenure on the scoring date. Mean predicted vs observed event rate, and Brier score.", "",
          "| Horizon | Tenure | Rows | Observed | " + " | ".join(f"{n}: mean p / Brier" for n in ["recency-elig", "rfm-elig", "fw365", "fw180"]) + " |",
          "|---|---|---:|---:|---|---|---|---|"]
    for h in HORIZONS:
        g = res[h]["val"][2]
        e = g[g["days_since_last"] >= D - h]
        for label, part in [("under 90 days", e[e["tenure_days"] < 90]), ("90–364 days", e[e["tenure_days"].between(90, 364)]),
                            ("365 days or more", e[e["tenure_days"] >= 365])]:
            y = part[f"y_{h}"].to_numpy(float)
            cells = [f"{part[n].mean():.3f} / {brier(y, part[n].to_numpy()):.4f}" if len(part) else "–" for n in ["recency-elig", "rfm-elig", "fw365", "fw180"]]
            L.append(f"| {tag(h)} | {label} | {len(part):,} | {pct(y.mean()) if len(part) else '–'} | " + " | ".join(cells) + " |")
    L += ["", "## 7. Exploratory: the previously inspected test period", "",
          "Reported after the selection above and not used for it. **These are not confirmation.** Test was examined on "
          "Nov 12 and Nov 19.", ""]
    for h in HORIZONS:
        m, t, g = res[h]["test"]
        e = g[g["days_since_last"] >= D - h]
        L += [f"### h = {tag(h)}: models fitted on train", "",
              f"{g['scoring_date'].nunique()} scoring dates, {len(e):,} eligible rows, {int(e[f'y_{h}'].sum()):,} events.", "",
              *table(m, order), "", *toptable(t, rank_order), ""]
    L += ["### Refit on everything through 30 Jun 2011, scored on test", "",
          f"Same specifications, fitted on all labelled rows with s + h ≤ {VAL_END:%d %b %Y} (a full year of seasons; for the "
          "365-day models, full windows from December 2010). Eligible rows.", "",
          f"| Horizon | Model | Fitted on rows | AUC | Brier | Mean predicted | Observed | Cal. slope | Δ Brier vs {INCUMBENT} [95% CI] | Capture @100 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for h in HORIZONS:
        m, t, _ = res[h]["refit"]
        for n in ["elig-constant", "recency-elig", "rfm-elig", *SPECS]:
            r = m.loc[n]
            rows_fit = f"{res[h]['params2'][n]['rows']:,}" if n in SPECS else "–"
            cap = pct(t.loc[(n, 100), "capture"]) if n != "elig-constant" else "–"
            L.append(f"| {tag(h)} | {n} | {rows_fit} | {f3(r.auc_elig)} | {f4(r.brier_elig)} | {f3(r.mean_p)} | {f3(r.observed)} | "
                     f"{'–' if pd.isna(r.slope) else f'{r.slope:.2f}'} | {ci(r, 'd_inc')} | {cap} |")
        L.append(f"| {tag(h)} | website rule | – | – | – | – | – | – | – | {pct(t.loc[('website rule', 100), 'capture'])} |")
    chosen = {h: r["chosen"] for h, r in res.items()}
    stale = [] if chosen == INTERPRETED else [
        f"> **Warning: the selection changed to {chosen} since Sections 8–10 were written for {INTERPRETED}. "
        "Re-read the tables above before relying on the text below.**", ""]
    return "\n".join(L + ["", *stale, INTERPRETATION]), res


INTERPRETED = {30: "recency-elig", 60: "fw180-days", 90: "fw180-days"}   # the selection Sections 8–10 were written for
INTERPRETATION = """## 8. Interpretation

**Ranking among eligible customers: yes, modestly, but not from the window.** Every model with frequency, spend and
tenure ranks eligible customers better than recency alone (validation AUC 0.75–0.77 vs 0.69 at 30 days, 0.80–0.81 vs
0.75 at 60, 0.85 vs 0.78 at 90). The old cumulative `rfm-elig` already had that advantage; fixed windows keep it and do
not add to it on validation (within 0.01 either way). On the already-inspected test period the fixed-window models
rank up to 0.02 better than `rfm-elig`.

**Probability accuracy and calibration: yes at 60 and 90 days with a 180-day window, no at 30 days.**

- *30 days:* no fixed-window candidate beats `recency-elig` on validation with an interval that excludes zero (best:
  −0.0011, interval −0.0031 to +0.0006). Almost every eligible customer becomes dormant (91%), so there is little to
  gain. `recency-elig` stays.
- *60 days:* the three 180-day models lower the eligible Brier score by about 0.008–0.009 (interval excludes zero);
  `fw180-days` is selected. `fw365` does not qualify (+0.0015, interval spans zero).
- *90 days (exploratory):* the same pattern, larger (−0.022).
- *Level:* on validation every model predicts too little dormancy (60 days: 0.68 for `fw180` and 0.72 for
  `recency-elig` against 0.78 observed). Training has no winter dates and validation is the post-Christmas period, so
  part of this is seasonal and no feature choice fixes it. Calibration slopes are between 1.0 and 1.2: predictions are
  slightly too cautious rather than too extreme.
- *What the window fixes:* `fw365` has exactly the coefficients of `rfm-elig`; giving it one year of activity instead
  of the whole history is the entire difference, and it removes most of `rfm-elig`'s deficit (60 days: 0.1552 vs
  0.1648). The 180-day models go further because their features mean the same thing in training and afterwards.

**Capture for limited outreach capacity: no.** At 50, 100 and 200 contacts per week, every model is within about half a
percentage point of the website rule on validation, sometimes below it at N = 50. The top of every list is customers
days from the threshold, where recency already decides. Weekly top-N values have no intervals; differences this small
should be read as none.

**By horizon.** The longer the horizon, the more a customer's purchase history matters relative to recency, and the
larger the gain. The 30-day problem is nearly settled by eligibility; the 90-day problem benefits most but is the least
well measured (13 validation dates with heavily overlapping windows).

**Seasonality.** One training year without December–February, a validation period with a 78% eligible event rate at 60
days and a test period with 70%. A model's level error changes sign between the two periods for `recency-elig`
(−0.06, then 0.00) and stays negative for the fixed-window models (−0.10, then −0.02). No model here has been fitted
and evaluated in the same season.

**Short histories.** Section 6: for customers observed under 90 days the feature models are clearly better (they know
these customers have one or two invoices; recency alone predicts 0.73 where 0.90 happens at 60 days). For customers
observed a year or more, `recency-elig` is close in level (0.70 vs 0.68 observed) while `fw180` is 9 points low, yet
`fw180` still has the better Brier score because it separates them better. The gain is therefore partly "knowing who
is new", which a tenure-aware rule could also capture.

**Invoice boundaries and cancellations.** Counting purchase days instead of invoices, or using net spend and a
cancellation count, changes the Brier score by at most 0.001 in either direction. `fw180-days` was selected by the
rule on a margin far inside the noise; `fw180`, `fw180-days` and `fw180-net` should be treated as one candidate. No
additional treatment is justified by these data.

**Repeated snapshots and uncertainty.** A customer contributes up to 21 validation rows with overlapping outcome
windows. The Brier intervals resample customers and are the only uncertainty statement here; they do not cover the
choice among candidates, the seasons, or the earlier looks at these periods.

**The exploratory test period and the refit (Section 7) do not confirm anything**, but they qualify the validation
result. With train-only fits, fixed-window models also beat `recency-elig` on test at every horizon, including 30
days. When all models are refitted through June 2011, the ordering between window lengths reverses (`fw365` best at 60
and 90 days) and even the cumulative `rfm-elig` beats `recency-elig`. So **the advantage of 180 over 365 days is a
consequence of the short training history, not a property of the window**, and a longer, seasonally complete training
period matters at least as much as the feature definition.

**Not addressed:** whether contacting any of these customers changes what they do. Every number describes who became
dormant, not who could have been kept.

### Recommendation

- **60-day horizon:** the 180-day fixed-window model (`fw180-days`, equivalently `fw180`) is ready for *additional
  evaluation* as a probability model. It is not ready for use.
- **30-day horizon:** no candidate improved on `recency-elig`. Keep it as the reference.
- **90-day horizon:** promising, exploratory only.
- **Contact lists:** no candidate improves on the rule-based watchlist. Keep the website as it is.

## 9. Confirmation plan

Nothing in Online Retail II is left unexamined, so confirmation has to come from data that do not exist in this
repository yet.

1. **Freeze.** Record one specification per horizon before any new data are seen: features and window (180 days),
   the fitting rows (all labelled rows with a complete window through the last complete day), coefficients and scaling,
   the eligibility rule, and whether a calibrator is used and on which rows. Store them as a file with a hash and a date.
   Also freeze the comparators: `recency-elig`, the eligible constant and the website rule.
2. **Score forward.** Each week, score the active customers with the frozen files and store the scores with the scoring
   date. Do not refit, recalibrate or change the feature code.
3. **Wait for outcomes to mature.** A score at date s is evaluated only after s + h + the 90-day threshold logic can be
   observed, i.e. at least h days later; rows whose follow-up is incomplete stay out.
4. **Evaluate once**, with the metrics, populations, top-N definition and decision rule of this report, written down
   before the outcomes are joined. Report every horizon, including failures.
5. **Size and season.** At least one full year of scoring dates, so each season is evaluated once; customer-level
   intervals as here. Fewer weeks give an answer about one season only.
6. **Shadow mode.** During this period the website keeps the rule-based watchlist, and nobody acts on the scores.
   If scores were used to choose whom to contact, the outcomes would no longer measure the model.

**Another dataset (external testing).** Not available in this repository, and none was downloaded or checked in this
iteration. What it would need: persistent customer identifiers; invoice or order dates; amounts; at least 18–24 months
(180-day window + training + a later evaluation year); a non-contractual setting with a meaningful share of repeat
buyers; a licence that allows the use. Candidates to check, with the reasons they may not qualify:

- *UCI "Online Retail" (2010–2011):* the same retailer and a subset of the period already used. Not independent.
- *CDNOW (1997–1998), the standard dataset of the buy-till-you-die literature:* reported as about 23,500 customers over
  18 months, but a single cohort of first-time buyers, so there are no established customers on the first dates, and it
  has no invoice identifiers or cancellations. Whether its history is long enough for a 180-day window plus evaluation
  must be checked.
- *Marketplace or grocery panels* (for example public e-commerce order datasets): often dominated by one-time buyers or
  by weekly household shopping, where a 90-day dormancy threshold means something different.

An external dataset would test **transfer** (does the specification work elsewhere after refitting, and do frozen
coefficients carry over?), not the Online Retail II model itself. A different retailer, period and purchase rhythm would
need its own dormancy threshold check before any comparison is meaningful. The upload page deliberately does not apply
the model to other data for this reason.

## 10. What this is and is not

| | Exploratory course demonstration (this report) | Evidence sufficient for production use |
|---|---|---|
| Evaluation data | Periods examined several times while the models were designed | Data untouched until a frozen model is scored |
| Selection | Rule written before the first run, but informed by earlier results | Rule and model frozen before outcomes exist |
| Seasons | Less than one year of training, no winter | At least one full year fitted and one evaluated |
| Uncertainty | Customer bootstrap for Brier differences only | Intervals for every reported metric, including lists |
| Calibration | Level off by up to 10 points on validation | Checked, and monitored, in the period of use |
| Scope | One UK online retailer, 2009–2011, 90-day threshold | The business, period and threshold it will be used for |
| Effect of acting | Not studied | A randomized or otherwise causal test of the outreach |
| Operation | Offline script | Monitoring, retraining policy, drift checks, data-quality gates |

Until then the website keeps the rule-based watchlist, the sensitivity page keeps its experimental model unchanged,
and no individual risk is published.
"""


if __name__ == "__main__":
    inv, canc, df = prepare()
    REPORT.write_text(report(inv, canc, df)[0])
    print(REPORT.read_text())
