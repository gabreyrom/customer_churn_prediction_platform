# Non-contractual Customer Dormancy Prediction

Final project for **AAI 627: Big Data Analytics for Engineering Design**,
part of the **M.S. in Data Science** at **Stevens Institute of Technology**.

A Streamlit website that turns retail purchase history into a current view of each customer:
who is **engaged**, who is **approaching dormancy**, and who is already **dormant**. Its goal is
to help a retention team decide whom to contact when it can reach only a limited number of
customers.

> **Current stage: November 19 milestone (offline model evaluation).** The website runs on the real
> **Online Retail II** data (cleaned, invoice level). Customer status and the watchlist are
> **rule-based**. Dormancy-risk models, their calibration, and top-N targeting are evaluated
> **offline only** (see [Offline modeling](#offline-modeling)); the website does not display model
> predictions or evaluation results. See [Project status](#project-status).
>
> **Live site:** https://gabriel-reynoso-customer-churn.streamlit.app (rule-based status and watchlist; no model predictions).

---

## Contents

1. [The problem](#the-problem)
2. [Key definitions](#key-definitions)
3. [Dataset](#dataset)
4. [Quick start](#quick-start)
5. [Website tour](#website-tour)
6. [Offline modeling](#offline-modeling)
7. [Project status](#project-status)
8. [Repository structure](#repository-structure)

---

## The problem

In **non-contractual** retail, customers never formally cancel; they simply stop buying. The
business has no explicit "churn" event, so it has to infer engagement from purchase behavior.

**Business question:** *Which customers appear engaged, which are approaching dormancy, and which
are already dormant?*

**Who would use it:** retention, CRM, and marketing teams at a retailer.

**Why ranking matters:** outreach is limited. A team may only be able to email, call, or send
offers to **N** customers per cycle, so the site has to **rank** customers, not just label them.

- Customers who are **still active but approaching the threshold** are the main outreach
  candidates.
- Customers who are **already dormant** are listed separately, because winning them back is a
  different task.

---

## Key definitions

| Term | Meaning |
|---|---|
| **Valid purchase** | A non-cancelled invoice with an identified customer, positive quantities and prices, and at least one product line (see [Cleaning rules](#cleaning-rules)) |
| **Observation ("as of") date** | The date the analysis is run from. Only transactions on or before it are used |
| **Recency** | Days from the last valid purchase to the as-of date |
| **Frequency** | Number of distinct valid invoices up to the as-of date |
| **Monetary** | Gross spend (£) on valid invoices up to the as-of date; returns are not netted |
| **Dormant** | Recency of at least **90 days** (provisional threshold; 60 and 120 days will also be compared) |

**Dormancy is a state, not an exit.** A dormant customer may purchase again.

### Provisional segments (threshold T, default 90 days)

| Segment | Rule |
|---|---|
| Engaged | recency < T − 30 |
| Approaching dormancy | T − 30 ≤ recency < T (still active) |
| Dormant | recency ≥ T (threshold already crossed) |

**Current status vs. future risk:** *status* describes where a customer stands today, based on
recency. *Dormancy risk* would be a forecast of who becomes dormant next. Only status is shown
today. Risk is a planned feature.

---

## Dataset

**Online Retail II**, from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
It contains transactions from a UK-based online retailer.

Credit: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D. Licensed under CC BY 4.0.

Verified by this project:

- Two sheets, `Year 2009-2010` and `Year 2010-2011`, with **1,067,371** rows in total
- Purchases from **1 Dec 2009 to 9 Dec 2011**. The last day is partial: purchases stop at 12:50
- Columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`,
  `Customer ID`, `Country`

The raw workbook is **not** committed. Download it to `data/online_retail_II.xlsx` (gitignored) and
run `python prepare_data.py` (needs `pip install openpyxl`) to rebuild the files in `processed/`.
The website reads only `processed/invoices.csv.gz`.

### Cleaning rules

Full counts are in [`processed/data_quality.md`](processed/data_quality.md).

| Rule | Rows removed |
|---|---:|
| Sheet 1 repeats 1–9 Dec 2010, which sheet 2 also holds; keep the sheet-2 copy (verified identical) | 22,523 |
| Missing `Customer ID` | 235,287 |
| Cancellation invoices (`C` prefix), stored separately in `processed/cancellations.csv.gz` | 18,446 |
| Quantity ≤ 0 or price ≤ 0 | 70 |
| `TEST*` and `ADJUST*` lines | 45 |
| Invoices with only postage, carriage, manual, bank-charge or discount lines | 350 |
| Purchases fully cancelled on the **same calendar day** (98 invoices, 77 customers) | 809 |

What remains is **36,496 invoices from 5,851 customers**, with a gross amount of **£16,945,042.16**.

- **Same-day full cancellations are removed.** A purchase is removed only when all of its lines
  (StockCode, quantity, price) are matched by cancellation lines from the same customer, on the
  same calendar day, issued at or after the purchase. This includes invoice 581483 (£168,469.60,
  customer 16446) and invoice 541431 (£77,183.60, customer 12346).
  - Because as-of dates are end-of-day, this never uses information that would be unavailable
    on the date being viewed.
  - **Ambiguous days are kept:** on 9 customer-days, each invoice matches on its own but the
    cancellations can't cover all of them together.
  - **Non-line-for-line cancellations are kept:** for example, invoice 556444 (£38,970) was
    reversed by a single manual `M` line, so it stays in the data.
- **Later cancellations are never applied.** A purchase cancelled on a later day stays in, so
  earlier as-of views are not affected by what happened afterwards.
- **Identical lines within a sheet are kept (11,602 rows).** They cannot be told apart from
  genuine repeat lines. Removing them would lower the gross total by £56,306.31.
- **Rounding:** each invoice amount is rounded to the penny once, and every total is the sum of
  those rounded amounts.

---

## Quick start

**Requirements:** Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Run the website
streamlit run app.py
```

Then open **http://localhost:8501**. Press `Ctrl+C` in the terminal to stop the server.

---

## Website tour

Current screenshots are in [`artifacts/submission5/`](artifacts/submission5/).

### 1. Main page (sidebar: "Home")

- **Headline and introduction:** "Protect Your Customer Lifecycle".
- **Upload transaction data:** shown but **disabled**, and labeled *Not yet available*.
- **Open customer status:** opens the results page.
- **Explore continuous monitoring:** informational only, **not implemented**.
- **What you'll see:** a preview of the features. Dormancy risk is marked **Planned**.

### 2. Customer status

Built from the Online Retail II data. The dataset's date range and the selected as-of date are
shown at the top, so the "current" status always means status on that date within the
dataset, not today.

| Control | Effect |
|---|---|
| As-of date | Uses only purchases on or before this date, then recalculates everything |
| Dormancy threshold (60 / 90 / 120 days) | Moves the segment boundaries and the earliest as-of date you can choose |
| Contact-list size N | Sets how many customers appear in the watchlist |
| Customer | Shows that customer's invoices up to the as-of date |

What the page shows:

- **Status counts:** engaged, approaching dormancy, dormant.
- **Recency distribution** with the threshold marked.
- **Rule-based watchlist:** the top N *active* customers approaching dormancy.
  - Ranked by recency, with customers closest to the threshold first.
  - Ties are broken by higher gross spend.
  - This is **not** predicted dormancy risk.
- **Already dormant:** a separate re-activation list.
- **Purchase timeline** for a selected customer.
- **Future dormancy risk:** prediction, model comparison, calibration and alert timing are all
  marked **Planned**.

![Main page](artifacts/submission5/main_page.png)

![Customer status page](artifacts/submission5/results_page.png)

The earlier mock-data version is in [`artifacts/submission4/`](artifacts/submission4/).

---

## Offline modeling

Dormancy-risk models and their evaluation. They run **offline**: nothing here is read by the Streamlit
pages, and the deployed website (https://gabriel-reynoso-customer-churn.streamlit.app) still shows the
rule-based watchlist.

```bash
python build_snapshots.py    # weekly snapshots + 30/60/90-day labels -> data/weekly_snapshots.csv.gz
python train_baselines.py    # Nov 12 baselines -> processed/baseline_results.md, data/models/baselines.json
python evaluate_models.py    # Nov 19 evaluation -> processed/model_evaluation.md, processed/reliability_eligible.png
python test_snapshots.py && python test_baselines.py && python test_evaluation.py   # checks
```

The modeling scripts need pandas and numpy; `evaluate_models.py` also needs matplotlib
(`pip install matplotlib`). These are not in `requirements.txt`, which lists only what the website needs.
`evaluate_models.py` takes about 15 seconds. It also re-checks the snapshot pipeline on real data (an
independent recomputation of 1,000 rows, and a truncation test for future data).

- **Snapshots:** every Thursday from 4 Mar 2010 to 8 Dec 2011 (the last complete day), each customer who
  is still active under the 90-day threshold. Features use only purchases up to the end of that day:
  recency, tenure, frequency, and gross spend.
- **Label:** the customer first reaches 90 days without a valid purchase within the next 30, 60 or 90
  days. Outcomes that run past 8 Dec 2011 are **censored** and excluded, not counted as negative.
  Method and audit: [`processed/snapshot_audit.md`](processed/snapshot_audit.md).
- **Calendar split** (no random rows): train labels end by 31 Dec 2010, validation by 30 Jun 2011,
  and test by 8 Dec 2011. Each label is fully observed before the next period starts.
- **Nov 12 baselines:** a constant event rate, a recency-only logistic regression, and a recency +
  tenure + frequency + monetary logistic regression. Recency was chosen on validation Brier score.
  Results: [`processed/baseline_results.md`](processed/baseline_results.md).
- **Nov 19 evaluation:** [`processed/model_evaluation.md`](processed/model_evaluation.md).
  - **Populations:** each model is scored on all active customers and on customers *eligible* to cross the
    threshold within the horizon (recency ≥ 90 − h).
  - **Calibration:** reliability tables and a plot, plus calibration slopes.
  - **Targeting:** weekly top-N (N = 50/100/200), compared with random selection and the website's own
    recency rule.
  - **Model study:** recalibration (Platt) and refitting on eligible customers. Fitting uses train,
    calibration uses validation, and comparison uses test.
  - **Main findings:**
    - Overall metrics mostly reflect eligibility.
    - The Nov 12 models are miscalibrated for eligible customers.
    - Refitting on eligible customers or recalibrating improves probabilities.
    - RFM's frequency, spend and tenure features are cumulative and drift over time.
    - No model beats the website's recency ordering for short contact lists by more than about 2 points.
- **The test period is not an untouched holdout.** It was first inspected on Nov 12, so every test result
  from Nov 19 is **exploratory**. The data end in December 2011, so no later holdout exists. Confirming a
  model needs a prospective (shadow) evaluation. The 90-day horizon is also exploratory.
- **Top-N capture measures predictive targeting, not the effect of contacting customers.**
- Generated tables, predictions and model files live in `data/`, which is gitignored.

---

## Project status

| Area | Status |
|---|---|
| Problem definition and dormancy rule | ✅ Defined (threshold is provisional) |
| Data audit and cleaning of Online Retail II | ✅ Done (`prepare_data.py`, `processed/data_quality.md`) |
| RFM status, segment counts, timelines on real data | ✅ Works |
| Rule-based top-N watchlist | ✅ Works. A rule, not a prediction |
| File upload | ⏳ Shown as disabled. Not functional |
| Time-aware use of cancellations and returns | ⏳ Planned |
| Weekly snapshots, censoring-aware labels, calendar split | ✅ Done, offline (`build_snapshots.py`) |
| Initial logistic-regression baselines (AUC, Brier) | ✅ Done, offline (`train_baselines.py`). Not shown on the site |
| Calibration, eligible-group and top-N evaluation; recalibration study | ✅ Done, offline (`evaluate_models.py`). Test results exploratory |
| Window-stable features and a prospective holdout | ⏳ Next step |
| Alert-timing evaluation | ⏳ Planned |
| Model predictions on the website | ⏳ Planned. The site still shows the rule-based watchlist |
| Deployment to Streamlit Community Cloud | ✅ Live at https://gabriel-reynoso-customer-churn.streamlit.app (rule-based watchlist only) |
| Continuous monitoring integration | ❌ Out of scope for the course |

All modeling is offline. The website does not show any model probability, accuracy, calibration,
or outreach-effect values.

---

## Repository structure

```
.
├── app.py                  # Entry point: page navigation and sidebar notice
├── main_page.py            # Landing page
├── results_page.py         # Customer status dashboard (segments, watchlist, timeline)
├── prepare_data.py         # Rebuilds processed/ from the raw UCI workbook
├── build_snapshots.py      # Offline: weekly snapshots and 30/60/90-day dormancy labels
├── train_baselines.py      # Offline: split comparison and logistic-regression baselines
├── evaluate_models.py      # Offline: calibration, eligible-group and top-N evaluation, recalibration study
├── test_snapshots.py       # Checks for labels, resets, censoring
├── test_baselines.py       # Checks for predictor allowlist and split boundaries
├── test_evaluation.py      # Checks for eligibility, role boundaries, calibration separation, top-N
├── processed/              # invoices.csv.gz (used by the site), cancellations.csv.gz, data_quality.md,
│                           #   snapshot_audit.md, baseline_results.md, model_evaluation.md,
│                           #   reliability_eligible.png
├── requirements.txt        # streamlit, pandas, altair
├── .streamlit/config.toml  # Theme
├── artifacts/submission5/  # Current screenshots of both pages
├── artifacts/submission4/  # Screenshots of the earlier mockup
├── .gitignore              # Ignores data/ (raw workbook, snapshots, models), virtual environments, caches, and secrets
└── README.md
```

---

*Course project for AAI 627 at Stevens Institute of Technology. Dataset source: Online Retail II,
UCI Machine Learning Repository.*
