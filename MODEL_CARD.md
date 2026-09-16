# 📋 Institutional Model Card (SR 11-7 Model Validation)

## 1. Model Overview
* **Model Name:** Institutional Quant NLP Calibrated Alpha Classifier (`CalibratedRandomForestAlphaClassifier`)
* **Model Version:** `2.1.0`
* **Model Type:** Calibrated Ensemble Decision Forest (`RandomForestClassifier` with Sigmoid / Isotonic Probability Mapping)
* **Domain:** Quantitative Asset Management / Event-Driven Alpha Extraction / BFSI
* **Regulatory Governance:** Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management Compliant

---

## 2. Intended Use & Scope
* **Primary Objective:** Detect corporate executive tone divergence and linguistic uncertainty in quarterly earnings calls to forecast post-earnings 5-day sector-neutral abnormal returns ($Sector\ CAR_{5D}$).
* **Primary Target:** 
  $$\text{Target} = \mathbb{I}(Sector\ CAR_{5D} > 0)$$
  where $Sector\ CAR_{5D} = \sum_{t=1}^5 (R_{i,t} - R_{\text{SectorETF},t})$.
* **Intended Users:** Quantitative Portfolio Managers, Equity Research Analysts, Institutional Risk Officers.
* **Non-Intended Use:** High-frequency execution, retail investment advice, unhedged directional positioning without portfolio constraints.

---

## 3. Data Universe & Temporal Scope
* **Equities Universe:** 46 liquid US Large/Mega-Cap Equities across 8 GICS Sectors (Technology, Financials, Healthcare, Consumer Discretionary, Consumer Staples, Energy, Industrials, Communication).
* **Benchmark Universe:** S&P 500 Index (`^GSPC`) and Sector ETFs (`XLK`, `XLF`, `XLV`, `XLE`, `XLP`, `XLY`, `XLI`, `XLC`).
* **Time Horizon:** 2018-Q3 to 2023-Q1 (316 audited, unique earnings call transcripts).
* **Fiscal Calendar Calibration:** Exact corporate fiscal year offsets applied (e.g., NVDA/CRM FY ends January, MSFT/PG FY ends June, AAPL/QCOM FY ends September).

---

## 4. Feature Architecture & Schema Order
The model ingests a standardized 7-dimensional feature vector in exact ordinal sequence:
1. `nsi_exec` ($[-0.38, +0.38]$): Net Sentiment Index of Executive Prepared Remarks.
2. `nsi_qa` ($[-0.38, +0.38]$): Net Sentiment Index of Analyst Q&A session.
3. `divergence` ($[-0.76, +0.76]$): Executive Tone Divergence ($NSI_{Exec} - NSI_{QA}$).
4. `qa_ratio` ($[0.1, 20.0]$): Word count ratio of Q&A session to Executive Remarks.
5. `exec_hedge_pct` ($[0.0\%, 10.0\%]$): Frequency of executive hedging words (`may`, `could`, `might`, `uncertain`).
6. `hedge_divergence` ($[-10.0\%, +10.0\%]$): Hedging Spread ($\text{Exec Hedge \%} - \text{QA Hedge \%}$).
7. `is_overconfident` ($\{0, 1\}$): Binary indicator ($\mathbb{I}(\text{Divergence} > +0.05)$).

---

## 5. Statistical Validation & Out-of-Sample Performance

### A. Purged & Embargoed Walk-Forward Cross-Validation (5 Folds)
To eliminate label overlap leakage from overlapping 5-day CAR return windows:
* **Purge Window:** 5 trading days prior to test split.
* **Embargo Window:** 5 trading days following test split.
* **Out-of-Sample Mean Precision:** $34.1\% \pm 22.1\%$ (Reflecting cross-regime market variance).
* **Mean Brier Score:** $0.2673$.

### B. Holdout Benchmark Comparison (95 Unseen Calls)
* **Monte Carlo Random (1,000 draws):** $54.8\%$ ($95\%$ Bootstrap CI: $[42.6\%, 67.4\%]$)
* **All-Events Long Benchmark:** $54.7\%$
* **Naive Sentiment Rule ($NSI_{QA} > 0$):** $54.4\%$
* **Calibrated Quant ML Model (Threshold 0.36):** $53.7\%$ (Recall: $42.3\%$, F1: $0.473$)

---

## 6. Known Model Limitations & Inherent Biases
1. **Survivorship Bias:** Restricted to prominent large-cap equities active in the Motley Fool historical archive.
2. **Regime Sensitivity:** Data window (2018–2023) spans COVID volatility and 2022 Federal Reserve rate-hike cycles, causing non-stationary factor payoffs.
3. **Thin Event Cross-Section:** On non-peak earnings days, cross-sectional portfolio breadth is constrained.

---

## 7. Model Monitoring & Governance Plan
* **Feature Drift Monitoring:** Monthly calculation of Population Stability Index (PSI) for `divergence` and `exec_hedge_pct`. PSI $> 0.25$ triggers mandatory factor review.
* **Performance Decay Trigger:** Out-of-sample rolling quarterly Brier score $> 0.30$ triggers automatic model retraining and hyperparameter calibration.
* **Model Retraining Frequency:** Quarterly refit following the conclusion of US GAAP earnings cycles.
