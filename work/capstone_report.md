# Capstone Report — Content Refresh & Opportunity Scoring

- **Author:** Abdelrahman Mokhtar
- **Lane:** Lane 2: Content Refresh / Opportunity Scoring
- **Repo:** [https://github.com/AMokhtar51/flyrank-ml-internship](https://github.com/AMokhtar51/flyrank-ml-internship)
- **Date:** August 2026

---

## 0. Abstract

Editorial and content marketing teams face severe bandwidth constraints when managing large publication portfolios, making manual page audits across thousands of URLs infeasible. Using the FlyRank March 2026 research snapshot comprising 30,000 anonymized page records across 32 clients, we formulate content decay prediction as an out-of-sample ranking problem. We train supervised classification models (Logistic Regression, Decision Tree, and Random Forest) alongside a transparent rule-based heuristic baseline, evaluating performance on a strictly partitioned client-holdout split (80% train / 20% test across 6 held-out clients, 2,325 test rows) via `make_client_aware_split` to eliminate cross-domain data leakage. The Random Forest model achieves a Precision@50 of 82.0% and Average Precision of 0.647 with an ROC-AUC of 0.754, delivering a substantial lift over the heuristic baseline (Precision@50 = 52.0%, Avg Precision = 0.401, ROC-AUC = 0.512) against a 39.1% held-out test base rate. The resulting prioritized action queue and interpretable reason codes equip editors with decision-support triage to audit decaying high-impact pages, expand thin content, and monitor stable assets.

---

## 1. Problem Framing

- **Decision Supported:** Editorial triage — determining which decaying, high-visibility pages content strategists should audit and refresh first under finite operational bandwidth.
- **Unit of Analysis:** One published page / content URL (`content_id`) evaluated over a historical 90-day feature window.
- **Output:** A prioritized ranking score ($0$ to $100$), a recommended action (`REFRESH`, `EXPAND_AND_REFRESH`, or `MONITOR`), and a transparent reason code explaining the primary risk factor.
- **Human Action:** An editor inspects the flagged page to refresh outdated factual claims, improve search snippets / metadata, add substantive depth to thin sections, or maintain current monitoring.
- **Cost of a Wrong Call:**
  - *False Positive:* An editor spends 2–4 hours investigating and rewriting a page that was already organically stable or growing, wasting scarce creative capacity while truly decaying pages lose audience share.
  - *False Negative:* A revenue-driving article experiencing stealth SERP rank erosion or CTR decay remains unaddressed, resulting in continuous organic traffic loss until discovered months later.
- **Why Data & Machine Learning Help:** Heuristics based solely on content age (\"refresh everything older than 6 months\") fail because evergreen content frequently maintains high engagement without updates, while young articles in competitive verticals may suffer rapid SERP displacement. Machine learning models capture non-linear feature interactions between search impression volume, average SERP position, CTR, word count, and engagement consistency.

---

## 2. Data Safety & Leakage Prevention

- **Dataset Release & Scope:** FlyRank anonymized research snapshot (`data/raw/content_refresh_anonymized.csv`), containing 30,000 distinct page observations across 32 enterprise publishing clients.
- **Feature Window:** Rolling 90-day pre-intervention historical aggregates (`impressions_90d`, `clicks_90d`, `sessions_90d`, `avg_position`, `ctr`, `days_since_last_update`, `engagement_rate`, `scroll_rate`).
- **Target Window:** Forward 30-day performance trajectory (`impressions_last_30d` vs `impressions_prev_30d`).
- **Deliberate Exclusions & Safety Guardrails:**
  - *Client & Content IDs:* `client_id` and `content_id` are pseudonymous hashes used solely for grouping/splitting and record joining; they are strictly excluded from the predictive feature space.
  - *Target Leakage Fields:* `trend_direction` and `trend_pct` are excluded because they directly define the target label `is_declining_label`.
  - *Target-Window Metrics:* All forward-looking fields (`impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`) are excluded from model training.
  - *Private Information:* No client domain names, article titles, author identities, or search query strings appear anywhere in the codebase or outputs.

---

## 3. Baseline

We establish a transparent, domain-driven heuristic rule to benchmark learned model performance. The baseline identifies pages with high historical search demand ($\text{impressions}_{90d} \ge 500$) that exhibit either average SERP position degradation ($\text{avg\_position} > 20$) or content staleness ($\text{days\_since\_last\_update} \ge 90$):

$$\text{Baseline Score} = \mathbb{I}(\text{visible}) \times \left( \mathbb{I}(\text{slip}) \times \text{impressions}_{90d} + 0.3 \times \mathbb{I}(\text{stale}) \times \text{impressions}_{90d} \right)$$

On the held-out client test split (2,325 rows across 6 unseen clients, base rate = 39.1%), the baseline achieves:
- **Precision@20:** 0.550 (55.0%)
- **Precision@50:** 0.520 (52.0%)
- **Precision@100:** 0.520 (52.0%)
- **Average Precision (PR-AUC):** 0.401
- **ROC-AUC:** 0.512

The baseline demonstrates that simple volume-weighted position filtering underperforms random chance on top-ranked items because high impression volume does not inherently correlate with severe decay risk.

---

## 4. Model & Feature Engineering

We evaluated supervised classification architectures suited for tabular search data:
1. **Logistic Regression (Standardized):** Linear model providing interpretable odds ratios and calibrated probabilities.
2. **Decision Tree (Max Depth = 6):** Non-linear tree establishing transparent decision boundaries.
3. **Random Forest (200 Trees, Max Depth = 12):** Ensemble model capturing non-linear interactions across SERP position, search volume, content length, and engagement rates.

### Feature Space (26 Base Features, One-Hot Encoded to 52 Dimensions):
- **Continuous Metrics (18):** `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`, `days_with_impressions`, `days_with_sessions`, `content_age_days`, `days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`.
- **Categorical Tiers (8):** `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`.
- **Target Definition:** `is_declining_label` = $1$ if `trend_pct < 0` (forward 30-day impression drop), $0$ otherwise.

---

## 5. Evaluation

### Split Strategy: Grouped Client-Holdout Split
To guarantee that models generalize across distinct domains and CMS architectures rather than memorizing domain-specific idiosyncrasies, we partition the dataset using the reference `make_client_aware_split` function from `scripts/03_train_model.py` (80% train / 20% test, `RANDOM_STATE = 42`) grouped by `client_id`.
- **Training Set:** 27,675 pages across 26 clients.
- **Held-Out Test Set:** 2,325 pages across 6 completely unseen clients (Base Rate = 0.391).

### Comparative Metric Receipts (Held-Out Test Split)

| Method | Precision@20 | Precision@50 | Precision@100 | Avg Precision (PR-AUC) | ROC-AUC | Precision | Recall | F1 | Accuracy | Base Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Rule Baseline** | 0.550 | 0.520 | 0.520 | 0.401 | 0.512 | — | — | — | — | 0.391 |
| **Decision Tree** | 0.400 | 0.420 | 0.500 | 0.562 | 0.736 | 0.568 | 0.743 | 0.644 | 0.679 | 0.391 |
| **Logistic Regression** | 0.350 | 0.360 | 0.420 | 0.520 | 0.700 | 0.562 | 0.656 | 0.605 | 0.666 | 0.391 |
| **Random Forest** | **0.800** | **0.820** | **0.840** | **0.647** | **0.754** | 0.527 | 0.804 | 0.637 | 0.641 | 0.391 |

### Error Analysis & Observations:
- **Top Queue Precision Lift:** Random Forest achieves **82.0% Precision@50** (and 84.0% at Precision@100), delivering a **1.58× lift** over the heuristic baseline (52.0%) and more than doubling the test base rate (39.1%).
- **False Positives:** Highly visible pages ranked on deep SERP positions ($>30$) that maintain steady niche traffic despite low CTR.
- **False Negatives:** Recently updated pages ($<30$ days) that suffered unexpected SERP displacement due to external competitor movements or core algorithm shifts.

---

## 6. Interpretation

### Feature Importance & Risk Factors
Feature importance analysis in Random Forest and coefficient inspection in Logistic Regression reveal the primary signals driving content decay:
1. **Search Visibility (`log_impressions_90d` — 11.4%):** High visibility provides strong statistical power to detect meaningful trend shifts.
2. **Average Position (`avg_position` — 11.0%):** Pages hovering on Page 2 or Page 3 (positions 11–30) are vulnerable to ranking drops that cause steep traffic loss.
3. **Activity Consistency (`days_with_impressions` — 10.8%):** Pages with sporadic daily search impressions have higher decline probabilities than steady daily performers.
4. **Content Staleness (`content_age_days` — 7.9%):** Older content that has not been refreshed experiences natural information decay.
5. **Content Depth (`word_count` — 4.6% & `char_count` — 4.2%):** Thin content exhibits higher vulnerability to SERP volatility.

---

## 7. Recommendation & Action Playbook

We translate predictive probabilities and heuristic signals into an operational action queue for editorial teams:

### Action Playbook Categories
1. **`REFRESH` (9,053 pages in portfolio):** High-visibility pages experiencing documented traffic decline or position slippage. Editorial action: Audit key sections, update outdated facts, and refine page metadata.
2. **`EXPAND_AND_REFRESH` (14 pages in portfolio):** High-visibility pages with thin content length ($<1,200$ words). Editorial action: Add comprehensive sections, FAQs, and multimedia assets.
3. **`MONITOR` (20,933 pages in portfolio):** Stable or growing content with healthy rank trajectories. Editorial action: Maintain observation without active revision.

### Operational Guardrails & No-Go Policy
- **Human Review Checklist:** Always verify search intent shifts, seasonal trends, and commercial conversion value before committing rewriting resources.
- **No-Go Exclusions:** Never programmatically refresh legal, privacy, terms, or brand core navigation pages. Hold recently updated pages ($<30$ days) in cooldown.
- **Monitoring & Retrain Triggers:** Trigger scheduled retraining quarterly, or immediately if held-out Precision@50 drops below 0.55 or median portfolio CTR drifts by $>20\%$.

---

## 8. Reproducibility

All analysis, modeling, and output artifacts can be reproduced deterministically from a fresh clone of this repository:

### Rerun Instructions
```bash
# 1. Clone repository
git clone https://github.com/AMokhtar51/flyrank-ml-internship.git
cd flyrank-ml-internship

# 2. Run the complete pipeline
python scripts/run_all.py

# 3. Execute the capstone notebook
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

- **Random Seed:** `RANDOM_STATE = 42` across all data partitions and model initializations.
- **Environment:** Python 3.13 / 3.14 with `pandas>=2.2.0`, `scikit-learn>=1.5.0`, `numpy>=1.26.0`, `reportlab>=4.0.0`.
- **Metric Receipts:** Verified against committed output files `work/outputs/receipts.json` and `data/processed/model_predictions.csv`.

---

## 9. Acknowledgments & Data Credit

Built on the FlyRank ML Internship dataset, provided by [https://flyrank.ai](https://flyrank.ai).
