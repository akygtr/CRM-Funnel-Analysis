# GTM Sales Funnel Analysis & Win Propensity Model

![Domain](https://img.shields.io/badge/Domain-GTM%20%2F%20Revenue%20Analytics-blue)
![ML](https://img.shields.io/badge/ML-Classification-green)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Project Summary

This project simulates the work of a **Senior Data Analyst embedded in a GTM (Go-To-Market) / Revenue Operations team**. It combines traditional funnel analytics with a machine learning layer to predict whether an active sales deal will be **Won or Lost** — enabling sales teams to prioritize their pipeline intelligently.

---

## Business Problem

Sales teams manage hundreds of open deals simultaneously but have limited bandwidth. Without data, reps rely on gut feel to decide which deals to chase. This leads to:

- Wasted time on low-probability deals
- Missed revenue from neglected high-probability deals
- No visibility into *why* deals are won or lost

**This project answers two core business questions:**
1. *Where in our sales funnel are we losing the most deals?*
2. *Which active deals are most likely to close — and why?*

---

## How This Helps in a Real Work Environment

| Stakeholder | How They Use This |
|---|---|
| **VP of Sales** | Reviews funnel drop-off to identify pipeline health issues |
| **Sales Manager** | Uses deal scores to coach reps on which deals to prioritize |
| **Sales Rep** | Focuses energy on  High probability deals |
| **RevOps Analyst** | Monitors win rate trends by product, region, and manager |
| **Data Team** | Maintains and retrains the model as new deals come in |

In practice, the deal scoring output (Cell 16) would be pushed back into Salesforce or a BI tool like Looker so sales reps see win probabilities directly in their CRM — no need to open a notebook.

---

##  Domain

**GTM Analytics / Revenue Operations (RevOps)**

This project sits at the intersection of:
- **Sales Analytics** — funnel conversion, win rates, pipeline velocity
- **Marketing Analytics** — channel attribution, lead quality
- **Predictive Analytics** — ML-powered deal scoring
- **Business Intelligence** — executive dashboards and KPI reporting

This is one of the most in-demand data skill sets for Senior Data Analyst roles at B2B SaaS and tech companies.

---

##  Dataset

**Source:** Maven Analytics — CRM Sales Opportunities  
**Type:** Fictional B2B company selling computer hardware  
**Size:** ~4,000 deals across 4 tables

| File | Description |
|---|---|
| `sales_pipeline.csv` | Core deals table — stage, dates, value |
| `sales_teams.csv` | Sales agent → manager → regional office |
| `accounts.csv` | Company info — sector, revenue, employees, location |
| `products.csv` | Product catalogue with series and price |

---

##  Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python 3.10** | Core language |
| **Pandas** | Data loading, cleaning, feature engineering |
| **Matplotlib / Seaborn** | Funnel charts, bar charts, box plots, trend lines |
| **Scikit-learn** | Model training, cross-validation, hyperparameter tuning, evaluation metrics |
| **XGBoost** | Gradient boosted tree classifier — primary model |
| **Jupyter Notebook** | Development environment and presentation layer |

---

##  Part 1 — Funnel Analysis (EDA)

### Sales Funnel Overview
![Funnel Overview](images/01_funnel_overview.png)

### What Was Built
A full exploratory analysis of the sales funnel covering:

- **Stage conversion rates** — % of deals that progress through each stage
- **Win/Loss by product** — which products close at the highest rate
- **Sales team leaderboard** — win rate and revenue by manager and regional office
- **Deal velocity** — how many days deals spend at each stage
- **Quarter-over-quarter trends** — revenue and deal volume over time

### Key Metrics Tracked
- Overall Win Rate
- Stage-level Conversion Rate
- Average Deal Value by Product
- Total Revenue Won by Manager
- Median Days to Close by Stage

---

##  Part 2 — Win Propensity ML Model

### Problem Type
**Binary Classification** — predict whether a closed deal outcome is `Won (1)` or `Lost (0)`

### Features Used
| Feature | Type | Business Meaning |
|---|---|---|
| `close_value` | Numeric | Deal size — larger deals may behave differently |
| `days_to_close` | Numeric | How long the deal took — velocity signal |
| `engage_month` | Numeric | Seasonality — some months close better |
| `engage_quarter` | Numeric | Quarterly patterns in sales |
| `product` | Categorical | Some products win more often |
| `manager` | Categorical | Manager quality affects win rate |
| `office_location` | Categorical | Regional performance differences |
| `sector` | Categorical | Some industries buy more readily |
| `revenue` | Numeric | Account size — enterprise vs SMB |
| `employees` | Numeric | Company size signal |

### Models Trained & Compared

| Model | Why It Was Included |
|---|---|
| **Logistic Regression** | Simple baseline — interpretable coefficients |
| **Random Forest** | Handles non-linearity, robust to outliers |
| **Gradient Boosting** | Strong ensemble method for tabular data |
| **XGBoost** | State-of-the-art for structured data, fast |
| **KNN** | Distance-based — tests if similar deals cluster |
| **SVM** | Margin-based classifier for comparison |

### How Models Were Trained

1. **Data Preparation** — filtered to Won/Lost only (removed open deals), encoded categoricals with LabelEncoder, filled nulls
2. **Train/Test Split** — 80/20 stratified split to preserve class balance
3. **Cross Validation** — StratifiedKFold (5 folds) on training set for unbiased evaluation
4. **Metrics Tracked** — AUC-ROC, F1 Score, Accuracy, Training Time
5. **Hyperparameter Tuning** — RandomizedSearchCV (30 iterations) on top 2 models:
   - Random Forest: tuned `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
   - XGBoost: tuned `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
6. **Final Evaluation** — Confusion Matrix + ROC Curve on held-out test set

### Why AUC-ROC Was the Primary Metric
Win/Loss data can be imbalanced. Accuracy alone is misleading — a model predicting "Lost" for everything scores high accuracy but is useless. AUC-ROC measures how well the model *ranks* deals by win probability regardless of threshold, which is exactly what deal scoring needs.

---

## 📈 Output — Deal Scoring

The trained model scores every **active deal** (still in Prospecting or Engaging stage) with a win probability and assigns a tier:

| Tier | Win Probability | Action |
|---|---|---|
|  High | > 70% | Prioritize — close this week |
|  Medium | 40–70% | Nurture — needs attention |
|  Low | < 40% | Deprioritize or reassign |

This output is the bridge between data science and sales execution — turning a model into a daily operational tool.

---

##  Project Structure

```
crm_funnel_project/
│
├── data/
│   ├── sales_pipeline.csv
│   ├── sales_teams.csv
│   ├── accounts.csv
│   ├── products.csv
│   └── data_dictionary.csv
│
├── funnel_analysis.ipynb       ← Main notebook (EDA + ML)
└── README.md
```

---

##  How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/crm-funnel-analysis.git
cd crm-funnel-analysis

# 2. Install dependencies
pip install pandas matplotlib seaborn scikit-learn xgboost jupyter

# 3. Launch notebook
jupyter notebook funnel_analysis.ipynb
```

---

##  Future Improvements

- [ ] Add SHAP values for per-deal explainability ("why does this deal score 82%?")
- [ ] Build a Streamlit dashboard for non-technical sales managers
- [ ] Retrain model monthly as new deals close (concept drift monitoring)
- [ ] Add multi-touch attribution analysis for marketing channel ROI
- [ ] Connect to live Salesforce data via `simple-salesforce` Python library

---

##  Author

Built as part of a GTM Domain Knowledge learning path targeting **Senior Data Analyst** roles in B2B SaaS and Revenue Operations.
