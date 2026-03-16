# Quick Commerce Delivery Analytics

End-to-end analytics project covering SQL, Python EDA, Machine Learning,
and a Streamlit dashboard — built on the **E-Commerce Shipping Dataset**.

---

## Project Structure

```
quick_commerce/
│
├── data/
│   ├── Train.csv              ← Raw dataset (place here)
│   ├── delivery.db            ← SQLite DB (auto-generated)
│   ├── features.csv           ← Feature-engineered CSV (auto-generated)
│   └── figures/               ← Matplotlib plots (auto-generated)
│
├── sql/
│   └── schema_and_queries.sql ← Full SQL schema + 10 analytical queries
│
├── ml/
│   ├── train_model.py         ← Model training (LR, DT, RF, GBT)
│   ├── model.pkl              ← Saved best model (auto-generated)
│   ├── model_meta.json        ← Metrics & metadata (auto-generated)
│   └── feature_importance.csv ← Feature importances (auto-generated)
│
├── dashboard/
│   └── app.py                 ← Streamlit dashboard (5 tabs)
│
├── db_setup.py                ← CSV → SQLite loader
├── analysis.py                ← EDA + feature engineering + plots
├── run_pipeline.py            ← One-shot pipeline runner
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```
This runs all three steps automatically.

### 3. Launch dashboard
```bash
streamlit run dashboard/app.py
```

---

## Dataset Columns

| Column | Description |
|---|---|
| `ID` | Order identifier |
| `Warehouse_block` | Warehouse (A / B / C / D / F) |
| `Mode_of_Shipment` | Flight / Ship / Road |
| `Customer_care_calls` | Support calls made |
| `Customer_rating` | 1–5 rating |
| `Cost_of_the_Product` | Order cost (₹) |
| `Prior_purchases` | Previous orders by customer |
| `Product_importance` | low / medium / high |
| `Gender` | M / F |
| `Discount_offered` | Discount % |
| `Weight_in_gms` | Package weight |
| `Reached.on.Time_Y.N` | **Target** — 1=Delayed, 0=On Time |

---

##  ML Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Decision Tree | Interpretable |
| Random Forest | Best performance |
| Gradient Boosting | Tuned ensemble |

---

##  Dashboard Tabs

| Tab | Contents |
|---|---|
| Overview | KPI cards, delay split donut, warehouse/shipment charts |
| Deep Analysis | Care calls, weight & discount distributions, correlation heatmap |
| SQL Insights | 5 pre-built SQL queries + custom query runner |
| ML Model | Model comparison, feature importance, live predictor |
| Raw Data | Filterable table + CSV download |

---

##  Key Findings

- ~60% of orders in the dataset are **delayed**
- Higher customer care calls → higher delay probability
- Heavier packages tend to be delivered on time more often
- Large discounts are associated with delayed shipments
- Warehouse blocks **D & F** show highest delay rates
