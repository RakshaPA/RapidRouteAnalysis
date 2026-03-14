"""
analysis.py
-----------
Exploratory Data Analysis & Feature Engineering.
Reads from the SQLite database, performs EDA, and saves
cleaned feature-engineered data for the ML model.

Run:  python analysis.py
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from db_setup import get_connection

OUTPUT_DIR = Path(__file__).parent / "data"
FIG_DIR    = Path(__file__).parent / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


# ─── Load data ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql("SELECT * FROM orders", conn)
    conn.close()
    return df


# ─── EDA ─────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("QUICK COMMERCE — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nShape          : {df.shape}")
    print(f"Missing values :\n{df.isnull().sum()}")
    print(f"\nClass balance  :\n{df['reached_on_time'].value_counts()}")
    print(f"\nDescriptive stats:\n{df.describe()}")


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Weight bucket
    df["weight_bucket"] = pd.cut(
        df["weight_gms"],
        bins=[0, 2000, 4000, 6000, 9000],
        labels=["Light", "Medium", "Heavy", "Very Heavy"],
    )

    # High value order flag
    df["high_value"] = (df["cost_of_product"] > 250).astype(int)

    # Discount tier
    df["discount_tier"] = pd.cut(
        df["discount_offered"],
        bins=[-1, 0, 10, 30, 66],
        labels=["No Discount", "Low", "Medium", "High"],
    )

    # Encode categoricals for ML
    df["warehouse_enc"]    = df["warehouse_block"].astype("category").cat.codes
    df["shipment_enc"]     = df["mode_of_shipment"].astype("category").cat.codes
    df["importance_enc"]   = df["product_importance"].map({"low": 0, "medium": 1, "high": 2})
    df["gender_enc"]       = (df["gender"] == "M").astype(int)

    print("\n✅  Feature engineering complete.")
    return df


# ─── Visualisations ──────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame) -> None:

    # 1. Delay rate overall (donut)
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df["reached_on_time"].value_counts()
    ax.pie(
        counts,
        labels=["Not Delayed", "Delayed"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.5},
        colors=["#4CAF50", "#F44336"],
    )
    ax.set_title("Overall Delivery Delay Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delay_rate_donut.png", dpi=150)
    plt.close()

    # 2. Delay by warehouse block
    wh = (
        df.groupby("warehouse_block")["reached_on_time"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"reached_on_time": "delay_pct"})
        .sort_values("delay_pct", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=wh, x="warehouse_block", y="delay_pct", palette="Reds_d", ax=ax)
    ax.set_title("Delay Rate by Warehouse Block", fontweight="bold")
    ax.set_ylabel("Delay %")
    ax.set_xlabel("Warehouse Block")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delay_by_warehouse.png", dpi=150)
    plt.close()

    # 3. Delay by shipment mode
    ship = (
        df.groupby("mode_of_shipment")["reached_on_time"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"reached_on_time": "delay_pct"})
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=ship, x="mode_of_shipment", y="delay_pct", palette="Blues_d", ax=ax)
    ax.set_title("Delay Rate by Shipment Mode", fontweight="bold")
    ax.set_ylabel("Delay %")
    ax.set_xlabel("Mode of Shipment")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delay_by_shipment.png", dpi=150)
    plt.close()

    # 4. Customer care calls vs delay
    cc = (
        df.groupby("customer_care_calls")["reached_on_time"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"reached_on_time": "delay_pct"})
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=cc, x="customer_care_calls", y="delay_pct", marker="o", color="#E74C3C", ax=ax)
    ax.set_title("Delay Rate vs Customer Care Calls", fontweight="bold")
    ax.set_ylabel("Delay %")
    ax.set_xlabel("Customer Care Calls")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delay_vs_care_calls.png", dpi=150)
    plt.close()

    # 5. Weight distribution (delayed vs on-time)
    fig, ax = plt.subplots(figsize=(8, 4))
    for val, label, color in [(0, "On Time", "#2ECC71"), (1, "Delayed", "#E74C3C")]:
        sns.kdeplot(
            df.loc[df["reached_on_time"] == val, "weight_gms"],
            label=label,
            fill=True,
            alpha=0.4,
            color=color,
            ax=ax,
        )
    ax.set_title("Weight Distribution: Delayed vs On Time", fontweight="bold")
    ax.set_xlabel("Weight (gms)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "weight_distribution.png", dpi=150)
    plt.close()

    # 6. Discount offered vs delay
    fig, ax = plt.subplots(figsize=(8, 4))
    for val, label, color in [(0, "On Time", "#2ECC71"), (1, "Delayed", "#E74C3C")]:
        sns.kdeplot(
            df.loc[df["reached_on_time"] == val, "discount_offered"],
            label=label,
            fill=True,
            alpha=0.4,
            color=color,
            ax=ax,
        )
    ax.set_title("Discount Offered: Delayed vs On Time", fontweight="bold")
    ax.set_xlabel("Discount (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "discount_distribution.png", dpi=150)
    plt.close()

    # 7. Correlation heatmap
    num_cols = ["cost_of_product", "prior_purchases", "discount_offered",
                "weight_gms", "customer_care_calls", "customer_rating", "reached_on_time"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 8. Product importance vs delay
    pi = (
        df.groupby("product_importance")["reached_on_time"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"reached_on_time": "delay_pct"})
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=pi, x="product_importance", y="delay_pct",
                order=["low", "medium", "high"], palette="Oranges_d", ax=ax)
    ax.set_title("Delay Rate by Product Importance", fontweight="bold")
    ax.set_ylabel("Delay %")
    ax.set_xlabel("Product Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delay_by_importance.png", dpi=150)
    plt.close()

    print(f"📊  8 plots saved to {FIG_DIR}")


# ─── SQL-based analytics ─────────────────────────────────────────────────────

def run_sql_analytics() -> dict:
    conn = get_connection()
    results = {}

    results["delay_by_warehouse"] = pd.read_sql("""
        SELECT warehouse_block,
               COUNT(*) AS total,
               ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
        FROM orders GROUP BY warehouse_block ORDER BY delay_pct DESC
    """, conn)

    results["delay_by_shipment"] = pd.read_sql("""
        SELECT mode_of_shipment,
               COUNT(*) AS total,
               ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
        FROM orders GROUP BY mode_of_shipment ORDER BY delay_pct DESC
    """, conn)

    results["discount_vs_delay"] = pd.read_sql("""
        SELECT CASE reached_on_time WHEN 1 THEN 'Delayed' ELSE 'On Time' END AS status,
               ROUND(AVG(discount_offered),2) AS avg_discount,
               ROUND(AVG(cost_of_product),2)  AS avg_cost,
               ROUND(AVG(weight_gms),2)        AS avg_weight
        FROM orders GROUP BY reached_on_time
    """, conn)

    results["care_calls_delay"] = pd.read_sql("""
        SELECT customer_care_calls,
               COUNT(*) AS total,
               ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
        FROM orders GROUP BY customer_care_calls ORDER BY customer_care_calls
    """, conn)

    results["weight_bucket_delay"] = pd.read_sql("""
        SELECT CASE
                 WHEN weight_gms < 2000 THEN 'Light'
                 WHEN weight_gms < 4000 THEN 'Medium'
                 WHEN weight_gms < 6000 THEN 'Heavy'
                 ELSE 'Very Heavy'
               END AS weight_bucket,
               COUNT(*) AS total,
               ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
        FROM orders GROUP BY weight_bucket ORDER BY delay_pct DESC
    """, conn)

    conn.close()

    print("\n📋  SQL Analytics Results:")
    for k, v in results.items():
        print(f"\n— {k.replace('_',' ').title()} —")
        print(v.to_string(index=False))

    return results


if __name__ == "__main__":
    df = load_data()
    run_eda(df)
    df = engineer_features(df)
    df.to_csv(OUTPUT_DIR / "features.csv", index=False)
    print(f"\n💾  Feature CSV saved to {OUTPUT_DIR / 'features.csv'}")
    plot_all(df)
    run_sql_analytics()
