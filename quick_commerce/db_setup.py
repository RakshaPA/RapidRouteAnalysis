"""
db_setup.py
-----------
Loads Train.csv into a local SQLite database (delivery.db)
and runs validation queries.

Run:  python db_setup.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH  = Path(__file__).parent / "data" / "delivery.db"
CSV_PATH = Path(__file__).parent / "data" / "Train.csv"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id                  INTEGER PRIMARY KEY,
            warehouse_block     TEXT    NOT NULL,
            mode_of_shipment    TEXT    NOT NULL,
            cost_of_product     INTEGER NOT NULL,
            prior_purchases     INTEGER NOT NULL,
            product_importance  TEXT    NOT NULL,
            gender              TEXT    NOT NULL,
            discount_offered    INTEGER NOT NULL,
            weight_gms          INTEGER NOT NULL,
            customer_care_calls INTEGER NOT NULL,
            customer_rating     INTEGER NOT NULL,
            reached_on_time     INTEGER NOT NULL
        )
    """)
    conn.commit()


def load_csv(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    # Rename columns to snake_case to match DB schema
    df.rename(columns={
        "ID":                    "id",
        "Warehouse_block":       "warehouse_block",
        "Mode_of_Shipment":      "mode_of_shipment",
        "Cost_of_the_Product":   "cost_of_product",
        "Prior_purchases":       "prior_purchases",
        "Product_importance":    "product_importance",
        "Gender":                "gender",
        "Discount_offered":      "discount_offered",
        "Weight_in_gms":         "weight_gms",
        "Customer_care_calls":   "customer_care_calls",
        "Customer_rating":       "customer_rating",
        "Reached.on.Time_Y.N":   "reached_on_time",
    }, inplace=True)

    # Drop if already loaded
    conn.execute("DELETE FROM orders")
    df.to_sql("orders", conn, if_exists="append", index=False)
    conn.commit()
    print(f"✅  Loaded {len(df):,} rows into 'orders' table.")
    return df


def validate(conn: sqlite3.Connection) -> None:
    queries = {
        "Total orders":    "SELECT COUNT(*) FROM orders",
        "Delayed orders":  "SELECT SUM(reached_on_time) FROM orders",
        "Delay rate (%)":  "SELECT ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) FROM orders",
    }
    print("\n📊 Validation:")
    for label, sql in queries.items():
        val = conn.execute(sql).fetchone()[0]
        print(f"   {label}: {val}")


if __name__ == "__main__":
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_connection()
    create_table(conn)
    load_csv(conn)
    validate(conn)
    conn.close()
    print(f"\n💾  Database saved to: {DB_PATH}")
