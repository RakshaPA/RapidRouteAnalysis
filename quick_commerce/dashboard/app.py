"""
dashboard/app.py
----------------
Streamlit dashboard for Quick Commerce Delivery Analytics.

Run:  streamlit run dashboard/app.py
"""

import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
ML_DIR    = ROOT / "ml"
DB_PATH   = DATA_DIR / "delivery.db"

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quick Commerce Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }
    .green-card  { background: linear-gradient(135deg,#11998e,#38ef7d) !important; }
    .red-card    { background: linear-gradient(135deg,#f7971e,#ffd200) !important; color:#333 !important; }
    .blue-card   { background: linear-gradient(135deg,#2196F3,#21CBF3) !important; }
    .stTabs [data-baseweb="tab"]        { font-size: 1rem; font-weight: 600; }
    section[data-testid="stSidebar"]    { background: #1a1a2e; }
    section[data-testid="stSidebar"] *  { color: #eee; }
</style>
""", unsafe_allow_html=True)


# ─── Data loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Train.csv").rename(columns={
        "ID": "id", "Warehouse_block": "warehouse_block",
        "Mode_of_Shipment": "mode_of_shipment",
        "Cost_of_the_Product": "cost_of_product",
        "Prior_purchases": "prior_purchases",
        "Product_importance": "product_importance",
        "Gender": "gender", "Discount_offered": "discount_offered",
        "Weight_in_gms": "weight_gms",
        "Customer_care_calls": "customer_care_calls",
        "Customer_rating": "customer_rating",
        "Reached.on.Time_Y.N": "reached_on_time",
    })


@st.cache_data
def load_features() -> pd.DataFrame | None:
    p = DATA_DIR / "features.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_model():
    p = ML_DIR / "model.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_meta() -> dict:
    p = ML_DIR / "model_meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data
def load_fi() -> pd.DataFrame | None:
    p = ML_DIR / "feature_importance.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def sql_query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(sql, conn)
    conn.close()
    return df


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def sidebar(df: pd.DataFrame):
    st.sidebar.image("https://img.icons8.com/color/96/delivery.png", width=64)
    st.sidebar.title("⚡ Quick Commerce")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filters")

    warehouses = ["All"] + sorted(df["warehouse_block"].unique().tolist())
    sel_wh     = st.sidebar.selectbox("Warehouse Block", warehouses)

    shipments  = ["All"] + sorted(df["mode_of_shipment"].unique().tolist())
    sel_ship   = st.sidebar.selectbox("Shipment Mode", shipments)

    importance = ["All"] + sorted(df["product_importance"].unique().tolist())
    sel_imp    = st.sidebar.selectbox("Product Importance", importance)

    st.sidebar.markdown("---")
    st.sidebar.info("📦 Dataset: 10,999 orders\n\n🔍 Target: Delivery Delay (1=Delayed)")

    # Apply filters
    mask = pd.Series([True] * len(df))
    if sel_wh   != "All": mask &= df["warehouse_block"]    == sel_wh
    if sel_ship != "All": mask &= df["mode_of_shipment"]   == sel_ship
    if sel_imp  != "All": mask &= df["product_importance"] == sel_imp
    return df[mask]


# ─── KPI cards ───────────────────────────────────────────────────────────────

def kpi_cards(df: pd.DataFrame):
    total   = len(df)
    delayed = df["reached_on_time"].sum()
    on_time = total - delayed
    delay_r = delayed / total * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, cls in [
        (c1, f"{total:,}",        "Total Orders",    "blue-card"),
        (c2, f"{delayed:,}",      "Delayed Orders",  "red-card"),
        (c3, f"{on_time:,}",      "On-Time Orders",  "green-card"),
        (c4, f"{delay_r:.1f}%",   "Delay Rate",      "metric-card"),
    ]:
        col.markdown(
            f'<div class="metric-card {cls}"><h2>{val}</h2><p>{label}</p></div>',
            unsafe_allow_html=True,
        )


# ─── Tab 1 — Overview ────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame):
    st.header("📊 Delivery Overview")

    # Donut + bar side by side
    col1, col2 = st.columns([1, 1.6])

    with col1:
        counts = df["reached_on_time"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({1: "Delayed", 0: "On Time"})
        fig = px.pie(
            counts, values="Count", names="Status",
            hole=0.5, color="Status",
            color_discrete_map={"Delayed": "#E74C3C", "On Time": "#2ECC71"},
            title="Delay vs On-Time Split",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        wh = (
            df.groupby("warehouse_block")["reached_on_time"]
            .agg(["sum", "count"])
            .reset_index()
        )
        wh.columns = ["Warehouse", "Delayed", "Total"]
        wh["On Time"] = wh["Total"] - wh["Delayed"]
        wh_m = wh.melt("Warehouse", ["Delayed", "On Time"], var_name="Status", value_name="Orders")
        fig2 = px.bar(
            wh_m, x="Warehouse", y="Orders", color="Status", barmode="group",
            color_discrete_map={"Delayed": "#E74C3C", "On Time": "#2ECC71"},
            title="Orders by Warehouse Block",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Shipment mode breakdown
    ship = (
        df.groupby("mode_of_shipment")["reached_on_time"]
        .mean().mul(100).reset_index()
        .rename(columns={"reached_on_time": "Delay %"})
    )
    fig3 = px.bar(
        ship, x="mode_of_shipment", y="Delay %", color="Delay %",
        color_continuous_scale="RdYlGn_r",
        labels={"mode_of_shipment": "Shipment Mode"},
        title="Delay Rate (%) by Shipment Mode",
        text_auto=".1f",
    )
    st.plotly_chart(fig3, use_container_width=True)


# ─── Tab 2 — Deep Analysis ───────────────────────────────────────────────────

def tab_analysis(df: pd.DataFrame):
    st.header("🔬 Deep Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Customer care calls
        cc = (
            df.groupby("customer_care_calls")["reached_on_time"]
            .mean().mul(100).reset_index()
            .rename(columns={"reached_on_time": "Delay %", "customer_care_calls": "Care Calls"})
        )
        fig = px.line(
            cc, x="Care Calls", y="Delay %", markers=True,
            title="Delay Rate vs Customer Care Calls",
            color_discrete_sequence=["#E74C3C"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Product importance
        pi = (
            df.groupby("product_importance")["reached_on_time"]
            .mean().mul(100).reset_index()
            .rename(columns={"reached_on_time": "Delay %", "product_importance": "Importance"})
        )
        fig2 = px.bar(
            pi, x="Importance", y="Delay %", color="Delay %",
            color_continuous_scale="Oranges",
            title="Delay Rate by Product Importance",
            text_auto=".1f",
            category_orders={"Importance": ["low", "medium", "high"]},
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Weight distribution
        fig3 = px.histogram(
            df, x="weight_gms",
            color=df["reached_on_time"].map({1: "Delayed", 0: "On Time"}),
            barmode="overlay",
            opacity=0.7,
            nbins=40,
            title="Weight Distribution (Delayed vs On Time)",
            color_discrete_map={"Delayed": "#E74C3C", "On Time": "#2ECC71"},
            labels={"color": "Status", "weight_gms": "Weight (gms)"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Discount distribution
        fig4 = px.histogram(
            df, x="discount_offered",
            color=df["reached_on_time"].map({1: "Delayed", 0: "On Time"}),
            barmode="overlay",
            opacity=0.7,
            nbins=35,
            title="Discount Distribution (Delayed vs On Time)",
            color_discrete_map={"Delayed": "#E74C3C", "On Time": "#2ECC71"},
            labels={"color": "Status", "discount_offered": "Discount (%)"},
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔗 Correlation Heatmap")
    num_cols = ["cost_of_product", "prior_purchases", "discount_offered",
                "weight_gms", "customer_care_calls", "customer_rating", "reached_on_time"]
    corr = df[num_cols].corr().round(2)
    fig5 = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="Feature Correlation Matrix",
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Prior purchases
    pp = (
        df.groupby("prior_purchases")["reached_on_time"]
        .mean().mul(100).reset_index()
        .rename(columns={"reached_on_time": "Delay %", "prior_purchases": "Prior Purchases"})
    )
    fig6 = px.bar(
        pp, x="Prior Purchases", y="Delay %",
        title="Delay Rate vs Prior Purchases",
        color="Delay %", color_continuous_scale="Blues",
        text_auto=".1f",
    )
    st.plotly_chart(fig6, use_container_width=True)


# ─── Tab 3 — SQL Insights ────────────────────────────────────────────────────

def tab_sql():
    st.header("🗃️ SQL-Based Insights")

    if not DB_PATH.exists():
        st.warning("⚠️  Database not found. Run `python db_setup.py` first.")
        return

    queries = {
        "Delay by Warehouse": """
            SELECT warehouse_block,
                   COUNT(*) AS total_orders,
                   SUM(reached_on_time) AS delayed_orders,
                   ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
            FROM orders GROUP BY warehouse_block ORDER BY delay_pct DESC
        """,
        "Delay by Shipment Mode": """
            SELECT mode_of_shipment,
                   COUNT(*) AS total,
                   ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
            FROM orders GROUP BY mode_of_shipment ORDER BY delay_pct DESC
        """,
        "Discount vs Delay": """
            SELECT CASE reached_on_time WHEN 1 THEN 'Delayed' ELSE 'On Time' END AS status,
                   ROUND(AVG(discount_offered),2) AS avg_discount,
                   ROUND(AVG(cost_of_product),2)  AS avg_cost,
                   ROUND(AVG(weight_gms),2)        AS avg_weight_gms
            FROM orders GROUP BY reached_on_time
        """,
        "Customer Care Calls Impact": """
            SELECT customer_care_calls,
                   COUNT(*) AS total,
                   ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
            FROM orders GROUP BY customer_care_calls ORDER BY customer_care_calls
        """,
        "Weight Bucket Analysis": """
            SELECT CASE
                     WHEN weight_gms < 2000 THEN 'Light (<2kg)'
                     WHEN weight_gms < 4000 THEN 'Medium (2-4kg)'
                     WHEN weight_gms < 6000 THEN 'Heavy (4-6kg)'
                     ELSE 'Very Heavy (>6kg)'
                   END AS weight_bucket,
                   COUNT(*) AS total,
                   ROUND(100.0*SUM(reached_on_time)/COUNT(*),2) AS delay_pct
            FROM orders GROUP BY weight_bucket ORDER BY delay_pct DESC
        """,
    }

    sel = st.selectbox("Choose a SQL Query", list(queries.keys()))
    st.code(queries[sel].strip(), language="sql")

    result = sql_query(queries[sel])
    st.dataframe(result, use_container_width=True)

    # Auto-visualise
    if "delay_pct" in result.columns:
        x_col = result.columns[0]
        fig   = px.bar(
            result, x=x_col, y="delay_pct", text_auto=".1f",
            color="delay_pct", color_continuous_scale="RdYlGn_r",
            labels={"delay_pct": "Delay %"},
            title=f"Delay Rate: {sel}",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Custom SQL
    st.subheader("🖊️ Run Custom SQL Query")
    custom = st.text_area(
        "Enter SQL query",
        "SELECT * FROM orders LIMIT 10",
        height=120,
    )
    if st.button("▶ Run Query"):
        try:
            res = sql_query(custom)
            st.dataframe(res, use_container_width=True)
            st.success(f"Returned {len(res)} rows.")
        except Exception as e:
            st.error(f"Error: {e}")


# ─── Tab 4 — ML Model ────────────────────────────────────────────────────────

def tab_model():
    st.header("🤖 Delay Prediction Model")

    meta  = load_meta()
    model = load_model()
    fi    = load_fi()

    if not meta:
        st.warning("⚠️  Model not trained yet. Run `python ml/train_model.py` first.")
        return

    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    all_models = meta.get("all_models", {})
    if all_models:
        perf_df = pd.DataFrame(all_models).T.reset_index()
        perf_df.columns = ["Model", "Accuracy (%)", "ROC-AUC", "CV ROC-AUC"]

        fig = px.bar(
            perf_df.melt("Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            title="Model Performance Metrics",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(perf_df, use_container_width=True)

    # Best model info
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Model",   meta.get("best_model", "N/A"))
    col2.metric("🎯 Accuracy",     f"{meta.get('accuracy', 0)}%")
    col3.metric("📈 ROC-AUC",      meta.get("roc_auc", 0))

    # Feature importance
    if fi is not None:
        st.subheader("🌟 Feature Importance")
        fig2 = px.bar(
            fi, x="importance", y="feature", orientation="h",
            title="Feature Importance (Best Tree Model)",
            color="importance", color_continuous_scale="Viridis",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    # Live predictor
    if model is not None:
        st.subheader("🔮 Live Delay Predictor")
        st.info("Fill in the order details below to get a delay probability.")

        col1, col2, col3 = st.columns(3)
        with col1:
            cost          = st.slider("Cost of Product (₹)",    100, 800, 250)
            prior         = st.slider("Prior Purchases",          1,  10,   3)
            discount      = st.slider("Discount Offered (%)",     0,  70,  10)
            weight        = st.slider("Weight (gms)",           500, 8000, 3000)
        with col2:
            care_calls    = st.slider("Customer Care Calls",     1,   7,   3)
            rating        = st.slider("Customer Rating",         1,   5,   3)
            high_value    = int(cost > 250)
        with col3:
            warehouse     = st.selectbox("Warehouse Block", ["A","B","C","D","F"])
            shipment      = st.selectbox("Shipment Mode",   ["Flight","Ship","Road"])
            importance    = st.selectbox("Product Importance", ["low","medium","high"])
            gender        = st.selectbox("Gender",          ["M","F"])

        wh_enc   = {"A":0,"B":1,"C":2,"D":3,"F":4}[warehouse]
        sh_enc   = {"Flight":0,"Road":1,"Ship":2}[shipment]
        imp_enc  = {"low":0,"medium":1,"high":2}[importance]
        gen_enc  = 1 if gender == "M" else 0

        features = np.array([[
            cost, prior, discount, weight,
            care_calls, rating,
            wh_enc, sh_enc, imp_enc, gen_enc, high_value,
        ]])

        if st.button("🔮 Predict Delay", type="primary"):
            prob    = model.predict_proba(features)[0][1]
            pred    = "🚨 DELAYED" if prob > 0.5 else "✅ ON TIME"
            color   = "#E74C3C" if prob > 0.5 else "#2ECC71"

            st.markdown(
                f"<h2 style='color:{color};text-align:center'>{pred}</h2>",
                unsafe_allow_html=True,
            )
            fig3 = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                title = {"text": "Delay Probability (%)"},
                gauge = {
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": color},
                    "steps": [
                        {"range": [0,  40], "color": "#EAFAF1"},
                        {"range": [40, 70], "color": "#FDEBD0"},
                        {"range": [70,100], "color": "#FADBD8"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
                },
            ))
            st.plotly_chart(fig3, use_container_width=True)


# ─── Tab 5 — Raw Data ────────────────────────────────────────────────────────

def tab_data(df: pd.DataFrame):
    st.header("📋 Raw Data Explorer")
    st.write(f"Showing {len(df):,} records (after filters)")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.dataframe(df.dtypes.reset_index().rename(columns={"index":"Column", 0:"DType"}),
                     use_container_width=True)

    st.subheader("Sample Records")
    n = st.slider("Rows to display", 5, 100, 20)
    st.dataframe(df.head(n), use_container_width=True)

    st.download_button(
        "⬇️ Download filtered CSV",
        df.to_csv(index=False).encode(),
        "filtered_data.csv",
        "text/csv",
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    df_raw  = load_raw()
    df_filt = sidebar(df_raw)

    st.title("🚀 Quick Commerce Delivery Analytics")
    st.markdown("*Analytics dashboard for Zepto / Blinkit / Swiggy-style delivery operations*")
    st.markdown("---")

    kpi_cards(df_filt)
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔬 Deep Analysis",
        "🗃️ SQL Insights",
        "🤖 ML Model",
        "📋 Raw Data",
    ])

    with tab1: tab_overview(df_filt)
    with tab2: tab_analysis(df_filt)
    with tab3: tab_sql()
    with tab4: tab_model()
    with tab5: tab_data(df_filt)


if __name__ == "__main__":
    main()
