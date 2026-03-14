"""
ml/train_model.py
-----------------
Trains Logistic Regression, Decision Tree, and Random Forest models
to predict delivery delays.  Best model is saved as model.pkl.

Run:  python ml/train_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics            import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)
from sklearn.pipeline           import Pipeline

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent

FEATURE_COLS = [
    "cost_of_product",
    "prior_purchases",
    "discount_offered",
    "weight_gms",
    "customer_care_calls",
    "customer_rating",
    "warehouse_enc",
    "shipment_enc",
    "importance_enc",
    "gender_enc",
    "high_value",
]
TARGET = "reached_on_time"


# ─── Load features ───────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    path = DATA_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            "features.csv not found. Run analysis.py first."
        )
    return pd.read_csv(path)


# ─── Prepare data ────────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ─── Build pipelines ─────────────────────────────────────────────────────────

def build_pipelines() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=6, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            )),
        ]),
    }


# ─── Evaluate ────────────────────────────────────────────────────────────────

def evaluate(pipelines: dict, X_train, X_test, y_train, y_test) -> dict:
    results = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        roc    = roc_auc_score(y_test, y_prob)
        cv     = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()

        results[name] = {
            "pipeline":  pipe,
            "accuracy":  round(acc * 100, 2),
            "roc_auc":   round(roc, 4),
            "cv_roc":    round(cv, 4),
            "report":    classification_report(y_test, y_pred),
            "conf_mat":  confusion_matrix(y_test, y_pred).tolist(),
        }

        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  ROC-AUC  : {roc:.4f}")
        print(f"  CV AUC   : {cv:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")

    return results


# ─── Feature importance ──────────────────────────────────────────────────────

def feature_importance(pipe, name: str) -> pd.DataFrame | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        fi = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"\n🌟 Feature Importances ({name}):\n{fi.to_string(index=False)}")
        return fi
    return None


# ─── Save best model ─────────────────────────────────────────────────────────

def save_best(results: dict) -> None:
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best      = results[best_name]

    print(f"\n🏆  Best model: {best_name}  (ROC-AUC = {best['roc_auc']})")

    # Save pipeline
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best["pipeline"], f)

    # Save metadata (for dashboard)
    meta = {
        "best_model":  best_name,
        "accuracy":    best["accuracy"],
        "roc_auc":     best["roc_auc"],
        "cv_roc":      best["cv_roc"],
        "feature_cols": FEATURE_COLS,
        "all_models": {
            k: {"accuracy": v["accuracy"], "roc_auc": v["roc_auc"], "cv_roc": v["cv_roc"]}
            for k, v in results.items()
        },
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾  Saved model.pkl and model_meta.json to {MODEL_DIR}")

    # Feature importance for best tree-based model
    tree_models = {k: v for k, v in results.items()
                   if k in ("Random Forest", "Gradient Boosting", "Decision Tree")}
    if tree_models:
        top = max(tree_models, key=lambda k: results[k]["roc_auc"])
        fi  = feature_importance(results[top]["pipeline"], top)
        if fi is not None:
            fi.to_csv(MODEL_DIR / "feature_importance.csv", index=False)


if __name__ == "__main__":
    df                             = load_features()
    X_train, X_test, y_train, y_test = prepare(df)
    pipelines                      = build_pipelines()
    results                        = evaluate(pipelines, X_train, X_test, y_train, y_test)
    save_best(results)
