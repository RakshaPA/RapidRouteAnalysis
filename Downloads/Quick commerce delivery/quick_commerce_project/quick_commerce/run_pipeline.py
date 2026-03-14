"""
run_pipeline.py
---------------
One-shot script that runs the entire project pipeline:
  1. Load CSV → SQLite database
  2. EDA + feature engineering
  3. Train ML models
  4. Print launch instructions for the dashboard

Run:  python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def step(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def run(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\n❌  Script failed: {script}")
        sys.exit(1)


if __name__ == "__main__":
    step("STEP 1 — Setting up SQLite database")
    run("db_setup.py")

    step("STEP 2 — EDA + Feature Engineering")
    run("analysis.py")

    step("STEP 3 — Training ML Models")
    run("ml/train_model.py")

    step("DONE  — Launch the dashboard")
    print("""
  All steps completed successfully! 🎉

  To launch the Streamlit dashboard, run:

      streamlit run dashboard/app.py

  Then open http://localhost:8501 in your browser.
""")
