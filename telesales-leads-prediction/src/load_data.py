"""Reload raw CSV data without dropping duplicates or filling missing values."""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db import create_db_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_leads():
    # Only empty CSV fields become NULL; literal text such as "NA" is preserved.
    leads = pd.read_csv(
        PROJECT_ROOT / "data" / "leads.csv",
        keep_default_na=False,
        na_values=[""],
    )
    leads.columns = leads.columns.str.lower().str.replace(" ", "_", regex=False)
    leads["created_at"] = pd.to_datetime(leads["created_at"], errors="raise")
    return leads


def main():
    engine = None
    try:
        leads = load_leads()
        engine = create_db_engine()
        # Explicit full reload; a failed insert rolls back to the previous data.
        with engine.begin() as connection:
            connection.execute(text(
                "TRUNCATE TABLE raw_leads RESTART IDENTITY"
            ))
            leads.to_sql(
                "raw_leads", connection, if_exists="append", index=False,
                chunksize=1000,
            )
        print(f"Loaded {len(leads):,} rows into raw_leads.")
    except Exception as exc:
        print(f"Data loading failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())

