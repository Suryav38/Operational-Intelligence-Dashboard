"""Load the structured CSVs into a SQLite database the AI assistant can query (read-only)."""
import sqlite3
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "manrag" / "structured"
db_path = root / "data" / "manrag" / "manrag_factory.db"

if db_path.exists():
    db_path.unlink()

conn = sqlite3.connect(db_path)
for csv in sorted(data_dir.glob("*.csv")):
    pd.read_csv(csv).to_sql(csv.stem, conn, if_exists="replace", index=False)
    print(f"Loaded {csv.name} into table {csv.stem}")
conn.close()
print("SQLite database ready:", db_path)
