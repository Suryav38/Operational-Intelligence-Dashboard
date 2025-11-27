import sqlite3
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "manrag" / "structured"
db_path = root / "data" / "manrag" / "manrag_factory.db"

conn = sqlite3.connect(db_path)

for csv in data_dir.glob("*.csv"):
    df = pd.read_csv(csv)
    table_name = csv.stem
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"--> Loaded {csv.name} into table {table_name}")

conn.close()
print("\n🎉 SQLite Database Ready:", db_path)
