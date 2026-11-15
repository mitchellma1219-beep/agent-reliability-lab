"""Read-only adapter for Google's OFFICIALLY SIMULATED Meridian marketing data.

It validates source schema but intentionally does not claim causal optimization.
Requires outbound internet for --fetch. Uses Python standard library only.
"""
import argparse
import csv
import io
from pathlib import Path
from urllib.request import urlopen

SOURCE="https://raw.githubusercontent.com/google/meridian/main/meridian/data/simulated_data/csv/national_all_channels.csv"

def inspect(text: str) -> dict:
    rows=list(csv.DictReader(io.StringIO(text)))
    required=["time","conversions","revenue_per_conversion","Promo",
              "competitor_sales_control",*[f"Channel{i}_spend" for i in range(5)]]
    if not rows or any(c not in rows[0] for c in required):
        raise ValueError("Unexpected Meridian dataset schema")
    dates=[r["time"] for r in rows]
    if dates!=sorted(dates) or len(dates)!=len(set(dates)):
        raise ValueError("Missing chronological uniqueness")
    for row in rows:
        for c in required[5:]:
            if float(row[c])<0:raise ValueError("Negative channel spend")
    return {"rows":len(rows),"channels":5,"synthetic":True,
            "first_date":dates[0],"last_date":dates[-1],
            "causal_evidence":False}

def download(destination: Path) -> dict:
    with urlopen(SOURCE,timeout=30) as f:
        payload=f.read().decode("utf-8")
    info=inspect(payload)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(payload,encoding="utf-8")
    return info

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fetch",action="store_true")
    p.add_argument("--path",type=Path,default=Path(__file__).resolve().parents[1]/"data/meridian_national_simulated.csv")
    args=p.parse_args()
    if not args.fetch:p.error("Use --fetch to download an official public SIMULATED dataset")
    print(download(args.path))
