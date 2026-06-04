import math
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from src.api.routes.sentiment import safe_float

PRICES_DIR = Path("data/raw/prices")

router = APIRouter()


@router.get("/prices/{ticker}/{date}")
def get_prices(ticker: str, date: str):
    p = PRICES_DIR / f"{ticker}_{date}.csv"
    if not p.exists():
        return {"prices": []}

    df = pd.read_csv(p, index_col=0, parse_dates=True)
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_convert(None) if idx.tz is not None else idx

    records = []
    for dt, row in df.iterrows():
        vol = row.get("Volume", 0)
        try:
            vol_int = int(float(vol)) if not math.isnan(float(vol)) else 0
        except (TypeError, ValueError):
            vol_int = 0
        records.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "open": safe_float(row.get("Open")),
                "high": safe_float(row.get("High")),
                "low": safe_float(row.get("Low")),
                "close": safe_float(row.get("Close")),
                "volume": vol_int,
            }
        )
    return {"prices": records}
