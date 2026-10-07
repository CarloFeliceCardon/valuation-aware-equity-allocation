from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def main():
    shiller = pd.read_excel(
        SHILLER_URL,
        sheet_name="Data",
        skiprows=7,
    )

    shiller = shiller[
        [
            "Date",
            "P",
            "E",
            "Rate GS10",
            "CAPE",
        ]
    ].copy()

    shiller["Date"] = pd.to_numeric(shiller["Date"], errors="coerce")
    shiller = shiller.dropna(subset=["Date"])

    year = np.floor(shiller["Date"]).astype(int)
    month = ((shiller["Date"] - year) * 100).round().astype(int)

    shiller["date"] = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": 1,
        }
    ) + pd.offsets.MonthEnd(0)

    valuation_data = shiller.rename(
        columns={
            "P": "shiller_price",
            "E": "shiller_earnings",
            "Rate GS10": "shiller_treasury_10y_yield",
            "CAPE": "shiller_cape",
        }
    )[
        [
            "date",
            "shiller_price",
            "shiller_earnings",
            "shiller_treasury_10y_yield",
            "shiller_cape",
        ]
    ].copy()

    valuation_data["shiller_earnings_yield"] = (
        valuation_data["shiller_earnings"]
        / valuation_data["shiller_price"]
    )

    output_path = DATA_DIR / "shiller_valuation_monthly.csv"
    valuation_data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(valuation_data.tail())


if __name__ == "__main__":
    main()