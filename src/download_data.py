from pathlib import Path
from io import StringIO

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = "2000-01-01"
END_DATE = "2026-10-01"


def download_spy_prices():
    """Download SPY adjusted prices and convert them to monthly observations."""
    spy = yf.download(
        "SPY",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    prices = spy["Close"]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices = prices.rename("spy_price").to_frame()
    prices.index.name = "date"

    monthly_prices = prices.resample("ME").last()
    monthly_prices["spy_return"] = monthly_prices["spy_price"].pct_change()

    return monthly_prices


def download_treasury_yield():
    """Download the 10-year US Treasury yield from FRED."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"

    treasury = pd.read_csv(url)

    date_column = treasury.columns[0]
    treasury[date_column] = pd.to_datetime(treasury[date_column])
    treasury["DGS10"] = pd.to_numeric(treasury["DGS10"], errors="coerce")

    treasury = treasury.rename(
        columns={
            date_column: "date",
            "DGS10": "treasury_10y_yield",
        }
    ).set_index("date")

    monthly_treasury = treasury.resample("ME").last().ffill()

    return monthly_treasury


def main():
    spy = download_spy_prices()
    treasury = download_treasury_yield()

    dataset = spy.join(treasury, how="inner")
    dataset["treasury_10y_yield"] = dataset["treasury_10y_yield"] / 100

    output_path = DATA_DIR / "market_data_monthly.csv"
    dataset.to_csv(output_path)

    print(f"Dataset saved to: {output_path}")
    print(dataset.tail())


if __name__ == "__main__":
    main()