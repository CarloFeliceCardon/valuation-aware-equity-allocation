from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main():
    market = pd.read_csv(
        DATA_DIR / "market_data_monthly.csv",
        parse_dates=["date"],
    )

    market["forward_12m_total_return"] = (
        market["spy_price"].shift(-12) / market["spy_price"] - 1
    )

    valuation = pd.read_csv(
        DATA_DIR / "shiller_valuation_monthly.csv",
        parse_dates=["date"],
    )

    dataset = market.merge(
        valuation,
        on="date",
        how="inner",
    )

    dataset["cape_earnings_yield"] = 1 / dataset["shiller_cape"]

    dataset["earnings_yield_minus_treasury"] = (
        dataset["shiller_earnings_yield"]
        - dataset["treasury_10y_yield"]
    )

    output_path = DATA_DIR / "research_dataset_monthly.csv"
    dataset.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Observations: {len(dataset)}")
    print(f"Period: {dataset['date'].min().date()} — {dataset['date'].max().date()}")
    print(dataset.tail())


if __name__ == "__main__":
    main()