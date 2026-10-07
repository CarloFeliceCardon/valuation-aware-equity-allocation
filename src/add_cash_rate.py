from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"

    cash_rate = pd.read_csv(url)

    date_column = cash_rate.columns[0]

    cash_rate[date_column] = pd.to_datetime(cash_rate[date_column])
    cash_rate["TB3MS"] = pd.to_numeric(cash_rate["TB3MS"], errors="coerce")

    cash_rate = cash_rate.rename(
        columns={
            date_column: "date",
            "TB3MS": "t_bill_3m_yield",
        }
    )

    cash_rate = cash_rate.set_index("date").resample("ME").last().ffill()
    cash_rate = cash_rate.reset_index()

    cash_rate["t_bill_3m_yield"] = cash_rate["t_bill_3m_yield"] / 100

    cash_rate["cash_monthly_return"] = (
        (1 + cash_rate["t_bill_3m_yield"]) ** (1 / 12) - 1
    )

    market = pd.read_csv(
        DATA_DIR / "market_data_monthly.csv",
        parse_dates=["date"],
    )

    market = market.merge(
        cash_rate[
            [
                "date",
                "t_bill_3m_yield",
                "cash_monthly_return",
            ]
        ],
        on="date",
        how="left",
    )

    output_path = DATA_DIR / "market_data_monthly.csv"
    market.to_csv(output_path, index=False)

    print(f"Dataset updated at: {output_path}")
    print(market.tail())


if __name__ == "__main__":
    main()