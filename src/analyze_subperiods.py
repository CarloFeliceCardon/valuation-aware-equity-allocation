from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PERIODS = {
    "2011–2016: low-rate expansion": ("2011-03-31", "2016-12-31"),
    "2017–2019: late cycle": ("2017-01-31", "2019-12-31"),
    "2020–2023: Covid, inflation and hikes": (
        "2020-01-31",
        "2023-09-30",
    ),
}


def annualized_return(returns):
    return (1 + returns).prod() ** (12 / len(returns)) - 1


def annualized_volatility(returns):
    return returns.std() * np.sqrt(12)


def sharpe_ratio(returns, cash_returns):
    excess_returns = returns - cash_returns

    return (
        excess_returns.mean()
        / excess_returns.std()
        * np.sqrt(12)
    )


def maximum_drawdown(returns):
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    return drawdown.min()


def calculate_metrics(returns, cash_returns):
    return {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, cash_returns),
        "maximum_drawdown": maximum_drawdown(returns),
    }


def main():
    data = pd.read_csv(
        OUTPUT_DIR / "gradual_allocation_backtest.csv",
        parse_dates=["date"],
    )

    portfolios = {
        "Strategy": "strategy_return",
        "Static benchmark": "static_benchmark_return",
        "SPY buy-and-hold": "spy_return",
    }

    results = []

    for period_name, (start_date, end_date) in PERIODS.items():
        period_data = data[
            (data["date"] >= start_date)
            & (data["date"] <= end_date)
        ].copy()

        for portfolio_name, return_column in portfolios.items():
            metrics = calculate_metrics(
                period_data[return_column],
                period_data["cash_monthly_return"],
            )

            results.append(
                {
                    "period": period_name,
                    "portfolio": portfolio_name,
                    "months": len(period_data),
                    **metrics,
                }
            )

    results = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "subperiod_performance.csv"
    results.to_csv(output_path, index=False)

    display = results.copy()

    for column in [
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
    ]:
        display[column] = display[column].map(
            lambda value: f"{value:.2%}"
        )

    display["sharpe_ratio"] = display["sharpe_ratio"].map(
        lambda value: f"{value:.2f}"
    )

    print(display.to_string(index=False))
    print(f"\nResults saved in: {output_path}")


if __name__ == "__main__":
    main()