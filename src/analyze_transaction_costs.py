from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

COST_SCENARIOS_BPS = [0, 5, 10]


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

    static_weight = data["equity_position"].mean()

    data["strategy_turnover"] = (
        data["equity_position"]
        .diff()
        .abs()
        .fillna(0)
    )

    static_portfolio_return = data["static_benchmark_return"]

    drifted_static_equity_weight = (
        static_weight
        * (1 + data["spy_return"])
        / (1 + static_portfolio_return)
    )

    data["static_turnover"] = (
        (static_weight - drifted_static_equity_weight)
        .abs()
        .shift(1)
        .fillna(0)
    )

    print(
        "Annualized strategy turnover: "
        f"{data['strategy_turnover'].mean() * 12:.2%}"
    )

    print(
        "Annualized static-benchmark turnover: "
        f"{data['static_turnover'].mean() * 12:.2%}"
    )

    results = []

    for cost_bps in COST_SCENARIOS_BPS:
        cost_rate = cost_bps / 10_000

        strategy_net_return = (
            data["strategy_return"]
            - cost_rate * data["strategy_turnover"]
        )

        static_net_return = (
            data["static_benchmark_return"]
            - cost_rate * data["static_turnover"]
        )

        for portfolio_name, returns in {
            "Strategy": strategy_net_return,
            "Static benchmark": static_net_return,
        }.items():
            metrics = calculate_metrics(
                returns,
                data["cash_monthly_return"],
            )

            results.append(
                {
                    "cost_bps": cost_bps,
                    "portfolio": portfolio_name,
                    **metrics,
                }
            )

    results = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "transaction_cost_sensitivity.csv"
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

    print("\nTransaction-cost sensitivity:")
    print(display.to_string(index=False))
    print(f"\nResults saved in: {output_path}")


if __name__ == "__main__":
    main()