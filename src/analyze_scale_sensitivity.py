from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAINING_MONTHS = 120
FORECAST_HORIZON_MONTHS = 12
SCALES = [0.03, 0.05, 0.07, 0.10]
TRANSACTION_COST_BPS = 5


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


def metrics(returns, cash_returns):
    return {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, cash_returns),
        "maximum_drawdown": maximum_drawdown(returns),
    }


def create_forecasts(data):
    forecasts = []
    first_forecast_index = TRAINING_MONTHS + FORECAST_HORIZON_MONTHS

    for i in range(first_forecast_index, len(data)):
        train_data = data.iloc[: i - FORECAST_HORIZON_MONTHS]
        current_data = data.iloc[i]

        x_train = sm.add_constant(
            train_data[
                [
                    "cape_earnings_yield",
                    "treasury_10y_yield",
                ]
            ]
        )

        model = sm.OLS(
            train_data["forward_12m_total_return"],
            x_train,
        ).fit()

        x_current = pd.DataFrame(
            {
                "const": [1.0],
                "cape_earnings_yield": [
                    current_data["cape_earnings_yield"]
                ],
                "treasury_10y_yield": [
                    current_data["treasury_10y_yield"]
                ],
            }
        )

        forecasts.append(
            {
                "date": current_data["date"],
                "predicted_12m_return": model.predict(x_current).iloc[0],
            }
        )

    return pd.DataFrame(forecasts)


def main():
    data = pd.read_csv(
        DATA_DIR / "research_dataset_monthly.csv",
        parse_dates=["date"],
    )

    required_columns = [
        "cape_earnings_yield",
        "treasury_10y_yield",
        "forward_12m_total_return",
        "t_bill_3m_yield",
        "cash_monthly_return",
        "spy_return",
    ]

    data = (
        data
        .dropna(subset=required_columns)
        .sort_values("date")
        .reset_index(drop=True)
    )

    forecasts = create_forecasts(data)

    base = data.merge(forecasts, on="date", how="left")

    results = []
    cost_rate = TRANSACTION_COST_BPS / 10_000

    for scale in SCALES:
        backtest = base.copy()

        backtest["equity_weight"] = np.clip(
            (
                backtest["predicted_12m_return"]
                - backtest["t_bill_3m_yield"]
            ) / scale,
            0,
            1,
        )

        backtest["equity_position"] = (
            backtest["equity_weight"].shift(1)
        )

        backtest["strategy_gross_return"] = (
            backtest["equity_position"] * backtest["spy_return"]
            + (1 - backtest["equity_position"])
            * backtest["cash_monthly_return"]
        )

        backtest["strategy_turnover"] = (
            backtest["equity_position"]
            .diff()
            .abs()
            .fillna(0)
        )

        backtest["strategy_net_return"] = (
            backtest["strategy_gross_return"]
            - cost_rate * backtest["strategy_turnover"]
        )

        backtest = backtest.dropna(
            subset=[
                "equity_position",
                "strategy_net_return",
            ]
        ).copy()

        average_weight = backtest["equity_position"].mean()

        backtest["static_return"] = (
            average_weight * backtest["spy_return"]
            + (1 - average_weight)
            * backtest["cash_monthly_return"]
        )

        for portfolio_name, return_column in {
            "Strategy": "strategy_net_return",
            "Static benchmark": "static_return",
        }.items():
            results.append(
                {
                    "scale": scale,
                    "portfolio": portfolio_name,
                    "average_spy_weight": average_weight,
                    "annualized_turnover": (
                        backtest["strategy_turnover"].mean() * 12
                        if portfolio_name == "Strategy"
                        else np.nan
                    ),
                    **metrics(
                        backtest[return_column],
                        backtest["cash_monthly_return"],
                    ),
                }
            )

    results = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "scale_sensitivity.csv"
    results.to_csv(output_path, index=False)

    display = results.copy()

    display["scale"] = display["scale"].map(
        lambda value: f"{value:.0%}"
    )

    for column in [
        "average_spy_weight",
        "annualized_turnover",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
    ]:
        display[column] = display[column].map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.2%}"
        )

    display["sharpe_ratio"] = display["sharpe_ratio"].map(
        lambda value: f"{value:.2f}"
    )

    print(
        f"Transaction cost assumption: "
        f"{TRANSACTION_COST_BPS} bps per dollar traded\n"
    )

    print(display.to_string(index=False))
    print(f"\nResults saved in: {output_path}")


if __name__ == "__main__":
    main()