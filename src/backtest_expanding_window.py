from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAINING_MONTHS = 120
FORECAST_HORIZON_MONTHS = 12


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
    running_maximum = wealth.cummax()
    drawdown = wealth / running_maximum - 1

    return drawdown.min()


def print_performance(name, returns, cash_returns):
    print(f"\n{name}")
    print(f"Annualized return:     {annualized_return(returns):.2%}")
    print(f"Annualized volatility: {annualized_volatility(returns):.2%}")
    print(f"Sharpe ratio:          {sharpe_ratio(returns, cash_returns):.2f}")
    print(f"Maximum drawdown:      {maximum_drawdown(returns):.2%}")


def main():
    data = pd.read_csv(
        DATA_DIR / "research_dataset_monthly.csv",
        parse_dates=["date"],
    )

    data = data.sort_values("date").reset_index(drop=True)

    required_columns = [
        "cape_earnings_yield",
        "treasury_10y_yield",
        "forward_12m_total_return",
        "t_bill_3m_yield",
        "cash_monthly_return",
        "spy_return",
    ]

    data = data.dropna(subset=required_columns).copy()

    forecasts = []

    first_forecast_index = TRAINING_MONTHS + FORECAST_HORIZON_MONTHS

    for i in range(first_forecast_index, len(data)):
        train_data = data.iloc[: i - FORECAST_HORIZON_MONTHS].copy()
        current_data = data.iloc[i]

        x_train = sm.add_constant(
            train_data[
                [
                    "cape_earnings_yield",
                    "treasury_10y_yield",
                ]
            ]
        )

        y_train = train_data["forward_12m_total_return"]

        model = sm.OLS(y_train, x_train).fit()

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

        predicted_12m_return = model.predict(x_current).iloc[0]
        cash_hurdle = current_data["t_bill_3m_yield"]

        forecasts.append(
            {
                "date": current_data["date"],
                "predicted_12m_return": predicted_12m_return,
                "cash_hurdle": cash_hurdle,
                "equity_signal": int(predicted_12m_return > cash_hurdle),
            }
        )

    forecasts = pd.DataFrame(forecasts)

    backtest = data.merge(
        forecasts,
        on="date",
        how="left",
    )

    backtest["equity_position"] = backtest["equity_signal"].shift(1)

    backtest["decision_date"] = backtest["date"].shift(1)

    backtest["decision_predicted_12m_return"] = (
        backtest["predicted_12m_return"].shift(1)
    )

    backtest["decision_cash_hurdle"] = (
        backtest["cash_hurdle"].shift(1)
    )

    backtest["strategy_return"] = (
        backtest["equity_position"] * backtest["spy_return"]
        + (1 - backtest["equity_position"])
        * backtest["cash_monthly_return"]
    )

    backtest = backtest.dropna(
        subset=[
            "equity_position",
            "strategy_return",
        ]
    ).copy()

    backtest["strategy_wealth"] = (
        1 + backtest["strategy_return"]
    ).cumprod()

    backtest["spy_wealth"] = (
        1 + backtest["spy_return"]
    ).cumprod()

    backtest["cash_wealth"] = (
        1 + backtest["cash_monthly_return"]
    ).cumprod()

    output_path = OUTPUT_DIR / "expanding_window_backtest.csv"
    backtest.to_csv(output_path, index=False)

    print(
        f"Backtest period: "
        f"{backtest['date'].min().date()} — "
        f"{backtest['date'].max().date()}"
    )

    print(
        f"Months invested in SPY: "
        f"{backtest['equity_position'].mean():.1%}"
    )

    print_performance(
        "EXPANDING-WINDOW STRATEGY",
        backtest["strategy_return"],
        backtest["cash_monthly_return"],
    )

    print_performance(
        "SPY BUY-AND-HOLD",
        backtest["spy_return"],
        backtest["cash_monthly_return"],
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        backtest["date"],
        backtest["strategy_wealth"],
        label="Expanding-window strategy",
    )

    plt.plot(
        backtest["date"],
        backtest["spy_wealth"],
        label="SPY buy-and-hold",
    )

    plt.plot(
        backtest["date"],
        backtest["cash_wealth"],
        label="Cash",
    )

    plt.title("Out-of-sample backtest")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)

    chart_path = OUTPUT_DIR / "expanding_window_backtest.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")

    print(f"\nResults saved to: {output_path}")
    print(f"Chart saved to: {chart_path}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(
        backtest["date"],
        backtest["strategy_wealth"],
        label="Strategy",
    )

    axes[0].plot(
        backtest["date"],
        backtest["spy_wealth"],
        label="SPY buy-and-hold",
    )

    axes[0].set_title("Strategy performance and equity exposure")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].step(
        backtest["date"],
        backtest["equity_position"],
        where="post",
        color="tab:blue",
    )

    axes[1].set_ylabel("SPY weight")
    axes[1].set_xlabel("Date")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Cash", "SPY"])
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(alpha=0.3)

    diagnostics_path = OUTPUT_DIR / "signal_diagnostics.png"

    plt.savefig(diagnostics_path, dpi=150, bbox_inches="tight")

    print(f"Signal diagnostics saved in: {diagnostics_path}")

    cash_months = backtest.loc[
        backtest["equity_position"] == 0,
        [
            "date",
            "decision_date",
            "decision_predicted_12m_return",
            "decision_cash_hurdle",
        ],
    ].copy()

    cash_months = cash_months.rename(
        columns={
            "date": "return_month",
            "decision_date": "decision_date",
            "decision_predicted_12m_return": "predicted_12m_return",
            "decision_cash_hurdle": "cash_hurdle",
        }
    )

    print("\nMonths held in cash:")
    print(cash_months.to_string(index=False))


if __name__ == "__main__":
    main()