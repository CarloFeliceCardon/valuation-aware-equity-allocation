from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_regression(data, predictors, model_name):
    x = sm.add_constant(data[predictors])
    y = data["forward_12m_total_return"]

    model = sm.OLS(y, x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 11},
    )

    print(f"\n{model_name}")
    print(model.summary())

    return model


def main():
    dataset = pd.read_csv(
        DATA_DIR / "research_dataset_monthly.csv",
        parse_dates=["date"],
    )

    analysis = dataset.dropna(
        subset=[
            "forward_12m_total_return",
            "cape_earnings_yield",
            "treasury_10y_yield",
        ]
    ).copy()

    analysis["cape_spread"] = (
        analysis["cape_earnings_yield"]
        - analysis["treasury_10y_yield"]
    )

    print(f"Used observations: {len(analysis)}")
    print(
        "Analysed period: "
        f"{analysis['date'].min().date()} — {analysis['date'].max().date()}"
    )

    run_regression(
        analysis,
        predictors=["cape_spread"],
        model_name="MODEL 1: 12-month forward return ~ CAPE spread",
    )

    run_regression(
        analysis,
        predictors=[
            "cape_earnings_yield",
            "treasury_10y_yield",
        ],
        model_name="MODEL 2: 12-month forward return ~ CAPE earnings yield + 10Y Treasury yield",
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        analysis["cape_spread"],
        analysis["forward_12m_total_return"],
        alpha=0.6,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_title("CAPE spread and total future return for SPY")
    ax.set_xlabel("CAPE earnings yield − Treasury 10Y")
    ax.set_ylabel("Total SPY return in the following 12 months")

    figure_path = OUTPUT_DIR / "cape_spread_vs_forward_returns.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")

    print(f"\nChart saved to: {figure_path}")


if __name__ == "__main__":
    main()