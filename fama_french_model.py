"""Build and evaluate Fama-French factor models for an equal-weighted equity portfolio.

This script downloads daily adjusted prices for a fixed portfolio of U.S. equities,
retrieves the official Fama-French 3-factor and 5-factor daily datasets, estimates
factor regressions, creates visualizations, and prints plain-English interpretations
of the results.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)
warnings.filterwarnings(
    "ignore",
    message="The argument 'date_parser' is deprecated",
    category=FutureWarning,
)

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
REQUIRED_MODULES = (
    "matplotlib",
    "numpy",
    "pandas",
    "statsmodels",
    "yfinance",
    "pandas_datareader",
)


def ensure_project_interpreter() -> None:
    """Re-run the script with the project virtualenv when dependencies are missing.

    Args:
        None.

    Returns:
        None. The current process is replaced when the virtualenv interpreter should
        be used instead.

    Raises:
        SystemExit: If required packages are missing and no usable project virtualenv
            interpreter is available.
    """

    missing_modules = [
        module_name
        for module_name in REQUIRED_MODULES
        if importlib.util.find_spec(module_name) is None
    ]

    if not missing_modules:
        return

    current_prefix = Path(sys.prefix).resolve()

    if VENV_PYTHON.exists() and current_prefix != VENV_DIR.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

    missing_list = ", ".join(missing_modules)
    raise SystemExit(
        "Missing required Python packages: "
        f"{missing_list}. Create the project virtualenv and install requirements with:\n"
        f'cd "{PROJECT_DIR}"\n'
        "python3 -m venv .venv\n"
        ". .venv/bin/activate\n"
        "pip install -r requirements.txt"
    )


ensure_project_interpreter()

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.regression.linear_model import RegressionResultsWrapper


TICKERS = ["AAPL", "MSFT", "JPM", "GS", "BAC", "XOM", "CVX", "JNJ", "UNH", "AMZN"]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
ROLLING_WINDOW = 60
OUTPUT_DIR = PROJECT_DIR / "output"

THREE_FACTOR_DATASET = "F-F_Research_Data_Factors_daily"
FIVE_FACTOR_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"


def download_portfolio_prices(
    tickers: Iterable[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Download daily adjusted closing prices for the requested tickers.

    Args:
        tickers: Iterable of ticker symbols to download from Yahoo Finance.
        start_date: Inclusive start date in YYYY-MM-DD format.
        end_date: Inclusive end date in YYYY-MM-DD format.

    Returns:
        A DataFrame indexed by trading date with one adjusted price column per ticker.

    Raises:
        ValueError: If no price data is returned or required columns are missing.
    """

    ticker_list = list(tickers)

    prices = yf.download(
        tickers=ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if prices.empty:
        raise ValueError("No price data was returned from yfinance.")

    if isinstance(prices.columns, pd.MultiIndex):
        if "Close" not in prices.columns.get_level_values(0):
            raise ValueError("Expected a 'Close' field in the downloaded price data.")
        close_prices = prices["Close"].copy()
    else:
        close_prices = prices.rename(columns={"Close": ticker_list[0]}).copy()

    close_prices = close_prices.dropna(how="all")
    close_prices = close_prices.sort_index()
    close_prices.index = pd.to_datetime(close_prices.index)

    if close_prices.empty:
        raise ValueError("Downloaded price data is empty after basic cleaning.")

    return close_prices


def calculate_equal_weighted_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """Convert price data into constituent and portfolio daily returns.

    Args:
        price_data: DataFrame of daily prices with one column per asset.

    Returns:
        A DataFrame of daily simple returns with an added `Portfolio` column for the
        equal-weighted portfolio return.
    """

    asset_returns = price_data.pct_change().dropna(how="all")
    asset_returns["Portfolio"] = asset_returns.mean(axis=1)
    return asset_returns


def download_fama_french_factors(dataset_name: str) -> pd.DataFrame:
    """Download a daily Fama-French factor dataset from Kenneth French's library.

    Args:
        dataset_name: Kenneth French dataset identifier understood by
            `pandas_datareader`, such as the daily 3-factor or 5-factor dataset name.

    Returns:
        A DataFrame indexed by date with factor values converted from percent to
        decimal form.

    Raises:
        ValueError: If the requested dataset cannot be parsed into a daily DataFrame.
    """

    raw_dataset = web.DataReader(dataset_name, "famafrench")[0].copy()

    if raw_dataset.empty:
        raise ValueError(f"Fama-French dataset '{dataset_name}' returned no rows.")

    raw_dataset.index = pd.to_datetime(raw_dataset.index)
    raw_dataset = raw_dataset.sort_index()

    return raw_dataset / 100.0


def align_portfolio_and_factors(
    portfolio_returns: pd.Series, factor_data: pd.DataFrame
) -> pd.DataFrame:
    """Align portfolio returns with factor data on shared trading dates.

    Args:
        portfolio_returns: Series of daily portfolio simple returns.
        factor_data: DataFrame of daily factor returns including `RF`.

    Returns:
        A merged DataFrame containing portfolio return, risk-free rate, factor series,
        portfolio excess return, and market total return for matched dates.
    """

    merged = pd.concat([portfolio_returns.rename("Portfolio"), factor_data], axis=1, join="inner")
    merged = merged.dropna().sort_index()
    merged["Excess_Portfolio"] = merged["Portfolio"] - merged["RF"]
    merged["Market_Return"] = merged["Mkt-RF"] + merged["RF"]
    return merged


def fit_factor_model(data: pd.DataFrame, factor_columns: list[str]) -> RegressionResultsWrapper:
    """Estimate an OLS factor model of excess portfolio returns on selected factors.

    Args:
        data: Aligned DataFrame containing `Excess_Portfolio` and factor columns.
        factor_columns: Ordered list of factor column names to include in the model.

    Returns:
        A fitted statsmodels OLS regression results object.
    """

    x_matrix = sm.add_constant(data[factor_columns])
    y_vector = data["Excess_Portfolio"]
    return sm.OLS(y_vector, x_matrix).fit()


def build_coefficient_summary(
    model: RegressionResultsWrapper, factor_columns: list[str]
) -> pd.DataFrame:
    """Create a clean coefficient summary table for the selected model factors.

    Args:
        model: Fitted statsmodels OLS regression object.
        factor_columns: Ordered list of factor names to display.

    Returns:
        A DataFrame indexed by factor name with coefficient, standard error,
        t-statistic, p-value, and 95% confidence interval bounds.
    """

    confidence_intervals = model.conf_int()
    summary = pd.DataFrame(
        {
            "Coefficient": model.params[factor_columns],
            "Std. Error": model.bse[factor_columns],
            "t-Statistic": model.tvalues[factor_columns],
            "p-Value": model.pvalues[factor_columns],
            "CI Lower (95%)": confidence_intervals.loc[factor_columns, 0],
            "CI Upper (95%)": confidence_intervals.loc[factor_columns, 1],
        }
    )
    return summary


def classify_portfolio_style(model: RegressionResultsWrapper) -> str:
    """Produce a plain-English style classification from factor loadings.

    Args:
        model: Fitted 3-factor model containing `Mkt-RF`, `SMB`, and `HML`.

    Returns:
        A descriptive sentence summarizing market sensitivity, size tilt, and
        value-versus-growth tilt.
    """

    market_beta = model.params["Mkt-RF"]
    smb_beta = model.params["SMB"]
    hml_beta = model.params["HML"]

    if market_beta < 0.8:
        market_label = "below-market sensitivity"
    elif market_beta <= 1.2:
        market_label = "market-like sensitivity"
    else:
        market_label = "above-market sensitivity"

    size_label = "small-cap tilt" if smb_beta > 0 else "large-cap tilt"
    value_label = "value tilt" if hml_beta > 0 else "growth tilt"

    return (
        f"The portfolio shows {market_label}, a {size_label}, "
        f"and a {value_label} based on the 3-factor loadings."
    )


def create_cumulative_return_chart(data: pd.DataFrame, output_path: Path) -> None:
    """Plot cumulative portfolio and market returns over the sample period.

    Args:
        data: Aligned DataFrame containing `Portfolio` and `Market_Return`.
        output_path: File path where the chart image will be saved.

    Returns:
        None. The function saves a PNG file to disk.
    """

    cumulative = (1 + data[["Portfolio", "Market_Return"]]).cumprod() - 1

    plt.figure(figsize=(11, 6))
    plt.plot(cumulative.index, cumulative["Portfolio"], label="Portfolio", linewidth=2.0)
    plt.plot(cumulative.index, cumulative["Market_Return"], label="Market", linewidth=2.0)
    plt.title("Cumulative Portfolio Return vs Market Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_rolling_beta_chart(data: pd.DataFrame, output_path: Path, window: int = 60) -> None:
    """Plot the rolling market beta of portfolio excess returns.

    Args:
        data: Aligned DataFrame containing `Excess_Portfolio` and `Mkt-RF`.
        output_path: File path where the chart image will be saved.
        window: Rolling window length in trading days for beta estimation.

    Returns:
        None. The function saves a PNG file to disk.
    """

    rolling_covariance = data["Excess_Portfolio"].rolling(window).cov(data["Mkt-RF"])
    rolling_variance = data["Mkt-RF"].rolling(window).var()
    rolling_beta = rolling_covariance / rolling_variance

    plt.figure(figsize=(11, 6))
    plt.plot(rolling_beta.index, rolling_beta, color="tab:blue", linewidth=1.8)
    plt.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.2, label="Beta = 1")
    plt.title(f"Rolling {window}-Day Market Beta")
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_factor_bar_chart(
    summary_table: pd.DataFrame, output_path: Path, factor_columns: list[str]
) -> None:
    """Plot factor coefficients with 95% confidence interval error bars.

    Args:
        summary_table: Coefficient summary DataFrame for the chosen factors.
        output_path: File path where the chart image will be saved.
        factor_columns: Ordered list of factor names to plot.

    Returns:
        None. The function saves a PNG file to disk.
    """

    coefficients = summary_table.loc[factor_columns, "Coefficient"]
    ci_half_width = (
        summary_table.loc[factor_columns, "CI Upper (95%)"]
        - summary_table.loc[factor_columns, "Coefficient"]
    )

    plt.figure(figsize=(10, 6))
    plt.bar(factor_columns, coefficients, yerr=ci_half_width, capsize=6, color="tab:green", alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title("3-Factor Coefficients with 95% Confidence Intervals")
    plt.xlabel("Factor")
    plt.ylabel("Coefficient")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_actual_vs_fitted_chart(
    model: RegressionResultsWrapper,
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot actual versus model-fitted excess returns with a fitted trend line.

    Args:
        model: Fitted statsmodels OLS regression object.
        data: Aligned DataFrame used to fit the regression model.
        output_path: File path where the chart image will be saved.

    Returns:
        None. The function saves a PNG file to disk.
    """

    fitted_values = model.fittedvalues
    actual_values = data["Excess_Portfolio"]
    line_slope, line_intercept = np.polyfit(fitted_values, actual_values, 1)
    x_line = np.linspace(fitted_values.min(), fitted_values.max(), 100)
    y_line = line_slope * x_line + line_intercept

    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_values, actual_values, alpha=0.5, color="tab:purple")
    plt.plot(x_line, y_line, color="tab:red", linewidth=2.0, label="Regression Line")
    plt.title("Actual Excess Returns vs Model-Fitted Excess Returns")
    plt.xlabel("Model-Fitted Excess Return")
    plt.ylabel("Actual Excess Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_model_comparison_table(
    three_factor_model: RegressionResultsWrapper,
    five_factor_model: RegressionResultsWrapper,
) -> pd.DataFrame:
    """Create a side-by-side comparison of the 3-factor and 5-factor models.

    Args:
        three_factor_model: Fitted Fama-French 3-factor regression model.
        five_factor_model: Fitted Fama-French 5-factor regression model.

    Returns:
        A DataFrame comparing alpha, alpha p-value, and R-squared across models.
    """

    return pd.DataFrame(
        {
            "3-Factor Model": {
                "Alpha": three_factor_model.params["const"],
                "Alpha p-Value": three_factor_model.pvalues["const"],
                "R-Squared": three_factor_model.rsquared,
                "Adj. R-Squared": three_factor_model.rsquared_adj,
            },
            "5-Factor Model": {
                "Alpha": five_factor_model.params["const"],
                "Alpha p-Value": five_factor_model.pvalues["const"],
                "R-Squared": five_factor_model.rsquared,
                "Adj. R-Squared": five_factor_model.rsquared_adj,
            },
        }
    )


def interpret_results(
    three_factor_model: RegressionResultsWrapper,
    three_factor_summary: pd.DataFrame,
    comparison_table: pd.DataFrame,
) -> str:
    """Generate a plain-English interpretation of the factor model results.

    Args:
        three_factor_model: Fitted Fama-French 3-factor regression model.
        three_factor_summary: Coefficient summary table for the 3-factor model.
        comparison_table: Side-by-side 3-factor versus 5-factor comparison table.

    Returns:
        A multi-line human-readable interpretation of alpha, dominant factors, and
        portfolio style.
    """

    alpha = three_factor_model.params["const"]
    alpha_pvalue = three_factor_model.pvalues["const"]
    significant_text = (
        "is statistically significant"
        if alpha_pvalue < 0.05
        else "is not statistically significant"
    )

    dominant_factor = three_factor_summary["Coefficient"].abs().idxmax()
    dominant_loading = three_factor_summary.loc[dominant_factor, "Coefficient"]
    style_text = classify_portfolio_style(three_factor_model)
    r_squared_gain = (
        comparison_table.loc["R-Squared", "5-Factor Model"]
        - comparison_table.loc["R-Squared", "3-Factor Model"]
    )

    interpretation = [
        "Plain-English Interpretation",
        (
            f"Alpha is {alpha:.6f} per day and {significant_text} at the 5% level "
            f"(p-value = {alpha_pvalue:.4f})."
        ),
        (
            f"The largest absolute factor loading is {dominant_factor} at "
            f"{dominant_loading:.3f}, which suggests this factor is the strongest "
            "driver of the portfolio's excess returns."
        ),
        style_text,
        (
            f"Moving from the 3-factor model to the 5-factor model changes R-squared "
            f"by {r_squared_gain:.4f}, showing how much additional explanatory power "
            "profitability and investment factors add."
        ),
    ]

    return "\n".join(interpretation)


def format_regression_overview(
    model: RegressionResultsWrapper,
) -> pd.Series:
    """Create a compact regression overview for console output.

    Args:
        model: Fitted statsmodels OLS regression object.

    Returns:
        A Series containing alpha, factor coefficients, R-squared, and p-values.
    """

    metrics = {
        "Alpha": model.params["const"],
        "Mkt-RF Beta": model.params["Mkt-RF"],
        "SMB Beta": model.params["SMB"],
        "HML Beta": model.params["HML"],
        "R-Squared": model.rsquared,
        "Alpha p-Value": model.pvalues["const"],
        "Mkt-RF p-Value": model.pvalues["Mkt-RF"],
        "SMB p-Value": model.pvalues["SMB"],
        "HML p-Value": model.pvalues["HML"],
    }
    return pd.Series(metrics)


def main() -> None:
    """Run the full Fama-French portfolio analysis workflow end to end.

    Args:
        None.

    Returns:
        None. The function prints summary outputs and saves figures to disk.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=== Data Collection ===")
    price_data = download_portfolio_prices(TICKERS, START_DATE, END_DATE)
    returns = calculate_equal_weighted_returns(price_data)
    three_factor_data = download_fama_french_factors(THREE_FACTOR_DATASET)
    five_factor_data = download_fama_french_factors(FIVE_FACTOR_DATASET)

    print("=== Data Cleaning & Alignment ===")
    aligned_three_factor = align_portfolio_and_factors(returns["Portfolio"], three_factor_data)
    aligned_five_factor = align_portfolio_and_factors(returns["Portfolio"], five_factor_data)
    print(f"Aligned 3-factor sample size: {len(aligned_three_factor):,} daily observations")
    print(f"Aligned 5-factor sample size: {len(aligned_five_factor):,} daily observations")

    print("=== Model Estimation ===")
    three_factor_columns = ["Mkt-RF", "SMB", "HML"]
    five_factor_columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    three_factor_model = fit_factor_model(aligned_three_factor, three_factor_columns)
    five_factor_model = fit_factor_model(aligned_five_factor, five_factor_columns)

    regression_overview = format_regression_overview(three_factor_model)
    three_factor_summary = build_coefficient_summary(three_factor_model, three_factor_columns)
    model_comparison = build_model_comparison_table(three_factor_model, five_factor_model)

    print("\n3-Factor Regression Overview")
    print(regression_overview.to_string(float_format=lambda value: f"{value:0.6f}"))

    print("\n3-Factor Coefficient Summary")
    print(three_factor_summary.to_string(float_format=lambda value: f"{value:0.6f}"))

    alpha_significance = (
        "statistically significant"
        if three_factor_model.pvalues["const"] < 0.05
        else "not statistically significant"
    )
    print(
        f"\nAlpha significance at the 5% level: The 3-factor alpha is "
        f"{alpha_significance}."
    )

    print("\nPortfolio Style Classification")
    print(classify_portfolio_style(three_factor_model))

    print("\n=== Visualization ===")
    create_cumulative_return_chart(
        aligned_three_factor, OUTPUT_DIR / "cumulative_portfolio_vs_market.png"
    )
    create_rolling_beta_chart(
        aligned_three_factor,
        OUTPUT_DIR / "rolling_60_day_market_beta.png",
        window=ROLLING_WINDOW,
    )
    create_factor_bar_chart(
        three_factor_summary,
        OUTPUT_DIR / "three_factor_coefficients.png",
        three_factor_columns,
    )
    create_actual_vs_fitted_chart(
        three_factor_model,
        aligned_three_factor,
        OUTPUT_DIR / "actual_vs_fitted_excess_returns.png",
    )
    print(f"Saved charts to: {OUTPUT_DIR.resolve()}")

    print("\n=== 5-Factor Extension ===")
    print(model_comparison.to_string(float_format=lambda value: f"{value:0.6f}"))

    print("\n=== Final Interpretation ===")
    print(interpret_results(three_factor_model, three_factor_summary, model_comparison))


if __name__ == "__main__":
    main()
