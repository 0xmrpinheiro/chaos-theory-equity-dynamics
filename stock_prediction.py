"""Compare an equity ARIMA example with a separate deterministic-chaos demo."""

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


OHLC = ["Open", "High", "Low", "Close"]
ARIMA_ORDER = (1, 1, 1)
MIN_MODEL_ROWS = 30


def sample_history():
    """Create fixed, synthetic daily bars; these are not quotes for a ticker."""
    rng = np.random.default_rng(42)
    shocks = rng.normal(0, 0.55, 180)
    changes = shocks[1:] + 0.4 * shocks[:-1] + 0.06
    close = 100 + np.r_[0.0, np.cumsum(changes)]
    open_ = np.r_[100.0, close[:-1]] + rng.normal(0, 0.15, len(close))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + rng.uniform(0.1, 0.7, len(close)),
            "Low": np.minimum(open_, close) - rng.uniform(0.1, 0.7, len(close)),
            "Close": close,
        },
        index=pd.bdate_range("2024-01-02", periods=len(close), name="Date"),
    )


def clean_history(frame):
    if frame is None or frame.empty:
        raise ValueError("No price bars were returned.")
    if missing := set(OHLC) - set(frame.columns):
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
    result = frame.loc[:, OHLC].apply(pd.to_numeric, errors="coerce")
    index = pd.to_datetime(result.index, errors="coerce")
    if index.isna().any() or index.has_duplicates:
        raise ValueError("Price bars must have valid, unique dates.")
    result.index = index
    result = result.sort_index()
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise ValueError("OHLC prices must be positive and finite.")
    if (result["High"] < result["Low"]).any():
        raise ValueError("A high price cannot be below its low price.")
    return result


def download_history(ticker, start, end):
    from yfinance import download

    return clean_history(download(
        ticker, start=start, end=end, interval="1d", auto_adjust=True,
        progress=False, threads=False, multi_level_index=False,
    ))


def generate_chaos(x0, r, n):
    """Generate an independent, unitless logistic-map sequence."""
    if n < 1 or not 0 < x0 < 1 or not 0 < r <= 4:
        raise ValueError("Use n >= 1, 0 < x0 < 1, and 0 < r <= 4.")
    values = np.empty(n)
    values[0] = x0
    for i in range(1, n):
        values[i] = r * values[i - 1] * (1 - values[i - 1])
    return values


def calculate_atr(bars, window=14):
    """Simple average of true ranges for the most recent complete window."""
    if window < 1 or len(bars) < window:
        raise ValueError("ATR needs a positive window and at least that many bars.")
    prior_close = bars["Close"].shift()
    true_range = pd.concat([
        bars["High"] - bars["Low"],
        (bars["High"] - prior_close).abs(),
        (bars["Low"] - prior_close).abs(),
    ], axis=1).max(axis=1)
    return float(true_range.iloc[-window:].mean())


def forecast_stock_prices(close, steps=5):
    if len(close) < MIN_MODEL_ROWS or steps < 1:
        raise ValueError(f"ARIMA needs at least {MIN_MODEL_ROWS} closes and a positive horizon.")
    model = ARIMA(np.asarray(close, dtype=float), order=ARIMA_ORDER).fit()
    return np.asarray(model.forecast(steps=steps))


@dataclass(frozen=True)
class HoldoutResult:
    dates: pd.Index
    actual: np.ndarray
    arima: np.ndarray
    last_close: np.ndarray


def evaluate_holdout(close, length):
    """Forecast a contiguous final block from one origin without seeing that block."""
    if length < 1 or len(close) - length < MIN_MODEL_ROWS:
        raise ValueError(f"Leave at least {MIN_MODEL_ROWS} closes before the holdout.")
    train, test = close.iloc[:-length], close.iloc[-length:]
    return HoldoutResult(
        dates=test.index,
        actual=test.to_numpy(),
        arima=forecast_stock_prices(train, steps=length),
        last_close=np.full(length, train.iloc[-1]),
    )


def save_plot(path, close, forecast, synthetic, perturbed):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    recent = close.iloc[-45:]
    history_steps = np.arange(1 - len(recent), 1)
    next_steps = np.arange(len(forecast) + 1)
    figure, (history_ax, chaos_ax) = plt.subplots(1, 2, figsize=(10, 3.1), layout="constrained")
    history_ax.plot(history_steps, recent, color="#2554a4", label="Input closes")
    history_ax.axvline(0, color="#777777", alpha=0.65, linewidth=1)
    history_ax.plot(next_steps, np.r_[recent.iloc[-1], forecast], "o--",
                    color="#b84d16", label="ARIMA forecast", markersize=4)
    history_ax.set(title="ARIMA fits only input closes", xlabel="Observation relative to last close", ylabel="Price")
    history_ax.legend(loc="best", frameon=False)

    chaos_ax.plot(synthetic, color="#2554a4", label="x₀ = 0.41", alpha=0.85)
    chaos_ax.plot(perturbed, color="#b84d16", label="x₀ = 0.410001", alpha=0.75, linestyle="--")
    chaos_ax.set(title="Logistic map, independent of ARIMA", xlabel="Iteration", ylabel="Unitless state", ylim=(0, 1.05))
    chaos_ax.legend(loc="best", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", action="store_true", help="Use the offline sample (default).")
    source.add_argument("--csv", type=Path, help="Read Date, Open, High, Low, and Close columns.")
    source.add_argument("--ticker", help="Download adjusted daily bars for a single ticker.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today(), help="Exclusive download end date.")
    parser.add_argument("--steps", type=positive_int, default=5, help="Future observations, not calendar days.")
    parser.add_argument("--atr-window", type=positive_int, default=14)
    parser.add_argument("--holdout", type=positive_int, help="Score a final block against a last-close baseline.")
    parser.add_argument("--plot", type=Path, help="Write a PNG visualization.")
    args = parser.parse_args(argv)

    try:
        if args.csv:
            bars = clean_history(pd.read_csv(args.csv, index_col="Date"))
            label = f"CSV ({args.csv})"
        elif args.ticker:
            ticker = args.ticker.strip().upper()
            if not ticker or any(character.isspace() or character == "," for character in ticker):
                raise ValueError("Supply a single ticker such as AAPL.")
            if args.start >= args.end:
                raise ValueError("--end must be after --start.")
            bars = download_history(ticker, args.start, args.end)
            label = f"Yahoo Finance adjusted daily bars ({ticker})"
        else:
            bars = sample_history()
            label = "synthetic OHLC sample (not market data)"

        close = bars["Close"]
        atr = calculate_atr(bars, args.atr_window)
        score = evaluate_holdout(close, args.holdout) if args.holdout else None
        forecast = forecast_stock_prices(close, args.steps)
        synthetic = generate_chaos(0.41, 3.9, 45)
        perturbed = generate_chaos(0.410001, 3.9, 45)
        if args.plot:
            save_plot(args.plot, close, forecast, synthetic, perturbed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Source: {label}")
    print(f"Bars: {len(bars)} ({bars.index[0]:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d})")
    print(f"Recent ATR ({args.atr_window} bars; simple mean): {atr:.4f}")
    print(f"ARIMA{ARIMA_ORDER} forecast for {args.steps} subsequent observations:")
    print("  " + ", ".join(f"{value:.4f}" for value in forecast))
    print("Independent logistic map (unitless, not a price forecast); first five values:")
    print("  " + ", ".join(f"{value:.4f}" for value in synthetic[:5]))
    if score is not None:
        print(f"Fixed-origin holdout: {score.dates[0]:%Y-%m-%d} to {score.dates[-1]:%Y-%m-%d}")
        for name, predicted in [("ARIMA", score.arima), ("Last training close", score.last_close)]:
            errors = score.actual - predicted
            print(f"  {name}: MAE={np.abs(errors).mean():.4f}, RMSE={np.sqrt(np.square(errors).mean()):.4f}")
    if args.plot:
        print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()