import subprocess
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from stock_prediction import (
    calculate_atr,
    clean_history,
    download_history,
    evaluate_holdout,
    generate_chaos,
    sample_history,
)


ROOT = Path(__file__).resolve().parents[1]


class EquityExperimentTests(TestCase):
    def test_atr_uses_recent_bars_and_both_sides_of_price_gaps(self):
        bars = pd.DataFrame({
            "High": [11, 12, 25, 11],
            "Low": [9, 10, 20, 8],
            "Close": [10, 11, 24, 9],
        })
        # True ranges: 2 (no prior close), 2, 14 (gap up), 16 (gap down).
        self.assertAlmostEqual(calculate_atr(bars, window=2), 15)
        self.assertAlmostEqual(calculate_atr(bars, window=4), 8.5)
        self.assertAlmostEqual(calculate_atr(bars.iloc[:1], window=1), 2)
        with self.assertRaises(ValueError):
            calculate_atr(bars, window=5)

    def test_logistic_map_is_independent_and_sensitive_to_the_initial_state(self):
        values = generate_chaos(0.41, 3.9, 45)
        other = generate_chaos(0.410001, 3.9, 45)
        np.testing.assert_allclose(values[1:], 3.9 * values[:-1] * (1 - values[:-1]))
        self.assertGreater(float(np.max(np.abs(values - other))), 0.02)

    def test_clean_history_sorts_bars_and_rejects_duplicate_dates(self):
        bars = sample_history().iloc[:3]
        pd.testing.assert_frame_equal(clean_history(bars.iloc[::-1]), bars, check_freq=False)
        with self.assertRaisesRegex(ValueError, "unique dates"):
            clean_history(pd.concat([bars, bars.iloc[:1]]))

    def test_holdout_fits_only_training_prefix_and_uses_its_final_close(self):
        close = sample_history()["Close"]
        with patch("stock_prediction.forecast_stock_prices", return_value=np.arange(5)) as model:
            result = evaluate_holdout(close, 5)
        pd.testing.assert_series_equal(model.call_args.args[0], close.iloc[:-5])
        self.assertEqual(model.call_args.kwargs["steps"], 5)
        np.testing.assert_array_equal(result.actual, close.iloc[-5:])
        np.testing.assert_array_equal(result.last_close, np.full(5, close.iloc[-6]))
        np.testing.assert_array_equal(result.dates, close.index[-5:])

    def test_download_requests_single_ticker_adjusted_daily_bars(self):
        bars = sample_history()
        start, end = date(2024, 1, 2), date(2024, 10, 1)
        with patch("yfinance.download", return_value=bars) as download:
            result = download_history("AAPL", start, end)
        pd.testing.assert_frame_equal(result, bars)
        download.assert_called_once_with(
            "AAPL", start=start, end=end, interval="1d", auto_adjust=True,
            progress=False, threads=False, multi_level_index=False,
        )

    def test_csv_quickstart_uses_local_bars(self):
        with TemporaryDirectory() as directory:
            prices = Path(directory) / "prices.csv"
            sample_history().to_csv(prices)
            process = subprocess.run(
                [sys.executable, str(ROOT / "stock_prediction.py"), "--csv", str(prices), "--holdout", "20"],
                cwd=ROOT, capture_output=True, text=True, check=True, timeout=60,
            )
        self.assertIn("Source: CSV", process.stdout)
        self.assertIn("Fixed-origin holdout: 2024-08-13 to 2024-09-09", process.stdout)

    def test_offline_cli_runs_arima_and_writes_plot(self):
        with TemporaryDirectory() as directory:
            chart = Path(directory) / "equity-demo.png"
            process = subprocess.run(
                [sys.executable, str(ROOT / "stock_prediction.py"), "--sample", "--holdout", "20", "--plot", str(chart)],
                cwd=ROOT, capture_output=True, text=True, check=True, timeout=60,
            )
            self.assertIn("synthetic OHLC sample (not market data)", process.stdout)
            self.assertIn("ARIMA(1, 1, 1) forecast", process.stdout)
            self.assertIn("Independent logistic map (unitless, not a price forecast)", process.stdout)
            self.assertIn("Last training close: MAE=", process.stdout)
            self.assertNotIn("Trading Instruction", process.stdout)
            self.assertTrue(chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))