"""T004B input-contract regression tests; no datasets, training, or result writes.

Run: .venv/Scripts/python.exe -B tests/test_inference_contract.py
Timing limits: SPEC-GROUP1-SOC-001 section 2. CSV layouts: T004B user approval.
"""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import run_initial_soc_inference as inference


class InferenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        u = np.linspace(0, 1, 500)
        cls.window = np.column_stack((3.7 + 0.04 * np.sin(2 * np.pi * u),
                                      -2 + np.cos(2 * np.pi * u), 22 + u))
        cls.times = np.arange(500) * 0.1
        cls.artifacts = inference.load_frozen_artifacts(ROOT, device="cpu")

    def predict(self, window=None, **kwargs):
        return inference.predict_initial_soc(
            self.window if window is None else window, artifacts=self.artifacts, **kwargs)

    def assert_invalid(self, value, error, message, **kwargs):
        # Reject malformed input before loading artifacts or executing a model.
        with mock.patch.object(inference, "load_frozen_artifacts") as loader:
            with self.assertRaisesRegex(error, message):
                inference.predict_initial_soc(value, **kwargs)
            loader.assert_not_called()

    def csv_file(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "window.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        return path

    def test_valid_array_and_tuple(self):
        before = self.window.copy()
        array = self.predict()
        self.assertEqual(array, self.predict(tuple(self.window.T)))
        self.assertEqual(array, self.predict(tuple(c.tolist() for c in self.window.T)))
        self.assertFalse(array["timing_verified"])
        self.assertEqual(array["operating_stats"]["sample_count"], 500)
        np.testing.assert_array_equal(self.window, before)

    def test_legacy_prediction_values_all_switches(self):
        # Captured from the unmodified T004A-C inference source before T004B.
        # These deterministic synthetic predictions are NOT accuracy evidence.
        for clip in (False, True):
            for correction in (False, True):
                with self.subTest(clip=clip, correction=correction):
                    result = self.predict(clip_physical=clip, apply_current_correction=correction)
                    self.assertEqual(result["raw_model_soc_pct"], 68.5365)
                    self.assertEqual(result["corrected_soc_pct"], 72.2238)
                    self.assertEqual(result["correction_delta_pct"], -3.6872)
                    self.assertEqual(result["predicted_soc_pct"], 72.2238 if correction else 68.5365)

    def test_wrong_window_lengths(self):
        for n in (0, 1, 10, 499, 501):
            with self.subTest(n=n):
                self.assert_invalid(np.zeros((n, 3)), ValueError, r"shape \(500, 3\)")
                self.assert_invalid(tuple(np.zeros(n) for _ in range(3)),
                                    ValueError, "exactly 500 observations")

    def test_wrong_channel_dimensions(self):
        for shape in ((500,), (3, 500), (500, 2), (500, 4), (500, 3, 1)):
            with self.subTest(shape=shape):
                self.assert_invalid(np.zeros(shape), ValueError, "shape")
        self.assert_invalid((np.zeros((500, 1)), np.zeros(500), np.zeros(500)),
                            ValueError, "one-dimensional")
        for n in (0, 2, 4):
            self.assert_invalid(tuple(np.zeros(500) for _ in range(n)),
                                ValueError, "exactly three channels")

    def test_mismatched_tuple_lengths_and_direct_extraction(self):
        channels = (np.zeros(500), np.zeros(499), np.zeros(500))
        self.assert_invalid(channels, ValueError, "identical lengths")
        with self.assertRaisesRegex(ValueError, "identical lengths"):
            inference.extract_group_c_features(*channels, np.zeros(3), np.ones(3))
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            inference.extract_group_c_features(*(np.zeros(10) for _ in range(3)),
                                               np.zeros(3), np.ones(3))

    def test_nonfinite_channels(self):
        for value in (np.nan, np.inf, -np.inf):
            for channel in range(3):
                with self.subTest(value=value, channel=channel):
                    window = self.window.copy()
                    window[123, channel] = value
                    self.assert_invalid(window, ValueError, "finite")
                    self.assert_invalid(tuple(window.T), ValueError, "finite")

    def test_nonnumeric_and_unsupported_measurements(self):
        for value in (None, "window", {}, self.window.tolist()):
            with self.subTest(type=type(value)):
                self.assert_invalid(value, TypeError, "numpy array or three-channel tuple")
        for dtype in (str, object, complex, bool):
            self.assert_invalid(self.window.astype(dtype), TypeError, "real numeric")
        self.assert_invalid(([[1], [2, 3]], np.zeros(500), np.zeros(500)),
                            ValueError, "rectangular")
        self.assert_invalid(np.ma.array(self.window, mask=False), TypeError, "masked")

    def test_valid_numeric_dtypes(self):
        for dtype in (np.float32, np.float64, np.int32, np.uint32):
            with self.subTest(dtype=dtype):
                window = np.full((500, 3), 3, dtype=dtype)
                self.assertTrue(np.isfinite(self.predict(window)["predicted_soc_pct"]))

    def test_optional_timing_status_and_prediction_invariance(self):
        legacy = self.predict()
        for times in (self.times, self.times.tolist(), self.times + 1000):
            before = np.asarray(times).copy()
            timed = self.predict(timestamps=times)
            self.assertTrue(timed.pop("timing_verified"))
            self.assertEqual(timed, {k: v for k, v in legacy.items() if k != "timing_verified"})
            np.testing.assert_array_equal(times, before)

    def test_timestamp_shapes_and_missing_values(self):
        for times in ([], np.zeros(499), np.zeros(501), np.zeros((500, 1)), 1.0):
            with self.subTest(shape=np.shape(times)):
                self.assert_invalid(self.window, ValueError, "shape", timestamps=times)
        for value in (np.nan, np.inf, -np.inf):
            times = self.times.copy()
            times[100] = value
            self.assert_invalid(self.window, ValueError, "finite", timestamps=times)
        for times in (self.times.astype(str), self.times.astype(complex), [None] * 500):
            self.assert_invalid(self.window, TypeError, "real numeric", timestamps=times)

    def test_duplicate_and_backward_timestamps(self):
        for value in (self.times[99], self.times[99] - 0.1):
            times = self.times.copy()
            times[100] = value
            self.assert_invalid(self.window, ValueError, "strictly increasing", timestamps=times)

    def test_timestamp_gap_and_elapsed_bounds(self):
        for duration in (40.0, 49.9, 60.0):
            self.assertTrue(self.predict(timestamps=np.linspace(0, duration, 500))["timing_verified"])
        for duration in (39.999, 60.001):
            self.assert_invalid(self.window, ValueError, "between 40 and 60",
                                timestamps=np.linspace(0, duration, 500))
        for gap, valid in ((0.5, True), (0.5001, False)):
            gaps = np.full(499, 0.1)
            gaps[0] = gap
            times = np.r_[0.0, np.cumsum(gaps)]
            if valid:
                self.assertTrue(self.predict(timestamps=times)["timing_verified"])
            else:
                self.assert_invalid(self.window, ValueError, "0.5 seconds", timestamps=times)

    def test_supported_csv_formats(self):
        for header in ([], [["Voltage", "Current", "Temperature"]]):
            path = self.csv_file(header + self.window.tolist())
            data, times = inference.load_window_csv(path)
            np.testing.assert_array_equal(data, self.window)
            self.assertIsNone(times)
            self.assertEqual(self.predict(data, timestamps=times), self.predict())
        path = self.csv_file([["Time", "Voltage", "Current", "Temperature"]]
                             + np.column_stack((self.times, self.window)).tolist())
        data, times = inference.load_window_csv(path)
        np.testing.assert_array_equal(data, self.window)
        np.testing.assert_array_equal(times, self.times)
        self.assertTrue(self.predict(data, timestamps=times)["timing_verified"])

    def test_csv_ambiguous_columns_and_headers(self):
        for rows in (
                np.column_stack((self.times, self.window)).tolist(),
                [["Current", "Voltage", "Temperature"]] + self.window.tolist(),
                [["Voltage", "Current", "Temp"]] + self.window.tolist(),
                [["Voltage", "Current", "Temperature", "Extra"]] + self.window.tolist()):
            with self.subTest(first_row=rows[0]):
                with self.assertRaisesRegex(ValueError, "CSV"):
                    inference.load_window_csv(self.csv_file(rows))

    def test_csv_bad_values_rows_and_counts(self):
        header = [["Voltage", "Current", "Temperature"]]
        for bad in ([1, 2], [1, 2, 3, 4], [1, "", 3], [1, "text", 3], [],
                    [1, "nan", 3], [1, "inf", 3], [1, "-inf", 3]):
            rows = self.window.tolist()
            rows[200] = bad
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "CSV|finite"):
                    inference.load_window_csv(self.csv_file(header + rows))
        for rows in ([], header, self.window[:499].tolist(), np.zeros((501, 3)).tolist()):
            with self.assertRaisesRegex(ValueError, "CSV"):
                inference.load_window_csv(self.csv_file(rows))

    def test_csv_timestamp_errors_are_not_discarded(self):
        for value in (np.nan, self.times[99], self.times[99] - 1, self.times[99] + 0.6):
            times = self.times.copy()
            times[100] = value
            rows = [["Time", "Voltage", "Current", "Temperature"]]
            rows += np.column_stack((times, self.window)).tolist()
            with self.assertRaisesRegex(ValueError, "finite|strictly increasing|0.5 seconds"):
                inference.load_window_csv(self.csv_file(rows))

    def test_cli_valid_and_invalid_csv(self):
        command = [sys.executable, "-B", str(ROOT / "scripts/run_initial_soc_inference.py"), "--csv"]
        for valid in (True, False):
            path = self.csv_file(self.window.tolist() if valid else self.window[:499].tolist())
            run = subprocess.run(command + [str(path)], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=30)
            if valid:
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("Timing: unverified", run.stdout)
            else:
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn("500 measurement rows", run.stderr)
                self.assertNotIn("Estimated Initial SoC", run.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
