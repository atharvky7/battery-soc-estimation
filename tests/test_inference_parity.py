"""T004A: frozen Group C parity checks; no training or artifact writes.

Run from any directory: <repository>/.venv/Scripts/python.exe -B
    <repository>/tests/test_inference_parity.py

Reference: run_feature_ablation.py, extract_all_13_features (first 12 columns),
Group C, StandardScaler.transform, and SummaryMLP. That script is parsed only,
never imported/executed. Dataset provenance (units, sign, training-only channel
extrema) is inspected in notebook 03; this test does not refit any scaler.
"""

import ast
import ctypes
import gc
import os
from pathlib import Path
import pickle
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts import run_initial_soc_inference as inference

NAMES = ["V0", "I0", "T0", "Vend", "Iend", "Tend", "dV", "dI", "dT",
         "meanV", "meanI", "stdV"]
# Absolute tolerances, rtol=0 throughout. Channels/features are normalized.
# One float32 epsilon is 1.19e-7; feature scaling can amplify that rounding.
NORMALIZATION_ATOL = 1.2e-7
FEATURE_ATOL = 2e-7
# A 2.98e-8 endpoint-delta round-trip error becomes 3.58e-6 through
# the small dT scale on deterministic synthetic data. 4e-6 covers this
# observed float32 amplification; features themselves retain the 2e-7 bound.
SCALED_ATOL = 4e-6
FORWARD_PP_ATOL = 5e-5
# API rounds percentages to four decimals (up to 5e-5 pp extra error).
API_PP_ATOL = 1e-4


def available_memory():
    """Available physical RAM in bytes, or None if not measurable."""
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "total_page", "available_page",
                 "total_virtual", "available_virtual", "available_extended")]
        status = Memory()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available
    else:
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            pass
    return None


def synthetic_windows():
    """Five deterministic normalized windows; not accuracy evidence."""
    t = np.linspace(0, 1, 500, dtype=np.float32)
    return np.stack([np.column_stack((
        0.3 + 0.06 * k + 0.02 * np.sin((k + 1) * np.pi * t),
        0.2 + 0.01 * k + 0.04 * np.cos(2 * np.pi * t),
        0.4 + 0.03 * k + 0.01 * t)) for k in range(5)])


def reference_features(normalized):
    """Independent training formula; preserve native NumPy reduction dtype.

    Group D's integral column is omitted. For the saved data both sequences
    and timestamps are float32, so that column does not promote Group C.
    np.std uses population variance (ddof=0), as in the training source.
    """
    first, last = normalized[:, 0, :], normalized[:, -1, :]
    delta = last - first
    return np.column_stack((first, last, delta,
                            np.mean(normalized[:, :, 0], axis=1),
                            np.mean(normalized[:, :, 1], axis=1),
                            np.std(normalized[:, :, 0], axis=1, ddof=0)))


def reference_standardize(features, scaler):
    # StandardScaler subtracts then divides in place, retaining input dtype.
    z = features.copy()
    z -= scaler.mean_
    z /= scaler.scale_
    return z


def reference_forward(z, state):
    """Functional forward pass from checkpoint tensors, independent of API."""
    with torch.no_grad():
        h = torch.as_tensor(z, dtype=torch.float32)
        for layer in (0, 2, 4):
            h = F.linear(h, state[f"net.{layer}.weight"],
                         state[f"net.{layer}.bias"])
            if layer != 4:
                h = F.relu(h)
        # Training multiplies the NumPy float32 predictions by 100.
        return h.numpy().ravel() * 100.0


class InferenceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = inference.load_frozen_artifacts(ROOT, device="cpu")
        cls.state = torch.load(ROOT / "results/models/ablation_group_c_seed_42.pt",
                               map_location="cpu", weights_only=True)
        path = ROOT / "data/initial_soc_datasets.pkl"
        free = available_memory()
        # PyTorch/artifacts are already resident. Reserve >4x pickle size plus
        # 128 MiB for unpickling/selecting five windows, with a 1 GiB floor.
        required = max(1024 * 2**20, 4 * path.stat().st_size + 128 * 2**20) if path.exists() else 0
        print(f"Available RAM: {free / 2**30 if free is not None else 'unknown'} GiB; required: {required / 2**30:.2f} GiB", flush=True)
        if path.exists() and free is not None and free >= required:
            # Trusted local pickle necessarily materializes all splits. Only
            # five X_val windows are selected; no labels/test arrays are queried.
            with path.open("rb") as handle:
                datasets = pickle.load(handle)
            val = datasets[50]["X_val"]
            n = len(val)
            cls.indices = sorted(set((0, n // 4, n // 2, 3 * n // 4, n - 1)))
            cls.normalized = val[cls.indices].copy()
            cls.timestamp_dtype = datasets[50]["t_val"].dtype
            cls.evidence = "saved validation"
            del val, datasets
            gc.collect()
        else:
            cls.indices = list(range(5))
            cls.normalized = synthetic_windows()
            cls.timestamp_dtype = np.dtype("float32")
            cls.evidence = "SYNTHETIC ONLY (dataset absent or insufficient/unknown RAM)"
        print(f"Evidence: {cls.evidence}; indices: {cls.indices}; dtype: {cls.normalized.dtype}", flush=True)
        cls.physical = (cls.normalized * cls.artifacts["feat_range"]
                        + cls.artifacts["feat_min"])
        cls.ref_features = reference_features(cls.normalized)
        cls.actual_features = np.concatenate([
            inference.extract_group_c_features(*window.T,
                cls.artifacts["feat_min"], cls.artifacts["feat_range"])[0]
            for window in cls.physical])
        cls.ref_z = reference_standardize(cls.ref_features, cls.artifacts["scaler_c"])
        cls.actual_z = cls.artifacts["scaler_c"].transform(cls.actual_features)

    def test_training_contract_without_executing_training(self):
        tree = ast.parse((ROOT / "scripts/run_feature_ablation.py").read_text(encoding="utf-8"))
        assignments = {n.targets[0].id: n.value for n in tree.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.literal_eval(assignments["FEATURE_NAMES"])[:12], NAMES)
        self.assertEqual(inference.FEATURE_NAMES, NAMES)
        groups = assignments["GROUPS"]
        group_c = next(v for k, v in zip(groups.keys, groups.values)
                       if ast.literal_eval(k) == "Group C")
        fields = {ast.literal_eval(k): v for k, v in zip(group_c.keys, group_c.values)}
        self.assertEqual(ast.literal_eval(fields["dim"]), 12)
        self.assertEqual(ast.dump(fields["indices"]),
                         ast.dump(ast.parse("list(range(0, 12))", mode="eval").body))
        feature_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == "extract_all_13_features")
        stack = next(n for n in ast.walk(feature_function) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "column_stack")
        self.assertEqual([n.id for n in stack.args[0].elts][:12],
                         ["V_0", "I_0", "T_0", "V_end", "I_end", "T_end",
                          "dV", "dI", "dT", "meanV", "meanI", "stdV"])

    def test_channel_normalization_and_scaler(self):
        a = self.artifacts
        with np.load(ROOT / "data/initial_soc_scaler.npz") as saved:
            np.testing.assert_array_equal(a["feat_min"], saved["feat_min"])
            np.testing.assert_array_equal(a["feat_range"], saved["feat_range"])
            self.assertEqual(float(saved["target_range"]), 100.0)
        self.assertEqual(a["feat_min"].shape, (3,))
        self.assertTrue(np.all(a["feat_range"] > 0))
        self.assertEqual(a["scaler_c"].n_features_in_, 12)
        self.assertEqual(a["scaler_c"].n_samples_seen_, 5182)
        self.assertEqual(self.normalized.dtype, np.float32)
        self.assertEqual(self.timestamp_dtype, np.float32)
        normalized = (self.physical - a["feat_min"]) / a["feat_range"]
        diff = float(np.max(np.abs(normalized - self.normalized)))
        print(f"Normalization max abs difference: {diff:.9g}")
        np.testing.assert_allclose(normalized, self.normalized, rtol=0, atol=NORMALIZATION_ATOL)

    def test_feature_and_standardization_parity(self):
        diff = np.max(np.abs(self.actual_features - self.ref_features), axis=0)
        print("Feature max abs differences:", dict(zip(NAMES, diff.tolist())))
        print(f"Feature max: {diff.max():.9g}; standardized max: {np.abs(self.actual_z - self.ref_z).max():.9g}")
        np.testing.assert_allclose(self.actual_features, self.ref_features, rtol=0, atol=FEATURE_ATOL)
        np.testing.assert_allclose(self.actual_z, self.ref_z, rtol=0, atol=SCALED_ATOL)

    def test_frozen_architecture_and_loading(self):
        model = self.artifacts["model"]
        self.assertFalse(model.training)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2945)
        self.assertEqual([type(layer) for layer in model.net],
                         [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear])
        self.assertEqual([(m.in_features, m.out_features) for m in model.net if isinstance(m, nn.Linear)],
                         [(12, 64), (64, 32), (32, 1)])
        self.assertEqual(set(model.state_dict()), set(self.state))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, self.state[key]), key)

    def test_reference_forward_and_no_gradient_api(self):
        model = self.artifacts["model"]
        observed = []
        hook = model.register_forward_hook(lambda m, args, out:
            observed.append((torch.is_grad_enabled(), out.requires_grad)))
        try:
            api = np.array([inference.predict_initial_soc(window, artifacts=self.artifacts,
                            clip_physical=False, apply_current_correction=False)["raw_model_soc_pct"]
                            for window in self.physical])
        finally:
            hook.remove()
        self.assertEqual(observed, [(False, False)] * len(self.physical))
        expected = reference_forward(self.ref_z, self.state)
        with torch.no_grad():
            actual = np.array([model(torch.tensor(z[None], dtype=torch.float32)).item() * 100.0
                               for z in self.actual_z])
        print(f"Unrounded forward max difference (pp): {np.abs(actual - expected).max():.9g}")
        print(f"API/reference max difference (pp): {np.abs(api - expected).max():.9g}")
        print(f"API/rounded-reference max difference (pp): {np.abs(api - np.round(expected.astype(float), 4)).max():.9g}")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=FORWARD_PP_ATOL)
        np.testing.assert_allclose(api, expected, rtol=0, atol=API_PP_ATOL)

    def test_array_tuple_equivalence_and_current_mean_reporting(self):
        for window in self.physical:
            opts = dict(artifacts=self.artifacts, clip_physical=False, apply_current_correction=False)
            array = inference.predict_initial_soc(window, **opts)
            tuple_input = inference.predict_initial_soc(tuple(window.T), **opts)
            self.assertEqual(array, tuple_input)
            self.assertEqual(array["operating_stats"]["mean_current_A"], round(float(window[:, 1].mean()), 4))

    def test_float32_and_float64_precision_paths(self):
        # Same representative physical window, explicitly cast to either dtype.
        a = self.artifacts
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                raw = self.physical[0].astype(dtype)
                normalized = (raw - a["feat_min"]) / a["feat_range"]
                ref = reference_features(normalized[None])
                actual, _ = inference.extract_group_c_features(*raw.T, a["feat_min"], a["feat_range"])
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual, ref, rtol=0, atol=FEATURE_ATOL)
                z = reference_standardize(ref, a["scaler_c"])
                np.testing.assert_allclose(a["scaler_c"].transform(actual), z, rtol=0, atol=SCALED_ATOL)
                expected = reference_forward(z, self.state)[0]
                result = inference.predict_initial_soc(raw, artifacts=a, clip_physical=False,
                                                       apply_current_correction=False)
                np.testing.assert_allclose(result["predicted_soc_pct"], expected, rtol=0, atol=API_PP_ATOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
