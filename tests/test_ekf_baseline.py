"""Focused T005A checks of notebook code, without running notebook cells.

Run: .venv/Scripts/python.exe -B tests/test_ekf_baseline.py
Only selected function/parameter AST nodes and the actual voltage expression
are executed. No trajectory, OCV dataset, plots, metrics, or files are generated.
"""

import ast
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02_Baselines.ipynb"


class EKFPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        trees = [ast.parse("".join(cell["source"])) for cell in notebook["cells"]
                 if cell["cell_type"] == "code"]
        functions = {node.name: node for tree in trees for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        namespace = {"np": np}
        # Only these literal/parameter assignments are evaluated, in notebook
        # order. In particular, x0 and all data-loading/evaluation are excluded.
        wanted = {"Q_max", "R_IN", "R_TH", "C_TH", "TAU", "ETA", "Q_MAX"}
        parameters = [node for tree in trees for node in tree.body
                      if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in wanted]
        if len(parameters) != len(wanted):
            raise AssertionError("Expected exactly the existing seven ECM parameter assignments")
        selected = parameters + [functions["coulomb_counting"], functions["ekf_predict_state"]]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        cls.predict = staticmethod(namespace["ekf_predict_state"])
        cls.cc = staticmethod(namespace["coulomb_counting"])
        cls.parameters = {key: namespace[key] for key in wanted}
        cls.loop = next(node for tree in trees for node in tree.body
                        if isinstance(node, ast.For) and any(
                            isinstance(item, ast.Assign) and any(
                                isinstance(target, ast.Name) and target.id == "V_pred"
                                for target in item.targets) for item in node.body))
        voltage = next(node.value for node in cls.loop.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "V_pred"
                               for target in node.targets))
        cls.voltage_code = compile(ast.Expression(body=voltage), str(NOTEBOOK), "eval")

    def step(self, current, dt=1.0, state=(50.0, 0.0), eta=None, capacity=None):
        p = self.parameters
        return self.predict(np.asarray(state, dtype=float), current, dt,
                            p["ETA"] if eta is None else eta,
                            p["Q_MAX"] if capacity is None else capacity,
                            p["R_TH"], p["TAU"])

    def voltage(self, state, current):
        # Constant OCV isolates the actual notebook's ohmic/polarization signs.
        return eval(self.voltage_code, {"ocv_lookup": lambda soc: 3.7,
                                       "x_pred": state, "I": current,
                                       "R_IN": self.parameters["R_IN"]})

    def test_main_loop_uses_tested_prediction(self):
        calls = [node for node in ast.walk(self.loop) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "ekf_predict_state"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([node.id for node in calls[0].args],
                         ["x", "I_prop", "dt", "ETA", "Q_MAX", "R_TH", "TAU"])
        assignment = next(node for node in ast.walk(self.loop) if isinstance(node, ast.Assign)
                          and node.value is calls[0])
        self.assertIsInstance(assignment.targets[0], ast.Tuple)
        self.assertEqual([node.id for node in assignment.targets[0].elts], ["x_pred", "alpha"])

    def test_discharge_decreases_soc(self):
        state, _ = self.step(-2.9, dt=360)
        self.assertAlmostEqual(state[0], 40.0)
        self.assertLess(state[0], 50.0)

    def test_charge_increases_soc(self):
        state, _ = self.step(2.9, dt=360)
        self.assertAlmostEqual(state[0], 60.0)
        self.assertGreater(state[0], 50.0)

    def test_zero_current_preserves_soc_before_update(self):
        state, _ = self.step(0, dt=10, state=(37.5, -0.1))
        self.assertEqual(state[0], 37.5)
        self.assertGreater(state[1], -0.1)
        self.assertLess(state[1], 0)

    def test_physical_bounds_and_outward_currents(self):
        for initial, current, expected in ((1, -2.9, 0), (99, 2.9, 100),
                                           (0, -1, 0), (100, 1, 100)):
            with self.subTest(initial=initial, current=current):
                state, _ = self.step(current, dt=3600, state=(initial, 0))
                self.assertEqual(state[0], expected)
        self.assertGreater(self.step(1, state=(0, 0))[0][0], 0)
        self.assertLess(self.step(-1, state=(100, 0))[0][0], 100)

    def test_ah_seconds_percentage_and_efficiency(self):
        # One ampere for 36 seconds is 0.01 Ah: 1 percentage point of 1 Ah.
        self.assertAlmostEqual(self.step(-1, dt=36, capacity=1)[0][0], 49.0)
        self.assertAlmostEqual(self.step(1, dt=36, capacity=1)[0][0], 51.0)
        self.assertAlmostEqual(self.step(-1, dt=36, capacity=1, eta=0.8)[0][0], 49.2)

    def test_matches_existing_coulomb_counting_one_step(self):
        for current in (-3.0, 0.0, 2.0):
            with self.subTest(current=current):
                cc = self.cc([current, current], [0.1, 0.1], self.parameters["Q_MAX"], 50.0)
                predicted, _ = self.step(current, dt=0.1)
                self.assertEqual(predicted[0], cc[1])

    def test_zero_dt_and_input_state_preservation(self):
        state = np.array([50.0, -0.04])
        before = state.copy()
        p = self.parameters
        predicted, alpha = self.predict(state, -5, 0, p["ETA"], p["Q_MAX"], p["R_TH"], p["TAU"])
        np.testing.assert_array_equal(predicted, before)
        np.testing.assert_array_equal(state, before)
        self.assertEqual(alpha, 1)

    def test_polarization_current_sign_and_time_constant(self):
        tau = self.parameters["TAU"]
        discharge, alpha = self.step(-2, dt=tau)
        charge, _ = self.step(2, dt=tau)
        self.assertAlmostEqual(alpha, math.exp(-1))
        self.assertLess(discharge[1], 0)
        self.assertGreater(charge[1], 0)
        self.assertAlmostEqual(discharge[1], -charge[1])
        relaxed, _ = self.step(0, dt=tau, state=(50, -0.1))
        self.assertAlmostEqual(relaxed[1], -0.1 / math.e)

    def test_polarization_steady_state_and_long_step(self):
        for current in (-2, 2):
            equilibrium = self.parameters["R_TH"] * current
            predicted, _ = self.step(current, state=(50, equilibrium))
            self.assertAlmostEqual(predicted[1], equilibrium)
            long_step, alpha = self.step(current, dt=1e6)
            self.assertEqual(alpha, 0)
            self.assertAlmostEqual(long_step[1], equilibrium)
            self.assertTrue(np.all(np.isfinite(long_step)))

    def test_terminal_voltage_discharge_drop_and_charge_rise(self):
        discharge, _ = self.step(-2, dt=self.parameters["TAU"])
        charge, _ = self.step(2, dt=self.parameters["TAU"])
        self.assertLess(self.voltage(discharge, -2), self.voltage([50, 0], -2))
        self.assertLess(self.voltage([50, 0], -2), 3.7)
        self.assertGreater(self.voltage(charge, 2), 3.7)
        self.assertEqual(self.voltage([50, 0], 0), 3.7)
        self.assertAlmostEqual(self.voltage([50, 0], -2), 3.64904)


if __name__ == "__main__":
    unittest.main(verbosity=2)
