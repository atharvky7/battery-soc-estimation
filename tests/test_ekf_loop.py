"""T005B deterministic checks of the actual notebook loop and voltage update.

Run: .venv/Scripts/python.exe -B tests/test_ekf_loop.py
AST extraction executes only selected parameters, functions, and EKF loop code.
No notebook data loading, full trajectory, metrics, or plotting is executed.
Synthetic observations verify equations and indexing, not experimental accuracy.
"""

import ast
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02_Baselines.ipynb"


def linear_ocv(soc):
    return 3.0 + 0.01 * soc


class EKFLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        trees = [ast.parse("".join(cell["source"])) for cell in notebook["cells"]
                 if cell["cell_type"] == "code"]
        wanted = {"Q_max", "R_IN", "R_TH", "C_TH", "TAU", "ETA", "Q_MAX",
                  "Q_noise", "R_noise", "P0"}
        parameters = [node for tree in trees for node in tree.body
                      if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in wanted]
        if len(parameters) != len(wanted):
            raise AssertionError("Expected exactly the existing ECM/noise parameter assignments")
        functions = {node.name: node for tree in trees for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        namespace = {"np": np}
        nodes = parameters + [functions["ekf_predict_state"], functions["coulomb_counting"]]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        cls.base = namespace
        tree = next(tree for tree in trees if any(
            isinstance(node, ast.FunctionDef) and node.name == "ekf_predict_state"
            for node in tree.body))
        loop = next(node for node in tree.body if isinstance(node, ast.For))
        start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "V_pred"
                             for target in node.targets))
        end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "P"
                           for target in node.targets))
        cls.update_code = compile(ast.Module(body=loop.body[start:end + 1], type_ignores=[]),
                                  str(NOTEBOOK), "exec")
        # Run the actual loop/initializers, adding observations only. Code after
        # the loop (RMSE, MAE, printing) is excluded from the executable AST.
        beginning = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "N"
                                 for target in node.targets))
        loop_nodes = copy.deepcopy(tree.body[beginning:tree.body.index(loop) + 1])
        probe = ast.parse('''trace.append({
            "x_pred": x_pred.copy(), "P_pred": P_pred.copy(),
            "V_pred": float(V_pred), "innov": float(innov),
            "H": H.copy(), "S": S.copy(), "K": K.copy(),
            "x": x.copy(), "P": P.copy()})''').body
        loop_nodes[-1].body.extend(probe)
        cls.loop_code = compile(ast.fix_missing_locations(ast.Module(
            body=loop_nodes, type_ignores=[])), str(NOTEBOOK), "exec")

    def update(self, innovation=0.0, state=(50.0, 0.02), current=-2.0, ocv=linear_ocv):
        env = dict(self.base, x_pred=np.asarray(state, dtype=float),
                   P_pred=np.diag([4.0, 0.01]), I=current,
                   ocv_lookup=ocv)
        predicted_voltage = ocv(state[0]) + current * self.base["R_IN"] + state[1]
        env["V_meas"] = predicted_voltage + innovation
        exec(self.update_code, env)
        return env

    def run_loop(self, times, currents, voltages, state=(50.0, 0.0), ocv=linear_ocv):
        times = np.asarray(times, dtype=float)
        # A deliberately absurd first dt must be ignored by the actual loop.
        dt = np.r_[12345.0, np.diff(times)]
        spy = mock.Mock(wraps=self.base["ekf_predict_state"])
        env = dict(self.base, time=times, current=np.asarray(currents, dtype=float),
                   voltage=np.asarray(voltages, dtype=float), dt_arr=dt,
                   x0=np.asarray(state, dtype=float), P0=self.base["P0"].copy(),
                   ocv_lookup=ocv, trace=[], ekf_predict_state=spy)
        exec(self.loop_code, env)
        env["prediction_calls"] = spy.call_args_list
        return env

    def zero_innovation_sequence(self):
        times = np.array([5.0, 5.2, 6.7, 8.0, 8.1])
        currents = np.array([-2.0, 5.0, 0.0, -3.0, 1.0])
        dt = np.r_[0.0, np.diff(times)]
        # Existing independent CC implementation fixes expected SoC. The
        # analytical RC relaxation solution supplies a synthetic voltage trace.
        soc = self.base["coulomb_counting"](currents, dt, self.base["Q_MAX"], 50.0)
        polarization = np.zeros(len(times))
        for k in range(1, len(times)):
            equilibrium = currents[k - 1] * self.base["R_TH"]
            polarization[k] = equilibrium + (polarization[k - 1] - equilibrium) * np.exp(
                -dt[k] / self.base["TAU"])
        voltages = linear_ocv(soc) + currents * self.base["R_IN"] + polarization
        return times, currents, voltages, soc, polarization

    def test_terminal_prediction_and_measured_minus_predicted_innovation(self):
        result = self.update(innovation=0.021)
        self.assertAlmostEqual(float(result["V_pred"]), 3.46904, places=12)
        self.assertAlmostEqual(float(result["innov"]), 0.021, places=12)

    def test_linear_ocv_jacobian(self):
        result = self.update()
        np.testing.assert_allclose(result["H"], [[0.01, 1.0]], rtol=0, atol=1e-14)
        self.assertEqual(result["H"].shape, (1, 2))

    def test_analytical_innovation_covariance_and_gain(self):
        # H=[.01,1], prior variances=[4,.01], R=.0001:
        # scalar S=.0105; cov(x,V)=[.04,.01], hence K=[80/21,20/21].
        result = self.update()
        np.testing.assert_allclose(result["S"], [[0.0105]], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["K"], [[80 / 21], [20 / 21]], rtol=0, atol=1e-12)
        self.assertEqual(result["S"].shape, (1, 1))
        self.assertEqual(result["K"].shape, (2, 1))

    def test_zero_innovation_preserves_predicted_state(self):
        result = self.update()
        np.testing.assert_array_equal(result["x"], [50.0, 0.02])
        self.assertEqual(float(result["innov"]), 0.0)

    def test_positive_and_negative_innovations(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                result = self.update(innovation=sign * 0.021)
                # Independent scalar gain fractions give shifts .08 pp, .02 V.
                np.testing.assert_allclose(result["x"], [50 + sign * 0.08, 0.02 + sign * 0.02],
                                           rtol=0, atol=1e-12)

    def test_posterior_soc_is_clipped(self):
        for soc, innovation, expected in ((99.9, 1, 100), (0.1, -1, 0)):
            result = self.update(innovation=innovation, state=(soc, 0))
            self.assertEqual(result["x"][0], expected)
            self.assertTrue(np.all(np.isfinite(result["P"])))

    def test_joseph_covariance_matches_conditional_gaussian_reference(self):
        # Analytic posterior covariance from Gaussian conditioning, rather than
        # reproducing the implementation's Joseph matrix multiplication.
        expected = [[404 / 105, -4 / 105], [-4 / 105, 1 / 2100]]
        result = self.update(innovation=0.021)
        np.testing.assert_allclose(result["P"], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result["P"], result["P"].T, rtol=0, atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(result["P"]).min(), -1e-12)

    def test_sample_zero_has_no_propagation_or_process_noise(self):
        result = self.run_loop([7], [-10], [linear_ocv(50) - 10 * self.base["R_IN"]])
        self.assertEqual(len(result["prediction_calls"]), 0)
        trace = result["trace"][0]
        np.testing.assert_array_equal(trace["x_pred"], [50, 0])
        np.testing.assert_array_equal(trace["P_pred"], self.base["P0"])
        self.assertEqual(result["soc_ekf"][0], 50)

    def test_sample_zero_still_uses_current_voltage_measurement(self):
        voltage = linear_ocv(50) - 2 * self.base["R_IN"] + 0.01
        result = self.run_loop([0], [-2], [voltage])
        trace = result["trace"][0]
        self.assertAlmostEqual(trace["V_pred"], 3.44904)
        self.assertAlmostEqual(trace["innov"], 0.01)
        self.assertGreater(trace["x"][0], 50)

    def test_interval_start_current_and_actual_interval_lengths(self):
        times, currents, voltages, _, _ = self.zero_innovation_sequence()
        result = self.run_loop(times, currents, voltages)
        calls = result["prediction_calls"]
        self.assertEqual(len(calls), len(times) - 1)
        for k, call in enumerate(calls, start=1):
            self.assertEqual(call.args[1], currents[k - 1])
            self.assertEqual(call.args[2], times[k] - times[k - 1])

    def test_zero_innovation_agrees_with_cc_and_sample_voltage_current(self):
        times, currents, voltages, soc, polarization = self.zero_innovation_sequence()
        result = self.run_loop(times, currents, voltages)
        np.testing.assert_allclose(result["soc_ekf"], soc, rtol=0, atol=1e-11)
        np.testing.assert_allclose(result["vth_ekf"], polarization, rtol=0, atol=1e-13)
        np.testing.assert_allclose([t["V_pred"] for t in result["trace"]], voltages,
                                   rtol=0, atol=1e-13)
        np.testing.assert_allclose([t["innov"] for t in result["trace"]], 0,
                                   rtol=0, atol=1e-13)

    def test_covariance_propagates_only_between_samples(self):
        times, currents, voltages, _, _ = self.zero_innovation_sequence()
        result = self.run_loop(times, currents, voltages)
        for k in range(1, len(times)):
            alpha = np.exp(-(times[k] - times[k - 1]) / self.base["TAU"])
            previous = result["trace"][k - 1]["P"]
            # Element-wise analytical propagation of diagonal state transition.
            expected = previous * np.array([[1, alpha], [alpha, alpha * alpha]])
            expected += self.base["Q_noise"]
            np.testing.assert_allclose(result["trace"][k]["P_pred"], expected,
                                       rtol=0, atol=1e-14)

    def test_nonzero_innovation_loop_has_finite_states_and_psd_covariances(self):
        times, currents, voltages, _, _ = self.zero_innovation_sequence()
        result = self.run_loop(times, currents, voltages + [0.02, -0.01, 0.03, -0.02, 0.01])
        for trace in result["trace"]:
            for key, shape in (("x", (2,)), ("P", (2, 2)), ("K", (2, 1)),
                               ("H", (1, 2)), ("S", (1, 1))):
                self.assertEqual(trace[key].shape, shape)
                self.assertTrue(np.all(np.isfinite(trace[key])), key)
            self.assertTrue(0 <= trace["x"][0] <= 100)
            np.testing.assert_allclose(trace["P"], trace["P"].T, rtol=0, atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(trace["P"]).min(), -1e-12)
            self.assertGreater(trace["S"][0, 0], 0)

    def test_loop_boundary_cases_with_clipped_ocv(self):
        ocv = lambda soc: linear_ocv(np.clip(soc, 0, 100))
        for initial, residual, expected in ((99.9, 1, 100), (0.1, -1, 0)):
            result = self.run_loop([0], [0], [ocv(initial) + residual], state=(initial, 0), ocv=ocv)
            self.assertEqual(result["soc_ekf"][0], expected)
            trace = result["trace"][0]
            self.assertTrue(np.all(np.isfinite(trace["P"])))
            self.assertGreaterEqual(np.linalg.eigvalsh(trace["P"]).min(), -1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
