"""T006B: execute only extracted path setup and synthetic integration code.

Run: .venv/Scripts/python.exe -B tests/test_path_portability.py
No notebook experiments or feature-ablation module execution is performed.
"""

import ast
import json
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_EDA_Panasonic.ipynb", "02_Baselines.ipynb",
    "03_Feature_Engineering.ipynb", "04_ML_Baselines.ipynb",
    "05_Robustness_Analysis.ipynb", "06_Residual_Analysis.ipynb",
    "07_Residual_Correction.ipynb",
)
SCRIPT = ROOT / "scripts" / "run_feature_ablation.py"


def notebook_tree(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
    return ast.parse(source)


def root_assignment(tree):
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "PROJECT_ROOT"
                       for target in node.targets)]
    if len(matches) != 1:
        raise AssertionError("Expected exactly one PROJECT_ROOT assignment")
    return matches[0]


def notebook_root(name, cwd):
    tree = notebook_tree(name)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_find_project_root")
    code = compile(ast.Module(body=[function, root_assignment(tree)], type_ignores=[]),
                   name, "exec")
    env = {"Path": Path}
    with mock.patch.object(Path, "cwd", return_value=cwd):
        exec(code, env)
    return env["PROJECT_ROOT"]


class PathPortability(unittest.TestCase):
    def test_notebooks_from_repository_root(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                self.assertEqual(notebook_root(name, ROOT), ROOT)

    def test_notebooks_from_notebooks_directory(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                self.assertEqual(notebook_root(name, ROOT / "notebooks"), ROOT)

    def test_unrelated_directory_and_incomplete_markers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            # Neither an arbitrary directory nor one marker alone is enough.
            for has_agents in (False, True):
                if has_agents:
                    (cwd / "AGENTS.md").touch()
                for name in NOTEBOOKS:
                    with self.subTest(notebook=name, has_agents=has_agents):
                        with self.assertRaisesRegex(RuntimeError, "Cannot locate project repository"):
                            notebook_root(name, cwd)

    def test_script_root_is_independent_of_working_directory(self):
        assignment = root_assignment(ast.parse(SCRIPT.read_text(encoding="utf-8")))
        code = compile(ast.Module(body=[assignment], type_ignores=[]), str(SCRIPT), "exec")
        with tempfile.TemporaryDirectory() as directory:
            for cwd in (ROOT, ROOT / "notebooks", Path(directory)):
                env = {"Path": Path, "__file__": str(SCRIPT)}
                with self.subTest(cwd=cwd), mock.patch.object(Path, "cwd", return_value=cwd):
                    exec(code, env)
                    self.assertEqual(env["PROJECT_ROOT"], ROOT)

    def test_no_absolute_project_root_assignments(self):
        trees = [notebook_tree(name) for name in NOTEBOOKS]
        trees.append(ast.parse(SCRIPT.read_text(encoding="utf-8")))
        for tree in trees:
            assignment = root_assignment(tree)
            for node in ast.walk(assignment.value):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertFalse(Path(node.value).is_absolute())
                    self.assertFalse(PureWindowsPath(node.value).is_absolute())

    def test_notebook_four_trapezoid_arguments(self):
        tree = notebook_tree("04_ML_Baselines.ipynb")
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "np"
                 and node.func.attr in ("trapz", "trapezoid")]
        self.assertEqual(len(calls), 1)
        expected = ast.parse("np.trapezoid(I_phys, x=t_seq, axis=1)", mode="eval").body
        self.assertEqual(ast.dump(calls[0]), ast.dump(expected))

    def test_synthetic_signed_integration_with_irregular_timestamps(self):
        tree = notebook_tree("04_ML_Baselines.ipynb")
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "build_summary_features")
        # Execute only the real integration and Ah conversion assignments.
        nodes = [node for node in function.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name)
                         and target.id in ("Q_window_As", "Q_window_Ah")
                         for target in node.targets)]
        self.assertEqual(len(nodes), 2)
        env = {"np": np, "I_phys": np.array([[0., 2., 4.], [0., -2., -4.]]),
               "t_seq": np.array([[0., 1., 3.], [0., 2., 5.]])}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "integration", "exec"), env)
        np.testing.assert_array_equal(env["Q_window_As"], [7., -11.])
        np.testing.assert_allclose(env["Q_window_Ah"], np.array([7., -11.]) / 3600,
                                   rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
