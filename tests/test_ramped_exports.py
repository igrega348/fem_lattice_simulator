import json
from pathlib import Path

from src.model import FEAModel
from src.solver import Solver


def test_ramped_writes_vtu_json_and_pvd(tmp_path: Path):
    # Minimal 2-node, 1-element model with one ramped prescribed displacement.
    model_dict = {
        "materials": [{"id": 1, "E": 210e9, "nu": 0.3, "model": "linear_elastic"}],
        "sections": [{"id": 1, "A": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}],
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [1.0, 0.0, 0.0]},
        ],
        "elements": [{"id": 1, "nodes": [1, 2], "material": 1, "section": 1}],
        "boundary_conditions": [
            {"node": 1, "dof": ["ux", "uy", "uz", "rx", "ry", "rz"], "value": 0.0},
            {"node": 2, "dof": ["uz"], "value": -0.01},
        ],
        "point_loads": [],
    }

    inp = tmp_path / "m.json"
    inp.write_text(json.dumps(model_dict))

    model = FEAModel.from_json(str(inp))
    solver = Solver(model)

    prefix = tmp_path / "run"
    solver.solve_ramped(
        2,
        tol=1e-10,
        max_iter=10,
        output_steps={0, 1, 2},
        output_prefix=str(prefix),
        write_vtu=True,
        write_json=True,
        timestep_mode="factor",
    )

    assert (tmp_path / "run.pvd").exists()
    assert (tmp_path / "run_t0000.vtu").exists()
    assert (tmp_path / "run_t0001.vtu").exists()
    assert (tmp_path / "run_t0002.vtu").exists()
    assert (tmp_path / "run_t0000.json").exists()
    assert (tmp_path / "run_t0001.json").exists()
    assert (tmp_path / "run_t0002.json").exists()

