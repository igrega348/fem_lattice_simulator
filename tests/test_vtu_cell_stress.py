import json
from pathlib import Path

import meshio
import numpy as np

from src.io import compute_element_axial_strain_stress, export_vtk
from src.model import FEAModel


def test_export_vtk_includes_axial_stress_cell_data(tmp_path: Path):
    model_dict = {
        "materials": [{"id": 1, "E": 200e9, "nu": 0.3, "model": "linear_elastic"}],
        "sections": [{"id": 1, "A": 1e-4, "Iy": 1e-10, "Iz": 1e-10, "J": 2e-10}],
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [1.0, 0.0, 0.0]},
        ],
        "elements": [{"id": 1, "nodes": [1, 2], "material": 1, "section": 1}],
        "boundary_conditions": [],
        "point_loads": [],
    }
    inp = tmp_path / "m.json"
    inp.write_text(json.dumps(model_dict))
    m = FEAModel.from_json(str(inp))

    # u: node2 ux = 1e-4
    u = np.zeros((12,), dtype=float)
    u[6 + 0] = 1e-4

    eps, sigma = compute_element_axial_strain_stress(m, u)
    out = tmp_path / "o.vtu"
    export_vtk(
        m,
        u,
        str(out),
        cell_data={"AxialStrain": eps, "AxialStress": sigma, "VonMisesEqv": np.abs(sigma)},
    )
    mesh = meshio.read(str(out))
    assert "AxialStrain" in mesh.cell_data_dict
    assert "AxialStress" in mesh.cell_data_dict
    assert "VonMisesEqv" in mesh.cell_data_dict

