import json
import os
from pathlib import Path

import numpy as np
import meshio
from src.model import FEAModel


def compute_element_axial_strain_stress(model: FEAModel, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-element axial strain and axial stress (scalar) using the
    element's local x-axis defined by its chord.

    This is a first-pass stress metric intended for visualization.
    """
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(model.nodes.keys())}
    u_n = u.reshape((len(model.nodes), 6))

    axial_strain = np.zeros(len(model.elements), dtype=float)
    axial_stress = np.zeros(len(model.elements), dtype=float)

    for i, (_el_id, el) in enumerate(model.elements.items()):
        n1_id, n2_id = el.node_ids
        n1 = model.nodes[n1_id]
        n2 = model.nodes[n2_id]
        x1 = n1.coords
        x2 = n2.coords
        d = x2 - x1
        L = float(np.linalg.norm(d))
        if L == 0.0:
            continue
        e1 = d / L

        i1 = node_id_to_idx[n1_id]
        i2 = node_id_to_idx[n2_id]
        u1 = u_n[i1, :3]
        u2 = u_n[i2, :3]
        du = float(np.dot(e1, (u2 - u1)))
        eps = du / L

        mat = model.materials[el.material_id]
        axial_strain[i] = eps
        axial_stress[i] = float(mat.E) * eps

    return axial_strain, axial_stress


def export_deformed_model_json(model: FEAModel, u: np.ndarray, filepath: str) -> None:
    """
    Write a lattice JSON matching the FEAModel input schema, with node ``coords``
    set to reference position plus translational displacement (ux, uy, uz).

    Topology (elements), materials, sections, boundary_conditions, and
    point_loads are copied unchanged so the file can be edited (e.g. BC values)
    and loaded again with ``FEAModel.from_json``.
    """
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(model.nodes.keys())}
    u_n = u.reshape((len(model.nodes), 6))

    nodes_out: list[dict] = []
    for node_id, node in model.nodes.items():
        idx = node_id_to_idx[node_id]
        xf = (node.coords + u_n[idx, :3]).tolist()
        nodes_out.append({"id": int(node_id), "coords": xf})

    materials_out = [
        {
            "id": int(m.id),
            "E": float(m.E),
            "nu": float(m.nu),
            "model": m.model,
        }
        for m in sorted(model.materials.values(), key=lambda x: x.id)
    ]
    sections_out = [
        {
            "id": int(s.id),
            "A": float(s.A),
            "Iy": float(s.Iy),
            "Iz": float(s.Iz),
            "J": float(s.J),
        }
        for s in sorted(model.sections.values(), key=lambda x: x.id)
    ]
    elements_out = [
        {
            "id": int(el.id),
            "nodes": [int(el.node_ids[0]), int(el.node_ids[1])],
            "material": int(el.material_id),
            "section": int(el.section_id),
        }
        for el in sorted(model.elements.values(), key=lambda x: x.id)
    ]

    boundary_conditions_out = [
        {
            "node": int(bc.node_id),
            "dof": list(bc.dof),
            "value": float(bc.value),
        }
        for bc in model.boundary_conditions
    ]

    point_loads_out = [
        {
            "node": int(pl.node_id),
            "dof": pl.dof,
            "value": float(pl.value),
        }
        for pl in model.point_loads
    ]

    data = {
        "materials": materials_out,
        "sections": sections_out,
        "nodes": nodes_out,
        "elements": elements_out,
        "boundary_conditions": boundary_conditions_out,
        "point_loads": point_loads_out,
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def export_vtk(
    model: FEAModel,
    u: np.ndarray,
    filepath: str,
    *,
    cell_data: dict[str, np.ndarray] | None = None,
):
    """
    Exports the deformed lattice model to VTK/VTU format using meshio.
    """
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(model.nodes.keys())}
    
    # Extract points (original coordinates)
    points = np.zeros((len(model.nodes), 3))
    for node_id, node in model.nodes.items():
        idx = node_id_to_idx[node_id]
        points[idx] = node.coords
        
    # Extract lines (elements connecting nodes)
    lines = np.zeros((len(model.elements), 2), dtype=int)
    for i, (el_id, el) in enumerate(model.elements.items()):
        n1_id, n2_id = el.node_ids
        lines[i] = [node_id_to_idx[n1_id], node_id_to_idx[n2_id]]
        
    cells = [("line", lines)]
    
    # Extract node data (displacements)
    u_reshaped = u.reshape((len(model.nodes), 6))
    translations = u_reshaped[:, :3]
    rotations = u_reshaped[:, 3:]
    
    point_data = {
        "Displacement": translations,
        "Rotation": rotations
    }

    mesh_cell_data = None
    if cell_data is not None:
        mesh_cell_data = {k: [np.asarray(v)] for k, v in cell_data.items()}
    
    # Write to VTK/VTU using meshio
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=mesh_cell_data,
    )
    
    # Save the mesh
    mesh.write(filepath)


def write_pvd_timeseries(filepath: str, datasets: list[tuple[float, str]]) -> None:
    """
    Write a ParaView `.pvd` collection indexing multiple `.vtu` files as timesteps.

    datasets: list of (timestep, vtu_path). Paths are written relative to the PVD.
    """
    pvd_path = Path(filepath)
    pvd_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">')
    lines.append("  <Collection>")

    base_dir = str(pvd_path.parent)
    for ts, file_path in datasets:
        rel = os.path.relpath(file_path, start=base_dir)
        lines.append(f'    <DataSet timestep="{ts}" group="" part="0" file="{rel}"/>')

    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    pvd_path.write_text("\n".join(lines) + "\n")
