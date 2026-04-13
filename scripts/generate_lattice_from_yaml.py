#!/usr/bin/env python3
"""
Build a fem_lattice_simulator JSON model from an xray_projection_render-style
lattice.yaml (unit cell cylinders + tessellation).

Example source file:
https://github.com/igrega348/xray_projection_render/blob/main/examples/lattice.yaml

Usage:
  uv run python scripts/generate_lattice_from_yaml.py \\
      --yaml path/to/lattice.yaml --out lattice_kelvin.json --nx 4 --ny 4 --nz 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _extract_cylinders(uc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return cylinder entries from uc.objects (object_collection layout)."""
    objects = uc.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("uc.objects must be a mapping (object_collection).")
    inner = objects.get("objects")
    if not isinstance(inner, list):
        raise ValueError("uc.objects.objects must be a list of primitives.")
    out: list[dict[str, Any]] = []
    for item in inner:
        if isinstance(item, dict) and item.get("type") == "cylinder":
            out.append(item)
    if not out:
        raise ValueError("No cylinders found under uc.objects.objects.")
    return out


def _uc_period(uc: dict[str, Any]) -> np.ndarray:
    for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        if key not in uc:
            raise ValueError(f"Unit cell missing '{key}' (need full axis bounds).")
    return np.array(
        [
            float(uc["xmax"]) - float(uc["xmin"]),
            float(uc["ymax"]) - float(uc["ymin"]),
            float(uc["zmax"]) - float(uc["zmin"]),
        ],
        dtype=float,
    )


def _uc_origin(uc: dict[str, Any]) -> np.ndarray:
    return np.array(
        [float(uc["xmin"]), float(uc["ymin"]), float(uc["zmin"])], dtype=float
    )


def _parse_point(p: Any) -> np.ndarray:
    if isinstance(p, (list, tuple)) and len(p) == 3:
        return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
    raise TypeError(f"Expected length-3 list for point, got {type(p)}: {p!r}")


def _subdivide_segment(
    p0: np.ndarray, p1: np.ndarray, n_seg: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split one beam into n_seg contiguous segments (n_seg >= 1)."""
    if n_seg < 1:
        raise ValueError("subdivide must be >= 1")
    if n_seg == 1:
        return [(p0.copy(), p1.copy())]
    t = np.linspace(0.0, 1.0, n_seg + 1)
    pts = p0 + np.outer(t, (p1 - p0))
    return [(pts[i].copy(), pts[i + 1].copy()) for i in range(n_seg)]


def _collect_beam_endpoints(
    uc: dict[str, Any], repeats: tuple[int, int, int], subdivide: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    period = _uc_period(uc)
    base = _uc_origin(uc)
    cylinders = _extract_cylinders(uc)
    nx, ny, nz = repeats
    beams: list[tuple[np.ndarray, np.ndarray]] = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = base + period * np.array([ix, iy, iz], dtype=float)
                for c in cylinders:
                    p0 = _parse_point(c["p0"]) + offset
                    p1 = _parse_point(c["p1"]) + offset
                    beams.extend(_subdivide_segment(p0, p1, subdivide))
    return beams


def _node_key(p: np.ndarray, decimals: int) -> tuple[float, float, float]:
    return (
        round(float(p[0]), decimals),
        round(float(p[1]), decimals),
        round(float(p[2]), decimals),
    )


def build_model_dict(
    uc: dict[str, Any],
    repeats: tuple[int, int, int],
    *,
    subdivide: int = 1,
    position_decimals: int = 9,
    material_id: int = 1,
    section_id: int = 1,
    materials: list[dict[str, Any]] | None = None,
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    beams = _collect_beam_endpoints(uc, repeats, subdivide)
    pos_to_id: dict[tuple[float, float, float], int] = {}
    nodes: list[dict[str, Any]] = []
    elements: list[dict[str, Any]] = []

    def get_node_id(p: np.ndarray) -> int:
        key = _node_key(p, position_decimals)
        if key in pos_to_id:
            return pos_to_id[key]
        nid = len(nodes) + 1
        pos_to_id[key] = nid
        nodes.append({"id": nid, "coords": [key[0], key[1], key[2]]})
        return nid

    # Tessellation repeats the full unit-cell cylinder list per tile, so struts on
    # shared faces appear from both cells. Nodes merge by position; beams must be
    # deduped by undirected node pair so stiffness is not double-counted.
    seen_edges: set[tuple[int, int]] = set()
    el_id = 0
    for a, b in beams:
        n1 = get_node_id(a)
        n2 = get_node_id(b)
        if n1 == n2:
            continue
        edge = (n1, n2) if n1 < n2 else (n2, n1)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        el_id += 1
        elements.append(
            {
                "id": el_id,
                "nodes": [n1, n2],
                "material": material_id,
                "section": section_id,
            }
        )

    if materials is None:
        materials = [
            {"id": 1, "E": 210e9, "nu": 0.3, "model": "linear_elastic"},
        ]
    if sections is None:
        sections = [
            {"id": 1, "A": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5},
        ]

    return {
        "materials": materials,
        "sections": sections,
        "nodes": nodes,
        "elements": elements,
        "boundary_conditions": [],
        "point_loads": [],
    }


def load_uc_from_lattice_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Root of lattice YAML must be a mapping.")
    if data.get("type") != "tessellated_obj_coll":
        pass  # still allow if uc is present
    uc = data.get("uc")
    if not isinstance(uc, dict):
        raise ValueError("Expected top-level 'uc' mapping.")
    return uc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate lattice.json from xray-style lattice.yaml (Kelvin / cylinder UC)."
    )
    parser.add_argument("--yaml", type=Path, required=True, help="Input lattice.yaml path.")
    parser.add_argument("--out", type=Path, required=True, help="Output .json path.")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument(
        "--subdivide",
        type=int,
        default=1,
        help="Beam segments per YAML strut (1 = one FE element per strut). "
        "Values >1 split each strut into collinear elements for mesh refinement.",
    )
    parser.add_argument(
        "--position-decimals",
        type=int,
        default=9,
        help="Rounding for merging coincident nodes after tessellation.",
    )
    args = parser.parse_args()

    uc = load_uc_from_lattice_yaml(args.yaml)
    model = build_model_dict(
        uc,
        (args.nx, args.ny, args.nz),
        subdivide=args.subdivide,
        position_decimals=args.position_decimals,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2) + "\n")
    print(
        f"Wrote {args.out} with {len(model['nodes'])} nodes, "
        f"{len(model['elements'])} elements "
        f"({args.nx}x{args.ny}x{args.nz} cells, subdivide={args.subdivide})."
    )


if __name__ == "__main__":
    main()
