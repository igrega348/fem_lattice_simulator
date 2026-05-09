#!/usr/bin/env python3
"""
Build a fem_lattice_simulator JSON model from an xray_projection_render-style
lattice.yaml (unit cell cylinders + tessellation).

Example source file:
https://github.com/igrega348/xray_projection_render/blob/main/examples/lattice.yaml

Node coordinates are normalized for ``xray_projection_render`` (~[-1,1]^3 world): centered at
the origin with the axis-aligned bounding box half-extent (max half-width along x/y/z)
mapped to ``--target-half-extent`` (default 0.8). Section areas and second moments scale
accordingly so the undeformed slender-beam geometry matches the shrunken coordinates.
``meta.unit_cell_period`` holds the tessellation period in the same normalized length units.

Usage:
  uv run python scripts/generate_lattice_from_yaml.py \\
      --yaml path/to/lattice.yaml --out lattice_kelvin.json --nx 4 --ny 4 --nz 4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Solid circular section; matches ``radius: 0.025`` on cylinders in typical Kelvin ``lattice.yaml``.
_DEFAULT_STRUT_RADIUS = 0.025


def _circular_beam_section_props(radius: float) -> dict[str, float]:
    r2 = radius * radius
    r4 = r2 * r2
    return {
        "A": math.pi * r2,
        "Iy": math.pi * r4 / 4.0,
        "Iz": math.pi * r4 / 4.0,
        "J": math.pi * r4 / 2.0,
    }


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


def _normalize_tessellation_beams(
    beams: list[tuple[np.ndarray, np.ndarray]],
    *,
    target_half_extent: float,
    enabled: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float, np.ndarray]:
    """
    Uniform scale and translate so the tight axis-aligned hull of beam endpoints fits in
    approximately [-target_half_extent, target_half_extent] on the widest axis.

    Returns (scaled beams, linear scale factor s, centroid of raw bbox).
    ``p_normalized = s * (p_raw - centroid)``.
    """
    if not enabled or not beams:
        return beams, 1.0, np.zeros(3, dtype=float)
    pts_list: list[np.ndarray] = []
    for a, b in beams:
        pts_list.append(a.reshape(1, 3))
        pts_list.append(b.reshape(1, 3))
    arr = np.vstack(pts_list)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    centroid = (lo + hi) * 0.5
    half_ranges = (hi - lo) * 0.5
    max_half = float(np.max(half_ranges))
    if max_half <= 0.0:
        return beams, 1.0, centroid
    s = float(target_half_extent / max_half)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in beams:
        out.append((s * (a - centroid), s * (b - centroid)))
    return out, s, centroid


def _scale_sections_for_geometry_scale(
    sections: list[dict[str, Any]], linear_scale: float
) -> None:
    """Scale section dicts consistently with a uniform spatial scale factor (lengths × s)."""
    if linear_scale == 1.0:
        return
    s2 = linear_scale * linear_scale
    s4 = s2 * s2
    for sec in sections:
        sec["A"] = float(sec["A"]) * s2
        sec["Iy"] = float(sec["Iy"]) * s4
        sec["Iz"] = float(sec["Iz"]) * s4
        sec["J"] = float(sec["J"]) * s4


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
    normalize_for_renderer: bool = True,
    target_half_extent: float = 0.8,
    position_decimals: int = 9,
    material_id: int = 1,
    section_id: int = 1,
    materials: list[dict[str, Any]] | None = None,
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    beams = _collect_beam_endpoints(uc, repeats, subdivide)
    beams, geom_scale, center_yaml = _normalize_tessellation_beams(
        beams,
        target_half_extent=target_half_extent,
        enabled=normalize_for_renderer,
    )
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
            {"id": 1, "E": 1.0, "nu": 0.3, "model": "linear_elastic"},
        ]
    if sections is None:
        circ = _circular_beam_section_props(_DEFAULT_STRUT_RADIUS)
        sections = [{"id": section_id, **circ}]
    _scale_sections_for_geometry_scale(sections, geom_scale)

    out: dict[str, Any] = {
        "materials": materials,
        "sections": sections,
        "nodes": nodes,
        "elements": elements,
        "boundary_conditions": [],
        "point_loads": [],
    }
    if normalize_for_renderer:
        period = _uc_period(uc) * geom_scale
        out["meta"] = {
            "unit_cell_period": period.tolist(),
            "renderer_normalization": {
                "scale": geom_scale,
                "center_yaml_space": center_yaml.tolist(),
                "target_half_extent": float(target_half_extent),
            },
        }
    return out


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
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep raw YAML tessellation coordinates (disables renderer-style centering/scaling).",
    )
    parser.add_argument(
        "--target-half-extent",
        type=float,
        default=0.8,
        help="After centering, max half-width of node hull along x/y/z is set to this (default 0.8).",
    )
    args = parser.parse_args()

    uc = load_uc_from_lattice_yaml(args.yaml)
    model = build_model_dict(
        uc,
        (args.nx, args.ny, args.nz),
        subdivide=args.subdivide,
        normalize_for_renderer=not args.no_normalize,
        target_half_extent=args.target_half_extent,
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
