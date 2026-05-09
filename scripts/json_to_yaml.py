#!/usr/bin/env python3
"""
Convert fem_lattice_simulator-generated JSON files back to YAML.

Examples:
  # Lattice-renderable YAML (matches lattice.yaml object model)
  uv run python scripts/json_to_yaml.py run_kelvin_t0000.json --radius 0.025
  uv run python scripts/json_to_yaml.py "run_kelvin_t*.json" --radius 0.025 --outdir yaml_out --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


class _NoAliasSafeDumper(yaml.SafeDumper):
    # Keep output closer to the hand-written lattice.yaml (no &id/* anchors).
    def ignore_aliases(self, data: Any) -> bool:  # type: ignore[override]
        return True


def _iter_input_files(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        # Allow shell-globbed args and also unexpanded patterns.
        p = Path(raw)
        if p.exists() and p.is_file():
            out.append(p)
            continue

        matches = sorted(Path().glob(raw))
        if matches:
            out.extend([m for m in matches if m.is_file()])
            continue

        raise FileNotFoundError(f"No such file (or glob produced no matches): {raw}")

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)
    return deduped


def _out_path(in_path: Path, outdir: Path | None, suffix: str) -> Path:
    name = in_path.name
    if name.lower().endswith(".json"):
        name = name[: -len(".json")]
    name = f"{name}{suffix}"
    return (outdir / name) if outdir is not None else in_path.with_name(name)


def _compute_bounds(points: list[list[float]], pad: float = 0.0) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return {
        "xmin": float(min(xs) - pad),
        "xmax": float(max(xs) + pad),
        "ymin": float(min(ys) - pad),
        "ymax": float(max(ys) + pad),
        "zmin": float(min(zs) - pad),
        "zmax": float(max(zs) + pad),
    }


def _json_model_to_lattice_object_collection(
    data: dict[str, Any],
    *,
    radius: float | None,
    radius_from_area: bool,
    rho: float,
) -> dict[str, Any]:
    nodes = data.get("nodes")
    elements = data.get("elements")
    sections = data.get("sections", [])

    if not isinstance(nodes, list) or not isinstance(elements, list):
        raise ValueError("Expected JSON model with top-level 'nodes' and 'elements' lists.")

    node_id_to_coords: dict[int, list[float]] = {}
    for n in nodes:
        if not isinstance(n, dict) or "id" not in n or "coords" not in n:
            raise ValueError("Each node must be a mapping with 'id' and 'coords'.")
        nid = int(n["id"])
        c = n["coords"]
        if not (isinstance(c, list) and len(c) == 3):
            raise ValueError(f"Node {nid} has invalid coords (expected length-3 list).")
        node_id_to_coords[nid] = [float(c[0]), float(c[1]), float(c[2])]

    section_id_to_area: dict[int, float] = {}
    if isinstance(sections, list):
        for s in sections:
            if isinstance(s, dict) and "id" in s and "A" in s:
                section_id_to_area[int(s["id"])] = float(s["A"])

    cylinders: list[dict[str, Any]] = []
    for el in elements:
        if not isinstance(el, dict):
            raise ValueError("Each element must be a mapping.")
        nids = el.get("nodes")
        if not (isinstance(nids, list) and len(nids) == 2):
            raise ValueError("Each element must have 'nodes': [n1, n2].")
        n1 = int(nids[0])
        n2 = int(nids[1])
        p0 = node_id_to_coords.get(n1)
        p1 = node_id_to_coords.get(n2)
        if p0 is None or p1 is None:
            raise ValueError(f"Element references unknown node id(s): {n1}, {n2}")

        r = radius
        if r is None:
            # Default that matches your checked-in lattice.yaml.
            r = 0.025
            if radius_from_area:
                sec_id = el.get("section")
                if sec_id is not None:
                    A = section_id_to_area.get(int(sec_id))
                    if A is not None and A > 0.0:
                        r = math.sqrt(A / math.pi)

        cylinders.append(
            {
                "type": "cylinder",
                # Copy to avoid YAML anchors/aliases due to shared list objects.
                "p0": [float(p0[0]), float(p0[1]), float(p0[2])],
                "p1": [float(p1[0]), float(p1[1]), float(p1[2])],
                "radius": float(r),
                "rho": float(rho),
            }
        )

    points = list(node_id_to_coords.values())
    bounds = _compute_bounds(points, pad=float(radius or 0.0))
    return {
        "type": "object_collection",
        "objects": cylinders,
        **bounds,
    }


def convert_file(
    in_path: Path,
    out_path: Path,
    *,
    overwrite: bool,
    radius: float | None,
    radius_from_area: bool,
    rho: float,
) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out_path} (pass --overwrite)"
        )

    data: Any = json.loads(in_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected JSON root object to be a mapping.")
    out_data = _json_model_to_lattice_object_collection(
        data, radius=radius, radius_from_area=radius_from_area, rho=rho
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(
            out_data,
            sort_keys=False,
            default_flow_style=False,
            Dumper=_NoAliasSafeDumper,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON files to YAML.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON file(s). You can also pass a quoted glob like 'run_kelvin_t*.json'.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, YAML is written next to each input file.",
    )
    parser.add_argument(
        "--suffix",
        default=".yaml",
        help="Output filename suffix (default: .yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YAML files if present.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Cylinder radius (default: 0.025 if omitted).",
    )
    parser.add_argument(
        "--radius-from-area",
        action="store_true",
        help="If set and --radius is omitted, derive cylinder radius from section area A via r=sqrt(A/pi).",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Cylinder density (rho) field for the lattice renderer (default: 1.0).",
    )
    args = parser.parse_args()

    outdir: Path | None = None
    if args.outdir is not None:
        outdir = args.outdir

    inputs = _iter_input_files(args.inputs)
    for in_path in inputs:
        out_path = _out_path(in_path, outdir, args.suffix)
        convert_file(
            in_path,
            out_path,
            overwrite=args.overwrite,
            radius=args.radius,
            radius_from_area=args.radius_from_area,
            rho=args.rho,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

