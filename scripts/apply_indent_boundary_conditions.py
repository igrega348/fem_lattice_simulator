#!/usr/bin/env python3
"""
Add boundary conditions to a fem_lattice_simulator JSON lattice:

  - Bottom plane (minimum z): uz = 0 (roller support in x,y).
  - Top plane (maximum z), rectangular patch in xy: prescribed uz for indentation
    (negative = downward if +z is up). Patch placement is either centered in the
    domain (middle cells) or anchored at xmin, ymin (corner).

Typical use after generating a tiled UC (e.g. 4x4x4 cells of side 0.4 m), middle 2x2 indenter:

  uv run python scripts/apply_indent_boundary_conditions.py \\
      --in out.json --out out.json --patch-cells-x 2 --patch-cells-y 2 \\
      --patch-placement center --indent-uz -0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _patch_xy_bounds(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    cell: float,
    patch_cells_x: int,
    patch_cells_y: int,
    placement: str,
) -> tuple[float, float, float, float]:
    """Return (x0, x1, y0, y1) inclusive patch limits in the xy plane."""
    pcx, pcy = patch_cells_x, patch_cells_y
    if placement == "origin":
        x0 = xmin + 0.0
        x1 = xmin + pcx * cell
        y0 = ymin + 0.0
        y1 = ymin + pcy * cell
        return x0, x1, y0, y1
    if placement == "center":
        ncx = max(1, round((xmax - xmin) / cell))
        ncy = max(1, round((ymax - ymin) / cell))
        if pcx > ncx or pcy > ncy:
            raise SystemExit(
                f"Patch ({pcx}x{pcy} cells) does not fit grid ({ncx}x{ncy} cells) at cell_size={cell}."
            )
        ix0 = (ncx - pcx) // 2
        iy0 = (ncy - pcy) // 2
        x0 = xmin + ix0 * cell
        x1 = x0 + pcx * cell
        y0 = ymin + iy0 * cell
        y1 = y0 + pcy * cell
        return x0, x1, y0, y1
    raise SystemExit(f"Unknown --patch-placement {placement!r} (use origin or center).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Input lattice .json")
    p.add_argument("--out", dest="out", type=Path, required=True, help="Output .json")
    p.add_argument(
        "--cell-size",
        type=float,
        default=0.4,
        help="UC edge length used to size the top patch (matches lattice.yaml uc box; not inferred).",
    )
    p.add_argument(
        "--patch-cells-x",
        type=int,
        default=2,
        help="Patch width in number of unit cells along x.",
    )
    p.add_argument("--patch-cells-y", type=int, default=2)
    p.add_argument(
        "--patch-placement",
        choices=("origin", "center"),
        default="center",
        help="origin: patch at xmin,ymin; center: middle patch (e.g. middle 2x2 on a 4x4 top).",
    )
    p.add_argument(
        "--indent-uz",
        type=float,
        default=-1e-3,
        help="Prescribed uz on top patch (e.g. negative if +z is up).",
    )
    p.add_argument(
        "--plane-tol",
        type=float,
        default=1e-5,
        help="Tolerance for classifying nodes on z=zmin or z=zmax.",
    )
    p.add_argument(
        "--pin-bottom-uxuy",
        action="store_true",
        help="Also fix ux,uy of one bottom node to remove rigid-body drift.",
    )
    p.add_argument(
        "--indenter-uxuy-zero",
        action="store_true",
        help="Also fix ux,uy=0 on the indenter patch nodes (pure vertical motion).",
    )
    args = p.parse_args()

    data = json.loads(args.inp.read_text())
    nodes = data["nodes"]
    if not nodes:
        raise SystemExit("No nodes in input.")

    xs = [float(n["coords"][0]) for n in nodes]
    ys = [float(n["coords"][1]) for n in nodes]
    zs = [float(n["coords"][2]) for n in nodes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    cell = float(args.cell_size)
    x0, x1, y0, y1 = _patch_xy_bounds(
        xmin,
        xmax,
        ymin,
        ymax,
        cell,
        args.patch_cells_x,
        args.patch_cells_y,
        args.patch_placement,
    )

    boundary_conditions: list[dict] = []
    bottom_ids: list[int] = []
    top_patch_ids: list[int] = []
    pinned_bottom_id: int | None = None

    for n in nodes:
        nid = n["id"]
        x, y, z = (float(n["coords"][0]), float(n["coords"][1]), float(n["coords"][2]))
        if abs(z - zmin) < args.plane_tol:
            boundary_conditions.append({"node": nid, "dof": ["uz"], "value": 0.0})
            bottom_ids.append(nid)
        elif (
            abs(z - zmax) < args.plane_tol
            and x0 - args.plane_tol <= x <= x1 + args.plane_tol
            and y0 - args.plane_tol <= y <= y1 + args.plane_tol
        ):
            boundary_conditions.append({"node": nid, "dof": ["uz"], "value": float(args.indent_uz)})
            if args.indenter_uxuy_zero:
                boundary_conditions.append({"node": nid, "dof": ["ux", "uy"], "value": 0.0})
            top_patch_ids.append(nid)

    if args.pin_bottom_uxuy and bottom_ids:
        # Pin the lexicographically smallest (x,y) node on the bottom plane.
        bottom_nodes = [n for n in nodes if abs(float(n["coords"][2]) - zmin) < args.plane_tol]
        pin = min(
            bottom_nodes,
            key=lambda n: (float(n["coords"][0]), float(n["coords"][1]), int(n["id"])),
        )
        pinned_bottom_id = int(pin["id"])
        boundary_conditions.append({"node": pinned_bottom_id, "dof": ["ux", "uy"], "value": 0.0})

    data["boundary_conditions"] = boundary_conditions
    if "point_loads" not in data:
        data["point_loads"] = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")

    print(
        f"Wrote {args.out}: z in [{zmin:g}, {zmax:g}], cell_size={cell:g}, "
        f"patch_placement={args.patch_placement}, top patch xy in "
        f"[{x0:g}, {x1:g}] x [{y0:g}, {y1:g}], "
        f"uz=0 on {len(bottom_ids)} bottom nodes, uz={args.indent_uz:g} on {len(top_patch_ids)} top-patch nodes, "
        f"pin_bottom_uxuy={args.pin_bottom_uxuy}({pinned_bottom_id}), indenter_uxuy_zero={args.indenter_uxuy_zero}."
    )


if __name__ == "__main__":
    main()
