import argparse
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
from src.io import compute_element_axial_strain_stress, export_deformed_model_json, export_vtk
from src.model import FEAModel
from src.solver import Solver


def _parse_int_set(s: str) -> set[int]:
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def main():
    parser = argparse.ArgumentParser(description="3D beam lattice FE solver")
    parser.add_argument("model", nargs="?", default="lattice.json", help="Input model JSON")
    parser.add_argument(
        "--output-prefix",
        default="output",
        help="Output prefix for ramped exports (writes {prefix}_t####.vtu/.json and {prefix}.pvd).",
    )
    parser.add_argument(
        "--ramp-steps",
        type=int,
        default=0,
        help="If >0, run ramped solve with T_max=ramp_steps (inclusive 0..T_max).",
    )
    parser.add_argument(
        "--output-steps",
        type=_parse_int_set,
        default=None,
        help="Comma-separated step indices to export (e.g. 0,10,20).",
    )
    parser.add_argument(
        "--output-every",
        type=int,
        default=0,
        help="If >0, export every k ramp steps (in addition to any --output-steps).",
    )
    parser.add_argument(
        "--timestep-mode",
        choices=("factor", "step"),
        default="factor",
        help="Time values in the PVD: factor uses t/T_max, step uses t.",
    )
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--no-vtu", action="store_true", help="Disable VTU export")
    parser.add_argument("--no-json", action="store_true", help="Disable deformed JSON export")

    # Backwards-compatible positional outputs for single-step mode only.
    parser.add_argument(
        "out_vtu",
        nargs="?",
        default=None,
        help="(single-step only) VTU path, default output.vtu",
    )
    parser.add_argument(
        "out_json",
        nargs="?",
        default=None,
        help="(single-step only) deformed JSON path, default output_deformed.json",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model!r}...")
    model = FEAModel.from_json(args.model)
    print(f"Model loaded: {len(model.nodes)} nodes, {len(model.elements)} elements.")

    print("Initializing solver...")
    solver = Solver(model)

    if args.ramp_steps and args.ramp_steps > 0:
        if args.output_steps is None:
            output_steps: set[int] = set()
        else:
            output_steps = set(args.output_steps)

        if args.output_every and args.output_every > 0:
            output_steps |= set(range(0, args.ramp_steps + 1, args.output_every))

        print(f"Solving (ramped, T_max={args.ramp_steps})...")
        start_time = time.time()
        solver.solve_ramped(
            args.ramp_steps,
            tol=args.tol,
            max_iter=args.max_iter,
            output_steps=output_steps,
            output_prefix=args.output_prefix,
            write_vtu=not args.no_vtu,
            write_json=not args.no_json,
            timestep_mode=args.timestep_mode,
        )
        end_time = time.time()
        print(f"Ramped solve completed in {end_time - start_time:.4f} seconds.")
        print("Done!")
        return

    out_vtu = args.out_vtu or "output.vtu"
    out_json = args.out_json or str(Path(out_vtu).with_name(Path(out_vtu).stem + "_deformed.json"))

    print("Solving...")
    start_time = time.time()
    u = solver.solve(tol=args.tol, max_iter=args.max_iter)
    end_time = time.time()
    print(f"Solve completed in {end_time - start_time:.4f} seconds.")

    if not args.no_vtu:
        print(f"Exporting results to {out_vtu!r}...")
        eps, sigma = compute_element_axial_strain_stress(model, u)
        export_vtk(
            model,
            u,
            out_vtu,
            cell_data={
                "AxialStrain": eps,
                "AxialStress": sigma,
                "VonMisesEqv": np.abs(sigma),
            },
        )
    if not args.no_json:
        print(f"Exporting deformed model JSON to {out_json!r}...")
        export_deformed_model_json(model, u, out_json)
    print("Done!")


if __name__ == "__main__":
    main()
