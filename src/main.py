import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
from src.io import export_deformed_model_json, export_vtk
from src.model import FEAModel
from src.solver import Solver


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "lattice.json"
    out_vtu = sys.argv[2] if len(sys.argv) > 2 else "output.vtu"
    if len(sys.argv) > 3:
        out_json = sys.argv[3]
    else:
        out_json = str(Path(out_vtu).with_name(Path(out_vtu).stem + "_deformed.json"))
    print(f"Loading model from {model_path!r}...")
    model = FEAModel.from_json(model_path)
    
    print(f"Model loaded: {len(model.nodes)} nodes, {len(model.elements)} elements.")
    
    print("Initializing solver...")
    solver = Solver(model)
    
    print("Solving...")
    start_time = time.time()
    u = solver.solve()
    end_time = time.time()
    
    print(f"Solve completed in {end_time - start_time:.4f} seconds.")
    
    print(f"Exporting results to {out_vtu!r}...")
    export_vtk(model, u, out_vtu)
    print(f"Exporting deformed model JSON to {out_json!r}...")
    export_deformed_model_json(model, u, out_json)
    print("Done!")

if __name__ == "__main__":
    main()
