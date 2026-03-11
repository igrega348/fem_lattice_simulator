import time
import jax
jax.config.update("jax_enable_x64", True)
from src.model import FEAModel
from src.solver import Solver
from src.io import export_vtk

def main():
    print("Loading model...")
    model = FEAModel.from_json("lattice.json")
    
    print(f"Model loaded: {len(model.nodes)} nodes, {len(model.elements)} elements.")
    
    print("Initializing solver...")
    solver = Solver(model)
    
    print("Solving...")
    start_time = time.time()
    u = solver.solve()
    end_time = time.time()
    
    print(f"Solve completed in {end_time - start_time:.4f} seconds.")
    
    print("Exporting results to output.vtu...")
    export_vtk(model, u, "output.vtu")
    print("Done!")

if __name__ == "__main__":
    main()
