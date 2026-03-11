import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from src.model import FEAModel
from src.assembly import Assembler

class Solver:
    def __init__(self, model: FEAModel):
        self.model = model
        self.assembler = Assembler(model)
        self.num_dofs = self.assembler.num_dofs
        
        self.dof_map = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5,
                        "fx": 0, "fy": 1, "fz": 2, "mx": 3, "my": 4, "mz": 5}
                        
        self._prepare_bcs()
        
    def _prepare_bcs(self):
        # Identify constrained DOFs and prescribed values
        self.constrained_dofs = []
        self.prescribed_values = []
        
        for bc in self.model.boundary_conditions:
            node_idx = self.assembler.node_id_to_idx[bc.node_id]
            for dof_str in bc.dof:
                local_dof = self.dof_map[dof_str]
                global_dof = node_idx * 6 + local_dof
                self.constrained_dofs.append(global_dof)
                self.prescribed_values.append(bc.value)
                
        self.constrained_dofs = np.array(self.constrained_dofs, dtype=int)
        self.prescribed_values = np.array(self.prescribed_values, dtype=float)
        
        # Free DOFs
        self.free_dofs = np.setdiff1d(np.arange(self.num_dofs), self.constrained_dofs)
        
        # Prepare external force vector
        self.F_ext = np.zeros(self.num_dofs)
        for pl in self.model.point_loads:
            node_idx = self.assembler.node_id_to_idx[pl.node_id]
            local_dof = self.dof_map[pl.dof]
            global_dof = node_idx * 6 + local_dof
            self.F_ext[global_dof] += pl.value
            
    def solve(self, tol=1e-6, max_iter=20):
        """
        Newton-Raphson Solver
        """
        u = np.zeros(self.num_dofs)
        
        # Apply non-zero prescribed displacements as initial guess
        if len(self.constrained_dofs) > 0:
            u[self.constrained_dofs] = self.prescribed_values
            
        for iteration in range(max_iter):
            # Assemble K and F_int based on current displacement u
            K_global, F_int = self.assembler.assemble(u)
            
            # Calculate residual R = F_ext - F_int
            R = self.F_ext - F_int
            
            # Check convergence (only on free DOFs)
            res_norm = np.linalg.norm(R[self.free_dofs])
            if res_norm < tol:
                print(f"Converged in {iteration} iterations. Residual norm: {res_norm:.2e}")
                break
                
            # Convert K to CSC format for slicing and solving
            K_csc = K_global.tocsc()
            
            # Partition matrices for reduced system solve
            # K_ff * delta_u_f = R_f
            K_ff = K_csc[self.free_dofs, :][:, self.free_dofs]
            R_f = R[self.free_dofs]
            
            # Solve for displacement increments
            delta_u_f = spsolve(K_ff, R_f)
            
            # Update displacements
            u[self.free_dofs] += delta_u_f
            
        else:
            print(f"Warning: Did not converge after {max_iter} iterations. Final residual norm: {res_norm:.2e}")
            
        return u
