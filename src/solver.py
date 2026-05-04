import numpy as np
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
        self._prepare_reference_positions()
        
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
        self.prescribed_final = self.prescribed_values.copy()
        self.prescribed_prev_target = np.zeros_like(self.prescribed_final)
        
        # Free DOFs
        self.free_dofs = np.setdiff1d(np.arange(self.num_dofs), self.constrained_dofs)
        
        # Prepare external force vector
        self.F_ext = np.zeros(self.num_dofs)
        for pl in self.model.point_loads:
            node_idx = self.assembler.node_id_to_idx[pl.node_id]
            local_dof = self.dof_map[pl.dof]
            global_dof = node_idx * 6 + local_dof
            self.F_ext[global_dof] += pl.value

    def _prepare_reference_positions(self):
        """
        Store reference nodal positions ordered by assembler.node_id_to_idx.
        """
        positions = np.zeros((self.assembler.num_nodes, 3), dtype=float)
        for node_id, idx in self.assembler.node_id_to_idx.items():
            positions[idx] = self.model.nodes[node_id].coords
        self.positions0 = positions.copy()
        self.positions_ref = positions.copy()
        self.assembler.set_reference_positions(self.positions_ref)

    def _solve_equilibrium(self, prescribed_values: np.ndarray, tol=1e-6, max_iter=20):
        """
        Solve one equilibrium problem on the current reference geometry.
        prescribed_values must align with self.constrained_dofs.
        Returns (u, converged, res_norm, iterations).
        """
        u = np.zeros(self.num_dofs)

        if len(self.constrained_dofs) > 0:
            u[self.constrained_dofs] = prescribed_values

        converged = False
        res_norm = np.inf
        iterations = 0
        for iteration in range(max_iter):
            iterations = iteration
            K_global, F_int = self.assembler.assemble(u)
            R = self.F_ext - F_int

            res_norm = np.linalg.norm(R[self.free_dofs])
            if res_norm < tol:
                converged = True
                break

            K_csc = K_global.tocsc()
            K_ff = K_csc[self.free_dofs, :][:, self.free_dofs]
            R_f = R[self.free_dofs]
            delta_u_f = spsolve(K_ff, R_f)
            u[self.free_dofs] += delta_u_f

        return u, converged, float(res_norm), int(iterations)
            
    def solve(self, tol=1e-6, max_iter=20):
        """
        Newton-Raphson Solver
        """
        u, converged, res_norm, iterations = self._solve_equilibrium(
            self.prescribed_values, tol=tol, max_iter=max_iter
        )
        if converged:
            print(f"Converged in {iterations} iterations. Residual norm: {res_norm:.2e}")
        else:
            print(
                f"Warning: Did not converge after {max_iter} iterations. Final residual norm: {res_norm:.2e}"
            )
        return u

    def solve_ramped(
        self,
        T_max: int,
        *,
        tol=1e-6,
        max_iter=20,
        output_steps: set[int] | None = None,
        output_prefix: str = "output",
        write_vtu: bool = True,
        write_json: bool = True,
        timestep_mode: str = "factor",
    ):
        """
        Ramped prescribed-displacement solve with incremental reference updates.

        - Prescribed BC values from the input JSON are treated as the final targets.
        - At step t: target = (t/T_max) * prescribed_final.
        - The step solves for the increment needed from the previous target.
        - After convergence, reference nodal coordinates are updated by the step
          translational displacement, and cumulative DOFs are accumulated for export.

        timestep_mode:
          - \"factor\": use t/T_max in the PVD (time in [0,1])
          - \"step\": use float(t) (integer step index)

        Returns u_cumulative.
        """
        from src.io import (
            compute_element_axial_strain_stress,
            export_deformed_model_json,
            export_vtk,
            write_pvd_timeseries,
        )

        if T_max <= 0:
            raise ValueError("T_max must be > 0")

        if output_steps is None:
            output_steps = set()

        u_cum = np.zeros(self.num_dofs)
        datasets: list[tuple[float, str]] = []

        self.positions_ref = self.positions0.copy()
        self.assembler.set_reference_positions(self.positions_ref)
        self.prescribed_prev_target = np.zeros_like(self.prescribed_final)

        for t in range(0, T_max + 1):
            factor = float(t) / float(T_max)
            prescribed_target = self.prescribed_final * factor
            prescribed_inc = prescribed_target - self.prescribed_prev_target

            u_step, converged, res_norm, iterations = self._solve_equilibrium(
                prescribed_inc, tol=tol, max_iter=max_iter
            )
            if converged:
                print(
                    f"[ramp t={t}/{T_max}] Converged in {iterations} iterations. Residual norm: {res_norm:.2e}"
                )
            else:
                print(
                    f"[ramp t={t}/{T_max}] Warning: Did not converge after {max_iter} iterations. Final residual norm: {res_norm:.2e}"
                )

            u_cum += u_step

            u_step_nodes = u_step.reshape((self.assembler.num_nodes, 6))
            self.positions_ref = self.positions_ref + u_step_nodes[:, :3]
            self.assembler.set_reference_positions(self.positions_ref)

            self.prescribed_prev_target = prescribed_target

            if t in output_steps:
                vtu_path = f"{output_prefix}_t{t:04d}.vtu"
                json_path = f"{output_prefix}_t{t:04d}.json"
                if write_vtu:
                    eps, sigma = compute_element_axial_strain_stress(self.model, u_cum)
                    export_vtk(
                        self.model,
                        u_cum,
                        vtu_path,
                        cell_data={
                            "AxialStrain": eps,
                            "AxialStress": sigma,
                            "VonMisesEqv": np.abs(sigma),
                        },
                    )
                if write_json:
                    export_deformed_model_json(self.model, u_cum, json_path)

                if timestep_mode == "factor":
                    ts = factor
                elif timestep_mode == "step":
                    ts = float(t)
                else:
                    raise ValueError(f"Unknown timestep_mode: {timestep_mode!r}")

                if write_vtu:
                    datasets.append((ts, vtu_path))

        if write_vtu and datasets:
            write_pvd_timeseries(f"{output_prefix}.pvd", datasets)

        return u_cum
