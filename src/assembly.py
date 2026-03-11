import numpy as np
import jax.numpy as jnp
from scipy.sparse import coo_matrix
from src.beam import jit_get_forces, jit_get_stiffness
from src.model import FEAModel

class Assembler:
    def __init__(self, model: FEAModel):
        self.model = model
        
        # Map node ID to a contiguous index 0...N-1
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(model.nodes.keys())}
        self.num_nodes = len(model.nodes)
        self.num_dofs = self.num_nodes * 6
        self.num_elements = len(model.elements)
        
        # Preallocate JAX inputs
        self.x1 = np.zeros((self.num_elements, 3))
        self.x2 = np.zeros((self.num_elements, 3))
        self.E = np.zeros(self.num_elements)
        self.nu = np.zeros(self.num_elements)
        self.A = np.zeros(self.num_elements)
        self.Iy = np.zeros(self.num_elements)
        self.Iz = np.zeros(self.num_elements)
        self.J = np.zeros(self.num_elements)
        
        # Precompute DOF maps for sparse assembly
        self.element_dofs = np.zeros((self.num_elements, 12), dtype=int)
        
        self._prepare_data()
        self._prepare_sparse_indices()
        
    def _prepare_data(self):
        for i, (el_id, el) in enumerate(self.model.elements.items()):
            n1_id, n2_id = el.node_ids
            n1 = self.model.nodes[n1_id]
            n2 = self.model.nodes[n2_id]
            
            self.x1[i] = n1.coords
            self.x2[i] = n2.coords
            
            mat = self.model.materials[el.material_id]
            self.E[i] = mat.E
            self.nu[i] = mat.nu
            
            sec = self.model.sections[el.section_id]
            self.A[i] = sec.A
            self.Iy[i] = sec.Iy
            self.Iz[i] = sec.Iz
            self.J[i] = sec.J
            
            n1_idx = self.node_id_to_idx[n1_id]
            n2_idx = self.node_id_to_idx[n2_id]
            
            # 6 DOFs per node
            self.element_dofs[i, :6] = np.arange(n1_idx * 6, n1_idx * 6 + 6)
            self.element_dofs[i, 6:] = np.arange(n2_idx * 6, n2_idx * 6 + 6)
            
        # Convert all prepared arrays to JAX arrays to avoid conversion cost during solve loop
        self.x1 = jnp.array(self.x1)
        self.x2 = jnp.array(self.x2)
        self.E = jnp.array(self.E)
        self.nu = jnp.array(self.nu)
        self.A = jnp.array(self.A)
        self.Iy = jnp.array(self.Iy)
        self.Iz = jnp.array(self.Iz)
        self.J = jnp.array(self.J)
        
    def _prepare_sparse_indices(self):
        # We need row and col indices for the 12x12 stiffness matrix of each element
        # element_dofs shape is (num_elements, 12)
        # We want to create pairs of (row, col) for all 144 entries per element
        rows = np.repeat(self.element_dofs[:, :, None], 12, axis=2)
        cols = np.repeat(self.element_dofs[:, None, :], 12, axis=1)
        
        self.row_indices = rows.flatten()
        self.col_indices = cols.flatten()
        
    def assemble(self, u_global: np.ndarray):
        """
        u_global: (num_dofs,) numpy array of current displacements
        Returns global stiffness matrix K (scipy.sparse.coo_matrix) and internal forces F_int (numpy array)
        """
        # 1. Extract element displacements from global displacement vector
        u_elements = jnp.array(u_global[self.element_dofs])
        
        # 2. Evaluate physics using JAX (JIT-compiled and vectorized)
        k_elements = jit_get_stiffness(u_elements, self.x1, self.x2, self.E, self.nu, self.A, self.Iy, self.Iz, self.J)
        f_elements = jit_get_forces(u_elements, self.x1, self.x2, self.E, self.nu, self.A, self.Iy, self.Iz, self.J)
        
        # Convert back to numpy (to interop with scipy sparse solvers)
        k_elements = np.array(k_elements)
        f_elements = np.array(f_elements)
        
        # 3. Assemble sparse global stiffness matrix
        K_global = coo_matrix((k_elements.flatten(), (self.row_indices, self.col_indices)), 
                              shape=(self.num_dofs, self.num_dofs))
                              
        # 4. Assemble global internal force vector
        F_int = np.zeros(self.num_dofs)
        np.add.at(F_int, self.element_dofs.flatten(), f_elements.flatten())
        
        return K_global, F_int
