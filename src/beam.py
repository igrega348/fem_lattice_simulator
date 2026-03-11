import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def linear_beam_energy(u, x1, x2, E, nu, A, Iy, Iz, J):
    """
    Computes the strain energy of a 3D linear elastic Euler-Bernoulli beam.
    
    Args:
        u: (12,) array of global displacements [u1_x, u1_y, u1_z, r1_x, r1_y, r1_z, u2_x, u2_y, u2_z, r2_x, r2_y, r2_z]
        x1, x2: (3,) arrays of initial global coordinates of node 1 and 2
        E, nu: Young's modulus and Poisson's ratio
        A, Iy, Iz, J: Cross-sectional area, moments of inertia, and polar moment of inertia
        
    Returns:
        Scalar value of strain energy
    """
    u1 = u[:6]
    u2 = u[6:]
    
    L = jnp.linalg.norm(x2 - x1)
    
    # Orientation triad
    v1 = (x2 - x1) / L
    
    # Choose a vector not parallel to v1 to create an orthogonal basis
    # If v1 is mostly along X, use Y axis. Otherwise use X axis.
    v_tmp = jnp.where(jnp.abs(v1[0]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    v3 = jnp.cross(v1, v_tmp)
    v3 = v3 / jnp.linalg.norm(v3)
    v2 = jnp.cross(v3, v1)
    
    # T is the transformation matrix from global to local coordinates
    T = jnp.vstack((v1, v2, v3))
    
    # Transform global displacements/rotations to local frame
    u1_loc_trans = T @ u1[:3]
    u1_loc_rot = T @ u1[3:]
    u2_loc_trans = T @ u2[:3]
    u2_loc_rot = T @ u2[3:]
    
    # Axial and torsional deformations
    du = u2_loc_trans[0] - u1_loc_trans[0]
    drx = u2_loc_rot[0] - u1_loc_rot[0]
    
    G = E / (2 * (1 + nu))
    
    # 1. Axial Energy
    U_axial = 0.5 * (E * A / L) * du**2
    
    # 2. Torsional Energy
    U_torsion = 0.5 * (G * J / L) * drx**2
    
    # 3. Bending Energy in xy plane (deflection v, rotation rz)
    d_bz = jnp.array([u1_loc_trans[1], u1_loc_rot[2], u2_loc_trans[1], u2_loc_rot[2]])
    K_bz = (E * Iz / L**3) * jnp.array([
        [12.0, 6*L, -12.0, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12.0, -6*L, 12.0, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])
    U_bend_z = 0.5 * jnp.dot(d_bz, jnp.dot(K_bz, d_bz))
    
    # 4. Bending Energy in xz plane (deflection w, rotation ry)
    # Note: rotation about y causes deflection w. 
    # Positive ry causes negative dw/dx, which slightly changes signs in the stiffness matrix
    d_by = jnp.array([u1_loc_trans[2], u1_loc_rot[1], u2_loc_trans[2], u2_loc_rot[1]])
    K_by = (E * Iy / L**3) * jnp.array([
        [12.0, -6*L, -12.0, -6*L],
        [-6*L, 4*L**2, 6*L, 2*L**2],
        [-12.0, 6*L, 12.0, 6*L],
        [-6*L, 2*L**2, 6*L, 4*L**2]
    ])
    U_bend_y = 0.5 * jnp.dot(d_by, jnp.dot(K_by, d_by))
    
    return U_axial + U_torsion + U_bend_z + U_bend_y


# JAX Auto-Differentiation
# The 1st derivative of strain energy w.r.t. displacement vector `u` gives the internal forces
get_internal_forces = jax.grad(linear_beam_energy, argnums=0)

# The 2nd derivative (Hessian) of strain energy w.r.t. displacement gives the tangent stiffness matrix
get_stiffness_matrix = jax.hessian(linear_beam_energy, argnums=0)

# Vectorized (batched) operations using jax.vmap
# We vectorize over all inputs (arrays of elements)
vmap_internal_forces = jax.vmap(get_internal_forces, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
vmap_stiffness_matrix = jax.vmap(get_stiffness_matrix, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))

# JIT compile the vectorized functions for maximum performance (C++ speeds)
jit_get_forces = jax.jit(vmap_internal_forces)
jit_get_stiffness = jax.jit(vmap_stiffness_matrix)
