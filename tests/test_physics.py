import pytest
import numpy as np
import jax

# Ensure JAX uses 64-bit precision for tests
jax.config.update("jax_enable_x64", True)

from src.model import FEAModel, Node, Element, Material, Section, BoundaryCondition, PointLoad
from src.solver import Solver

def create_cantilever(L, E, nu, A, Iy, Iz, J, load_dof, load_val):
    model = FEAModel()
    model.materials[1] = Material(id=1, E=E, nu=nu)
    model.sections[1] = Section(id=1, A=A, Iy=Iy, Iz=Iz, J=J)
    model.nodes[1] = Node(id=1, coords=np.array([0.0, 0.0, 0.0]))
    model.nodes[2] = Node(id=2, coords=np.array([L, 0.0, 0.0]))
    model.elements[1] = Element(id=1, node_ids=[1, 2], material_id=1, section_id=1)
    
    # Fix node 1
    model.boundary_conditions.append(
        BoundaryCondition(node_id=1, dof=["ux", "uy", "uz", "rx", "ry", "rz"], value=0.0)
    )
    # Load node 2
    model.point_loads.append(PointLoad(node_id=2, dof=load_dof, value=load_val))
    
    return model

@pytest.fixture
def beam_props():
    return {
        "L": 2.0,
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iy": 1e-5,
        "Iz": 2e-5,
        "J": 3e-5
    }

def test_axial_tension(beam_props):
    P = 1000.0  # N
    model = create_cantilever(**beam_props, load_dof="fx", load_val=P)
    solver = Solver(model)
    u = solver.solve()
    
    # Analytical solution: delta = P * L / (E * A)
    expected_delta = (P * beam_props["L"]) / (beam_props["E"] * beam_props["A"])
    
    # Node 2 UX is at index 6
    actual_delta = u[6]
    
    np.testing.assert_allclose(actual_delta, expected_delta, rtol=1e-5)

def test_torsion(beam_props):
    T = 500.0  # Nm
    model = create_cantilever(**beam_props, load_dof="mx", load_val=T)
    solver = Solver(model)
    u = solver.solve()
    
    # Analytical solution: theta = T * L / (G * J)
    G = beam_props["E"] / (2 * (1 + beam_props["nu"]))
    expected_theta = (T * beam_props["L"]) / (G * beam_props["J"])
    
    # Node 2 RX is at index 9
    actual_theta = u[9]
    
    np.testing.assert_allclose(actual_theta, expected_theta, rtol=1e-5)

def test_bending_y(beam_props):
    # Bending in XY plane (force in Y direction)
    # Deflection v corresponds to bending about Z axis (Iz)
    P = 100.0  # N
    model = create_cantilever(**beam_props, load_dof="fy", load_val=P)
    solver = Solver(model)
    u = solver.solve()
    
    # Analytical solution for tip deflection: delta = P * L^3 / (3 * E * I)
    expected_delta = (P * beam_props["L"]**3) / (3 * beam_props["E"] * beam_props["Iz"])
    
    # Analytical solution for tip rotation: theta = P * L^2 / (2 * E * I)
    expected_theta = (P * beam_props["L"]**2) / (2 * beam_props["E"] * beam_props["Iz"])
    
    # Node 2 UY is at index 7, RZ is at index 11
    actual_delta = u[7]
    actual_theta = u[11]
    
    np.testing.assert_allclose(actual_delta, expected_delta, rtol=1e-5)
    np.testing.assert_allclose(actual_theta, expected_theta, rtol=1e-5)

def test_bending_z(beam_props):
    # Bending in XZ plane (force in Z direction)
    # Deflection w corresponds to bending about Y axis (Iy)
    P = 100.0  # N
    model = create_cantilever(**beam_props, load_dof="fz", load_val=P)
    solver = Solver(model)
    u = solver.solve()
    
    # Analytical solution for tip deflection: delta = P * L^3 / (3 * E * I)
    expected_delta = (P * beam_props["L"]**3) / (3 * beam_props["E"] * beam_props["Iy"])
    
    # Analytical solution for tip rotation: theta = - P * L^2 / (2 * E * I)
    # (Positive force in Z causes negative rotation about Y in right-handed system)
    expected_theta = -(P * beam_props["L"]**2) / (2 * beam_props["E"] * beam_props["Iy"])
    
    # Node 2 UZ is at index 8, RY is at index 10
    actual_delta = u[8]
    actual_theta = u[10]
    
    np.testing.assert_allclose(actual_delta, expected_delta, rtol=1e-5)
    np.testing.assert_allclose(actual_theta, expected_theta, rtol=1e-5)
