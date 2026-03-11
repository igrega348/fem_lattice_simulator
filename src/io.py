import numpy as np
import meshio
from src.model import FEAModel

def export_vtk(model: FEAModel, u: np.ndarray, filepath: str):
    """
    Exports the deformed lattice model to VTK/VTU format using meshio.
    """
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(model.nodes.keys())}
    
    # Extract points (original coordinates)
    points = np.zeros((len(model.nodes), 3))
    for node_id, node in model.nodes.items():
        idx = node_id_to_idx[node_id]
        points[idx] = node.coords
        
    # Extract lines (elements connecting nodes)
    lines = np.zeros((len(model.elements), 2), dtype=int)
    for i, (el_id, el) in enumerate(model.elements.items()):
        n1_id, n2_id = el.node_ids
        lines[i] = [node_id_to_idx[n1_id], node_id_to_idx[n2_id]]
        
    cells = [("line", lines)]
    
    # Extract node data (displacements)
    u_reshaped = u.reshape((len(model.nodes), 6))
    translations = u_reshaped[:, :3]
    rotations = u_reshaped[:, 3:]
    
    point_data = {
        "Displacement": translations,
        "Rotation": rotations
    }
    
    # Write to VTK/VTU using meshio
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data
    )
    
    # Save the mesh
    mesh.write(filepath)
