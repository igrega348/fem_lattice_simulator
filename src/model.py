import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Material:
    id: int
    E: float
    nu: float
    model: str = "linear_elastic"

@dataclass
class Section:
    id: int
    A: float
    Iy: float
    Iz: float
    J: float

@dataclass
class Node:
    id: int
    coords: np.ndarray

@dataclass
class Element:
    id: int
    node_ids: List[int]
    material_id: int
    section_id: int

@dataclass
class BoundaryCondition:
    node_id: int
    dof: List[str]
    value: float

@dataclass
class PointLoad:
    node_id: int
    dof: str
    value: float

class FEAModel:
    def __init__(self):
        self.materials: Dict[int, Material] = {}
        self.sections: Dict[int, Section] = {}
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.boundary_conditions: List[BoundaryCondition] = []
        self.point_loads: List[PointLoad] = []

    @classmethod
    def from_json(cls, filepath: str) -> "FEAModel":
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        model = cls()
        
        for m_data in data.get("materials", []):
            model.materials[m_data["id"]] = Material(
                id=m_data["id"],
                E=float(m_data["E"]),
                nu=float(m_data["nu"]),
                model=m_data.get("model", "linear_elastic")
            )
            
        for s_data in data.get("sections", []):
            model.sections[s_data["id"]] = Section(
                id=s_data["id"],
                A=float(s_data["A"]),
                Iy=float(s_data["Iy"]),
                Iz=float(s_data["Iz"]),
                J=float(s_data["J"])
            )
            
        for n_data in data.get("nodes", []):
            model.nodes[n_data["id"]] = Node(
                id=n_data["id"],
                coords=np.array(n_data["coords"], dtype=float)
            )
            
        for e_data in data.get("elements", []):
            model.elements[e_data["id"]] = Element(
                id=e_data["id"],
                node_ids=e_data["nodes"],
                material_id=e_data["material"],
                section_id=e_data["section"]
            )
            
        for bc_data in data.get("boundary_conditions", []):
            dofs = bc_data["dof"]
            if isinstance(dofs, str):
                dofs = [dofs]
            model.boundary_conditions.append(BoundaryCondition(
                node_id=bc_data["node"],
                dof=dofs,
                value=float(bc_data["value"])
            ))
            
        for pl_data in data.get("point_loads", []):
            model.point_loads.append(PointLoad(
                node_id=pl_data["node"],
                dof=pl_data["dof"],
                value=float(pl_data["value"])
            ))
            
        return model
