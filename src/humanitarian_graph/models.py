from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    DEPOT = "depot"
    C = "C"
    H = "H"


class NodeSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(slots=True)
class ETruck:
    B: float = 600.0
    Qt: int = 4000
    ev: float = 1.0
    v: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Drone:
    w: int = 10
    q: int = 10
    tmax: int = 600
    v: float = 15.0
    Bv: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Node:
    node_id: str
    node_type: NodeType
    x_km: float
    y_km: float
    p: int
    demand: int
    truck_accessible: bool
    size: str = ""
    cluster_id: str = ""
    member_node_ids: list[str] = field(default_factory=list)
    p_m: int = 0
    P_o: int = 0
    R: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Edge:
    edge_id: str
    from_node: str
    to_node: str
    distance_km: float
    truck_traversable: bool
    drone_traversable: bool
    role: str = "road"
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScenarioGraph:
    name: str
    nodes: list[Node]
    edges: list[Edge]
    e_truck: ETruck = field(default_factory=ETruck)
    drone: Drone = field(default_factory=Drone)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "e_truck": self.e_truck.to_dict(),
            "drone": self.drone.to_dict(),
            "metadata": self.metadata,
        }

    def node_index(self) -> dict[str, Node]:
        return {node.node_id: node for node in self.nodes}


@dataclass(slots=True)
class ScenarioConfig:
    name: str = "simple_cluster_scenario"
    random_seed: int = 20260402
    grid_count: int = 3
    demand_sizes_by_cluster: tuple[int, ...] = (5, 5, 5)
    drone_only_per_cluster: int = 1
    center_max_distance_km: float = 4.5
    local_radius_range_km: tuple[float, float] = (1.5, 4.2)
    truck_cross_link_distance_limit_km: float = 18.0
    material_units_per_person: float = 1.0
    material_unit_kg: float = 5.0
    hub_population_fixed: int = 300
