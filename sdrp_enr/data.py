from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanitarian_graph import load_scenario_json  # noqa: E402


ALPHA = 1.0
BETA = 1.0
RHO = 0.1
T_MAX_HOURS = 24.0
SUCCESSOR_GAP_K = 3


@dataclass(frozen=True)
class ProblemData:
    scenario: str
    path: Path
    graph: object
    nodes: dict[str, object]
    depot: str
    h_nodes: list[str]
    c_nodes: list[str]
    c_truck: list[str]
    truck_nodes: list[str]
    launch_nodes: list[str]
    truck_ids: list[int]
    drones_by_truck: dict[int, list[int]]
    truck_arcs: set[tuple[str, str]]
    truck_time: dict[tuple[str, str], float]
    truck_energy: dict[tuple[str, str], float]
    truck_capacity: float
    truck_battery: float
    drone_payload: float
    drone_tmax_hours: float
    drone_battery: float
    drone_speed_kmh: float
    truck_speed_kmh: float
    h_energy_demand: dict[str, float]

    def xy(self, node_id: str) -> tuple[float, float]:
        node = self.nodes[node_id]
        return float(node.x_km), float(node.y_km)

    def euclidean_km(self, a: str, b: str) -> float:
        ax, ay = self.xy(a)
        bx, by = self.xy(b)
        return math.hypot(ax - bx, ay - by)

    def drone_time(self, a: str, i: str, b: str) -> float:
        return (self.euclidean_km(a, i) + self.euclidean_km(i, b)) / (self.drone_speed_kmh * 3.6)

    def drone_energy(self, fly_time_hours: float) -> float:
        if self.drone_tmax_hours <= 0:
            return float("inf")
        return (fly_time_hours / self.drone_tmax_hours) * self.drone_battery

    def material_demand(self, node_id: str) -> float:
        return float(self.nodes[node_id].demand)

    def population(self, node_id: str) -> float:
        return float(self.nodes[node_id].p)


def load_data(path: str | Path) -> ProblemData:
    scenario_path = Path(path)
    graph = load_scenario_json(scenario_path)
    nodes = graph.node_index()
    depot = "D0"
    h_nodes = [n.node_id for n in graph.nodes if n.node_type.value == "H"]
    c_nodes = [n.node_id for n in graph.nodes if n.node_type.value == "C"]
    c_truck = [n for n in c_nodes if nodes[n].truck_accessible]
    truck_nodes = [depot] + h_nodes + c_truck
    launch_nodes = h_nodes + c_truck

    assumptions = graph.metadata.get("assumptions", {}) if isinstance(graph.metadata, dict) else {}
    truck_count = int(assumptions.get("truck_count", 1))
    drones_per_truck = int(assumptions.get("drones_per_truck", 2))
    truck_ids = list(range(truck_count))
    drones_by_truck = {v: list(range(drones_per_truck)) for v in truck_ids}

    truck_arcs: set[tuple[str, str]] = set()
    truck_time: dict[tuple[str, str], float] = {}
    truck_energy: dict[tuple[str, str], float] = {}
    for edge in graph.edges:
        if not edge.truck_traversable:
            continue
        if edge.from_node not in truck_nodes or edge.to_node not in truck_nodes or edge.from_node == edge.to_node:
            continue
        factor = float(edge.notes.get("truck_time_factor", 1.0)) if isinstance(edge.notes, dict) else 1.0
        arc = (edge.from_node, edge.to_node)
        truck_arcs.add(arc)
        truck_time[arc] = (float(edge.distance_km) / graph.e_truck.v) * factor
        truck_energy[arc] = float(edge.distance_km) * graph.e_truck.ev * factor

    drone_body_weight = graph.drone.w * drones_per_truck
    h_energy_demand = {
        h: float(nodes[h].p_m) * RHO * float(nodes[h].R)
        for h in h_nodes
    }
    return ProblemData(
        scenario=graph.name,
        path=scenario_path,
        graph=graph,
        nodes=nodes,
        depot=depot,
        h_nodes=h_nodes,
        c_nodes=c_nodes,
        c_truck=c_truck,
        truck_nodes=truck_nodes,
        launch_nodes=launch_nodes,
        truck_ids=truck_ids,
        drones_by_truck=drones_by_truck,
        truck_arcs=truck_arcs,
        truck_time=truck_time,
        truck_energy=truck_energy,
        truck_capacity=float(graph.e_truck.Qt - drone_body_weight),
        truck_battery=float(graph.e_truck.B),
        drone_payload=float(graph.drone.q),
        drone_tmax_hours=float(graph.drone.tmax) / 3600.0,
        drone_battery=float(graph.drone.Bv),
        drone_speed_kmh=float(graph.drone.v),
        truck_speed_kmh=float(graph.e_truck.v),
        h_energy_demand=h_energy_demand,
    )


def microgrid_utility(c: float) -> float:
    c = max(0.0, min(1.0, c))
    if c <= 0.5:
        return 1.25 * c
    return 0.625 + 0.75 * (c - 0.5)
