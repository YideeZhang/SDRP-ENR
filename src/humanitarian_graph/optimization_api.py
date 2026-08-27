from __future__ import annotations

import networkx as nx

from .models import ScenarioGraph


def build_node_table(graph: ScenarioGraph) -> list[dict[str, object]]:
    return [node.to_dict() for node in graph.nodes]


def build_edge_table(graph: ScenarioGraph) -> list[dict[str, object]]:
    return [edge.to_dict() for edge in graph.edges]


def node_sets(graph: ScenarioGraph) -> dict[str, list[str]]:
    depot = [node.node_id for node in graph.nodes if node.node_type.value == "depot"]
    hubs = [node.node_id for node in graph.nodes if node.node_type.value == "H"]
    demands = [node.node_id for node in graph.nodes if node.node_type.value == "C"]
    return {"D": depot, "H": hubs, "C": demands, "N": depot + hubs + demands}


def feasible_arcs(graph: ScenarioGraph, mode: str = "truck") -> list[tuple[str, str]]:
    if mode not in {"truck", "drone"}:
        raise ValueError("mode must be 'truck' or 'drone'")
    return [
        (edge.from_node, edge.to_node)
        for edge in graph.edges
        if (mode == "truck" and edge.truck_traversable) or (mode == "drone" and edge.drone_traversable)
    ]


def build_distance_matrix(graph: ScenarioGraph, mode: str = "truck") -> dict[tuple[str, str], float]:
    if mode not in {"truck", "drone"}:
        raise ValueError("mode must be 'truck' or 'drone'")
    return {
        (edge.from_node, edge.to_node): edge.distance_km
        for edge in graph.edges
        if (mode == "truck" and edge.truck_traversable) or (mode == "drone" and edge.drone_traversable)
    }


def build_population(graph: ScenarioGraph) -> dict[str, int]:
    return {node.node_id: node.p for node in graph.nodes if node.node_type.value in {"C", "H"}}


def build_material_demand(graph: ScenarioGraph) -> dict[str, int]:
    return {node.node_id: node.demand for node in graph.nodes if node.node_type.value == "C"}


def build_h_parameters(graph: ScenarioGraph) -> dict[str, dict[str, object]]:
    return {
        node.node_id: {
            "member_node_ids": node.member_node_ids,
            "p_m": node.p_m,
            "P_o": node.P_o,
            "R": node.R,
            "self_demand": node.demand,
        }
        for node in graph.nodes
        if node.node_type.value == "H"
    }


def build_truck_nx(graph: ScenarioGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in graph.nodes:
        g.add_node(node.node_id, **node.to_dict())
    for edge in graph.edges:
        if edge.truck_traversable:
            g.add_edge(edge.from_node, edge.to_node, **edge.to_dict())
    return g
