from __future__ import annotations

import math
import random
from dataclasses import asdict

from .models import Drone, ETruck, Edge, Node, NodeSize, NodeType, ScenarioConfig, ScenarioGraph


C_SIZE_TO_VALUE = {
    NodeSize.LARGE: 60,
    NodeSize.MEDIUM: 30,
    NodeSize.SMALL: 20,
}


def _material_demand(population: int, material_units_per_person: float) -> int:
    return int(round(population * material_units_per_person))


def build_humanitarian_graph(config: ScenarioConfig) -> ScenarioGraph:
    rng = random.Random(config.random_seed)
    nodes, cluster_map = _build_nodes(config, rng)
    edges = _build_edges(config, nodes, cluster_map)
    _populate_h_metadata(nodes, cluster_map)

    metadata = {
        "generator": "humanitarian_graph.build_humanitarian_graph",
        "config": asdict(config),
        "requirements": {
            "all_cluster_points_to_center_distance_leq_km": config.center_max_distance_km,
            "material_units_per_person": config.material_units_per_person,
            "material_unit_kg": config.material_unit_kg,
            "hub_population_fixed": config.hub_population_fixed,
            "C_node_demand_equals_mu_times_population": True,
            "H_node_power_equals_cluster_population": True,
        },
    }
    return ScenarioGraph(
        name=config.name,
        nodes=nodes,
        edges=edges,
        e_truck=ETruck(),
        drone=Drone(),
        metadata=metadata,
    )


def _build_nodes(
    config: ScenarioConfig,
    rng: random.Random,
) -> tuple[list[Node], dict[str, list[str]]]:
    restoration_times = [10.0, 12.0, 14.0]
    nodes: list[Node] = [
        Node(
            node_id="D0",
            node_type=NodeType.DEPOT,
            x_km=20.0,
            y_km=4.0,
            p=0,
            demand=0,
            truck_accessible=True,
            notes={"role": "base"},
        )
    ]

    hub_positions = [(10.0, 12.0), (20.0, 15.5), (30.0, 12.0)]
    cluster_map: dict[str, list[str]] = {}
    node_counter = 0

    for cluster_idx in range(config.grid_count):
        hub_id = f"H{cluster_idx}"
        hx, hy = hub_positions[cluster_idx]
        hub_size = NodeSize.MEDIUM
        hub_value = config.hub_population_fixed
        hub = Node(
            node_id=hub_id,
            node_type=NodeType.H,
            x_km=hx,
            y_km=hy,
            p=hub_value,
            demand=_material_demand(hub_value, config.material_units_per_person),
            truck_accessible=True,
            size=hub_size.value,
            cluster_id=hub_id,
            R=restoration_times[cluster_idx],
            notes={"role": "grid_center"},
        )
        nodes.append(hub)

        demand_ids: list[str] = []
        count = config.demand_sizes_by_cluster[cluster_idx]
        drone_only_slots = set(rng.sample(range(count), min(config.drone_only_per_cluster, count)))
        base_angles = [2 * math.pi * idx / count for idx in range(count)]
        rng.shuffle(base_angles)

        for local_idx in range(count):
            angle = base_angles[local_idx]
            radius = rng.uniform(*config.local_radius_range_km)
            x = hx + radius * math.cos(angle)
            y = hy + radius * math.sin(angle)
            size = rng.choice([NodeSize.SMALL, NodeSize.MEDIUM, NodeSize.LARGE])
            value = C_SIZE_TO_VALUE[size]
            truck_accessible = local_idx not in drone_only_slots
            node_id = f"C{node_counter}"
            node_counter += 1

            node = Node(
                node_id=node_id,
                node_type=NodeType.C,
                x_km=round(x, 3),
                y_km=round(y, 3),
                p=value,
                demand=_material_demand(value, config.material_units_per_person),
                truck_accessible=truck_accessible,
                size=size.value,
                cluster_id=hub_id,
                notes={"distance_to_center_km": round(radius, 3)},
            )
            nodes.append(node)
            demand_ids.append(node_id)

        cluster_map[hub_id] = demand_ids

    return nodes, cluster_map


def _build_edges(
    config: ScenarioConfig,
    nodes: list[Node],
    cluster_map: dict[str, list[str]],
) -> list[Edge]:
    node_by_id = {node.node_id: node for node in nodes}
    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()

    for hub_id in cluster_map:
        _add_pair(edges, seen_pairs, node_by_id["D0"], node_by_id[hub_id], truck=True, drone=True, role="main")

    ordered_hubs = list(cluster_map.keys())
    for left, right in zip(ordered_hubs, ordered_hubs[1:]):
        _add_pair(edges, seen_pairs, node_by_id[left], node_by_id[right], truck=True, drone=True, role="main")

    for hub_id, demand_ids in cluster_map.items():
        hub = node_by_id[hub_id]
        truck_nodes = [node_by_id[nid] for nid in demand_ids if node_by_id[nid].truck_accessible]
        drone_only_nodes = [node_by_id[nid] for nid in demand_ids if not node_by_id[nid].truck_accessible]

        for node in truck_nodes:
            _add_pair(edges, seen_pairs, hub, node, truck=True, drone=True, role="cluster")

        if len(truck_nodes) >= 2:
            ordered = sorted(truck_nodes, key=lambda n: math.atan2(n.y_km - hub.y_km, n.x_km - hub.x_km))
            for idx, node in enumerate(ordered):
                nxt = ordered[(idx + 1) % len(ordered)]
                _add_pair(edges, seen_pairs, node, nxt, truck=True, drone=True, role="cluster")

        for node in drone_only_nodes:
            _add_pair(edges, seen_pairs, hub, node, truck=False, drone=True, role="drone_only")
            if truck_nodes:
                nearest = min(truck_nodes, key=lambda other: _distance(node, other))
                _add_pair(edges, seen_pairs, nearest, node, truck=False, drone=True, role="drone_only")

    # Cross-cluster C-C and C-H truck links for material transport.
    for left_hub, right_hub in zip(ordered_hubs, ordered_hubs[1:]):
        left_nodes = [node_by_id[nid] for nid in cluster_map[left_hub] if node_by_id[nid].truck_accessible]
        right_nodes = [node_by_id[nid] for nid in cluster_map[right_hub] if node_by_id[nid].truck_accessible]
        if left_nodes and right_nodes:
            left_anchor = min(left_nodes, key=lambda n: _distance(n, node_by_id[right_hub]))
            right_anchor = min(right_nodes, key=lambda n: _distance(n, node_by_id[left_hub]))
            _add_pair(edges, seen_pairs, left_anchor, right_anchor, truck=True, drone=True, role="cross_cluster")
            _add_pair(edges, seen_pairs, left_anchor, node_by_id[right_hub], truck=True, drone=True, role="cross_cluster")
            _add_pair(edges, seen_pairs, node_by_id[left_hub], right_anchor, truck=True, drone=True, role="cross_cluster")

    return edges


def _populate_h_metadata(nodes: list[Node], cluster_map: dict[str, list[str]]) -> None:
    node_by_id = {node.node_id: node for node in nodes}
    for hub_id, member_ids in cluster_map.items():
        hub = node_by_id[hub_id]
        member_population = sum(node_by_id[node_id].p for node_id in member_ids)
        hub.member_node_ids = member_ids
        hub.p_m = member_population + hub.p
        hub.P_o = hub.p_m // 5


def _add_pair(
    edges: list[Edge],
    seen_pairs: set[tuple[str, str]],
    source: Node,
    target: Node,
    *,
    truck: bool,
    drone: bool,
    role: str,
) -> None:
    pair = tuple(sorted((source.node_id, target.node_id)))
    if pair in seen_pairs:
        return
    seen_pairs.add(pair)
    distance = round(_distance(source, target), 3)
    common = {
        "distance_km": distance,
        "truck_traversable": truck,
        "drone_traversable": drone,
        "role": role,
        "notes": {},
    }
    edges.append(Edge(edge_id=f"E_{source.node_id}_{target.node_id}", from_node=source.node_id, to_node=target.node_id, **common))
    edges.append(Edge(edge_id=f"E_{target.node_id}_{source.node_id}", from_node=target.node_id, to_node=source.node_id, **common))


def _distance(node_a: Node, node_b: Node) -> float:
    return math.hypot(node_a.x_km - node_b.x_km, node_a.y_km - node_b.y_km)
