from __future__ import annotations

import json
from pathlib import Path

from .models import Drone, ETruck, Edge, Node, NodeType, ScenarioGraph


def save_scenario_json(graph: ScenarioGraph, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")


def load_scenario_json(path: str | Path) -> ScenarioGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [
        Node(
            node_id=item["node_id"],
            node_type=NodeType(item["node_type"]),
            x_km=item["x_km"],
            y_km=item["y_km"],
            p=item["p"],
            demand=item["demand"],
            truck_accessible=item["truck_accessible"],
            size=item.get("size", ""),
            cluster_id=item.get("cluster_id", ""),
            member_node_ids=item.get("member_node_ids", []),
            p_m=item.get("p_m", 0),
            P_o=item.get("P_o", item.get("P_h", 0)),
            R=item.get("R", 0.0),
            notes=item.get("notes", {}),
        )
        for item in data["nodes"]
    ]
    edges = [
        Edge(
            edge_id=item["edge_id"],
            from_node=item["from_node"],
            to_node=item["to_node"],
            distance_km=item["distance_km"],
            truck_traversable=item["truck_traversable"],
            drone_traversable=item["drone_traversable"],
            role=item.get("role", "road"),
            notes=item.get("notes", {}),
        )
        for item in data["edges"]
    ]
    return ScenarioGraph(
        name=data["name"],
        nodes=nodes,
        edges=edges,
        e_truck=ETruck(**data.get("e_truck", {})),
        drone=Drone(**data.get("drone", {})),
        metadata=data.get("metadata", {}),
    )
