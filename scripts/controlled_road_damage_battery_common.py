from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BASE_INSTANCE = ROOT / "data" / "benchmark" / "suite_v2_n20_i2_t2.json"
DATA_DIR = ROOT / "data" / "controlled_road_damage_battery_n20"
RESULT_DIR = ROOT / "results" / "controlled_road_damage_battery_n20_exact"
FIGURE_DIR = RESULT_DIR / "figures"

DAMAGE_LEVELS = (0, 4, 8, 12, 16)
DAMAGE_SEEDS = (20260826, 20260827, 20260828, 20260829, 20260830)
SOLVER_SEED = 20260826
TIME_LIMIT_SEC = 1800.0
TARGET_MIP_GAP = 0.001
ENERGY_TOLERANCE = 1e-5
VALIDATION_TOLERANCE = 1e-6

EXPECTED_TOTAL_NODES = 20
EXPECTED_H_COUNT = 3
EXPECTED_CT_COUNT = 12
EXPECTED_DRONE_ONLY_COUNT = 4
EXPECTED_TRUCK_COUNT = 2
EXPECTED_DRONES_PER_TRUCK = 3
EXPECTED_TRUCK_LINKS = 27
EXPECTED_BACKBONE_LINKS = 5
EXPECTED_ELIGIBLE_LINKS = 22


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_id(a: str, b: str) -> str:
    return "--".join(sorted((str(a), str(b))))


def group_undirected_edges(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in payload.get("edges", []):
        grouped[link_id(edge["from_node"], edge["to_node"])].append(edge)
    return dict(grouped)


def node_composition(payload: dict[str, Any]) -> dict[str, int]:
    nodes = payload.get("nodes", [])
    h_count = sum(str(node.get("node_type", "")).upper() == "H" for node in nodes)
    c_nodes = [node for node in nodes if str(node.get("node_type", "")).upper() == "C"]
    ct_count = sum(bool(node.get("truck_accessible", False)) for node in c_nodes)
    return {
        "total_nodes": len(nodes),
        "depot_count": sum(str(node.get("node_type", "")).lower() == "depot" for node in nodes),
        "h_count": h_count,
        "ct_count": ct_count,
        "drone_only_count": len(c_nodes) - ct_count,
        "truck_count": int(payload.get("metadata", {}).get("assumptions", {}).get("truck_count", -1)),
        "drones_per_truck": int(payload.get("metadata", {}).get("assumptions", {}).get("drones_per_truck", -1)),
    }


def edge_inventory(payload: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    backbone: list[str] = []
    eligible: list[str] = []
    rows: list[dict[str, Any]] = []
    for uid, edges in sorted(group_undirected_edges(payload).items()):
        roles = sorted({str(edge.get("role", "")) for edge in edges})
        truck_flags = [bool(edge.get("truck_traversable", False)) for edge in edges]
        drone_flags = [bool(edge.get("drone_traversable", False)) for edge in edges]
        distances = sorted({float(edge.get("distance_km", math.nan)) for edge in edges})
        factors = sorted(
            {
                float(edge.get("notes", {}).get("truck_time_factor", 1.0))
                for edge in edges
                if bool(edge.get("truck_traversable", False))
            }
        )
        is_symmetric = (
            len(edges) == 2
            and edges[0]["from_node"] == edges[1]["to_node"]
            and edges[0]["to_node"] == edges[1]["from_node"]
        )
        is_truck_link = all(truck_flags) and len(truck_flags) == 2
        role = roles[0] if len(roles) == 1 else "|".join(roles)
        if is_truck_link and role == "main":
            backbone.append(uid)
        if is_truck_link and role == "secondary_penalized":
            eligible.append(uid)
        rows.append(
            {
                "link_id": uid,
                "directed_edge_count": len(edges),
                "role": role,
                "truck_direction_count": sum(truck_flags),
                "drone_direction_count": sum(drone_flags),
                "distance_values": ";".join(f"{value:.12g}" for value in distances),
                "original_factor_values": ";".join(f"{value:.12g}" for value in factors),
                "reciprocal_pair": bool(is_symmetric),
                "protected_backbone": uid in backbone,
                "eligible_for_damage": uid in eligible,
            }
        )
    return backbone, eligible, rows


def canonical_invariant_hashes(payload: dict[str, Any]) -> dict[str, str]:
    edge_invariants = []
    for edge in sorted(payload.get("edges", []), key=lambda item: str(item.get("edge_id", ""))):
        edge_invariants.append(
            {
                "edge_id": edge.get("edge_id"),
                "from_node": edge.get("from_node"),
                "to_node": edge.get("to_node"),
                "distance_km": edge.get("distance_km"),
                "drone_traversable": edge.get("drone_traversable"),
                "role": edge.get("role"),
            }
        )
    return {
        "nodes_hash": stable_hash(payload.get("nodes", [])),
        "truck_hash": stable_hash(payload.get("e_truck", {})),
        "drone_hash": stable_hash(payload.get("drone", {})),
        "edge_invariants_hash": stable_hash(edge_invariants),
    }


def canonicalize_undamaged(payload: dict[str, Any], eligible: Iterable[str]) -> dict[str, Any]:
    canonical = json.loads(json.dumps(payload))
    eligible_set = set(eligible)
    for edge in canonical.get("edges", []):
        if link_id(edge["from_node"], edge["to_node"]) in eligible_set:
            edge["truck_traversable"] = True
            edge.setdefault("notes", {})["truck_time_factor"] = 1.0
    canonical.setdefault("metadata", {}).setdefault("controlled_damage_experiment", {}).update(
        {
            "source_instance": str(BASE_INSTANCE),
            "canonical_undamaged": True,
            "damage_definition": "number of closed eligible undirected secondary truck-road links",
        }
    )
    return canonical


def apply_damage(
    canonical: dict[str, Any],
    *,
    scenario_name: str,
    q: int,
    realization: str,
    closed_links: Iterable[str],
) -> dict[str, Any]:
    damaged = json.loads(json.dumps(canonical))
    closed_set = set(closed_links)
    damaged["name"] = scenario_name
    for edge in damaged.get("edges", []):
        uid = link_id(edge["from_node"], edge["to_node"])
        if uid in closed_set:
            edge["truck_traversable"] = False
            edge.setdefault("notes", {})["truck_time_factor"] = 1.0
            edge["notes"]["controlled_damage_closed"] = True
        else:
            edge.setdefault("notes", {}).pop("controlled_damage_closed", None)
    damaged.setdefault("metadata", {}).setdefault("controlled_damage_experiment", {}).update(
        {
            "damage_q": int(q),
            "damage_realization": str(realization),
            "closed_link_ids": sorted(closed_set),
            "closed_link_count": len(closed_set),
        }
    )
    return damaged


def reachable_nodes(payload: dict[str, Any]) -> set[str]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in payload.get("edges", []):
        if bool(edge.get("truck_traversable", False)):
            adjacency[str(edge["from_node"])].add(str(edge["to_node"]))
    seen = {"D0"}
    queue = deque(["D0"])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency.get(node, set())):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def validate_graph(
    payload: dict[str, Any],
    *,
    expected_q: int,
    expected_closed: Iterable[str],
    invariant_hashes: dict[str, str],
    backbone: Iterable[str],
    eligible: Iterable[str],
) -> dict[str, Any]:
    grouped = group_undirected_edges(payload)
    expected_closed_set = set(expected_closed)
    backbone_set = set(backbone)
    eligible_set = set(eligible)
    actual_closed = {
        uid
        for uid in eligible_set
        if uid in grouped and sum(bool(edge.get("truck_traversable", False)) for edge in grouped[uid]) == 0
    }
    asymmetric_closed = [
        uid
        for uid in eligible_set
        if uid in grouped
        and sum(bool(edge.get("truck_traversable", False)) for edge in grouped[uid]) not in {0, 2}
    ]
    backbone_open = all(
        uid in grouped and sum(bool(edge.get("truck_traversable", False)) for edge in grouped[uid]) == 2
        for uid in backbone_set
    )
    factors_one = all(
        math.isclose(float(edge.get("notes", {}).get("truck_time_factor", 1.0)), 1.0, abs_tol=1e-12)
        for uid in eligible_set
        for edge in grouped.get(uid, [])
    )
    reachable = reachable_nodes(payload)
    h_nodes = sorted(
        str(node["node_id"])
        for node in payload.get("nodes", [])
        if str(node.get("node_type", "")).upper() == "H"
    )
    composition = node_composition(payload)
    current_hashes = canonical_invariant_hashes(payload)
    checks = {
        "composition_valid": composition
        == {
            "total_nodes": EXPECTED_TOTAL_NODES,
            "depot_count": 1,
            "h_count": EXPECTED_H_COUNT,
            "ct_count": EXPECTED_CT_COUNT,
            "drone_only_count": EXPECTED_DRONE_ONLY_COUNT,
            "truck_count": EXPECTED_TRUCK_COUNT,
            "drones_per_truck": EXPECTED_DRONES_PER_TRUCK,
        },
        "closed_count_valid": len(actual_closed) == int(expected_q),
        "closed_set_valid": actual_closed == expected_closed_set,
        "closed_symmetry_valid": not asymmetric_closed,
        "backbone_open": backbone_open,
        "depot_has_truck_connection": len(reachable) > 1,
        "all_h_reachable": set(h_nodes).issubset(reachable),
        "eligible_factors_restored": factors_one,
        "nodes_unchanged": current_hashes["nodes_hash"] == invariant_hashes["nodes_hash"],
        "truck_parameters_unchanged": current_hashes["truck_hash"] == invariant_hashes["truck_hash"],
        "drone_parameters_unchanged": current_hashes["drone_hash"] == invariant_hashes["drone_hash"],
        "edge_invariants_unchanged": current_hashes["edge_invariants_hash"] == invariant_hashes["edge_invariants_hash"],
    }
    return {
        **composition,
        "expected_q": int(expected_q),
        "actual_closed_link_count": len(actual_closed),
        "actual_closed_link_ids": ";".join(sorted(actual_closed)),
        "asymmetric_closed_link_ids": ";".join(sorted(asymmetric_closed)),
        "reachable_truck_node_count": len(reachable),
        "reachable_h_count": sum(node in reachable for node in h_nodes),
        **checks,
        "structural_validation_passed": all(checks.values()),
    }


def require_base_instance(payload: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    composition = node_composition(payload)
    expected = {
        "total_nodes": EXPECTED_TOTAL_NODES,
        "depot_count": 1,
        "h_count": EXPECTED_H_COUNT,
        "ct_count": EXPECTED_CT_COUNT,
        "drone_only_count": EXPECTED_DRONE_ONLY_COUNT,
        "truck_count": EXPECTED_TRUCK_COUNT,
        "drones_per_truck": EXPECTED_DRONES_PER_TRUCK,
    }
    if composition != expected:
        raise RuntimeError(f"Base-instance composition mismatch: expected {expected}, observed {composition}")
    backbone, eligible, rows = edge_inventory(payload)
    truck_links = len(backbone) + len(eligible)
    if (
        truck_links != EXPECTED_TRUCK_LINKS
        or len(backbone) != EXPECTED_BACKBONE_LINKS
        or len(eligible) != EXPECTED_ELIGIBLE_LINKS
    ):
        raise RuntimeError(
            "Base graph inventory mismatch: "
            f"expected truck/backbone/eligible={EXPECTED_TRUCK_LINKS}/{EXPECTED_BACKBONE_LINKS}/{EXPECTED_ELIGIBLE_LINKS}, "
            f"observed {truck_links}/{len(backbone)}/{len(eligible)}"
        )
    bad_pairs = [row["link_id"] for row in rows if not row["reciprocal_pair"]]
    if bad_pairs:
        raise RuntimeError(f"Non-reciprocal edge pairs found: {bad_pairs}")
    return backbone, eligible, rows
