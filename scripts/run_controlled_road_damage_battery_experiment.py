from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from humanitarian_graph import load_scenario_json  # noqa: E402
from scripts.controlled_road_damage_battery_common import (  # noqa: E402
    DAMAGE_SEEDS,
    DATA_DIR,
    ENERGY_TOLERANCE,
    RESULT_DIR,
    SOLVER_SEED,
    TARGET_MIP_GAP,
    TIME_LIMIT_SEC,
    VALIDATION_TOLERANCE,
    file_hash,
    link_id,
    read_json,
    write_json,
)
from scripts.solve_gurobi_model import (  # noqa: E402
    RHO,
    TRUCK_BATTERY_RESERVE,
    save_solution,
    solve_model,
)


CSV_ENCODING = "utf-8-sig"


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def status_name(code: int | float | str) -> str:
    try:
        value = int(code)
    except (TypeError, ValueError):
        return str(code)
    return {
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        9: "TIME_LIMIT",
        13: "SUBOPTIMAL",
    }.get(value, str(value))


def route_nodes_by_truck(routes: pd.DataFrame) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    if routes.empty:
        return result
    for row in routes.itertuples(index=False):
        truck = int(row.truck)
        result[truck].update((str(row.from_), str(row.to))) if hasattr(row, "from_") else None
    if not result:
        for _, row in routes.iterrows():
            result[int(row["truck"])].update((str(row["from"]), str(row["to"])))
    return result


def reconstruct_energy(
    payload: dict[str, Any],
    solution: dict[str, Any],
    truck_count: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    nodes = {str(node["node_id"]): node for node in payload["nodes"]}
    edges = {(str(edge["from_node"]), str(edge["to_node"])): edge for edge in payload["edges"]}
    routes = solution["routes"]
    star = solution["star_delivery"]
    rendezvous = solution["rendezvous_delivery"]
    h_service = solution["h_service"]
    authoritative = solution["energy_breakdown"].set_index("truck") if not solution["energy_breakdown"].empty else pd.DataFrame()
    drone_speed_kmh = float(payload["drone"]["v"]) * 3.6
    drone_tmax_hours = float(payload["drone"]["tmax"]) / 3600.0
    drone_battery = float(payload["drone"]["Bv"])
    truck_ev = float(payload["e_truck"]["ev"])
    usable = float(payload["e_truck"]["B"]) - float(TRUCK_BATTERY_RESERVE)

    def air_distance(a: str, b: str) -> float:
        return math.hypot(
            float(nodes[a]["x_km"]) - float(nodes[b]["x_km"]),
            float(nodes[a]["y_km"]) - float(nodes[b]["y_km"]),
        )

    rows: list[dict[str, Any]] = []
    max_reconcile = 0.0
    max_share_residual = 0.0
    for truck in range(truck_count):
        selected_routes = routes.loc[pd.to_numeric(routes.get("truck"), errors="coerce") == truck] if not routes.empty else pd.DataFrame()
        drive = 0.0
        distance = 0.0
        travel_time = 0.0
        for route in selected_routes.to_dict("records"):
            edge = edges[(str(route["from"]), str(route["to"]))]
            factor = float(edge.get("notes", {}).get("truck_time_factor", 1.0))
            length = float(edge["distance_km"])
            distance += length
            drive += length * truck_ev * factor
            travel_time += length / float(payload["e_truck"]["v"]) * factor

        drone = 0.0
        selected_star = star.loc[pd.to_numeric(star.get("truck"), errors="coerce") == truck] if not star.empty else pd.DataFrame()
        for sortie in selected_star.to_dict("records"):
            fly_time = 2.0 * air_distance(str(sortie["launch"]), str(sortie["i"])) / drone_speed_kmh
            drone += fly_time / drone_tmax_hours * drone_battery * float(sortie["times"])
        selected_rv = rendezvous.loc[pd.to_numeric(rendezvous.get("truck"), errors="coerce") == truck] if not rendezvous.empty else pd.DataFrame()
        for sortie in selected_rv.to_dict("records"):
            fly_time = (
                air_distance(str(sortie["a"]), str(sortie["i"]))
                + air_distance(str(sortie["i"]), str(sortie["b"]))
            ) / drone_speed_kmh
            drone += fly_time / drone_tmax_hours * drone_battery

        grid = 0.0
        selected_h = h_service.loc[pd.to_numeric(h_service.get("truck"), errors="coerce") == truck] if not h_service.empty else pd.DataFrame()
        for service in selected_h.to_dict("records"):
            grid += float(nodes[str(service["h"])]["P_o"]) * float(service["tau"])
        total = drive + drone + grid
        unused = usable - total
        shares = {
            "phi_drive": drive / usable,
            "phi_drone": drone / usable,
            "phi_grid": grid / usable,
            "phi_unused": unused / usable,
        }
        share_residual = abs(sum(shares.values()) - 1.0)
        auth_drive = float(authoritative.loc[truck, "drive_energy_kwh"]) if not authoritative.empty else math.nan
        auth_drone = float(authoritative.loc[truck, "drone_energy_kwh"]) if not authoritative.empty else math.nan
        auth_grid = float(authoritative.loc[truck, "grid_energy_kwh"]) if not authoritative.empty else math.nan
        reconciliation = max(abs(drive - auth_drive), abs(drone - auth_drone), abs(grid - auth_grid))
        max_reconcile = max(max_reconcile, reconciliation)
        max_share_residual = max(max_share_residual, share_residual)
        rows.append(
            {
                "truck": truck,
                "usable_battery_kwh": usable,
                "E_drive_kwh": drive,
                "E_drone_kwh": drone,
                "E_star_kwh": sum(
                    (2.0 * air_distance(str(row["launch"]), str(row["i"])) / drone_speed_kmh)
                    / drone_tmax_hours
                    * drone_battery
                    * float(row["times"])
                    for row in selected_star.to_dict("records")
                ),
                "E_rendezvous_kwh": sum(
                    ((air_distance(str(row["a"]), str(row["i"])) + air_distance(str(row["i"]), str(row["b"]))) / drone_speed_kmh)
                    / drone_tmax_hours
                    * drone_battery
                    for row in selected_rv.to_dict("records")
                ),
                "E_grid_kwh": grid,
                "E_unused_kwh": unused,
                **shares,
                "share_sum": sum(shares.values()),
                "share_sum_residual": share_residual,
                "authoritative_E_drive_kwh": auth_drive,
                "authoritative_E_drone_kwh": auth_drone,
                "authoritative_E_grid_kwh": auth_grid,
                "max_energy_reconciliation_error_kwh": reconciliation,
                "route_distance_km": distance,
                "route_travel_time_h": travel_time,
            }
        )
    return pd.DataFrame(rows), {
        "max_energy_reconciliation_error_kwh": max_reconcile,
        "max_share_sum_residual": max_share_residual,
    }


def objective_and_service_metrics(payload: dict[str, Any], solution: dict[str, Any]) -> dict[str, float]:
    nodes = {str(node["node_id"]): node for node in payload["nodes"]}
    material = solution["material"]
    coverage = solution["coverage"]
    material_component = sum(
        float(nodes[str(row["node"])]["p"]) * float(row["coverage"])
        for row in material.to_dict("records")
    )
    microgrid_component = sum(
        float(nodes[str(row["h"])]["p_m"]) * float(row["g"])
        for row in coverage.to_dict("records")
    )
    material_denominator = sum(float(node.get("p", 0.0)) for node in payload["nodes"] if str(node.get("node_type", "")).lower() != "depot")
    microgrid_denominator = sum(float(node.get("p_m", 0.0)) for node in payload["nodes"] if str(node.get("node_type", "")).upper() == "H")
    full_service = material_denominator + microgrid_denominator
    return {
        "material_objective_component": material_component,
        "microgrid_objective_component": microgrid_component,
        "objective_component_sum": material_component + microgrid_component,
        "objective_reconciliation_error": abs(float(solution["objective"]) - material_component - microgrid_component),
        "normalized_objective": float(solution["objective"]) / full_service,
        "material_coverage_ratio": material_component / material_denominator,
        "microgrid_utility_ratio": microgrid_component / microgrid_denominator,
        "full_service_objective": full_service,
    }


def route_validation(payload: dict[str, Any], solution: dict[str, Any], truck_count: int) -> dict[str, int]:
    routes = solution["routes"]
    available = {
        (str(edge["from_node"]), str(edge["to_node"]))
        for edge in payload["edges"]
        if bool(edge.get("truck_traversable", False))
    }
    closed_arc_use = 0
    flow_violation = 0
    disconnected_subtour = 0
    repeated_node = 0
    for truck in range(truck_count):
        selected = routes.loc[pd.to_numeric(routes.get("truck"), errors="coerce") == truck] if not routes.empty else pd.DataFrame()
        arcs = [(str(row["from"]), str(row["to"])) for row in selected.to_dict("records")]
        closed_arc_use += sum(arc not in available for arc in arcs)
        indegree = Counter(b for _, b in arcs)
        outdegree = Counter(a for a, _ in arcs)
        for node in set(indegree) | set(outdegree):
            if indegree[node] != outdegree[node]:
                flow_violation += 1
            if node != "D0" and (indegree[node] > 1 or outdegree[node] > 1):
                repeated_node += 1
        adjacency: dict[str, list[str]] = defaultdict(list)
        for a, b in arcs:
            adjacency[a].append(b)
        reachable = {"D0"}
        queue = deque(["D0"])
        while queue:
            a = queue.popleft()
            for b in adjacency.get(a, []):
                if b not in reachable:
                    reachable.add(b)
                    queue.append(b)
        route_nodes = {node for arc in arcs for node in arc}
        disconnected_subtour += len(route_nodes - reachable)
    return {
        "closed_or_unavailable_route_arc_count": closed_arc_use,
        "route_flow_violation_count": flow_violation,
        "disconnected_route_node_count": disconnected_subtour,
        "repeated_route_node_count": repeated_node,
    }


def solution_validation(
    payload: dict[str, Any],
    solution: dict[str, Any],
    battery: pd.DataFrame,
    energy_diag: dict[str, float],
    service: dict[str, float],
) -> dict[str, Any]:
    truck_count = int(payload["metadata"]["assumptions"]["truck_count"])
    routes = solution["routes"]
    h_service = solution["h_service"]
    coverage = solution["coverage"]
    material = solution["material"]
    star = solution["star_delivery"]
    rendezvous = solution["rendezvous_delivery"]
    route_checks = route_validation(payload, solution, truck_count)
    visited_ct = {
        node
        for row in routes.to_dict("records")
        for node in (str(row["from"]), str(row["to"]))
        if node.startswith("C")
    }
    duplicate_positive_tau = 0
    tau_on_unvisited = 0
    if not h_service.empty:
        positive = h_service.loc[pd.to_numeric(h_service["tau"], errors="coerce") > VALIDATION_TOLERANCE]
        duplicate_positive_tau = int((positive.groupby("h")["truck"].nunique() > 1).sum())
        tau_on_unvisited = int(
            (
                (pd.to_numeric(h_service["tau"], errors="coerce") > VALIDATION_TOLERANCE)
                & (pd.to_numeric(h_service["z"], errors="coerce") <= 0.5)
            ).sum()
        )
    h_cap_violation = 0
    if not coverage.empty:
        h_cap_violation = int(
            (
                (pd.to_numeric(coverage["energy_supplied"], errors="coerce") > pd.to_numeric(coverage["energy_demand_at_arrival"], errors="coerce") + VALIDATION_TOLERANCE)
                | (pd.to_numeric(coverage["energy_supplied"], errors="coerce") > pd.to_numeric(coverage["energy_demand"], errors="coerce") + VALIDATION_TOLERANCE)
            ).sum()
        )
    material_cap_violation = 0
    node_map = {str(node["node_id"]): node for node in payload["nodes"]}
    for row in material.to_dict("records"):
        q = float(row["q_total"])
        demand = float(node_map[str(row["node"])]["demand"])
        material_cap_violation += int(q < -VALIDATION_TOLERANCE or q > demand + VALIDATION_TOLERANCE)
    drone_to_visited_ct = sum(str(row["i"]) in visited_ct for row in star.to_dict("records"))
    drone_to_visited_ct += sum(str(row["i"]) in visited_ct for row in rendezvous.to_dict("records"))
    self_rendezvous = sum(
        str(row["a"]) == str(row["i"]) or str(row["b"]) == str(row["i"])
        for row in rendezvous.to_dict("records")
    )
    battery_violation = int((battery["E_unused_kwh"] < -ENERGY_TOLERANCE).sum())
    energy_reconcile_violation = int(energy_diag["max_energy_reconciliation_error_kwh"] > ENERGY_TOLERANCE)
    share_violation = int(energy_diag["max_share_sum_residual"] > ENERGY_TOLERANCE)
    objective_violation = int(service["objective_reconciliation_error"] > 1e-4)
    metrics = {
        **route_checks,
        "duplicate_positive_tau_count": duplicate_positive_tau,
        "tau_on_unvisited_h_count": tau_on_unvisited,
        "h_energy_cap_violation_count": h_cap_violation,
        "material_demand_violation_count": material_cap_violation,
        "drone_to_truck_visited_ct_count": int(drone_to_visited_ct),
        "self_rendezvous_count": int(self_rendezvous),
        "truck_battery_violation_count": battery_violation,
        "energy_reconciliation_violation_count": energy_reconcile_violation,
        "battery_share_sum_violation_count": share_violation,
        "objective_reconciliation_violation_count": objective_violation,
    }
    metrics["validation_metric_sum"] = float(sum(metrics.values()))
    metrics.update(energy_diag)
    metrics["objective_reconciliation_error"] = service["objective_reconciliation_error"]
    return metrics


def add_per_truck_service(
    payload: dict[str, Any],
    solution: dict[str, Any],
    battery: pd.DataFrame,
) -> pd.DataFrame:
    nodes = {str(node["node_id"]): node for node in payload["nodes"]}
    qd = float(payload["drone"]["q"])
    direct = solution["truck_delivery"]
    star = solution["star_delivery"]
    rendezvous = solution["rendezvous_delivery"]
    h_service = solution["h_service"]
    coverage = solution["coverage"].set_index("h") if not solution["coverage"].empty else pd.DataFrame()
    returns = solution["returns"].set_index("truck") if not solution["returns"].empty else pd.DataFrame()
    rows = []
    for energy in battery.to_dict("records"):
        truck = int(energy["truck"])
        direct_qty = sum(float(row["qty"]) for row in direct.to_dict("records") if int(row["truck"]) == truck)
        star_qty = sum(qd * float(row["times"]) for row in star.to_dict("records") if int(row["truck"]) == truck)
        rv_qty = sum(float(row["qty"]) for row in rendezvous.to_dict("records") if int(row["truck"]) == truck)
        material_contribution = sum(
            float(row["qty"]) * float(nodes[str(row["node"])]["p"]) / max(float(nodes[str(row["node"])]["demand"]), 1e-12)
            for row in direct.to_dict("records")
            if int(row["truck"]) == truck
        )
        material_contribution += sum(
            qd * float(row["times"]) * float(nodes[str(row["i"])]["p"]) / max(float(nodes[str(row["i"])]["demand"]), 1e-12)
            for row in star.to_dict("records")
            if int(row["truck"]) == truck
        )
        material_contribution += sum(
            float(row["qty"]) * float(nodes[str(row["i"])]["p"]) / max(float(nodes[str(row["i"])]["demand"]), 1e-12)
            for row in rendezvous.to_dict("records")
            if int(row["truck"]) == truck
        )
        grid_contribution = 0.0
        for row in h_service.to_dict("records"):
            if int(row["truck"]) == truck and float(row.get("gamma", 0.0)) > 0.5:
                h = str(row["h"])
                grid_contribution += float(nodes[h]["p_m"]) * float(coverage.loc[h, "g"])
        rows.append(
            {
                **energy,
                "direct_material_quantity": direct_qty,
                "star_material_quantity": star_qty,
                "rendezvous_material_quantity": rv_qty,
                "total_material_quantity": direct_qty + star_qty + rv_qty,
                "material_objective_contribution": material_contribution,
                "microgrid_objective_contribution": grid_contribution,
                "total_service_objective_contribution": material_contribution + grid_contribution,
                "star_count": sum(float(row["times"]) for row in star.to_dict("records") if int(row["truck"]) == truck),
                "rendezvous_count": sum(1 for row in rendezvous.to_dict("records") if int(row["truck"]) == truck),
                "positive_tau_count": sum(1 for row in h_service.to_dict("records") if int(row["truck"]) == truck and float(row["tau"]) > VALIDATION_TOLERANCE),
                "mission_return_time_h": float(returns.loc[truck, "return_time"]) if not returns.empty else math.nan,
                "used_truck": int(energy["route_distance_km"] > VALIDATION_TOLERANCE),
            }
        )
    return pd.DataFrame(rows)


def summarize_case(item: dict[str, Any], payload: dict[str, Any], solution: dict[str, Any], per_truck: pd.DataFrame, validation: dict[str, Any]) -> dict[str, Any]:
    service = objective_and_service_metrics(payload, solution)
    stats = solution.get("model_stats", {})
    star_count = float(pd.to_numeric(solution["star_delivery"].get("times"), errors="coerce").fillna(0).sum()) if not solution["star_delivery"].empty else 0.0
    rendezvous_count = len(solution["rendezvous_delivery"])
    h_service = solution["h_service"]
    positive_tau = h_service.loc[pd.to_numeric(h_service.get("tau"), errors="coerce") > VALIDATION_TOLERANCE] if not h_service.empty else pd.DataFrame()
    routes = solution["routes"]
    returns = solution["returns"]
    return {
        "scenario": item["scenario"],
        "total_nodes": int(item["total_nodes"]),
        "truck_count": int(item["truck_count"]),
        "damage_q": int(item["damage_q"]),
        "damage_realization": str(item["damage_realization"]),
        "closed_link_count": int(item["closed_link_count"]),
        "closed_link_ids": str(item["closed_link_ids"]),
        "scenario_json_sha256": file_hash(Path(str(item["json"]))),
        "solver_seed": int(stats.get("solver_seed", SOLVER_SEED)),
        "status": status_name(solution["status"]),
        "status_code": int(solution["status"]),
        "incumbent": float(solution["objective"]),
        "objective": float(solution["objective"]),
        "best_bound": float(stats.get("best_bound", math.nan)),
        "mip_gap": float(stats.get("gap", math.nan)),
        "runtime_sec": float(stats.get("runtime_sec", math.nan)),
        "sol_count": int(stats.get("sol_count", 0)),
        "num_vars": int(stats.get("num_vars", 0)),
        "num_constrs": int(stats.get("num_constrs", 0)),
        **service,
        "route_distance_km": float(per_truck["route_distance_km"].sum()),
        "route_travel_time_h": float(per_truck["route_travel_time_h"].sum()),
        "route_driving_energy_kwh": float(per_truck["E_drive_kwh"].sum()),
        "mission_makespan_h": float(pd.to_numeric(returns.get("return_time"), errors="coerce").max()) if not returns.empty else math.nan,
        "route_arc_count": len(routes),
        "star_count": star_count,
        "rendezvous_count": rendezvous_count,
        "total_drone_sorties": star_count + rendezvous_count,
        "positive_tau_count": len(positive_tau),
        "charged_h_count": int(positive_tau["h"].nunique()) if not positive_tau.empty else 0,
        "total_grid_support_dwell_h": float(pd.to_numeric(positive_tau.get("tau"), errors="coerce").sum()) if not positive_tau.empty else 0.0,
        "total_grid_support_energy_kwh": float(per_truck["E_grid_kwh"].sum()),
        "used_truck_count": int(per_truck["used_truck"].sum()),
        "E_drive_kwh": float(per_truck["E_drive_kwh"].sum()),
        "E_drone_kwh": float(per_truck["E_drone_kwh"].sum()),
        "E_grid_kwh": float(per_truck["E_grid_kwh"].sum()),
        "E_unused_kwh": float(per_truck["E_unused_kwh"].sum()),
        "usable_battery_kwh": float(per_truck["usable_battery_kwh"].sum()),
        "phi_drive": float(per_truck["E_drive_kwh"].sum() / per_truck["usable_battery_kwh"].sum()),
        "phi_drone": float(per_truck["E_drone_kwh"].sum() / per_truck["usable_battery_kwh"].sum()),
        "phi_grid": float(per_truck["E_grid_kwh"].sum() / per_truck["usable_battery_kwh"].sum()),
        "phi_unused": float(per_truck["E_unused_kwh"].sum() / per_truck["usable_battery_kwh"].sum()),
        **validation,
    }


def write_raw_outputs(result_dir: Path, run_rows: list[dict], truck_rows: list[dict], validation_rows: list[dict]) -> None:
    runs = pd.DataFrame(run_rows).sort_values(["damage_q", "damage_realization"], kind="stable")
    trucks = pd.DataFrame(truck_rows).sort_values(["damage_q", "damage_realization", "truck"], kind="stable")
    validations = pd.DataFrame(validation_rows).sort_values(["damage_q", "damage_realization"], kind="stable")
    runs.to_csv(result_dir / "all_runs.csv", index=False, encoding=CSV_ENCODING)
    trucks.to_csv(result_dir / "battery_allocation_by_truck.csv", index=False, encoding=CSV_ENCODING)
    validations.to_csv(result_dir / "validation.csv", index=False, encoding=CSV_ENCODING)
    status_cols = [
        "scenario", "damage_q", "damage_realization", "status", "status_code", "incumbent",
        "best_bound", "mip_gap", "runtime_sec", "sol_count", "solver_seed",
    ]
    runs[status_cols].to_csv(result_dir / "solver_status.csv", index=False, encoding=CSV_ENCODING)


def gate_case(row: dict[str, Any], structural_valid: bool) -> tuple[bool, list[str]]:
    reasons = []
    if not structural_valid:
        reasons.append("structural validation failed")
    if float(row["validation_metric_sum"]) > 0.0:
        reasons.append(f"validation_metric_sum={row['validation_metric_sum']}")
    if float(row["max_energy_reconciliation_error_kwh"]) > ENERGY_TOLERANCE:
        reasons.append(f"energy reconciliation error={row['max_energy_reconciliation_error_kwh']}")
    if float(row["max_share_sum_residual"]) > ENERGY_TOLERANCE:
        reasons.append(f"battery share residual={row['max_share_sum_residual']}")
    status_ok = row["status"] == "OPTIMAL" or (
        math.isfinite(float(row["mip_gap"])) and float(row["mip_gap"]) <= TARGET_MIP_GAP + 1e-12
    )
    if not status_ok:
        reasons.append(f"status/gap gate failed: status={row['status']}, gap={row['mip_gap']}")
    return not reasons, reasons


def run_case(item: dict[str, Any], result_dir: Path, output_flag: int) -> tuple[dict[str, Any], list[dict], dict[str, Any]]:
    path = Path(str(item["json"]))
    payload = read_json(path)
    graph = load_scenario_json(path)
    start = time.perf_counter()
    solution = solve_model(
        graph,
        time_limit_sec=TIME_LIMIT_SEC,
        mip_gap=TARGET_MIP_GAP,
        solver_seed=SOLVER_SEED,
        successor_gap=3,
        no_truck_visited_ct_drone_service=True,
        exclude_self_rendezvous=True,
        output_flag=output_flag,
        alpha=1.0,
        beta=1.0,
    )
    if not math.isfinite(float(solution.get("objective", math.nan))) or "routes" not in solution:
        raise RuntimeError(
            f"No feasible incumbent for {item['scenario']}: status={solution.get('status')}, stats={solution.get('model_stats')}"
        )
    scenario_dir = result_dir / "raw_solutions" / str(item["scenario"])
    save_solution(solution, scenario_dir)
    battery, energy_diag = reconstruct_energy(payload, solution, int(item["truck_count"]))
    battery = add_per_truck_service(payload, solution, battery)
    service = objective_and_service_metrics(payload, solution)
    validation = solution_validation(payload, solution, battery, energy_diag, service)
    row = summarize_case(item, payload, solution, battery, validation)
    row["wall_runtime_sec"] = time.perf_counter() - start
    battery.insert(0, "scenario", str(item["scenario"]))
    battery.insert(1, "damage_q", int(item["damage_q"]))
    battery.insert(2, "damage_realization", str(item["damage_realization"]))
    validation_row = {
        "scenario": str(item["scenario"]),
        "damage_q": int(item["damage_q"]),
        "damage_realization": str(item["damage_realization"]),
        **validation,
    }
    return row, battery.to_dict("records"), validation_row


def load_existing(result_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    paths = (
        result_dir / "all_runs.csv",
        result_dir / "battery_allocation_by_truck.csv",
        result_dir / "validation.csv",
    )
    frames = [read_csv(path) for path in paths]
    return tuple(frame.to_dict("records") for frame in frames)  # type: ignore[return-value]


def write_failure_report(result_dir: Path, failed: list[tuple[str, list[str]]]) -> None:
    lines = [
        "# Pilot failure report",
        "",
        "Pilot gate 未通过，因此未启动剩余 16 个场景。阈值没有被放宽，也没有重新生成道路损伤序列。",
        "",
        "## Diagnostics",
        "",
    ]
    for scenario, reasons in failed:
        lines.append(f"- `{scenario}`: {'; '.join(reasons)}")
    lines.extend(
        [
            "",
            f"- Required MIP gap: `{TARGET_MIP_GAP:.4f}`.",
            f"- Energy reconciliation tolerance: `{ENERGY_TOLERANCE}` kWh.",
            "- Existing pilot incumbents and solver logs are retained for diagnosis.",
        ]
    )
    (result_dir / "pilot_failure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    result_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, dtype={"damage_realization": str})
    graph_audit = pd.read_csv(manifest_path.parent / "graph_audit.csv", dtype={"damage_realization": str})
    if len(manifest) != 21 or not as_bool(graph_audit["structural_validation_passed"]).all():
        raise RuntimeError("The generated suite is incomplete or structurally invalid; refusing to solve")
    config = {
        "scientific_objective": "controlled road-disruption battery-allocation experiment",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_hash(manifest_path),
        "full_exact_milp": True,
        "time_limit_sec": TIME_LIMIT_SEC,
        "target_mip_gap": TARGET_MIP_GAP,
        "solver_seed": SOLVER_SEED,
        "solver_seed_role": "solver reproducibility only; not an experimental replicate",
        "damage_realizations": list(DAMAGE_SEEDS),
        "damage_realization_role": "experimental repetitions",
        "successor_gap": 3,
        "alpha": 1.0,
        "beta": 1.0,
        "no_truck_visited_ct_drone_service": True,
        "exclude_self_rendezvous": True,
        "no_drone_experiment": False,
        "outcome_adaptive_regeneration": False,
    }
    write_json(result_dir / "experiment_config.json", config)

    run_rows: list[dict]
    truck_rows: list[dict]
    validation_rows: list[dict]
    if args.resume:
        run_rows, truck_rows, validation_rows = load_existing(result_dir)
    else:
        run_rows, truck_rows, validation_rows = [], [], []
    completed = {str(row["scenario"]) for row in run_rows}
    pilot_mask = as_bool(manifest["pilot_case"])
    pilot = manifest.loc[pilot_mask].sort_values("damage_q", kind="stable")
    if len(pilot) != 5:
        raise RuntimeError(f"Expected exactly five pilot cases, found {len(pilot)}")

    for item in pilot.to_dict("records"):
        scenario = str(item["scenario"])
        if scenario in completed:
            print(f"[pilot] skip completed {scenario}", flush=True)
            continue
        print(f"[pilot] start {scenario}", flush=True)
        try:
            row, trucks, validation = run_case(item, result_dir, args.output_flag)
        except Exception as exc:
            write_failure_report(result_dir, [(scenario, [f"solver or extraction exception: {type(exc).__name__}: {exc}"])])
            raise
        run_rows.append(row)
        truck_rows.extend(trucks)
        validation_rows.append(validation)
        completed.add(scenario)
        write_raw_outputs(result_dir, run_rows, truck_rows, validation_rows)
        print(
            f"[pilot] {scenario}: status={row['status']} obj={row['objective']:.6f} "
            f"gap={row['mip_gap']:.6g} validation={row['validation_metric_sum']}",
            flush=True,
        )

    rows_by_scenario = {str(row["scenario"]): row for row in run_rows}
    audit_by_scenario = graph_audit.set_index("scenario")
    failures: list[tuple[str, list[str]]] = []
    for scenario in pilot["scenario"].astype(str):
        row = rows_by_scenario[scenario]
        structural_value = str(audit_by_scenario.loc[scenario, "structural_validation_passed"]).strip().lower()
        passed, reasons = gate_case(row, structural_value in {"true", "1", "yes"})
        if not passed:
            failures.append((scenario, reasons))
    if failures:
        write_failure_report(result_dir, failures)
        print(f"pilot_gate=FAILED failed_cases={len(failures)}", flush=True)
        raise SystemExit(2)
    failure_path = result_dir / "pilot_failure_report.md"
    if failure_path.exists():
        failure_path.unlink()
    print("pilot_gate=PASSED", flush=True)
    if args.pilot_only:
        return

    remaining = manifest.loc[~pilot_mask].sort_values(
        ["damage_q", "damage_realization"], kind="stable"
    )
    for item in remaining.to_dict("records"):
        scenario = str(item["scenario"])
        if scenario in completed:
            print(f"[full] skip completed {scenario}", flush=True)
            continue
        print(f"[full] start {scenario}", flush=True)
        row, trucks, validation = run_case(item, result_dir, args.output_flag)
        run_rows.append(row)
        truck_rows.extend(trucks)
        validation_rows.append(validation)
        completed.add(scenario)
        write_raw_outputs(result_dir, run_rows, truck_rows, validation_rows)
        print(
            f"[full] {scenario}: status={row['status']} obj={row['objective']:.6f} "
            f"gap={row['mip_gap']:.6g} validation={row['validation_metric_sum']}",
            flush=True,
        )
    if len(completed) != 21:
        raise RuntimeError(f"Expected 21 completed unique cases, found {len(completed)}")
    print(f"full_experiment=COMPLETED cases={len(completed)} output_dir={result_dir}", flush=True)


if __name__ == "__main__":
    main()
