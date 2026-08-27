from __future__ import annotations

import math
from pathlib import Path
import time

import gurobipy as gp
import networkx as nx
import pandas as pd
from gurobipy import GRB

from humanitarian_graph import (
    ScenarioConfig,
    build_edge_table,
    build_h_parameters,
    build_humanitarian_graph,
    build_node_table,
    build_truck_nx,
    feasible_arcs,
    load_scenario_json,
    node_sets,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


ALPHA = 1.0
BETA = 1.0
BIG_M = 1e4
T_MAX_HOURS = 24.0
TRUCK_BATTERY_RESERVE = 0.0
RHO = 0.1
RENDEZVOUS_MAX_SUCCESSOR_GAP = 3


def main() -> None:
    wall_start = time.perf_counter()
    medium_path = DATA_DIR / "medium_benchmark_scenario.json"
    toy_path = DATA_DIR / "manual_toy_scenario.json"
    if medium_path.exists():
        graph = load_scenario_json(medium_path)
    elif toy_path.exists():
        graph = load_scenario_json(toy_path)
    else:
        graph = build_humanitarian_graph(ScenarioConfig())
    print(f"[solver] build graph done at +{time.perf_counter() - wall_start:.1f}s")
    solution = solve_model(graph)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_solution(solution, OUT_DIR)
    print(f"status={solution['status']}")
    print(f"objective={solution['objective']:.4f}")
    print(f"[solver] total elapsed = {time.perf_counter() - wall_start:.1f}s")
    print(f"saved_dir={OUT_DIR}")


def solve_model(
    graph,
    time_limit_sec: float | None = None,
    mip_gap: float | None = None,
    solver_seed: int | None = None,
    successor_gap: int = RENDEZVOUS_MAX_SUCCESSOR_GAP,
    no_same_truck_ct_rendezvous: bool = False,
    no_truck_visited_ct_drone_service: bool = False,
    drone_range_multiplier: float = 1.0,
    drone_battery_multiplier: float = 1.0,
    exclude_self_rendezvous: bool = False,
    output_flag: int = 1,
    alpha: float = ALPHA,
    beta: float = BETA,
):
    solve_start = time.perf_counter()
    assumptions = graph.metadata.get("assumptions", {}) if isinstance(graph.metadata, dict) else {}
    truck_count = int(assumptions.get("truck_count", 2))
    drones_per_truck = int(assumptions.get("drones_per_truck", 3))
    truck_ids = list(range(truck_count))
    drones_by_truck = {v: list(range(drones_per_truck)) for v in truck_ids}

    sets = node_sets(graph)
    nodes = {row["node_id"]: row for row in build_node_table(graph)}
    edges = build_edge_table(graph)
    h_params = build_h_parameters(graph)

    H = sets["H"]
    C = sets["C"]
    S = H + C
    C_T = [node_id for node_id in C if nodes[node_id]["truck_accessible"]]
    N_T = ["D0"] + H + C_T
    ORDER_NODES = H + C_T
    STAR_LAUNCH_NODES = H + C_T
    A_T = [(i, j) for i, j in feasible_arcs(graph, mode="truck") if i in N_T and j in N_T and i != j]

    truck_speed = graph.e_truck.v
    truck_energy_per_km = graph.e_truck.ev
    truck_capacity = graph.e_truck.Qt - sum(graph.drone.w for _ in drones_by_truck[0])
    drone_capacity = graph.drone.q
    drone_tmax_hours = (graph.drone.tmax / 3600.0) * max(float(drone_range_multiplier), 0.0)
    drone_battery = graph.drone.Bv * max(float(drone_battery_multiplier), 0.0)
    dist = {(e["from_node"], e["to_node"]): e["distance_km"] for e in edges}
    time_factor = {}
    for edge in graph.edges:
        factor = edge.notes.get("truck_time_factor", 1.0) if isinstance(edge.notes, dict) else 1.0
        time_factor[(edge.from_node, edge.to_node)] = float(factor)
    t_truck = {(i, j): (dist[(i, j)] / truck_speed) * time_factor.get((i, j), 1.0) for (i, j) in A_T}
    e_truck = {(i, j): dist[(i, j)] * truck_energy_per_km * time_factor.get((i, j), 1.0) for (i, j) in A_T}

    def km(a, b):
        ax, ay = nodes[a]["x_km"], nodes[a]["y_km"]
        bx, by = nodes[b]["x_km"], nodes[b]["y_km"]
        return math.hypot(ax - bx, ay - by)

    star_keys = [(a, i, v, d) for a in STAR_LAUNCH_NODES for i in C if a != i for v in truck_ids for d in drones_by_truck[v]]
    star_time = {(a, i, v, d): 2.0 * km(a, i) / (graph.drone.v * 3.6) for a, i, v, d in star_keys}
    star_energy = {
        (a, i, v, d): (star_time[(a, i, v, d)] / drone_tmax_hours) * drone_battery
        for a, i, v, d in star_keys
    }

    rendezvous_fly_time = {}
    rendezvous_keys = []
    for a in N_T:
        for i in C:
            for b in N_T:
                if a == b:
                    continue
                if exclude_self_rendezvous and (a == i or b == i):
                    continue
                fly_time = (km(a, i) + km(i, b)) / (graph.drone.v * 3.6)
                if fly_time <= drone_tmax_hours + 1e-9:
                    for v in truck_ids:
                        for d in drones_by_truck[v]:
                            key = (a, i, b, v, d)
                            rendezvous_keys.append(key)
                            rendezvous_fly_time[key] = fly_time
    rendezvous_energy = {
        (a, i, b, v, d): (rendezvous_fly_time[(a, i, b, v, d)] / drone_tmax_hours) * drone_battery
        for a, i, b, v, d in rendezvous_keys
    }
    model = gp.Model("truck_drone_microgrid")
    model.Params.OutputFlag = int(output_flag)
    if time_limit_sec is not None:
        model.Params.TimeLimit = float(time_limit_sec)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if solver_seed is not None:
        model.Params.Seed = int(solver_seed)

    x = model.addVars(A_T, truck_ids, vtype=GRB.BINARY, name="x")
    z = model.addVars(H, truck_ids, vtype=GRB.BINARY, name="z")
    gamma = model.addVars(H, truck_ids, vtype=GRB.BINARY, name="gamma")
    eta_ct = model.addVars(C_T, vtype=GRB.BINARY, name="eta_ct")
    tau = model.addVars(H, truck_ids, lb=0.0, name="tau")
    sigma = model.addVars(STAR_LAUNCH_NODES, truck_ids, lb=0.0, name="sigma")
    A = model.addVars(N_T, truck_ids, lb=0.0, name="A")
    L = model.addVars(N_T, truck_ids, lb=0.0, name="L")
    w = model.addVars(N_T, truck_ids, lb=0.0, name="w")
    T_return = model.addVars(truck_ids, lb=0.0, name="T_return")
    u = model.addVars(
        ORDER_NODES,
        truck_ids,
        vtype=GRB.INTEGER,
        lb=0,
        ub=len(ORDER_NODES),
        name="u",
    )
    route_len = model.addVars(
        truck_ids,
        vtype=GRB.INTEGER,
        lb=0,
        ub=len(ORDER_NODES),
        name="route_len",
    )

    q_total = model.addVars(S, lb=0.0, name="q_total")
    q_truck = model.addVars([(i, v) for i in H + C_T for v in truck_ids], lb=0.0, name="q_truck")
    y = model.addVars(star_keys, vtype=GRB.INTEGER, lb=0, name="y")
    r = model.addVars(rendezvous_keys, vtype=GRB.BINARY, name="r")
    s = model.addVars(S, lb=0.0, ub=1.0, name="s")

    demand_rate = {h: h_params[h]["p_m"] * RHO for h in H}
    E_dem = {h: demand_rate[h] * h_params[h]["R"] for h in H}
    h_arrival = model.addVars(H, lb=0.0, ub=T_MAX_HOURS, name="h_charge_arrival")
    h_tau_total = model.addVars(H, lb=0.0, ub=T_MAX_HOURS, name="h_tau_total")
    h_remaining_window_raw = model.addVars(H, lb=-T_MAX_HOURS, ub=T_MAX_HOURS, name="h_remaining_window_raw")
    h_remaining_window = model.addVars(H, lb=0.0, ub=T_MAX_HOURS, name="h_remaining_window")
    h_energy_demand_at_arrival = model.addVars(H, lb=0.0, name="h_energy_demand_at_arrival")
    E_sup = model.addVars(H, lb=0.0, name="E_sup")
    c = model.addVars(H, lb=0.0, ub=1.0, name="c")
    g_cov = model.addVars(H, lb=0.0, ub=1.0, name="g")

    # Objective.
    model.setObjective(
        float(alpha) * gp.quicksum(h_params[h]["p_m"] * g_cov[h] for h in H)
        + float(beta) * gp.quicksum(nodes[i]["p"] * s[i] for i in S),
        GRB.MAXIMIZE,
    )

    # Piecewise-linear normalized g(c): (0,0), (0.5,0.625), (1,1).
    for h in H:
        model.addGenConstrPWL(c[h], g_cov[h], [0.0, 0.5, 1.0], [0.0, 0.625, 1.0], name=f"pwl_{h}")

    # Depot flow.
    for v in truck_ids:
        model.addConstr(gp.quicksum(x["D0", b, v] for b in N_T if ("D0", b, v) in x) == 1, name=f"depot_out_{v}")
        model.addConstr(gp.quicksum(x[a, "D0", v] for a in N_T if (a, "D0", v) in x) == 1, name=f"depot_in_{v}")

    # Flow conservation.
    for v in truck_ids:
        for a in H + C_T:
            out_expr = gp.quicksum(x[a, b, v] for b in N_T if (a, b, v) in x)
            in_expr = gp.quicksum(x[b, a, v] for b in N_T if (b, a, v) in x)
            model.addConstr(out_expr == in_expr, name=f"flow_{a}_{v}")
            if a in C_T:
                model.addConstr(in_expr <= 1, name=f"ct_visit_once_in_{a}_{v}")
                model.addConstr(out_expr <= 1, name=f"ct_visit_once_out_{a}_{v}")

    # H visit linkage.
    for v in truck_ids:
        for h in H:
            model.addConstr(gp.quicksum(x[h, b, v] for b in N_T if (h, b, v) in x) == z[h, v], name=f"hzout_{h}_{v}")
            model.addConstr(gp.quicksum(x[a, h, v] for a in N_T if (a, h, v) in x) == z[h, v], name=f"hzin_{h}_{v}")

    # A microgrid can be physically visited by multiple trucks, but charged by at most one.
    for h in H:
        model.addConstr(gp.quicksum(gamma[h, v] for v in truck_ids) <= 1, name=f"h_charge_unique_{h}")
        for v in truck_ids:
            model.addConstr(gamma[h, v] <= z[h, v], name=f"gamma_visit_link_{h}_{v}")

    # Truck-accessible demand nodes may be physically visited by multiple trucks.
    # eta_ct records whether at least one truck physically reaches a C^T node.
    for i in C_T:
        visit_terms = [gp.quicksum(x[a, i, v] for a in N_T if (a, i, v) in x) for v in truck_ids]
        for v, visit_i_v in zip(truck_ids, visit_terms):
            model.addConstr(eta_ct[i] >= visit_i_v, name=f"eta_ct_lb_{i}_{v}")
        model.addConstr(eta_ct[i] <= gp.quicksum(visit_terms), name=f"eta_ct_ub_{i}")

    # Route order variables for successor-gap rendezvous restriction.
    order_big_m = len(ORDER_NODES) + 1
    for v in truck_ids:
        visit_terms = []
        for node_id in ORDER_NODES:
            if node_id in H:
                visit_node = z[node_id, v]
            else:
                visit_node = gp.quicksum(x[a, node_id, v] for a in N_T if (a, node_id, v) in x)
            visit_terms.append(visit_node)
            model.addConstr(u[node_id, v] >= visit_node, name=f"u_lb_{node_id}_{v}")
            model.addConstr(u[node_id, v] <= len(ORDER_NODES) * visit_node, name=f"u_ub_{node_id}_{v}")
            model.addConstr(u[node_id, v] <= route_len[v], name=f"u_route_len_{node_id}_{v}")
        model.addConstr(route_len[v] == gp.quicksum(visit_terms), name=f"route_len_def_{v}")

        for node_id in ORDER_NODES:
            if ("D0", node_id, v) in x:
                model.addConstr(
                    u[node_id, v] >= 1 - order_big_m * (1 - x["D0", node_id, v]),
                    name=f"u_from_depot_lb_{node_id}_{v}",
                )
                model.addConstr(
                    u[node_id, v] <= 1 + order_big_m * (1 - x["D0", node_id, v]),
                    name=f"u_from_depot_ub_{node_id}_{v}",
                )
            if (node_id, "D0", v) in x:
                model.addConstr(
                    route_len[v] >= u[node_id, v] - order_big_m * (1 - x[node_id, "D0", v]),
                    name=f"route_len_last_lb_{node_id}_{v}",
                )
                model.addConstr(
                    route_len[v] <= u[node_id, v] + order_big_m * (1 - x[node_id, "D0", v]),
                    name=f"route_len_last_ub_{node_id}_{v}",
                )

        for i in ORDER_NODES:
            for j in ORDER_NODES:
                if i == j or (i, j, v) not in x:
                    continue
                model.addConstr(
                    u[j, v] >= u[i, v] + 1 - order_big_m * (1 - x[i, j, v]),
                    name=f"u_arc_lb_{i}_{j}_{v}",
                )
                model.addConstr(
                    u[j, v] <= u[i, v] + 1 + order_big_m * (1 - x[i, j, v]),
                    name=f"u_arc_ub_{i}_{j}_{v}",
                )

    # Time constraints.
    for v in truck_ids:
        model.addConstr(A["D0", v] == 0.0, name=f"start_time_{v}")
        model.addConstr(L["D0", v] == 0.0, name=f"leave_depot_{v}")
        for (a, b) in A_T:
            if b != "D0":
                model.addConstr(
                    A[b, v] >= L[a, v] + t_truck[(a, b)] - BIG_M * (1 - x[a, b, v]),
                    name=f"time_{a}_{b}_{v}",
                )
            else:
                model.addConstr(
                    T_return[v] >= L[a, v] + t_truck[(a, b)] - BIG_M * (1 - x[a, b, v]),
                    name=f"return_{a}_{v}",
                )
        for h in H:
            model.addConstr(L[h, v] == A[h, v] + tau[h, v] + sigma[h, v] + w[h, v], name=f"leave_h_{h}_{v}")
            model.addConstr(tau[h, v] <= T_MAX_HOURS * gamma[h, v], name=f"tau_link_{h}_{v}")
            model.addConstr(A[h, v] <= T_MAX_HOURS * z[h, v], name=f"arrive_h_link_{h}_{v}")
            model.addConstr(sigma[h, v] <= T_MAX_HOURS * z[h, v], name=f"sigma_h_link_{h}_{v}")
        for i in C_T:
            visit_i = gp.quicksum(x[a, i, v] for a in N_T if (a, i, v) in x)
            model.addConstr(L[i, v] == A[i, v] + sigma[i, v] + w[i, v], name=f"leave_c_{i}_{v}")
            model.addConstr(sigma[i, v] <= T_MAX_HOURS * visit_i, name=f"sigma_c_link_{i}_{v}")
        model.addConstr(T_return[v] <= T_MAX_HOURS, name=f"max_time_{v}")

    # Microgrid energy coverage constraints.
    for h in H:
        for v in truck_ids:
            model.addConstr(
                h_arrival[h] >= A[h, v] - BIG_M * (1 - gamma[h, v]),
                name=f"h_charge_arrival_lb_{h}_{v}",
            )
            model.addConstr(
                h_arrival[h] <= A[h, v] + BIG_M * (1 - gamma[h, v]),
                name=f"h_charge_arrival_ub_{h}_{v}",
            )
        model.addConstr(h_tau_total[h] == gp.quicksum(tau[h, v] for v in truck_ids), name=f"h_tau_sum_{h}")
        model.addConstr(
            h_remaining_window_raw[h] == h_params[h]["R"] - h_arrival[h],
            name=f"h_remaining_window_raw_{h}",
        )
        model.addGenConstrMax(h_remaining_window[h], [h_remaining_window_raw[h]], 0.0, name=f"h_remaining_window_{h}")
        model.addConstr(
            h_energy_demand_at_arrival[h] == demand_rate[h] * h_remaining_window[h],
            name=f"h_energy_demand_at_arrival_{h}",
        )
        model.addConstr(
            E_sup[h] == h_params[h]["P_o"] * h_tau_total[h],
            name=f"energy_supply_{h}",
        )
        model.addConstr(E_sup[h] <= h_energy_demand_at_arrival[h], name=f"arrival_cap_{h}")
        model.addConstr(E_sup[h] == E_dem[h] * c[h], name=f"coverage_{h}")

    # Material aggregation.
    for i in S:
        truck_part = gp.quicksum(q_truck[i, v] for v in truck_ids if (i, v) in q_truck)
        star_part = gp.quicksum(
            drone_capacity * y[(a, i, v, d)]
            for a in STAR_LAUNCH_NODES
            for v in truck_ids
            for d in drones_by_truck[v]
            if (a, i, v, d) in y
        )
        rendezvous_part = gp.quicksum(
            drone_capacity * r[(a, i, b, v, d)]
            for a in N_T
            for b in N_T
            if a != b
            for v in truck_ids
            for d in drones_by_truck[v]
            if (a, i, b, v, d) in r
        )
        if i in H:
            star_part = 0.0
            rendezvous_part = 0.0
        model.addConstr(q_total[i] == truck_part + star_part + rendezvous_part, name=f"q_balance_{i}")
        model.addConstr(q_total[i] <= nodes[i]["demand"], name=f"q_cap_{i}")
        model.addConstr(q_total[i] == nodes[i]["demand"] * s[i], name=f"s_link_{i}")

    # Truck capacity with drone body weight.
    for v in truck_ids:
        expr = gp.quicksum(q_truck[i, v] for i in H + C_T if (i, v) in q_truck)
        expr += gp.quicksum(drone_capacity * y[key] for key in star_keys if key[2] == v)
        expr += gp.quicksum(drone_capacity * r[key] for key in rendezvous_keys if key[3] == v)
        model.addConstr(expr <= truck_capacity, name=f"truck_cap_{v}")

    # Direct truck delivery only if visited.
    for v in truck_ids:
        for i in H + C_T:
            model.addConstr(
                q_truck[i, v] <= nodes[i]["demand"] * gp.quicksum(x[a, i, v] for a in N_T if (a, i, v) in x),
                name=f"truck_visit_link_{i}_{v}",
            )

    # Star mode can be launched from any visited star-launch node.
    for key in star_keys:
        a, i, v, d = key
        if a in H:
            launch_visit = z[a, v]
        else:
            launch_visit = gp.quicksum(x[b, a, v] for b in N_T if (b, a, v) in x)
        model.addConstr(y[key] <= BIG_M * launch_visit, name=f"star_visit_{a}_{i}_{v}_{d}")
        if no_truck_visited_ct_drone_service and i in C_T:
            model.addConstr(y[key] <= BIG_M * (1 - eta_ct[i]), name=f"star_no_truck_visited_ct_{a}_{i}_{v}_{d}")
        model.addConstr(star_energy[key] * y[key] <= graph.drone.Bv * y[key], name=f"star_battery_{a}_{i}_{v}_{d}")

    # Star missions must finish before the truck leaves the launch node.
    # At H nodes, tau is charging time and sigma is extra non-charging stay reserved for star work.
    for v in truck_ids:
        for d in drones_by_truck[v]:
            for a in STAR_LAUNCH_NODES:
                if a in H:
                    star_time_limit = tau[a, v] + sigma[a, v]
                else:
                    star_time_limit = sigma[a, v]
                model.addConstr(
                    gp.quicksum(star_time[(a, i, v, d)] * y[(a, i, v, d)] for i in C if (a, i, v, d) in y) <= star_time_limit,
                    name=f"star_time_sum_{a}_{v}_{d}",
                )

    # Rendezvous linkage.
    for key in rendezvous_keys:
        a, i, b, v, d = key
        model.addConstr(r[key] <= gp.quicksum(x[a, j, v] for j in N_T if (a, j, v) in x), name=f"r_from_{a}_{i}_{b}_{v}_{d}")
        model.addConstr(r[key] <= gp.quicksum(x[j, b, v] for j in N_T if (j, b, v) in x), name=f"r_to_{a}_{i}_{b}_{v}_{d}")
        if no_same_truck_ct_rendezvous and i in C_T:
            truck_visit_i = gp.quicksum(x[j, i, v] for j in N_T if (j, i, v) in x)
            model.addConstr(r[key] <= 1 - truck_visit_i, name=f"r_no_same_truck_ct_{a}_{i}_{b}_{v}_{d}")
        if no_truck_visited_ct_drone_service and i in C_T:
            model.addConstr(r[key] <= 1 - eta_ct[i], name=f"r_no_truck_visited_ct_{a}_{i}_{b}_{v}_{d}")
        if a == "D0":
            model.addConstr(
                u[b, v] >= 1 - order_big_m * (1 - r[key]),
                name=f"r_gap_depot_lb_{a}_{i}_{b}_{v}_{d}",
            )
            model.addConstr(
                u[b, v] <= successor_gap + order_big_m * (1 - r[key]),
                name=f"r_gap_depot_ub_{a}_{i}_{b}_{v}_{d}",
            )
        elif b == "D0":
            model.addConstr(
                route_len[v] <= u[a, v] + successor_gap - 1 + order_big_m * (1 - r[key]),
                name=f"r_gap_return_ub_{a}_{i}_{b}_{v}_{d}",
            )
        else:
            model.addConstr(
                u[b, v] >= u[a, v] + 1 - order_big_m * (1 - r[key]),
                name=f"r_gap_lb_{a}_{i}_{b}_{v}_{d}",
            )
            model.addConstr(
                u[b, v] <= u[a, v] + successor_gap + order_big_m * (1 - r[key]),
                name=f"r_gap_ub_{a}_{i}_{b}_{v}_{d}",
            )

        if b == "D0":
            model.addConstr(
                T_return[v] >= L[a, v] + rendezvous_fly_time[key] - BIG_M * (1 - r[key]),
                name=f"r_sync_return_{a}_{i}_{b}_{v}_{d}",
            )
        elif b in H:
            model.addConstr(
                w[b, v] >= L[a, v] + rendezvous_fly_time[key] - A[b, v] - tau[b, v] - sigma[b, v] - BIG_M * (1 - r[key]),
                name=f"r_wait_h_{a}_{i}_{b}_{v}_{d}",
            )
            model.addConstr(L[b, v] >= L[a, v] + rendezvous_fly_time[key] - BIG_M * (1 - r[key]), name=f"r_sync_{a}_{i}_{b}_{v}_{d}")
        else:
            model.addConstr(
                w[b, v] >= L[a, v] + rendezvous_fly_time[key] - A[b, v] - sigma[b, v] - BIG_M * (1 - r[key]),
                name=f"r_wait_c_{a}_{i}_{b}_{v}_{d}",
            )
            model.addConstr(L[b, v] >= L[a, v] + rendezvous_fly_time[key] - BIG_M * (1 - r[key]), name=f"r_sync_{a}_{i}_{b}_{v}_{d}")
        model.addConstr(rendezvous_energy[key] <= drone_battery + BIG_M * (1 - r[key]), name=f"r_bat_{a}_{i}_{b}_{v}_{d}")
        model.addConstr(rendezvous_fly_time[key] <= drone_tmax_hours + BIG_M * (1 - r[key]), name=f"r_tmax_{a}_{i}_{b}_{v}_{d}")

    # Launch / recovery node uniqueness for each truck-drone pair.
    for v in truck_ids:
        for d in drones_by_truck[v]:
            for a in N_T:
                model.addConstr(
                    gp.quicksum(r[(a, i, b, v, d)] for i in C for b in N_T if a != b and (a, i, b, v, d) in r) <= 1,
                    name=f"r_launch_unique_{a}_{v}_{d}",
                )
            for b in N_T:
                model.addConstr(
                    gp.quicksum(r[(a, i, b, v, d)] for a in N_T for i in C if a != b and (a, i, b, v, d) in r) <= 1,
                    name=f"r_recover_unique_{b}_{v}_{d}",
                )

    # Shared battery constraints.
    for v in truck_ids:
        drive_energy = gp.quicksum(e_truck[(a, b)] * x[a, b, v] for (a, b) in A_T)
        drone_energy = gp.quicksum(star_energy[key] * y[key] for key in star_keys if key[2] == v)
        drone_energy += gp.quicksum(rendezvous_energy[key] * r[key] for key in rendezvous_keys if key[3] == v)
        grid_energy = gp.quicksum(h_params[h]["P_o"] * tau[h, v] for h in H)
        model.addConstr(
            drive_energy + drone_energy + grid_energy <= graph.e_truck.B - TRUCK_BATTERY_RESERVE,
            name=f"battery_{v}",
        )

    # Subtour elimination via lazy constraints.
    model.Params.LazyConstraints = 1
    model._x = x
    model._A_T = A_T
    model._N_T = N_T
    model._V = truck_ids
    model._solve_start = solve_start
    model._last_progress_print = solve_start

    def callback(m, where):
        if where != GRB.Callback.MIPSOL:
            if where == GRB.Callback.MIP:
                now = time.perf_counter()
                if now - m._last_progress_print >= 15.0:
                    incumbent = m.cbGet(GRB.Callback.MIP_OBJBST)
                    bound = m.cbGet(GRB.Callback.MIP_OBJBND)
                    nodecnt = int(m.cbGet(GRB.Callback.MIP_NODCNT))
                    solcnt = int(m.cbGet(GRB.Callback.MIP_SOLCNT))
                    gap = float("inf")
                    if abs(incumbent) > 1e-9 and incumbent < GRB.INFINITY:
                        gap = abs(bound - incumbent) / abs(incumbent)
                    gap_text = f"{gap:.4%}" if gap != float("inf") else "inf"
                    print(
                        f"[progress] t={now - m._solve_start:.1f}s nodes={nodecnt} sols={solcnt} "
                        f"inc={incumbent:.6f} bound={bound:.6f} gap={gap_text}",
                        flush=True,
                    )
                    m._last_progress_print = now
            return
        for v in m._V:
            selected = [(a, b) for (a, b) in m._A_T if m.cbGetSolution(m._x[a, b, v]) > 0.5]
            g = nx.DiGraph()
            g.add_nodes_from(m._N_T)
            g.add_edges_from(selected)
            for comp in nx.strongly_connected_components(g):
                if "D0" not in comp and len(comp) >= 2:
                    expr = gp.quicksum(m._x[a, b, v] for a in comp for b in comp if a != b and (a, b, v) in m._x)
                    m.cbLazy(expr <= len(comp) - 1)

    model.optimize(callback)
    model.update()

    if model.Status == GRB.INFEASIBLE:
        iis_path = OUT_DIR / "model_infeasible.ilp"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model.computeIIS()
        model.write(str(iis_path))
        print(f"[solver] IIS written to {iis_path}")

    gap = float("nan")
    best_bound = float("nan")
    try:
        if model.SolCount > 0:
            gap = float(model.MIPGap)
            best_bound = float(model.ObjBound)
    except gp.GurobiError:
        gap = float("nan")
        best_bound = float("nan")

    return extract_solution(
        model,
        graph,
        nodes,
        h_params,
        A_T,
        truck_ids,
        drones_by_truck,
        t_truck,
        e_truck,
        star_time,
        star_energy,
        rendezvous_fly_time,
        rendezvous_energy,
        x,
        z,
        gamma,
        tau,
        A,
        L,
        w,
        u,
        route_len,
        demand_rate,
        h_arrival,
        h_tau_total,
        h_remaining_window,
        h_energy_demand_at_arrival,
        E_sup,
        E_dem,
        c,
        g_cov,
        q_total,
        q_truck,
        y,
        r,
        T_return,
        sigma,
        {
            "x": len(A_T) * len(truck_ids),
            "z": len(H) * len(truck_ids),
            "gamma": len(H) * len(truck_ids),
            "tau": len(H) * len(truck_ids),
            "sigma": len(STAR_LAUNCH_NODES) * len(truck_ids),
            "A": len(N_T) * len(truck_ids),
            "L": len(N_T) * len(truck_ids),
            "w": len(N_T) * len(truck_ids),
            "T_return": len(truck_ids),
            "u": len(ORDER_NODES) * len(truck_ids),
            "route_len": len(truck_ids),
            "q_total": len(S),
            "q_truck": len(H + C_T) * len(truck_ids),
            "y": len(star_keys),
            "r": len(rendezvous_keys),
            "s": len(S),
            "h_arrival": len(H),
            "h_tau_total": len(H),
            "h_remaining_window_raw": len(H),
            "h_remaining_window": len(H),
            "h_energy_demand_at_arrival": len(H),
            "E_sup": len(H),
            "c": len(H),
            "g_cov": len(H),
        },
        {
            "runtime_sec": float(model.Runtime),
            "gap": gap,
            "best_bound": best_bound,
            "num_vars": int(model.NumVars),
            "num_constrs": int(model.NumConstrs),
            "num_bin_vars": int(model.NumBinVars),
            "num_int_vars": int(model.NumIntVars),
            "status": int(model.Status),
            "sol_count": int(model.SolCount),
            "successor_gap": int(successor_gap),
            "target_mip_gap": float(mip_gap) if mip_gap is not None else math.nan,
            "solver_seed": int(solver_seed) if solver_seed is not None else -1,
        },
    )


def extract_solution(
    model,
    graph,
    nodes,
    h_params,
    A_T,
    truck_ids,
    drones_by_truck,
    t_truck,
    e_truck,
    star_time,
    star_energy,
    rendezvous_fly_time,
    rendezvous_energy,
    x,
    z,
    gamma,
    tau,
    A,
    L,
    w,
    u,
    route_len,
    demand_rate,
    h_arrival,
    h_tau_total,
    h_remaining_window,
    h_energy_demand_at_arrival,
    E_sup,
    E_dem,
    c,
    g_cov,
    q_total,
    q_truck,
    y,
    r,
    T_return,
    sigma,
    variable_group_counts,
    model_stats,
):
    status = model.Status
    if status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        return {"status": status, "objective": float("nan")}
    if model.SolCount == 0:
        empty_df = pd.DataFrame()
        return {
            "status": status,
            "objective": float("nan"),
            "model_stats": model_stats,
            "variable_group_counts": variable_group_counts,
            "routes": empty_df,
            "h_service": empty_df,
            "coverage": empty_df,
            "material": empty_df,
            "truck_delivery": empty_df,
            "star_stay": empty_df,
            "star_delivery": empty_df,
            "rendezvous_delivery": empty_df,
            "returns": empty_df,
            "truck_schedule": empty_df,
            "drone_schedule": empty_df,
            "energy_breakdown": empty_df,
            "route_order": empty_df,
        }

    def sorted_or_empty(rows, columns, by):
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=columns)
        return df.sort_values(by).reset_index(drop=True)

    route_rows = []
    dist = {(edge.from_node, edge.to_node): edge.distance_km for edge in graph.edges}
    for v in truck_ids:
        for a, b in A_T:
            if x[a, b, v].X > 0.5:
                route_rows.append(
                    {
                        "truck": v,
                        "from": a,
                        "to": b,
                        "distance_km": dist[(a, b)],
                        "travel_time": t_truck[(a, b)],
                        "depart_time": L[a, v].X,
                        "arrive_time": A[b, v].X if b != "D0" else T_return[v].X,
                    }
                )

    h_rows = [
        {
            "h": h,
            "truck": v,
            "z": round(z[h, v].X),
            "gamma": round(gamma[h, v].X),
            "arrival_time": A[h, v].X,
            "tau": tau[h, v].X,
            "is_charging_owner": int(gamma[h, v].X > 0.5),
            "sigma_star": sigma[h, v].X,
        }
        for h in c.keys()
        for v in truck_ids
        if z[h, v].X > 1e-6
        or gamma[h, v].X > 1e-6
        or A[h, v].X > 1e-6
        or tau[h, v].X > 1e-6
        or sigma[h, v].X > 1e-6
    ]
    star_stay_rows = [
        {"node": a, "truck": v, "sigma_star": sigma[a, v].X}
        for a, v in sigma.keys()
        if sigma[a, v].X > 1e-6
    ]
    route_length_rows = [{"truck": v, "route_len": route_len[v].X} for v in truck_ids]
    coverage_rows = [
        {
            "h": h,
            "charge_arrival_time": h_arrival[h].X,
            "tau_total": h_tau_total[h].X,
            "remaining_window": h_remaining_window[h].X,
            "demand_rate": demand_rate[h],
            "energy_demand_at_arrival": h_energy_demand_at_arrival[h].X,
            "energy_demand": E_dem[h],
            "energy_supplied": E_sup[h].X,
            "c": c[h].X,
            "g": g_cov[h].X,
        }
        for h in c.keys()
    ]
    material_rows = [{"node": i, "q_total": q_total[i].X, "coverage": q_total[i].X / max(1, nodes[i]["demand"])} for i in q_total.keys()]
    truck_delivery_rows = [{"node": i, "truck": v, "qty": q_truck[i, v].X} for i, v in q_truck.keys() if q_truck[i, v].X > 1e-6]
    star_rows = [{"launch": a, "i": i, "truck": v, "drone": d, "times": y[(a, i, v, d)].X} for a, i, v, d in y.keys() if y[(a, i, v, d)].X > 1e-6]
    rendezvous_rows = [
        {
            "a": a,
            "i": i,
            "b": b,
            "truck": v,
            "drone": d,
            "qty": graph.drone.q * r[(a, i, b, v, d)].X,
            "start_time": L[a, v].X,
            "end_time": max(A[b, v].X if b != "D0" else T_return[v].X, L[a, v].X + rendezvous_fly_time[(a, i, b, v, d)]),
            "wait_time": max((A[b, v].X if b != "D0" else T_return[v].X) - (L[a, v].X + rendezvous_fly_time[(a, i, b, v, d)]), 0.0),
        }
        for a, i, b, v, d in r.keys()
        if r[(a, i, b, v, d)].X > 1e-6
    ]
    returns = [{"truck": v, "return_time": T_return[v].X} for v in truck_ids]
    truck_schedule_rows = []
    visited_by_truck = {v: set() for v in truck_ids}
    for row in route_rows:
        visited_by_truck[row["truck"]].add(row["from"])
        visited_by_truck[row["truck"]].add(row["to"])

    for v in truck_ids:
        truck_schedule_rows.append(
            {
                "truck": v,
                "node": "D0",
                "arrival_time": 0.0,
                "departure_time": 0.0,
                "wait_time": 0.0,
                "service_time": 0.0,
                "node_type": "depot",
            }
        )
        for node_id in sorted(visited_by_truck[v] - {"D0"}):
            grid_time = tau[node_id, v].X if (node_id, v) in tau else 0.0
            star_time_stay = sigma[node_id, v].X if (node_id, v) in sigma else 0.0
            if node_id == "D0":
                continue
            truck_schedule_rows.append(
                {
                    "truck": v,
                    "node": node_id,
                    "arrival_time": A[node_id, v].X,
                    "departure_time": L[node_id, v].X,
                    "wait_time": w[node_id, v].X,
                    "service_time": grid_time + star_time_stay,
                    "grid_service_time": grid_time,
                    "star_service_time": star_time_stay,
                    "order_position": u[node_id, v].X if (node_id, v) in u else None,
                    "node_type": nodes[node_id]["node_type"],
                }
            )
        truck_schedule_rows.append(
            {
                "truck": v,
                "node": "D0_return",
                "arrival_time": T_return[v].X,
                "departure_time": T_return[v].X,
                "wait_time": 0.0,
                "service_time": 0.0,
                "node_type": "depot_return",
            }
        )

    drone_schedule_rows = []
    for v in truck_ids:
        for d in drones_by_truck[v]:
            for a in sorted({key[0] for key in y.keys() if key[2] == v and key[3] == d}):
                occupied = sum(star_time[(a, i, v, d)] * y[(a, i, v, d)].X for i in [k[1] for k in y.keys() if k[0] == a and k[2] == v and k[3] == d])
                total_sorties = sum(y[(a, i, v, d)].X for i in [k[1] for k in y.keys() if k[0] == a and k[2] == v and k[3] == d])
                if occupied > 1e-6:
                    drone_schedule_rows.append(
                        {
                            "truck": v,
                            "drone": d,
                            "mode": "star_block",
                            "launch": a,
                            "service": a,
                            "recover": a,
                            "start_time": A[a, v].X,
                            "end_time": A[a, v].X + occupied,
                            "occupied_time": occupied,
                            "fly_time": occupied,
                            "wait_time": 0.0,
                            "sorties": total_sorties,
                        }
                    )

    for a, i, b, v, d in r.keys():
        if r[(a, i, b, v, d)].X > 1e-6:
            start_time = L[a, v].X
            fly_time = rendezvous_fly_time[(a, i, b, v, d)]
            recover_arrival = A[b, v].X if b != "D0" else T_return[v].X
            wait_time = max(recover_arrival - start_time - fly_time, 0.0)
            end_time = max(recover_arrival, start_time + fly_time)
            drone_schedule_rows.append(
                {
                    "truck": v,
                    "drone": d,
                    "mode": "rendezvous",
                    "launch": a,
                    "service": i,
                    "recover": b,
                    "start_time": start_time,
                    "end_time": end_time,
                    "occupied_time": end_time - start_time,
                    "fly_time": fly_time,
                    "wait_time": wait_time,
                    "sorties": 1.0,
                }
            )

    energy_rows = []
    for v in truck_ids:
        drive_energy = sum(e_truck[(a, b)] * x[a, b, v].X for (a, b) in A_T)
        drone_energy = sum(star_energy[key] * y[key].X for key in y.keys() if key[2] == v)
        drone_energy += sum(rendezvous_energy[key] * r[key].X for key in r.keys() if key[3] == v)
        grid_energy = sum(h_params[h]["P_o"] * tau[h, v].X for h in h_params.keys())
        total_energy = drive_energy + drone_energy + grid_energy
        energy_rows.append(
            {
                "truck": v,
                "drive_energy_kwh": drive_energy,
                "grid_energy_kwh": grid_energy,
                "drone_energy_kwh": drone_energy,
                "total_energy_kwh": total_energy,
                "battery_capacity_kwh": graph.e_truck.B,
                "battery_remaining_kwh": graph.e_truck.B - total_energy,
                "battery_binding": total_energy >= graph.e_truck.B - 1e-6,
            }
        )

    return {
        "status": status,
        "objective": model.ObjVal,
        "model_stats": model_stats,
        "variable_group_counts": variable_group_counts,
        "routes": pd.DataFrame(route_rows),
        "h_service": pd.DataFrame(h_rows),
        "coverage": pd.DataFrame(coverage_rows),
        "material": pd.DataFrame(material_rows),
        "truck_delivery": pd.DataFrame(truck_delivery_rows),
        "star_stay": pd.DataFrame(star_stay_rows),
        "star_delivery": pd.DataFrame(star_rows),
        "rendezvous_delivery": pd.DataFrame(rendezvous_rows),
        "returns": pd.DataFrame(returns),
            "truck_schedule": sorted_or_empty(
            truck_schedule_rows,
            ["truck", "node", "arrival_time", "departure_time", "wait_time", "service_time", "grid_service_time", "star_service_time", "order_position", "node_type"],
            ["truck", "arrival_time", "departure_time"],
        ),
        "drone_schedule": sorted_or_empty(
            drone_schedule_rows,
            ["truck", "drone", "mode", "launch", "service", "recover", "start_time", "end_time", "occupied_time", "fly_time", "wait_time", "sorties"],
            ["truck", "drone", "start_time", "end_time"],
        ),
        "energy_breakdown": pd.DataFrame(energy_rows),
        "route_order": pd.DataFrame(route_length_rows),
    }


def save_solution(solution, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "routes" not in solution:
        return
    for key in ["routes", "h_service", "coverage", "material", "truck_delivery", "star_stay", "star_delivery", "rendezvous_delivery", "returns", "truck_schedule", "drone_schedule", "energy_breakdown", "route_order"]:
        target = out_dir / f"{key}.csv"
        try:
            solution[key].to_csv(target, index=False)
        except PermissionError:
            fallback = out_dir / f"{key}_latest.csv"
            solution[key].to_csv(fallback, index=False)
            print(f"[solver] file locked, wrote fallback output to {fallback}")
    if "model_stats" in solution:
        pd.DataFrame([solution["model_stats"]]).to_csv(out_dir / "model_stats.csv", index=False)
    if "variable_group_counts" in solution:
        pd.DataFrame(
            [{"variable_group": key, "count": value} for key, value in solution["variable_group_counts"].items()]
        ).sort_values("count", ascending=False).to_csv(out_dir / "variable_group_counts.csv", index=False)


if __name__ == "__main__":
    main()
    def sorted_or_empty(rows, columns, by):
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=columns)
        return df.sort_values(by).reset_index(drop=True)
