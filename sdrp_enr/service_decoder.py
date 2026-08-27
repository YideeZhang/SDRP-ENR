from __future__ import annotations

import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from sdrp_enr.data import ALPHA, BETA, RHO, SUCCESSOR_GAP_K, T_MAX_HOURS, ProblemData
from sdrp_enr.evaluator import Evaluator
from sdrp_enr.solution import Solution, RendezvousTask, StarTask, TruckDelivery

from .decoder_result import BIG_M, ServiceMILPResult


class FixedRouteServiceMILP:
    """Fixed-route service decoder with frozen light-order rendezvous.

    The fixed route positions
    replace the baseline model's route-order variables: a rendezvous sortie is
    allowed only when launch and recovery are on the same truck route and
    1 <= recovery_position - launch_position <= K.
    """

    def __init__(
        self,
        data: ProblemData,
        time_limit_sec: float | None = 10.0,
        output_flag: int = 0,
        successor_gap: int = SUCCESSOR_GAP_K,
        allow_star: bool = True,
        allow_rendezvous: bool = True,
        allow_microgrid_charging: bool = True,
        no_same_truck_ct_rendezvous: bool = False,
        no_truck_visited_ct_drone_service: bool = True,
        drone_range_multiplier: float = 1.0,
        drone_battery_multiplier: float = 1.0,
        alpha: float = ALPHA,
        beta: float = BETA,
    ) -> None:
        self.data = data
        self.time_limit_sec = time_limit_sec
        self.output_flag = output_flag
        self.successor_gap = int(successor_gap)
        self.allow_star = bool(allow_star)
        self.allow_rendezvous = bool(allow_rendezvous)
        self.allow_microgrid_charging = bool(allow_microgrid_charging)
        self.no_same_truck_ct_rendezvous = bool(no_same_truck_ct_rendezvous)
        self.no_truck_visited_ct_drone_service = bool(no_truck_visited_ct_drone_service)
        self.drone_range_multiplier = float(drone_range_multiplier)
        self.drone_battery_multiplier = float(drone_battery_multiplier)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaluator = Evaluator(data)

    @property
    def effective_drone_tmax_hours(self) -> float:
        return self.data.drone_tmax_hours * max(self.drone_range_multiplier, 0.0)

    @property
    def effective_drone_battery(self) -> float:
        return self.data.drone_battery * max(self.drone_battery_multiplier, 0.0)

    def effective_drone_energy(self, fly_time_hours: float) -> float:
        tmax = self.effective_drone_tmax_hours
        if tmax <= 0:
            return float("inf")
        return (fly_time_hours / tmax) * self.effective_drone_battery

    def solve(self, routes: dict[int, list[str]]) -> ServiceMILPResult:
        start = time.perf_counter()
        route_check = self._validate_fixed_routes(routes)
        if route_check:
            sol = Solution(routes={v: list(r) for v, r in routes.items()}, feasible=False, notes=route_check)
            return ServiceMILPResult(
                solution=sol,
                status="route_invalid",
                status_code=-1,
                objective=float("nan"),
                runtime_sec=time.perf_counter() - start,
                gap=float("nan"),
                notes=route_check,
            )
        try:
            return self._solve_model(routes, start)
        except gp.GurobiError as exc:
            sol = Solution(routes={v: list(r) for v, r in routes.items()}, feasible=False, notes=[str(exc)])
            return ServiceMILPResult(
                solution=sol,
                status="gurobi_error",
                status_code=-2,
                objective=float("nan"),
                runtime_sec=time.perf_counter() - start,
                gap=float("nan"),
                notes=[str(exc)],
            )

    def _validate_fixed_routes(self, routes: dict[int, list[str]]) -> list[str]:
        notes: list[str] = []
        for v in self.data.truck_ids:
            route = routes.get(v, [])
            if not route or route[0] != self.data.depot or route[-1] != self.data.depot:
                notes.append(f"truck {v}: route must start/end at depot")
                continue
            internal_depot_count = sum(1 for node in route[1:-1] if node == self.data.depot)
            if internal_depot_count:
                notes.append(f"truck {v}: internal depot count {internal_depot_count}")
            seen_anchors: set[str] = set()
            for a, b in zip(route, route[1:]):
                if (a, b) not in self.data.truck_arcs:
                    notes.append(f"truck {v}: illegal fixed arc {a}->{b}")
            for node in route[1:-1]:
                if node in self.data.launch_nodes:
                    if node in seen_anchors:
                        notes.append(f"truck {v}: duplicate route anchor {node}")
                    seen_anchors.add(node)
        return notes

    def _solve_model(self, routes: dict[int, list[str]], start: float) -> ServiceMILPResult:
        model = gp.Model("fixed_route_service_milp_v2")
        model.Params.OutputFlag = int(self.output_flag)
        if self.time_limit_sec is not None:
            model.Params.TimeLimit = float(self.time_limit_sec)

        position_keys = [(v, k) for v in self.data.truck_ids for k in range(len(routes[v]))]
        service_position_keys = [
            (v, k)
            for v in self.data.truck_ids
            for k, node in enumerate(routes[v])
            if 0 < k < len(routes[v]) - 1 and node in self.data.launch_nodes
        ]
        h_positions_by_node: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for v, k in service_position_keys:
            node = routes[v][k]
            if node in self.data.h_nodes:
                h_positions_by_node[node].append((v, k))
        truck_visited_ct = {
            node
            for route in routes.values()
            for node in route[1:-1]
            if node in self.data.c_truck
        }

        A = model.addVars(position_keys, lb=0.0, ub=T_MAX_HOURS, name="A")
        L = model.addVars(position_keys, lb=0.0, ub=T_MAX_HOURS, name="L")
        w = model.addVars(position_keys, lb=0.0, ub=T_MAX_HOURS, name="w")
        tau = model.addVars(
            [(v, k) for v, k in service_position_keys if routes[v][k] in self.data.h_nodes],
            lb=0.0,
            ub=T_MAX_HOURS,
            name="tau",
        )
        gamma = model.addVars(list(tau.keys()), vtype=GRB.BINARY, name="gamma")
        sigma = model.addVars(service_position_keys, lb=0.0, ub=T_MAX_HOURS, name="sigma")

        q_keys = [
            (n, v)
            for n in self.data.h_nodes + self.data.c_truck
            for v in self.data.truck_ids
            if n in routes[v][1:-1]
        ]
        q_truck = model.addVars(q_keys, lb=0.0, name="qT")

        y_keys: list[tuple[int, int, str, int]] = []
        star_time: dict[tuple[int, int, str, int], float] = {}
        star_energy: dict[tuple[int, int, str, int], float] = {}
        if self.allow_star:
            for v, k in service_position_keys:
                launch = routes[v][k]
                for i in self.data.c_nodes:
                    if i == launch:
                        continue
                    if self.no_truck_visited_ct_drone_service and i in truck_visited_ct:
                        continue
                    fly = self.data.drone_time(launch, i, launch)
                    if fly > self.effective_drone_tmax_hours + 1e-9:
                        continue
                    for d in self.data.drones_by_truck[v]:
                        key = (v, k, i, d)
                        y_keys.append(key)
                        star_time[key] = fly
                        star_energy[key] = self.effective_drone_energy(fly)
        y = model.addVars(y_keys, vtype=GRB.INTEGER, lb=0, name="y")

        r_keys: list[tuple[int, int, str, int, int]] = []
        rv_time: dict[tuple[int, int, str, int, int], float] = {}
        rv_energy: dict[tuple[int, int, str, int, int], float] = {}
        if self.allow_rendezvous:
            for v in self.data.truck_ids:
                route = routes[v]
                for k, launch in enumerate(route[:-1]):
                    if launch not in [self.data.depot] + self.data.launch_nodes:
                        continue
                    for l in range(k + 1, min(len(route), k + self.successor_gap + 1)):
                        recover = route[l]
                        if recover not in [self.data.depot] + self.data.launch_nodes:
                            continue
                        if launch == recover:
                            continue
                        for i in self.data.c_nodes:
                            if i == launch or i == recover:
                                continue
                            if self.no_truck_visited_ct_drone_service and i in truck_visited_ct:
                                continue
                            if self.no_same_truck_ct_rendezvous and i in self.data.c_truck and i in route[1:-1]:
                                continue
                            fly = self.data.drone_time(launch, i, recover)
                            energy = self.effective_drone_energy(fly)
                            if fly > self.effective_drone_tmax_hours + 1e-9:
                                continue
                            if energy > self.effective_drone_battery + 1e-9:
                                continue
                            for d in self.data.drones_by_truck[v]:
                                key = (v, k, i, l, d)
                                r_keys.append(key)
                                rv_time[key] = fly
                                rv_energy[key] = energy
        r = model.addVars(r_keys, vtype=GRB.BINARY, name="r")

        c_mat = model.addVars(self.data.h_nodes + self.data.c_nodes, lb=0.0, ub=1.0, name="c_mat")
        c_h = model.addVars(self.data.h_nodes, lb=0.0, ub=1.0, name="c_h")
        g_h = model.addVars(self.data.h_nodes, lb=0.0, ub=1.0, name="g_h")
        rem = model.addVars(self.data.h_nodes, lb=0.0, ub=T_MAX_HOURS, name="h_remaining")
        active = model.addVars(self.data.h_nodes, vtype=GRB.BINARY, name="h_arrival_active")
        h_charge_arrival = model.addVars(self.data.h_nodes, lb=0.0, ub=T_MAX_HOURS, name="h_charge_arrival")

        model.setObjective(
            self.alpha * gp.quicksum(float(self.data.nodes[h].p_m) * g_h[h] for h in self.data.h_nodes)
            + self.beta * gp.quicksum(self.data.population(n) * c_mat[n] for n in self.data.h_nodes + self.data.c_nodes),
            GRB.MAXIMIZE,
        )
        for h in self.data.h_nodes:
            model.addGenConstrPWL(c_h[h], g_h[h], [0.0, 0.5, 1.0], [0.0, 0.625, 1.0], name=f"pwl_{h}")

        for v in self.data.truck_ids:
            route = routes[v]
            model.addConstr(A[v, 0] == 0.0, name=f"A0_{v}")
            model.addConstr(L[v, 0] == 0.0, name=f"L0_{v}")
            model.addConstr(w[v, 0] == 0.0, name=f"w0_{v}")
            for k in range(1, len(route)):
                prev_node = route[k - 1]
                node = route[k]
                model.addConstr(A[v, k] == L[v, k - 1] + self.data.truck_time[(prev_node, node)], name=f"time_{v}_{k}")
                if node == self.data.depot:
                    model.addConstr(L[v, k] == A[v, k], name=f"return_L_{v}_{k}")
                    model.addConstr(A[v, k] <= T_MAX_HOURS, name=f"return_cap_{v}_{k}")
                elif node in self.data.h_nodes:
                    model.addConstr(L[v, k] == A[v, k] + tau[v, k] + sigma[v, k] + w[v, k], name=f"leave_h_{v}_{k}")
                elif node in self.data.c_truck:
                    model.addConstr(L[v, k] == A[v, k] + sigma[v, k] + w[v, k], name=f"leave_cT_{v}_{k}")
                else:
                    model.addConstr(L[v, k] == A[v, k], name=f"leave_passthrough_{v}_{k}")
                    model.addConstr(w[v, k] == 0.0, name=f"w_passthrough_{v}_{k}")

        for v, k in service_position_keys:
            launch = routes[v][k]
            for d in self.data.drones_by_truck[v]:
                expr = gp.quicksum(star_time[key] * y[key] for key in y_keys if key[0] == v and key[1] == k and key[3] == d)
                if launch in self.data.h_nodes:
                    model.addConstr(expr <= tau[v, k] + sigma[v, k], name=f"star_time_h_{v}_{k}_{d}")
                else:
                    model.addConstr(expr <= sigma[v, k], name=f"star_time_cT_{v}_{k}_{d}")

        for v, k in tau.keys():
            model.addConstr(tau[v, k] <= T_MAX_HOURS * gamma[v, k], name=f"tau_gamma_{v}_{k}")
            if not self.allow_microgrid_charging:
                model.addConstr(tau[v, k] == 0.0, name=f"tau_disabled_{v}_{k}")
                model.addConstr(gamma[v, k] == 0.0, name=f"gamma_disabled_{v}_{k}")
        for h in self.data.h_nodes:
            h_occurrences = h_positions_by_node.get(h, [])
            if h_occurrences:
                model.addConstr(gp.quicksum(gamma[v, k] for v, k in h_occurrences) <= 1, name=f"single_charge_owner_{h}")
                for v, k in h_occurrences:
                    model.addConstr(h_charge_arrival[h] >= A[v, k] - BIG_M * (1 - gamma[v, k]), name=f"h_chg_arr_lb_{h}_{v}_{k}")
                    model.addConstr(h_charge_arrival[h] <= A[v, k] + BIG_M * (1 - gamma[v, k]), name=f"h_chg_arr_ub_{h}_{v}_{k}")
            else:
                model.addConstr(c_h[h] == 0.0, name=f"unvisited_h_cov_{h}")

        for key in r_keys:
            v, k, _i, l, d = key
            recover = routes[v][l]
            fly = rv_time[key]
            if recover == self.data.depot:
                model.addConstr(A[v, l] >= L[v, k] + fly - BIG_M * (1 - r[key]), name=f"r_sync_return_{v}_{k}_{l}_{d}")
            elif recover in self.data.h_nodes:
                model.addConstr(
                    w[v, l] >= L[v, k] + fly - A[v, l] - tau[v, l] - sigma[v, l] - BIG_M * (1 - r[key]),
                    name=f"r_wait_h_{v}_{k}_{l}_{d}",
                )
                model.addConstr(L[v, l] >= L[v, k] + fly - BIG_M * (1 - r[key]), name=f"r_sync_h_{v}_{k}_{l}_{d}")
            else:
                model.addConstr(
                    w[v, l] >= L[v, k] + fly - A[v, l] - sigma[v, l] - BIG_M * (1 - r[key]),
                    name=f"r_wait_cT_{v}_{k}_{l}_{d}",
                )
                model.addConstr(L[v, l] >= L[v, k] + fly - BIG_M * (1 - r[key]), name=f"r_sync_cT_{v}_{k}_{l}_{d}")

        launch_groups: dict[tuple[int, int, int], list[tuple[int, int, str, int, int]]] = defaultdict(list)
        recover_groups: dict[tuple[int, int, int], list[tuple[int, int, str, int, int]]] = defaultdict(list)
        for key in r_keys:
            v, k, _i, l, d = key
            launch_groups[(v, k, d)].append(key)
            recover_groups[(v, l, d)].append(key)
        for group_key, keys in launch_groups.items():
            model.addConstr(gp.quicksum(r[key] for key in keys) <= 1, name=f"r_launch_unique_{group_key}")
        for group_key, keys in recover_groups.items():
            model.addConstr(gp.quicksum(r[key] for key in keys) <= 1, name=f"r_recover_unique_{group_key}")

        for n in self.data.h_nodes + self.data.c_nodes:
            truck_part = gp.quicksum(q_truck[n, v] for v in self.data.truck_ids if (n, v) in q_truck)
            star_part = 0.0
            rv_part = 0.0
            if n in self.data.c_nodes:
                star_part = gp.quicksum(self.data.drone_payload * y[key] for key in y_keys if key[2] == n)
                rv_part = gp.quicksum(self.data.drone_payload * r[key] for key in r_keys if key[2] == n)
            demand = max(self.data.material_demand(n), 1e-9)
            model.addConstr(truck_part + star_part + rv_part <= self.data.material_demand(n), name=f"mat_cap_{n}")
            model.addConstr(truck_part + star_part + rv_part == demand * c_mat[n], name=f"mat_cov_{n}")

        for n, v in q_keys:
            model.addConstr(q_truck[n, v] <= self.data.material_demand(n), name=f"q_visit_{n}_{v}")

        for h in self.data.h_nodes:
            h_occurrences = h_positions_by_node.get(h, [])
            supplied = gp.quicksum(float(self.data.nodes[h].P_o) * tau[v, k] for v, k in h_occurrences)
            demand_rate = float(self.data.nodes[h].p_m) * RHO
            e_dem = max(self.data.h_energy_demand.get(h, 0.0), 1e-9)
            raw = float(self.data.nodes[h].R) - h_charge_arrival[h]
            model.addConstr(rem[h] >= raw, name=f"rem_lb_raw_{h}")
            model.addConstr(rem[h] <= raw + BIG_M * (1 - active[h]), name=f"rem_ub_raw_{h}")
            model.addConstr(rem[h] <= BIG_M * active[h], name=f"rem_ub_active_{h}")
            model.addConstr(supplied <= demand_rate * rem[h], name=f"arrival_cap_{h}")
            model.addConstr(supplied == e_dem * c_h[h], name=f"h_energy_cov_{h}")

        for v in self.data.truck_ids:
            truck_load = gp.quicksum(q_truck[n, v] for n in self.data.h_nodes + self.data.c_truck if (n, v) in q_truck)
            truck_load += gp.quicksum(self.data.drone_payload * y[key] for key in y_keys if key[0] == v)
            truck_load += gp.quicksum(self.data.drone_payload * r[key] for key in r_keys if key[0] == v)
            model.addConstr(truck_load <= self.data.truck_capacity, name=f"truck_capacity_{v}")

            drive_energy = sum(self.data.truck_energy[(a, b)] for a, b in zip(routes[v], routes[v][1:]))
            grid_energy = gp.quicksum(float(self.data.nodes[routes[v][k]].P_o) * tau[v, k] for _v, k in tau.keys() if _v == v)
            drone_energy = gp.quicksum(star_energy[key] * y[key] for key in y_keys if key[0] == v)
            drone_energy += gp.quicksum(rv_energy[key] * r[key] for key in r_keys if key[0] == v)
            model.addConstr(drive_energy + grid_energy + drone_energy <= self.data.truck_battery, name=f"battery_{v}")

        model.optimize()

        status_code = int(model.Status)
        status = self._status_name(status_code)
        gap = float("nan")
        if model.SolCount > 0:
            try:
                gap = float(model.MIPGap)
            except gp.GurobiError:
                gap = float("nan")
            sol = self._extract_solution(routes, q_keys, q_truck, tau, y_keys, y, r_keys, r)
            sol = self.evaluator.evaluate(sol)
            objective = float(model.ObjVal)
            sol.objective = objective
        else:
            sol = Solution(routes={v: list(rte) for v, rte in routes.items()}, feasible=False, notes=[status])
            objective = float("nan")

        return ServiceMILPResult(
            solution=sol,
            status=status,
            status_code=status_code,
            objective=objective,
            runtime_sec=time.perf_counter() - start,
            gap=gap,
            notes=list(sol.notes),
            num_vars=int(model.NumVars),
            num_constrs=int(model.NumConstrs),
        )

    def _extract_solution(self, routes, q_keys, q_truck, tau, y_keys, y, r_keys, r) -> Solution:
        sol = Solution(routes={v: list(rte) for v, rte in routes.items()})
        for n, v in q_keys:
            value = float(q_truck[n, v].X)
            if value > 1e-7:
                sol.truck_deliveries.append(TruckDelivery(node=n, truck=v, quantity=value))
        for v, k in tau.keys():
            value = float(tau[v, k].X)
            if value > 1e-7:
                sol.tau[(routes[v][k], v)] = value
        for key in y_keys:
            value = int(round(float(y[key].X)))
            if value > 0:
                v, k, service, d = key
                sol.star_tasks.append(StarTask(launch=routes[v][k], service=service, truck=v, drone=d, sorties=value))
        for key in r_keys:
            value = int(round(float(r[key].X)))
            if value > 0:
                v, k, service, l, d = key
                sol.rendezvous_tasks.append(
                    RendezvousTask(launch=routes[v][k], service=service, recover=routes[v][l], truck=v, drone=d)
                )
        return sol

    def _status_name(self, status_code: int) -> str:
        names = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INTERRUPTED: "INTERRUPTED",
        }
        return names.get(status_code, str(status_code))
