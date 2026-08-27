from __future__ import annotations

from collections import defaultdict

from .data import ProblemData, ALPHA, BETA, RHO, SUCCESSOR_GAP_K, T_MAX_HOURS, microgrid_utility
from .solution import Solution, TruckDelivery, TruckSchedule


class Evaluator:
    def __init__(self, data: ProblemData) -> None:
        self.data = data

    def evaluate(self, solution: Solution) -> Solution:
        sol = solution.copy()
        sol.notes = []
        sol.validation_metrics = self.validation_metrics(sol)
        self._validate_routes(sol)
        self._remove_orphans(sol)
        self._sync_truck_direct_deliveries(sol)
        sol.validation_metrics = self.validation_metrics(sol)
        self._rebuild_all_schedules(sol)
        self._score_and_validate_resources(sol)
        return sol

    def validation_metrics(self, sol: Solution) -> dict[str, float]:
        tau_positive = [(h, v) for (h, v), value in sol.tau.items() if value > 1e-9]
        route_arc_violations = 0
        duplicate_h_within_route = 0
        internal_depot_count = 0
        for v in self.data.truck_ids:
            route = sol.routes.get(v, [])
            route_seen_h: set[str] = set()
            for node in route[1:-1]:
                if node == self.data.depot:
                    internal_depot_count += 1
                if node in self.data.h_nodes:
                    if node in route_seen_h:
                        duplicate_h_within_route += 1
                    route_seen_h.add(node)
            for a, b in zip(route, route[1:]):
                if (a, b) not in self.data.truck_arcs:
                    route_arc_violations += 1
        unanchored_star = sum(
            1
            for item in sol.star_tasks
            if item.launch not in sol.visited_by_truck(item.truck)
            or item.launch not in self.data.launch_nodes
            or item.service not in self.data.c_nodes
        )
        unanchored_rv = sum(
            1
            for item in sol.rendezvous_tasks
            if not self._rv_anchor_ok(sol, item.truck, item.launch, item.recover)
            or item.service not in self.data.c_nodes
        )
        rv_launch_usage: dict[tuple[int, int, str], int] = defaultdict(int)
        rv_recover_usage: dict[tuple[int, int, str], int] = defaultdict(int)
        for item in sol.rendezvous_tasks:
            rv_launch_usage[(item.truck, item.drone, item.launch)] += 1
            rv_recover_usage[(item.truck, item.drone, item.recover)] += 1
        rv_availability_conflicts = sum(max(0, count - 1) for count in rv_launch_usage.values())
        rv_availability_conflicts += sum(max(0, count - 1) for count in rv_recover_usage.values())
        tau_on_unvisited = sum(
            1
            for h, v in tau_positive
            if h not in self.data.h_nodes or h not in sol.visited_by_truck(v)
        )
        duplicate_h_tau = 0
        for h in self.data.h_nodes:
            tau_trucks = {v for th, v in tau_positive if th == h}
            if len(tau_trucks) > 1:
                duplicate_h_tau += len(tau_trucks) - 1
        h_energy_cap_violation = 0
        if sol.schedules:
            for h in self.data.h_nodes:
                tau_trucks = [v for th, v in tau_positive if th == h]
                if len(tau_trucks) != 1:
                    continue
                owner = tau_trucks[0]
                arrival = sol.schedules.get(owner, TruckSchedule()).arrival.get(h, T_MAX_HOURS)
                tau = sol.tau.get((h, owner), 0.0)
                lam = float(self.data.nodes[h].p_m) * RHO
                cap = lam * max(float(self.data.nodes[h].R) - arrival, 0.0)
                supplied = float(self.data.nodes[h].P_o) * tau
                if supplied > cap + 1e-7 or supplied > self.data.h_energy_demand.get(h, 0.0) + 1e-7:
                    h_energy_cap_violation += 1
        return {
            "duplicate_h_tau_count": float(duplicate_h_tau),
            "unanchored_star_count": float(unanchored_star),
            "unanchored_rendezvous_count": float(unanchored_rv),
            "tau_on_unvisited_h_count": float(tau_on_unvisited),
            "internal_depot_count": float(internal_depot_count),
            "duplicate_h_within_route_count": float(duplicate_h_within_route),
            "route_arc_violation_count": float(route_arc_violations),
            "h_energy_cap_violation_count": float(h_energy_cap_violation),
            "rendezvous_drone_conflict_count": float(rv_availability_conflicts),
        }

    def _h_visit_owners(self, sol: Solution) -> dict[str, list[int]]:
        owners: dict[str, list[int]] = {}
        for h in self.data.h_nodes:
            owners[h] = [v for v in self.data.truck_ids if h in sol.routes.get(v, [])]
        return owners

    def _validate_routes(self, sol: Solution) -> None:
        for v in self.data.truck_ids:
            route = sol.routes.setdefault(v, [self.data.depot, self.data.depot])
            if not route or route[0] != self.data.depot or route[-1] != self.data.depot:
                sol.notes.append(f"truck {v}: route must start/end at D0")
            if any(node == self.data.depot for node in route[1:-1]):
                sol.notes.append(f"truck {v}: internal depot is not allowed")
            route_seen_h: set[str] = set()
            for node in route[1:-1]:
                if node in self.data.h_nodes:
                    if node in route_seen_h:
                        sol.notes.append(f"truck {v}: duplicate H visit within route {node}")
                    route_seen_h.add(node)
            for a, b in zip(route, route[1:]):
                if (a, b) not in self.data.truck_arcs:
                    sol.notes.append(f"truck {v}: missing truck arc {a}->{b}")

    def _remove_orphans(self, sol: Solution) -> None:
        sol.truck_deliveries = []
        sol.star_tasks = [
            item for item in sol.star_tasks
            if item.launch in sol.visited_by_truck(item.truck)
            and item.launch in self.data.launch_nodes
            and item.service in self.data.c_nodes
        ]
        sol.rendezvous_tasks = [
            item for item in sol.rendezvous_tasks
            if self._rv_anchor_ok(sol, item.truck, item.launch, item.recover)
            and item.service in self.data.c_nodes
        ]
        sol.tau = {
            (h, v): value
            for (h, v), value in sol.tau.items()
            if value > 1e-9 and h in self.data.h_nodes and h in sol.visited_by_truck(v)
        }

    def _sync_truck_direct_deliveries(self, sol: Solution) -> None:
        deliveries: list[TruckDelivery] = []
        served_nodes: set[str] = set()
        for v in self.data.truck_ids:
            for node in sol.routes.get(v, [])[1:-1]:
                if node in served_nodes:
                    continue
                if node not in self.data.h_nodes and node not in self.data.c_truck:
                    continue
                demand = self.data.material_demand(node)
                if demand <= 1e-9:
                    continue
                deliveries.append(TruckDelivery(node=node, truck=v, quantity=demand))
                served_nodes.add(node)
        sol.truck_deliveries = deliveries

    def cleanup_orphan_services(self, solution: Solution) -> Solution:
        sol = solution.copy()
        self._remove_orphans(sol)
        self._sync_truck_direct_deliveries(sol)
        sol.validation_metrics = self.validation_metrics(sol)
        return sol

    def _rv_anchor_ok(self, sol: Solution, truck: int, launch: str, recover: str) -> bool:
        route = sol.routes.get(truck, [])
        if launch not in route or recover not in route:
            return False
        a_pos = route.index(launch)
        b_pos = len(route) - 1 if recover == self.data.depot else route.index(recover)
        return 1 <= b_pos - a_pos <= SUCCESSOR_GAP_K

    def _rebuild_all_schedules(self, sol: Solution) -> None:
        sol.schedules = {v: self.rebuild_truck_schedule(sol, v) for v in self.data.truck_ids}

    def rebuild_truck_schedule(self, sol: Solution, truck: int) -> TruckSchedule:
        route = sol.routes.get(truck, [self.data.depot, self.data.depot])
        sch = TruckSchedule()
        if not route:
            return sch

        star_time_by_node_drone: dict[tuple[str, int], float] = defaultdict(float)
        for task in sol.star_tasks:
            if task.truck != truck:
                continue
            fly = self.data.drone_time(task.launch, task.service, task.launch)
            star_time_by_node_drone[(task.launch, task.drone)] += fly * max(0, task.sorties)

        rv_by_recover: dict[str, list] = defaultdict(list)
        for task in sol.rendezvous_tasks:
            if task.truck == truck:
                rv_by_recover[task.recover].append(task)

        current = 0.0
        for idx, node in enumerate(route):
            if idx == 0:
                sch.arrival[node] = 0.0
                sch.departure[node] = 0.0
                sch.sigma[node] = 0.0
                sch.waiting[node] = 0.0
                continue
            prev = route[idx - 1]
            current = sch.departure.get(prev, current) + self.data.truck_time.get((prev, node), float("inf"))
            if node == self.data.depot:
                # Recovery at depot can extend return time.
                rv_wait = 0.0
                for task in rv_by_recover.get(node, []):
                    launch_depart = sch.departure.get(task.launch, 0.0)
                    fly = self.data.drone_time(task.launch, task.service, task.recover)
                    rv_wait = max(rv_wait, launch_depart + fly - current)
                sch.return_time = current + max(0.0, rv_wait)
                sch.arrival[node] = sch.return_time
                sch.departure[node] = sch.return_time
                sch.waiting[node] = max(0.0, rv_wait)
                continue

            sch.arrival[node] = current
            tau = sol.tau.get((node, truck), 0.0) if node in self.data.h_nodes else 0.0
            sigma = max([0.0] + [t for (launch, _d), t in star_time_by_node_drone.items() if launch == node])
            if node in self.data.h_nodes:
                sigma = max(0.0, sigma - tau)
            sch.sigma[node] = sigma

            base_departure = current + tau + sigma
            rv_wait = 0.0
            for task in rv_by_recover.get(node, []):
                launch_depart = sch.departure.get(task.launch, 0.0)
                fly = self.data.drone_time(task.launch, task.service, task.recover)
                rv_wait = max(rv_wait, launch_depart + fly - base_departure)
            sch.waiting[node] = max(0.0, rv_wait)
            sch.departure[node] = base_departure + sch.waiting[node]
        return sch

    def _score_and_validate_resources(self, sol: Solution) -> None:
        material_qty: dict[str, float] = defaultdict(float)
        truck_load: dict[int, float] = defaultdict(float)
        drone_energy: dict[int, float] = defaultdict(float)
        validation_errors: list[str] = []

        for item in sol.truck_deliveries:
            qty = max(0.0, item.quantity)
            material_qty[item.node] += qty
            truck_load[item.truck] += qty

        for item in sol.star_tasks:
            fly = self.data.drone_time(item.launch, item.service, item.launch)
            if fly > self.data.drone_tmax_hours + 1e-9:
                validation_errors.append(f"star endurance exceeded {item.launch}->{item.service}")
            qty = self.data.drone_payload * max(0, item.sorties)
            material_qty[item.service] += qty
            truck_load[item.truck] += qty
            drone_energy[item.truck] += self.data.drone_energy(fly) * max(0, item.sorties)

        for item in sol.rendezvous_tasks:
            fly = self.data.drone_time(item.launch, item.service, item.recover)
            if fly > self.data.drone_tmax_hours + 1e-9:
                validation_errors.append(f"rendezvous endurance exceeded {item.launch}->{item.service}->{item.recover}")
            material_qty[item.service] += self.data.drone_payload
            truck_load[item.truck] += self.data.drone_payload
            drone_energy[item.truck] += self.data.drone_energy(fly)

        rv_launch_usage: dict[tuple[int, int, str], int] = defaultdict(int)
        rv_recover_usage: dict[tuple[int, int, str], int] = defaultdict(int)
        for item in sol.rendezvous_tasks:
            rv_launch_usage[(item.truck, item.drone, item.launch)] += 1
            rv_recover_usage[(item.truck, item.drone, item.recover)] += 1
        for key, count in rv_launch_usage.items():
            if count > 1:
                validation_errors.append(f"rendezvous launch availability conflict {key}")
        for key, count in rv_recover_usage.items():
            if count > 1:
                validation_errors.append(f"rendezvous recovery availability conflict {key}")

        for node in self.data.h_nodes + self.data.c_nodes:
            if material_qty.get(node, 0.0) > self.data.material_demand(node) + 1e-9:
                validation_errors.append(f"{node}: material over-delivery")

        for node in self.data.h_nodes + self.data.c_nodes:
            material_qty[node] = min(material_qty.get(node, 0.0), self.data.material_demand(node))

        microgrid_score = 0.0
        grid_energy_by_truck: dict[int, float] = defaultdict(float)
        for h in self.data.h_nodes:
            tau_trucks = [v for (th, v), value in sol.tau.items() if th == h and value > 1e-9]
            if len(tau_trucks) > 1:
                validation_errors.append(f"{h}: duplicate positive tau trucks {tau_trucks}")
                supplied = 0.0
            elif not tau_trucks:
                supplied = 0.0
            else:
                owner = tau_trucks[0]
                arrival = sol.schedules[owner].arrival.get(h, T_MAX_HOURS)
                lam = float(self.data.nodes[h].p_m) * RHO
                arrival_cap = lam * max(float(self.data.nodes[h].R) - arrival, 0.0)
                raw_supplied = float(self.data.nodes[h].P_o) * sol.tau.get((h, owner), 0.0)
                if raw_supplied > arrival_cap + 1e-7:
                    validation_errors.append(f"{h}: H energy arrival cap exceeded")
                if raw_supplied > self.data.h_energy_demand.get(h, 0.0) + 1e-7:
                    validation_errors.append(f"{h}: H total energy demand exceeded")
                supplied = min(raw_supplied, self.data.h_energy_demand.get(h, 0.0), arrival_cap)
                grid_energy_by_truck[owner] += raw_supplied
            denom = self.data.h_energy_demand.get(h, 0.0)
            c = supplied / denom if denom > 1e-9 else 0.0
            microgrid_score += ALPHA * float(self.data.nodes[h].p_m) * microgrid_utility(c)

        material_score = 0.0
        for node in self.data.h_nodes + self.data.c_nodes:
            demand = self.data.material_demand(node)
            ratio = material_qty.get(node, 0.0) / demand if demand > 1e-9 else 0.0
            material_score += BETA * self.data.population(node) * min(1.0, ratio)

        for v in self.data.truck_ids:
            route = sol.routes.get(v, [])
            drive_energy = sum(self.data.truck_energy.get((a, b), float("inf")) for a, b in zip(route, route[1:]))
            total_energy = drive_energy + drone_energy[v] + grid_energy_by_truck[v]
            if truck_load[v] > self.data.truck_capacity + 1e-9:
                validation_errors.append(f"truck {v}: material capacity exceeded")
            if total_energy > self.data.truck_battery + 1e-9:
                validation_errors.append(f"truck {v}: battery exceeded")
            if sol.schedules.get(v, TruckSchedule()).return_time > T_MAX_HOURS + 1e-9:
                validation_errors.append(f"truck {v}: return time exceeded")

        sol.served_material_score = material_score
        sol.microgrid_score = microgrid_score
        sol.objective = material_score + microgrid_score
        all_notes = sol.notes + validation_errors
        sol.notes = all_notes
        sol.validation_metrics = self.validation_metrics(sol)
        sol.feasible = len(all_notes) == 0
        sol.feasibility_status = "feasible" if sol.feasible else "partial_or_infeasible"
