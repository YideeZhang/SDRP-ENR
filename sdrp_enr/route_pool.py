from __future__ import annotations
import math
import random
from .data import BETA, T_MAX_HOURS
from .solution import Solution
from .pool_base import (RouteCandidate, RoutePoolBase, globally_visited_ct,
    is_duplicate_h_bridge_candidate, route_metrics, routes_to_string,
    signature_to_string, valid_h_normal_routes)

GEN_BALANCED = "BalancedBiasedRandomizedInsertion"

class RoutePoolGenerator(RoutePoolBase):
    def __init__(
        self,
        *args,
        balanced_random_count: int = 100,
        rcl_size: int = 15,
        ct_target_ratio: float = 0.25,
        max_ct_chain_len_preference: int = 3,
        target_route_time_utilization: float = 0.85,
        target_energy_utilization: float = 0.85,
        insertion_temperature: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.balanced_random_count = balanced_random_count
        self.rcl_size = rcl_size
        self.ct_target_ratio = ct_target_ratio
        self.max_ct_chain_len_preference = max_ct_chain_len_preference
        self.target_route_time_utilization = target_route_time_utilization
        self.target_energy_utilization = target_energy_utilization
        self.insertion_temperature = insertion_temperature
        self.step_trace_rows: list[dict] = []
        self.balanced_final_rows: list[dict] = []

    def generate_pool(self) -> list[RouteCandidate]:
        raw: list[RouteCandidate] = []
        h_only = self.generate_h_only()
        raw.extend([cand for cand in h_only if cand.generator == "H_only_current_initial"])
        raw.extend(self.generate_edge_replacement())
        raw.extend(self.generate_randomized())
        raw.extend(self.generate_balanced_biased_randomized())
        if self.enable_integrated_h_bridge:
            raw.extend(self.generate_integrated_h_bridge_variants(raw))

        unique: dict[tuple[tuple[int, tuple[str, ...]], ...], RouteCandidate] = {}
        for cand in raw:
            self.generated_by_generator[cand.generator] = self.generated_by_generator.get(cand.generator, 0) + 1
            if not valid_h_normal_routes(self.data, self.normalized_routes(cand.routes)):
                self.invalid_by_generator[cand.generator] = self.invalid_by_generator.get(cand.generator, 0) + 1
                continue
            signature = self.route_signature(cand.routes)
            if signature not in unique:
                unique[signature] = cand
            if len(unique) >= self.max_pool_size:
                break
        return list(unique.values())

    def generate_balanced_biased_randomized(self) -> list[RouteCandidate]:
        candidates: list[RouteCandidate] = []
        for candidate_id in range(self.balanced_random_count):
            sol = Solution(routes={v: [self.data.depot, self.data.depot] for v in self.data.truck_ids})
            failed_h = self._balanced_phase(sol, "H", list(self.data.h_nodes), candidate_id)
            self._balanced_ct_phase(sol, candidate_id)
            routes = self.normalized_routes(sol.routes)
            metrics = self._extended_route_metrics(routes)
            self.balanced_final_rows.append(
                {
                    "scenario": self.scenario,
                    "candidate_id": candidate_id,
                    "failed_h_count": failed_h,
                    **metrics,
                }
            )
            candidates.append(
                RouteCandidate(
                    self.scenario,
                    GEN_BALANCED,
                    routes,
                    {"candidate_id": candidate_id, "failed_h_count": failed_h},
                )
            )
        return candidates

    def _balanced_phase(self, sol: Solution, phase: str, targets: list[str], candidate_id: int) -> int:
        failed = 0
        remaining = list(targets)
        while remaining:
            scored = self._scored_insertion_candidates(sol, remaining, phase)
            if not scored:
                failed += len(
                    [
                        node
                        for node in remaining
                        if not (node in self.data.h_nodes and any(node in route[1:-1] for route in sol.routes.values()))
                        and node not in self.visited_anchors(sol)
                    ]
                )
                break
            selected, rank = self._choose_from_rcl(scored)
            self.apply_insertion(sol, selected["option"])
            self._record_step(sol, selected, candidate_id, phase, rank)
            remaining = [node for node in remaining if node not in self.visited_anchors(sol)]
        return failed

    def _balanced_bridge_phase(self, sol: Solution, candidate_id: int) -> None:
        attempts = 0
        max_attempts = max(1, len(self.data.h_nodes))
        while attempts < max_attempts:
            targets = [h for h in self.data.h_nodes if any(is_duplicate_h_bridge_candidate(self.data, sol.routes, int(v), h) for v in self.data.truck_ids)]
            if not targets:
                break
            scored = self._scored_insertion_candidates(sol, targets, "H_bridge")
            if not scored:
                break
            selected, rank = self._choose_from_rcl(scored)
            self.apply_insertion(sol, selected["option"])
            self._record_step(sol, selected, candidate_id, "H_bridge", rank)
            attempts += 1

    def _balanced_ct_phase(self, sol: Solution, candidate_id: int) -> None:
        max_attempts = 2 * len(self.data.c_truck)
        attempts = 0
        while attempts < max_attempts:
            anchors = self.visited_anchors(sol)
            anchor_count = len(anchors)
            ct_count = sum(1 for node in anchors if node in self.data.c_truck)
            if anchor_count > 0 and ct_count >= self.ct_target_ratio * anchor_count:
                break
            time_utils, energy_utils = self._utilizations(sol.routes)
            if time_utils and all(value >= self.target_route_time_utilization for value in time_utils.values()):
                break
            if energy_utils and all(value >= self.target_energy_utilization for value in energy_utils.values()):
                break
            targets = list(self.rank_ct_nodes())
            scored = self._scored_insertion_candidates(sol, targets, "CT")
            if not scored:
                break
            selected, rank = self._choose_from_rcl(scored)
            self.apply_insertion(sol, selected["option"])
            self._record_step(sol, selected, candidate_id, "CT", rank)
            attempts += 1

    def _scored_insertion_candidates(self, sol: Solution, targets: list[str], phase: str) -> list[dict]:
        raw: list[dict] = []
        for anchor in targets:
            if phase != "H_bridge" and anchor in self.data.h_nodes and anchor in self.visited_anchors(sol):
                continue
            for option in self.insertion_options(sol, anchor):
                trial_routes = self._routes_after_option(sol, option)
                truck = int(option["truck"])
                time_utils, energy_utils = self._utilizations(trial_routes)
                route_times = self._route_times(trial_routes)
                route_energies = self._route_energies(trial_routes)
                raw.append(
                    {
                        "anchor": anchor,
                        "anchor_type": self._anchor_type(anchor),
                        "phase": phase,
                        "option": option,
                        "opportunity_raw": self._opportunity(anchor, sol),
                        "resource_target_reward_raw": -abs(time_utils.get(truck, 0.0) - self.target_route_time_utilization)
                        - abs(energy_utils.get(truck, 0.0) - self.target_energy_utilization),
                        "balance_reward_raw": -self._std(list(route_times.values())) - 0.02 * self._std(list(route_energies.values())),
                        "ct_chain_reward_raw": self._ct_chain_reward(trial_routes, truck, anchor),
                        "bridge_reward_raw": 1.0 if is_duplicate_h_bridge_candidate(self.data, sol.routes, truck, anchor) else 0.0,
                        "excessive_cost_penalty_raw": -10.0 * max(0.0, time_utils.get(truck, 0.0) - 1.0)
                        - 10.0 * max(0.0, energy_utils.get(truck, 0.0) - 1.0),
                        "add_time": float(option["add_time"]),
                        "add_energy": float(option["add_energy"]),
                    }
                )
        if not raw:
            return []
        self._attach_rank_scores(raw)
        for item in raw:
            item["final_score"] = (
                1.0 * item["opportunity_score"]
                + 0.8 * item["resource_target_reward"]
                + 0.8 * item["balance_reward"]
                + 0.3 * item["ct_chain_reward"]
                + 0.15 * item["bridge_reward"]
                - 0.3 * item["add_time_score"]
                - 0.2 * item["add_energy_score"]
                + item["excessive_cost_penalty_raw"]
            )
        raw.sort(key=lambda item: item["final_score"], reverse=True)
        return raw

    def _attach_rank_scores(self, rows: list[dict]) -> None:
        for raw_key, score_key in [
            ("opportunity_raw", "opportunity_score"),
            ("resource_target_reward_raw", "resource_target_reward"),
            ("balance_reward_raw", "balance_reward"),
            ("ct_chain_reward_raw", "ct_chain_reward"),
            ("bridge_reward_raw", "bridge_reward"),
            ("add_time", "add_time_score"),
            ("add_energy", "add_energy_score"),
        ]:
            values = [float(row[raw_key]) for row in rows]
            ranks = self._rank_scores(values)
            for row, rank in zip(rows, ranks):
                row[score_key] = rank

    def _rank_scores(self, values: list[float]) -> list[float]:
        if len(values) <= 1:
            return [1.0 for _ in values]
        order = sorted(range(len(values)), key=lambda idx: values[idx])
        ranks = [0.0] * len(values)
        for rank, idx in enumerate(order):
            ranks[idx] = rank / (len(values) - 1)
        return ranks

    def _choose_from_rcl(self, scored: list[dict]) -> tuple[dict, int]:
        rcl = scored[: min(self.rcl_size, len(scored))]
        if len(rcl) == 1:
            return rcl[0], 1
        max_score = max(row["final_score"] for row in rcl)
        temp = max(self.insertion_temperature, 1e-9)
        weights = [math.exp((row["final_score"] - max_score) / temp) for row in rcl]
        total = sum(weights)
        pick = self.rng.random() * total
        acc = 0.0
        for idx, (row, weight) in enumerate(zip(rcl, weights), start=1):
            acc += weight
            if pick <= acc:
                return row, idx
        return rcl[-1], len(rcl)

    def _record_step(self, sol: Solution, selected: dict, candidate_id: int, phase: str, rcl_rank: int) -> None:
        metrics = self._extended_route_metrics(self.normalized_routes(sol.routes))
        option = selected["option"]
        route = sol.routes[int(option["truck"])]
        pos = int(option["pos"])
        insert_position = f"{route[max(pos - 2, 0)]}->{route[min(pos - 1, len(route) - 1)]}@{pos}"
        self.step_trace_rows.append(
            {
                "scenario": self.scenario,
                "candidate_id": candidate_id,
                "step": len([r for r in self.step_trace_rows if r["scenario"] == self.scenario and r["candidate_id"] == candidate_id]) + 1,
                "phase": phase,
                "selected_anchor": selected["anchor"],
                "selected_anchor_type": selected["anchor_type"],
                "truck": int(option["truck"]),
                "insert_position": insert_position,
                "add_time": selected["add_time"],
                "add_energy": selected["add_energy"],
                "opportunity_score": selected["opportunity_score"],
                "resource_target_reward": selected["resource_target_reward"],
                "balance_reward": selected["balance_reward"],
                "ct_chain_reward": selected["ct_chain_reward"],
                "bridge_reward": selected.get("bridge_reward", 0.0),
                "bridge_reward_raw": selected.get("bridge_reward_raw", 0.0),
                "excessive_cost_penalty": selected["excessive_cost_penalty_raw"],
                "final_score": selected["final_score"],
                "rcl_rank": rcl_rank,
                "route_time_utilization_after": metrics["max_route_time_utilization"],
                "energy_utilization_after": metrics["max_energy_utilization"],
                "route_balance_std_after": metrics["route_travel_time_balance_std"],
                "ct_anchor_count_after": metrics["ct_anchor_count"],
                "ct_chain_count_after": metrics["ct_chain_count"],
            }
        )

    def _routes_after_option(self, sol: Solution, option: dict) -> dict[int, list[str]]:
        routes = {v: list(route) for v, route in sol.routes.items()}
        v = int(option["truck"])
        pos = int(option["pos"])
        segment = list(option["segment"])
        route = routes[v]
        routes[v] = route[: pos - 1] + segment + route[pos + 1 :]
        return self.normalized_routes(routes)

    def _opportunity(self, anchor: str, sol: Solution) -> float:
        direct = BETA * self.data.population(anchor) if self.data.material_demand(anchor) > 1e-9 else 0.0
        current_cover = set()
        for node in self.visited_anchors(sol):
            current_cover.update(self.route_ops._cover(node))
        new_cover = set(self.route_ops._cover(anchor)) - current_cover
        star = sum(BETA * self.data.population(i) for i in new_cover)
        return direct + star

    def _ct_chain_reward(self, routes: dict[int, list[str]], truck: int, anchor: str) -> float:
        if anchor not in self.data.c_truck:
            return 0.0
        route = routes[truck]
        positions = [idx for idx, node in enumerate(route) if node == anchor]
        if not positions:
            return 0.0
        idx = positions[0]
        left = idx
        while left - 1 >= 0 and route[left - 1] in self.data.c_truck:
            left -= 1
        right = idx
        while right + 1 < len(route) and route[right + 1] in self.data.c_truck:
            right += 1
        length = right - left + 1
        if length <= self.max_ct_chain_len_preference and (
            (idx - 1 >= 0 and route[idx - 1] in self.data.c_truck) or (idx + 1 < len(route) and route[idx + 1] in self.data.c_truck)
        ):
            return 0.2
        if length > self.max_ct_chain_len_preference:
            return -0.5
        return 0.0

    def _extended_route_metrics(self, routes: dict[int, list[str]]) -> dict:
        metrics = route_metrics(self.data, routes)
        times = self._route_times(routes)
        energies = self._route_energies(routes)
        metrics["route_travel_time_balance_std"] = self._std(list(times.values()))
        metrics["route_energy_balance_std"] = self._std(list(energies.values()))
        metrics["max_route_time_utilization"] = max((value / max(T_MAX_HOURS, 1e-9) for value in times.values()), default=0.0)
        metrics["max_energy_utilization"] = max((value / max(self.data.truck_battery, 1e-9) for value in energies.values()), default=0.0)
        return metrics

    def _utilizations(self, routes: dict[int, list[str]]) -> tuple[dict[int, float], dict[int, float]]:
        times = self._route_times(routes)
        energies = self._route_energies(routes)
        return (
            {v: value / max(T_MAX_HOURS, 1e-9) for v, value in times.items()},
            {v: value / max(self.data.truck_battery, 1e-9) for v, value in energies.items()},
        )

    def _route_times(self, routes: dict[int, list[str]]) -> dict[int, float]:
        return {
            int(v): sum(self.data.truck_time[(a, b)] for a, b in zip(route, route[1:]) if (a, b) in self.data.truck_time)
            for v, route in routes.items()
        }

    def _route_energies(self, routes: dict[int, list[str]]) -> dict[int, float]:
        return {
            int(v): sum(self.data.truck_energy[(a, b)] for a, b in zip(route, route[1:]) if (a, b) in self.data.truck_energy)
            for v, route in routes.items()
        }

    def _anchor_type(self, anchor: str) -> str:
        if anchor in self.data.h_nodes:
            return "H"
        if anchor in self.data.c_truck:
            return "CT"
        return "other"

    def _std(self, values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
