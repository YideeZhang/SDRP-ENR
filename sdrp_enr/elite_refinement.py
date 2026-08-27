from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass
import pandas as pd
from .data import BETA
from .service_decoder import FixedRouteServiceMILP
from .pool_base import (can_insert_anchor, global_visited_h, globally_visited_ct,
    is_duplicate_h_bridge_candidate, route_metrics, route_visited_h, routes_to_string,
    segment_has_internal_depot, signature_to_string, validation_metric_sum)

@dataclass
class EvalRecord:
    status: str
    status_code: int
    objective: float
    runtime_sec: float
    gap: float
    validation_metric_sum: float
    material_objective: float
    microgrid_objective: float
    cache_hit: bool
    star_count: int = 0
    rendezvous_count: int = 0
    positive_tau_count: int = 0


def parse_routes(text: str) -> dict[int, list[str]]:
    routes = {}
    for part in str(text).split(" ; "):
        if ":" not in part:
            continue
        truck, nodes = part.split(":", 1)
        routes[int(truck)] = nodes.split("|") if nodes else []
    return routes


def normalize_routes(data, routes: dict[int, list[str]]) -> dict[int, list[str]]:
    out = {}
    for v in data.truck_ids:
        route = list(routes.get(v, [data.depot]))
        if len(route) == 0:
            route = [data.depot]
        if route == [data.depot, data.depot]:
            route = [data.depot]
        if route[0] != data.depot:
            route = [data.depot] + route
        if len(route) > 1 and route[-1] != data.depot:
            route.append(data.depot)
        out[int(v)] = route
    return out


def route_signature(data, routes: dict[int, list[str]]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    normalized = normalize_routes(data, routes)
    return tuple((int(v), tuple(normalized.get(v, [data.depot]))) for v in sorted(data.truck_ids))


def valid_routes(data, routes: dict[int, list[str]]) -> bool:
    for route in normalize_routes(data, routes).values():
        route_seen_h = set()
        route_seen_ct = set()
        if not route or route[0] != data.depot:
            return False
        if len(route) > 1 and route[-1] != data.depot:
            return False
        if any(node == data.depot for node in route[1:-1]):
            return False
        for a, b in zip(route, route[1:]):
            if (a, b) not in data.truck_time:
                return False
        for node in route[1:-1]:
            if node in data.h_nodes:
                if node in route_seen_h:
                    return False
                route_seen_h.add(node)
            if node in data.c_truck:
                if node in route_seen_ct:
                    return False
                route_seen_ct.add(node)
    return True


def remove_segment(data, routes: dict[int, list[str]], truck: int, start: int, end: int) -> dict[int, list[str]] | None:
    route = routes[truck]
    p = route[start - 1]
    s = route[end + 1]
    if (p, s) not in data.truck_time:
        return None
    cand = {v: list(r) for v, r in routes.items()}
    cand[truck] = route[:start] + route[end + 1 :]
    return normalize_routes(data, cand)


def direct_insertions(data, routes: dict[int, list[str]], node: str, preferred_truck: int | None = None) -> list[dict[int, list[str]]]:
    out = []
    trucks = [preferred_truck] if preferred_truck is not None else list(routes)
    for v in trucks:
        if not can_insert_anchor(data, routes, int(v), node):
            continue
        route = routes[v]
        for pos in range(1, len(route)):
            p, s = route[pos - 1], route[pos]
            if (p, node) in data.truck_time and (node, s) in data.truck_time:
                cand = {truck: list(r) for truck, r in routes.items()}
                cand[v] = route[:pos] + [node] + route[pos:]
                cand = normalize_routes(data, cand)
                if valid_routes(data, cand):
                    out.append(cand)
    return out


def unique_routes(data, candidates: list[dict[int, list[str]]]) -> list[dict[int, list[str]]]:
    out = {}
    for cand in candidates:
        out.setdefault(route_signature(data, cand), normalize_routes(data, cand))
    return list(out.values())


def route_times(data, routes: dict[int, list[str]]) -> dict[int, float]:
    return {v: sum(data.truck_time.get((a, b), 0.0) for a, b in zip(route, route[1:])) for v, route in routes.items()}


def route_energies(data, routes: dict[int, list[str]]) -> dict[int, float]:
    return {v: sum(data.truck_energy.get((a, b), 0.0) for a, b in zip(route, route[1:])) for v, route in routes.items()}


def anchor_counts(data, routes: dict[int, list[str]]) -> dict[int, int]:
    return {v: sum(1 for node in route[1:-1] if node in data.launch_nodes) for v, route in routes.items()}


def load_metrics(data, routes: dict[int, list[str]]) -> dict:
    times = route_times(data, routes)
    energies = route_energies(data, routes)
    counts = anchor_counts(data, routes)
    max_time = max(max(times.values(), default=0.0), 1e-9)
    max_energy = max(max(energies.values(), default=0.0), 1e-9)
    max_count = max(max(counts.values(), default=0), 1e-9)
    scores = {
        v: 0.4 * (times.get(v, 0.0) / max_time)
        + 0.4 * (energies.get(v, 0.0) / max_energy)
        + 0.2 * (counts.get(v, 0) / max_count)
        for v in routes
    }
    used = sum(1 for v in routes if counts.get(v, 0) > 0)
    return {
        "route_times": times,
        "route_energies": energies,
        "anchor_counts": counts,
        "load_scores": scores,
        "load_std": std(list(scores.values())),
        "used_truck_count": int(used),
        "empty_truck_count": int(len(routes) - used),
    }


def insert_fragment(data, routes: dict[int, list[str]], truck: int, fragment: list[str]) -> list[dict[int, list[str]]]:
    out = []
    route = list(routes[truck])
    if route == [data.depot]:
        route = [data.depot, data.depot]
    if any(node == data.depot for node in fragment):
        return out
    if any(node in data.h_nodes and node in route_visited_h(data, route) for node in fragment):
        return out
    if any(node in data.c_truck and node in route[1:-1] for node in fragment):
        return out
    for pos in range(1, len(route)):
        cand = {v: list(r) for v, r in routes.items()}
        cand[truck] = route[:pos] + list(fragment) + route[pos:]
        cand = normalize_routes(data, cand)
        if valid_routes(data, cand):
            out.append(cand)
    return out


def activation_candidate_score(data, before: dict[int, list[str]], after: dict[int, list[str]], donor: int, receiver: int, fragment: list[str], before_metrics: dict, after_metrics: dict) -> float:
    before_empty = float(before_metrics.get("empty_truck_count", 0))
    after_empty = float(after_metrics.get("empty_truck_count", 0))
    before_std = float(before_metrics.get("load_std", 0.0))
    after_std = float(after_metrics.get("load_std", 0.0))
    time_before = route_times(data, before)
    time_after = route_times(data, after)
    energy_before = route_energies(data, before)
    energy_after = route_energies(data, after)
    add_time = max(0.0, sum(time_after.values()) - sum(time_before.values()))
    add_energy = max(0.0, sum(energy_after.values()) - sum(energy_before.values()))
    service_proxy = sum(BETA * data.population(node) for node in fragment if node in data.launch_nodes)
    empty_reward = 10.0 * max(0.0, before_empty - after_empty)
    balance_reward = 5.0 * max(0.0, before_std - after_std)
    receiver_reward = 1.0 if before_metrics["anchor_counts"].get(receiver, 0) == 0 and after_metrics["anchor_counts"].get(receiver, 0) > 0 else 0.0
    return service_proxy + empty_reward + balance_reward + receiver_reward - 0.1 * add_time - 0.01 * add_energy


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def extended_route_metrics(data, routes: dict[int, list[str]]) -> dict:
    metrics = route_metrics(data, routes)
    times = route_times(data, routes)
    energies = route_energies(data, routes)
    metrics["route_time_balance_std"] = std(list(times.values()))
    metrics["route_energy_balance_std"] = std(list(energies.values()))
    return metrics


def select_elites(candidates: pd.DataFrame, elite_k: int) -> pd.DataFrame:
    valid = candidates[
        candidates["service_milp_status"].eq("OPTIMAL")
        & candidates["validation_metric_sum"].le(1e-9)
        & candidates["service_milp_objective"].notna()
    ].copy()
    valid = valid.sort_values("service_milp_objective", ascending=False)
    valid = valid.drop_duplicates("route_signature")
    return valid.head(elite_k)

class EliteRefiner:
    def __init__(
        self,
        data,
        scenario: str,
        seed: int,
        service_milp_time_limit_sec: float,
        max_new_candidates: int,
        allowed_operators: set[str] | None = None,
        enable_integrated_h_bridge: bool = True,
        enable_activation_rebalance: bool = False,
        max_activation_rebalance_candidates: int = 20,
    ) -> None:
        self.data = data
        self.scenario = scenario
        self.rng = random.Random(seed)
        self.service = FixedRouteServiceMILP(data, time_limit_sec=service_milp_time_limit_sec, output_flag=0)
        self.max_new_candidates = max_new_candidates
        self.allowed_operators = set(allowed_operators) if allowed_operators is not None else {"Rebalance-LNS", "Drop-and-reinsert-LNS"}
        self.enable_integrated_h_bridge = enable_integrated_h_bridge
        self.enable_activation_rebalance = enable_activation_rebalance
        self.max_activation_rebalance_candidates = max_activation_rebalance_candidates
        self.cache: dict[tuple[tuple[int, tuple[str, ...]], ...], EvalRecord] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.service_rows: list[dict] = []
        self.last_operator_info: dict = {}
        self.last_candidate_meta: dict[tuple[tuple[int, tuple[str, ...]], ...], dict] = {}

    def preload(self, candidates: pd.DataFrame) -> None:
        for row in candidates.to_dict("records"):
            routes = parse_routes(str(row["route_nodes_by_truck"]))
            signature = route_signature(self.data, routes)
            if signature in self.cache:
                continue
            self.cache[signature] = EvalRecord(
                status=str(row.get("service_milp_status", "")),
                status_code=2 if str(row.get("service_milp_status", "")) == "OPTIMAL" else 0,
                objective=float(row.get("service_milp_objective", math.nan)),
                runtime_sec=float(row.get("runtime_sec", 0.0)),
                gap=math.nan,
                validation_metric_sum=float(row.get("validation_metric_sum", math.inf)),
                material_objective=float(row.get("material_objective", math.nan)),
                microgrid_objective=float(row.get("microgrid_objective", math.nan)),
                cache_hit=True,
                star_count=int(float(row.get("star_count", 0) or 0)),
                rendezvous_count=int(float(row.get("rendezvous_count", 0) or 0)),
                positive_tau_count=int(float(row.get("positive_tau_count", 0) or 0)),
            )

    def evaluate(self, routes: dict[int, list[str]], source: str, operator: str = "") -> EvalRecord:
        normalized = normalize_routes(self.data, routes)
        signature = route_signature(self.data, normalized)
        if signature in self.cache:
            self.cache_hits += 1
            cached = self.cache[signature]
            return EvalRecord(**{**cached.__dict__, "cache_hit": True})
        self.cache_misses += 1
        result = self.service.solve(normalized)
        counts = result.solution.counts() if result.solution is not None else {}
        record = EvalRecord(
            status=result.status,
            status_code=result.status_code,
            objective=float(result.objective),
            runtime_sec=float(result.runtime_sec),
            gap=float(result.gap),
            validation_metric_sum=validation_metric_sum(result),
            material_objective=float(getattr(result.solution, "served_material_score", math.nan)),
            microgrid_objective=float(getattr(result.solution, "microgrid_score", math.nan)),
            cache_hit=False,
            star_count=int(counts.get("star_rows", 0)),
            rendezvous_count=int(counts.get("rendezvous_rows", 0)),
            positive_tau_count=int(len(getattr(result.solution, "tau", {}) or {})),
        )
        self.cache[signature] = record
        self.service_rows.append(
            {
                "scenario": self.scenario,
                "source": source,
                "operator": operator,
                "route_signature": signature_to_string(signature),
                "status": record.status,
                "status_code": record.status_code,
                "objective": record.objective,
                "runtime_sec": record.runtime_sec,
                "gap": record.gap,
                "validation_metric_sum": record.validation_metric_sum,
                "cache_hit": False,
                "star_count": record.star_count,
                "rendezvous_count": record.rendezvous_count,
                "positive_tau_count": record.positive_tau_count,
            }
        )
        return record

    def refine(self, elite_rows: pd.DataFrame, rounds_per_elite: int) -> tuple[dict, list[dict], list[dict]]:
        move_rows: list[dict] = []
        new_candidate_count = 0
        best_global: dict | None = None
        accepted = 0
        improved = 0
        operator_improvements: dict[str, int] = {}

        for elite_rank, row in enumerate(elite_rows.to_dict("records"), start=1):
            current_routes = normalize_routes(self.data, parse_routes(str(row["route_nodes_by_truck"])))
            current_eval = self.evaluate(current_routes, "elite_initial")
            current_obj = current_eval.objective
            if best_global is None or current_obj > best_global["objective"]:
                best_global = {"routes": current_routes, "objective": current_obj, "operator": "elite_initial"}

            for round_id in range(1, rounds_per_elite + 1):
                if new_candidate_count >= self.max_new_candidates:
                    break
                operator, reason = self.choose_operator(current_routes, elite_rows)
                routes_before = current_routes
                before_obj = current_obj
                candidates = self.generate_candidates(operator, current_routes)
                operator_info = dict(self.last_operator_info)
                before_dup_h = extended_route_metrics(self.data, current_routes).get("duplicate_h_visit_count", 0)
                bridge_candidate_count = sum(
                    1
                    for cand in candidates
                    if extended_route_metrics(self.data, cand).get("duplicate_h_visit_count", 0) > before_dup_h
                )
                remaining = self.max_new_candidates - new_candidate_count
                candidates = candidates[: max(0, remaining)]
                evals = []
                for cand in candidates:
                    rec = self.evaluate(cand, "lns_candidate", operator)
                    new_candidate_count += 0 if rec.cache_hit else 1
                    if rec.status == "OPTIMAL" and rec.validation_metric_sum <= 1e-9 and math.isfinite(rec.objective):
                        evals.append((rec.objective, cand, rec))
                selected_obj = math.nan
                accepted_move = False
                improved_move = False
                after_routes = current_routes
                selected_meta = {}
                if evals:
                    selected_obj, selected_routes, _rec = max(evals, key=lambda item: item[0])
                    selected_meta = self.last_candidate_meta.get(route_signature(self.data, selected_routes), {})
                    if selected_obj >= current_obj - 1e-9:
                        accepted_move = True
                        after_routes = selected_routes
                        if selected_obj > current_obj + 1e-9:
                            improved_move = True
                            improved += 1
                            operator_improvements[operator] = operator_improvements.get(operator, 0) + 1
                        accepted += 1
                        current_routes = selected_routes
                        current_obj = selected_obj
                        if best_global is None or selected_obj > best_global["objective"] + 1e-9:
                            best_global = {"routes": selected_routes, "objective": selected_obj, "operator": operator}
                metrics = extended_route_metrics(self.data, after_routes if accepted_move else routes_before)
                operator_info.update(selected_meta)
                if operator_info.get("rebalance_branch") == "activation":
                    before_load = load_metrics(self.data, routes_before)
                    after_load = load_metrics(self.data, after_routes if accepted_move else routes_before)
                    operator_info.update(
                        {
                            "empty_truck_count_before": before_load["empty_truck_count"],
                            "empty_truck_count_after": after_load["empty_truck_count"],
                            "used_truck_count_before": before_load["used_truck_count"],
                            "used_truck_count_after": after_load["used_truck_count"],
                            "load_std_before": before_load["load_std"],
                            "load_std_after": after_load["load_std"],
                            "activation_candidate_evaluated_count": len(candidates),
                            "objective_delta": (selected_obj - before_obj) if math.isfinite(selected_obj) else math.nan,
                        }
                    )
                move_rows.append(
                    {
                        "scenario": self.scenario,
                        "elite_rank": elite_rank,
                        "round": round_id,
                        "operator": operator,
                        "operator_reason": reason,
                        "candidate_count": len(candidates),
                        "bridge_candidate_count": bridge_candidate_count,
                        "selected_candidate_objective": selected_obj,
                        "current_objective_before": before_obj,
                        "objective_after": current_obj,
                        "accepted": accepted_move,
                        "improved": improved_move,
                        "bridge_accepted": bool(accepted_move and extended_route_metrics(self.data, after_routes).get("duplicate_h_visit_count", 0) > before_dup_h),
                        "bridge_improved": bool(improved_move and extended_route_metrics(self.data, after_routes).get("duplicate_h_visit_count", 0) > before_dup_h),
                        "ct_anchor_count_after": metrics["ct_anchor_count"],
                        "ct_chain_count_after": metrics["ct_chain_count"],
                        "route_time_balance_std_after": metrics["route_time_balance_std"],
                        "route_energy_balance_std_after": metrics["route_energy_balance_std"],
                        "rebalance_branch": operator_info.get("rebalance_branch", ""),
                        "receiver_truck": operator_info.get("receiver_truck", ""),
                        "receiver_was_empty": operator_info.get("receiver_was_empty", ""),
                        "donor_truck": operator_info.get("donor_truck", ""),
                        "donor_load_score": operator_info.get("donor_load_score", math.nan),
                        "receiver_load_score": operator_info.get("receiver_load_score", math.nan),
                        "fragment_len": operator_info.get("fragment_len", math.nan),
                        "fragment_nodes": operator_info.get("fragment_nodes", ""),
                        "activation_candidate_count": operator_info.get("activation_candidate_count", math.nan),
                        "activation_candidate_evaluated_count": operator_info.get("activation_candidate_evaluated_count", math.nan),
                        "empty_truck_count_before": operator_info.get("empty_truck_count_before", math.nan),
                        "empty_truck_count_after": operator_info.get("empty_truck_count_after", math.nan),
                        "used_truck_count_before": operator_info.get("used_truck_count_before", math.nan),
                        "used_truck_count_after": operator_info.get("used_truck_count_after", math.nan),
                        "load_std_before": operator_info.get("load_std_before", math.nan),
                        "load_std_after": operator_info.get("load_std_after", math.nan),
                        "activation_objective_delta": operator_info.get("objective_delta", math.nan),
                    }
                )
            if new_candidate_count >= self.max_new_candidates:
                break

        best_operator = max(operator_improvements.items(), key=lambda item: item[1])[0] if operator_improvements else ""
        summary = {
            "scenario": self.scenario,
            "elite_count": len(elite_rows),
            "generated_lns_candidate_count": new_candidate_count,
            "evaluated_lns_candidate_count": self.cache_misses,
            "accepted_move_count": accepted,
            "improved_move_count": improved,
            "cache_hit_count": self.cache_hits,
            "cache_miss_count": self.cache_misses,
            "best_operator": best_operator,
            "mean_candidate_runtime": mean([row["runtime_sec"] for row in self.service_rows]),
        }
        return best_global or {"routes": {}, "objective": math.nan, "operator": ""}, move_rows, [summary]

    def choose_operator(self, routes, elite_rows):
        metrics = extended_route_metrics(self.data, routes)
        operators = [op for op in ["Rebalance-LNS", "Drop-and-reinsert-LNS"] if op in self.allowed_operators]
        if not operators:
            return "LNS-disabled", "no_allowed_operator"
        if self.rng.random() < 0.2:
            return self.rng.choice(operators), "random_exploration_20pct"
        if "Rebalance-LNS" in self.allowed_operators and (metrics["route_time_balance_std"] > 1.0 or metrics["route_energy_balance_std"] > 30.0):
            return "Rebalance-LNS", "high_route_time_or_energy_balance_std"
        if "Drop-and-reinsert-LNS" in self.allowed_operators:
            return "Drop-and-reinsert-LNS", "fallback_single_elite_structure"
        return operators[0], "fallback_first_allowed_operator"

    def generate_candidates(self, operator, routes):
        self.last_operator_info = {}
        self.last_candidate_meta = {}
        if operator == "Rebalance-LNS":
            return self.rebalance_candidates(routes)
        if operator == "Drop-and-reinsert-LNS":
            return self.drop_reinsert_candidates(routes)
        return []

    def rebalance_candidates(self, routes: dict[int, list[str]]) -> list[dict[int, list[str]]]:
        if self.enable_activation_rebalance:
            metrics = load_metrics(self.data, routes)
            if metrics["empty_truck_count"] > 0 or metrics["load_std"] > 0.35:
                self.last_operator_info = {
                    "rebalance_branch": "activation",
                    "activation_candidate_count": 0,
                    "activation_candidate_evaluated_count": 0,
                    "empty_truck_count_before": metrics["empty_truck_count"],
                    "used_truck_count_before": metrics["used_truck_count"],
                    "load_std_before": metrics["load_std"],
                }
                return self.activation_rebalance_candidates(routes, metrics)
            self.last_operator_info = {"rebalance_branch": "ordinary", "load_std_before": metrics["load_std"]}
        times = route_times(self.data, routes)
        energies = route_energies(self.data, routes)
        high = max(routes, key=lambda v: times.get(v, 0.0) + energies.get(v, 0.0) / 30.0)
        low = min(routes, key=lambda v: times.get(v, 0.0) + energies.get(v, 0.0) / 30.0)
        route_high = routes[high]
        anchors = [(idx, node) for idx, node in enumerate(route_high[1:-1], start=1) if node in self.data.launch_nodes]
        anchors.sort(key=lambda item: (0 if item[1] in self.data.c_truck else 1, item[0]))
        out = []
        for idx, node in anchors:
            bases = []
            removed = remove_segment(self.data, routes, high, idx, idx)
            if removed is not None:
                bases.append(removed)
            if self.enable_integrated_h_bridge and node in self.data.h_nodes:
                bases.append(normalize_routes(self.data, routes))
            for base in bases:
                for cand in direct_insertions(self.data, base, node, preferred_truck=low):
                    if valid_routes(self.data, cand):
                        out.append(cand)
            if self.enable_integrated_h_bridge and node in self.data.h_nodes:
                continue
            if removed is None:
                continue
            for cand in direct_insertions(self.data, removed, node, preferred_truck=low):
                if valid_routes(self.data, cand):
                    out.append(cand)
        return unique_routes(self.data, out)

    def activation_rebalance_candidates(self, routes: dict[int, list[str]], metrics: dict) -> list[dict[int, list[str]]]:
        scores = metrics["load_scores"]
        anchor_counts = metrics["anchor_counts"]
        empty = [v for v in sorted(routes) if anchor_counts.get(v, 0) == 0]
        receiver = empty[0] if empty else min(routes, key=lambda v: scores.get(v, 0.0))
        receiver_was_empty = receiver in empty
        donors = [v for v in routes if v != receiver and anchor_counts.get(v, 0) > 0]
        donors.sort(key=lambda v: scores.get(v, 0.0), reverse=True)
        donors = donors[:3]
        scored: list[tuple[float, dict[int, list[str]], dict]] = []
        for donor in donors:
            route = routes[donor]
            for start in range(1, len(route) - 1):
                for length in (1, 2, 3):
                    end = start + length - 1
                    if end >= len(route) - 1:
                        continue
                    fragment = route[start : end + 1]
                    if not all(node in self.data.launch_nodes for node in fragment):
                        continue
                    base = remove_segment(self.data, routes, donor, start, end)
                    if base is None:
                        continue
                    for cand in insert_fragment(self.data, base, receiver, fragment):
                        if not valid_routes(self.data, cand):
                            continue
                        after = load_metrics(self.data, cand)
                        score = activation_candidate_score(self.data, routes, cand, donor, receiver, fragment, metrics, after)
                        meta = {
                            "rebalance_branch": "activation",
                            "receiver_truck": receiver,
                            "receiver_was_empty": bool(receiver_was_empty),
                            "donor_truck": donor,
                            "donor_load_score": scores.get(donor, math.nan),
                            "receiver_load_score": scores.get(receiver, math.nan),
                            "fragment_len": len(fragment),
                            "fragment_nodes": "|".join(fragment),
                        }
                        scored.append((score, cand, meta))
        scored.sort(key=lambda item: item[0], reverse=True)
        limited = scored[: self.max_activation_rebalance_candidates]
        unique: dict[tuple[tuple[int, tuple[str, ...]], ...], dict[int, list[str]]] = {}
        self.last_candidate_meta = {}
        for _score, cand, meta in limited:
            sig = route_signature(self.data, cand)
            if sig not in unique:
                unique[sig] = normalize_routes(self.data, cand)
                self.last_candidate_meta[sig] = meta
        self.last_operator_info.update(
            {
                "activation_candidate_count": len(scored),
                "activation_candidate_evaluated_count": len(unique),
            }
        )
        return list(unique.values())

    def drop_reinsert_candidates(self, routes: dict[int, list[str]]) -> list[dict[int, list[str]]]:
        out = []
        for v, route in routes.items():
            for start in range(1, len(route) - 1):
                for length in (1, 2):
                    end = start + length - 1
                    if end >= len(route) - 1:
                        continue
                    removed_nodes = route[start : end + 1]
                    if not all(node in self.data.launch_nodes for node in removed_nodes):
                        continue
                    base = remove_segment(self.data, routes, v, start, end)
                    if base is None:
                        continue
                    partials = [base]
                    for node in removed_nodes:
                        next_partials = []
                        for partial in partials:
                            next_partials.extend(direct_insertions(self.data, partial, node))
                        partials = unique_routes(self.data, [cand for cand in next_partials if valid_routes(self.data, cand)])
                    out.extend(partials)
        return unique_routes(self.data, out)
