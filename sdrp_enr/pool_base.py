from __future__ import annotations
import itertools
import math
import random
from dataclasses import dataclass, field
from .data import BETA
from .solution import Solution
from .insertion import InsertionPaths
from .seed_routes import HSeedConstructor
from .decoder_result import ServiceMILPResult

def segment_has_internal_depot(data, segment: list[str]) -> bool:
    return any(node == data.depot for node in segment[1:-1])


def route_has_internal_depot(data, route: list[str]) -> bool:
    return any(node == data.depot for node in route[1:-1])


def internal_depot_count(data, routes: dict[int, list[str]]) -> int:
    return int(sum(sum(1 for node in route[1:-1] if node == data.depot) for route in routes.values()))


def globally_visited_ct(data, routes: dict[int, list[str]]) -> set[str]:
    return {node for route in routes.values() for node in route[1:-1] if node in data.c_truck}


def global_visited_ct(data, routes: dict[int, list[str]]) -> set[str]:
    return globally_visited_ct(data, routes)


def route_visited_h(data, route: list[str]) -> set[str]:
    return {node for node in route[1:-1] if node in data.h_nodes}


def global_visited_h(data, routes: dict[int, list[str]]) -> set[str]:
    return {node for route in routes.values() for node in route[1:-1] if node in data.h_nodes}


def can_insert_anchor(data, routes: dict[int, list[str]], truck: int, anchor: str) -> bool:
    if anchor in data.h_nodes:
        return anchor not in route_visited_h(data, routes.get(truck, []))
    if anchor in data.c_truck:
        return anchor not in routes.get(truck, [])[1:-1]
    return False


def is_duplicate_h_bridge_candidate(data, routes: dict[int, list[str]], truck: int, anchor: str) -> bool:
    return (
        anchor in data.h_nodes
        and anchor in global_visited_h(data, routes)
        and anchor not in route_visited_h(data, routes.get(truck, []))
    )


def valid_h_normal_routes(data, routes: dict[int, list[str]]) -> bool:
    for raw_route in routes.values():
        route = list(raw_route)
        if route == [data.depot, data.depot]:
            route = [data.depot]
        if not route or route[0] != data.depot or route[-1] != data.depot:
            return False
        if any(node == data.depot for node in route[1:-1]):
            return False
        route_seen_h: set[str] = set()
        route_seen_ct: set[str] = set()
        for a, b in zip(route, route[1:]):
            if (a, b) not in data.truck_arcs:
                return False
        for node in route[1:-1]:
            if node in data.h_nodes:
                if node in route_seen_h:
                    return False
                route_seen_h.add(node)
            elif node in data.c_truck:
                if node in route_seen_ct:
                    return False
                route_seen_ct.add(node)
    return True


@dataclass
class RouteCandidate:
    scenario: str
    generator: str
    routes: dict[int, list[str]]
    trace: dict = field(default_factory=dict)


def route_metrics(data, routes: dict[int, list[str]]) -> dict:
    anchor_count = 0
    h_count = 0
    ct_count = 0
    h_occurrences = []
    travel = 0.0
    energy = 0.0
    ct_segments = []
    for route in routes.values():
        anchors = [node for node in route[1:-1] if node in data.launch_nodes]
        anchor_count += len(anchors)
        route_h = [node for node in anchors if node in data.h_nodes]
        h_occurrences.extend(route_h)
        h_count += len(route_h)
        ct_mask = [node in data.c_truck for node in anchors]
        ct_count += sum(1 for flag in ct_mask if flag)
        ct_segments.extend(consecutive_segments(ct_mask))
        travel += sum(data.truck_time[(a, b)] for a, b in zip(route, route[1:]) if (a, b) in data.truck_time)
        energy += sum(data.truck_energy[(a, b)] for a, b in zip(route, route[1:]) if (a, b) in data.truck_energy)
    return {
        "route_anchor_count": anchor_count,
        "h_anchor_count": h_count,
        "h_visit_count": h_count,
        "unique_h_visit_count": len(set(h_occurrences)),
        "duplicate_h_visit_count": h_count - len(set(h_occurrences)),
        "ct_anchor_count": ct_count,
        "ct_chain_count": sum(1 for length in ct_segments if length >= 2),
        "max_ct_chain_length": max(ct_segments + [0]),
        "route_travel_time": travel,
        "route_driving_energy": energy,
    }


def consecutive_segments(mask: list[bool]) -> list[int]:
    out = []
    current = 0
    for flag in mask:
        if flag:
            current += 1
        elif current:
            out.append(current)
            current = 0
    if current:
        out.append(current)
    return out


def validation_metric_sum(result: ServiceMILPResult) -> float:
    if result.solution is None:
        return math.inf
    return sum(float(v) for v in (result.solution.validation_metrics or {}).values())


def routes_to_string(routes: dict[int, list[str]]) -> str:
    return " ; ".join(f"{v}:{'|'.join(route)}" for v, route in sorted(routes.items()))


def signature_to_string(signature: tuple[tuple[int, tuple[str, ...]], ...]) -> str:
    return " ; ".join(f"{v}:{'|'.join(route)}" for v, route in signature)

class RoutePoolBase:
    def __init__(
        self,
        data,
        scenario: str,
        seed: int,
        service_milp_time_limit_sec: float,
        max_pool_size: int,
        random_route_count: int,
        edge_chain_max_len: int,
        h_duplicate_top_k: int = 10,
        max_h_duplicate_candidates: int = 100,
        h_duplicate_lambda_t: float = 0.1,
        h_duplicate_lambda_e: float = 0.01,
        enable_integrated_h_bridge: bool = True,
        bridge_h_choice_probability: float = 0.25,
    ) -> None:
        self.data = data
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_pool_size = max_pool_size
        self.random_route_count = random_route_count
        self.edge_chain_max_len = edge_chain_max_len
        self.h_duplicate_top_k = h_duplicate_top_k
        self.max_h_duplicate_candidates = max_h_duplicate_candidates
        self.h_duplicate_lambda_t = h_duplicate_lambda_t
        self.h_duplicate_lambda_e = h_duplicate_lambda_e
        self.enable_integrated_h_bridge = enable_integrated_h_bridge
        self.bridge_h_choice_probability = bridge_h_choice_probability
        self.route_ops = InsertionPaths(data)
        self.generated_by_generator: dict[str, int] = {}
        self.invalid_by_generator: dict[str, int] = {}

    def generate_integrated_h_bridge_variants(self, bases: list[RouteCandidate]) -> list[RouteCandidate]:
        return self._generate_h_bridge_variants(bases, standalone_name=None)

    def _generate_h_bridge_variants(self, bases: list[RouteCandidate], standalone_name: str | None) -> list[RouteCandidate]:
        scored: list[tuple[float, RouteCandidate]] = []
        seen: set[tuple[tuple[int, tuple[str, ...]], ...]] = set()
        for base in bases:
            routes = self.normalized_routes(base.routes)
            if not valid_h_normal_routes(self.data, routes):
                continue
            h_occurrences = [node for route in routes.values() for node in route[1:-1] if node in self.data.h_nodes]
            h_candidates = sorted(set(h_occurrences), key=lambda h: self.h_duplicate_proxy(h, routes), reverse=True)
            for h in h_candidates[: self.h_duplicate_top_k]:
                sol = Solution(routes={int(v): list(route) for v, route in routes.items()})
                for option in self.insertion_options(sol, h):
                    cand_routes = self._routes_after_option(sol, option)
                    if not valid_h_normal_routes(self.data, cand_routes):
                        continue
                    signature = self.route_signature(cand_routes)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    add_time = float(option.get("add_time", 0.0))
                    add_energy = float(option.get("add_energy", 0.0))
                    score = self.h_duplicate_proxy(h, routes) - self.h_duplicate_lambda_t * add_time - self.h_duplicate_lambda_e * add_energy
                    scored.append(
                        (
                            score,
                            RouteCandidate(
                                self.scenario,
                                standalone_name or base.generator,
                                self.normalized_routes(cand_routes),
                                {
                                    "base_generator": base.generator,
                                    "duplicate_h": h,
                                    "integrated_h_bridge": standalone_name is None,
                                    "add_time": add_time,
                                    "add_energy": add_energy,
                                    "proxy_score": score,
                                },
                            ),
                        )
                    )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [cand for _score, cand in scored[: self.max_h_duplicate_candidates]]

    def generate_h_only(self) -> list[RouteCandidate]:
        initial = HSeedConstructor(self.data, self.seed).construct()
        built, trace = self.construct_by_cheapest(list(self.data.h_nodes))
        routes = self.normalized_routes(initial.routes)
        if not valid_h_normal_routes(self.data, routes):
            routes = self.normalized_routes(built.routes)
        return [RouteCandidate(self.scenario, "H_only_current_initial", routes),
                RouteCandidate(self.scenario, "H_only_cheapest", self.normalized_routes(built.routes), trace)]

    def generate_edge_replacement(self) -> list[RouteCandidate]:
        candidates: list[RouteCandidate] = []
        base = self.generate_h_only()[0]
        current = {node for route in base.routes.values() for node in route[1:-1] if node in self.data.launch_nodes}
        available = [node for node in self.rank_ct_nodes() if node not in current]
        top_nodes = available[: min(12, len(available))]
        edge_items = []
        for v, route in base.routes.items():
            for pos in range(1, len(route)):
                p, s = route[pos - 1], route[pos]
                direct_t = self.data.truck_time.get((p, s), float("inf"))
                if not math.isfinite(direct_t):
                    continue
                for chain in self.edge_chains(p, s, top_nodes):
                    add_t = self.segment_time([p] + chain + [s]) - direct_t
                    if add_t < -1e-9:
                        continue
                    score = self.chain_potential(chain)
                    edge_items.append((score, -add_t, v, pos, chain, "ct_chain"))
                if self.enable_integrated_h_bridge:
                    h_bridges = sorted(
                        [
                            h
                            for h in global_visited_h(self.data, base.routes)
                            if is_duplicate_h_bridge_candidate(self.data, base.routes, int(v), h)
                        ],
                        key=lambda h: self.h_duplicate_proxy(h, base.routes),
                        reverse=True,
                    )
                    for h in h_bridges[: self.h_duplicate_top_k]:
                        segment = self.route_ops._expanded_segment(p, h, s)
                        if not segment or segment_has_internal_depot(self.data, segment):
                            continue
                        inner_anchors = [node for node in segment[1:-1] if node in self.data.launch_nodes]
                        if h not in inner_anchors:
                            continue
                        if any(not can_insert_anchor(self.data, base.routes, int(v), node) for node in inner_anchors):
                            continue
                        add_t = self.segment_time(segment) - direct_t
                        if add_t < -1e-9:
                            continue
                        score = self.h_duplicate_proxy(h, base.routes) - self.h_duplicate_lambda_t * add_t
                        edge_items.append((score, -add_t, v, pos, segment[1:-1], "h_bridge"))
        edge_items.sort(reverse=True)
        for idx, (_score, _neg_add_t, v, pos, chain, edge_kind) in enumerate(edge_items[: max(30, self.max_pool_size // 4)]):
            routes = {truck: list(route) for truck, route in base.routes.items()}
            route = routes[v]
            routes[v] = route[:pos] + list(chain) + route[pos:]
            candidates.append(RouteCandidate(self.scenario, "EdgeReplacement", self.normalized_routes(routes), {"edge_candidate_rank": idx, "edge_replacement_kind": edge_kind}))
        return candidates

    def generate_randomized(self) -> list[RouteCandidate]:
        candidates: list[RouteCandidate] = []
        ct_ranked = self.rank_ct_nodes()
        for idx in range(self.random_route_count):
            sol = Solution(routes={v: [self.data.depot, self.data.depot] for v in self.data.truck_ids})
            h_order = list(self.data.h_nodes)
            self.rng.shuffle(h_order)
            ct_count = self.rng.randint(0, len(ct_ranked)) if ct_ranked else 0
            ct_pool = list(ct_ranked[: max(ct_count, min(len(ct_ranked), 8))])
            self.rng.shuffle(ct_pool)
            targets: list[str] = []
            h_idx = 0
            ct_idx = 0
            while h_idx < len(h_order) or ct_idx < ct_count:
                choose_ct = self.rng.random() < 0.5
                if choose_ct and ct_idx < ct_count:
                    targets.append(ct_pool[ct_idx])
                    ct_idx += 1
                elif h_idx < len(h_order):
                    targets.append(h_order[h_idx])
                    h_idx += 1
                elif ct_idx < ct_count:
                    targets.append(ct_pool[ct_idx])
                    ct_idx += 1
            if self.enable_integrated_h_bridge:
                bridge_targets = [h for h in h_order if self.rng.random() < self.bridge_h_choice_probability]
                self.rng.shuffle(bridge_targets)
                targets.extend(bridge_targets)
            failed = []
            for target in targets:
                option = self.randomized_insertion_option(sol, target)
                if option is None:
                    failed.append(target)
                    continue
                self.apply_insertion(sol, option)
            candidates.append(RouteCandidate(self.scenario, "RandomizedInsertion", self.normalized_routes(sol.routes), {"random_index": idx, "failed_insert_count": len(failed)}))
        return candidates

    def construct_by_cheapest(self, targets: list[str]) -> tuple[Solution, dict]:
        sol = Solution(routes={v: [self.data.depot, self.data.depot] for v in self.data.truck_ids})
        failed = []
        inserted = []
        for target in targets:
            option = self.cheapest_insertion(sol, target)
            if option is None:
                failed.append(target)
                continue
            before = self.visited_anchors(sol)
            self.apply_insertion(sol, option)
            after = self.visited_anchors(sol)
            inserted.extend(sorted(after - before))
        return sol, {"failed_insert_count": len(failed), "inserted_anchor_count": len(set(inserted))}

    def cheapest_insertion(self, sol: Solution, target: str) -> dict | None:
        options = self.insertion_options(sol, target)
        if not options:
            return None
        return min(options, key=lambda item: (item["add_time"], item["add_energy"]))

    def randomized_insertion_option(self, sol: Solution, target: str) -> dict | None:
        options = self.insertion_options(sol, target)
        if not options:
            return None
        options.sort(key=lambda item: (item["add_time"], item["add_energy"]))
        top = options[: min(5, len(options))]
        return self.rng.choice(top)

    def insertion_options(self, sol: Solution, target: str) -> list[dict]:
        options = []
        for v, route in sol.routes.items():
            if not can_insert_anchor(self.data, sol.routes, int(v), target):
                continue
            for pos in range(1, len(route)):
                p, s = route[pos - 1], route[pos]
                segment = self.route_ops._expanded_segment(p, target, s)
                if not segment:
                    continue
                if segment_has_internal_depot(self.data, segment):
                    continue
                inner_anchors = [node for node in segment[1:-1] if node in self.data.launch_nodes]
                if target not in inner_anchors:
                    continue
                if any(not can_insert_anchor(self.data, sol.routes, int(v), node) for node in inner_anchors):
                    continue
                route_h = route_visited_h(self.data, route)
                segment_h_seen: set[str] = set()
                segment_ct_seen: set[str] = set()
                bad_duplicate = False
                for node in inner_anchors:
                    if node in self.data.h_nodes:
                        if node in route_h or node in segment_h_seen:
                            bad_duplicate = True
                            break
                        segment_h_seen.add(node)
                    elif node in self.data.c_truck:
                        if node in segment_ct_seen:
                            bad_duplicate = True
                            break
                        segment_ct_seen.add(node)
                if bad_duplicate:
                    continue
                cand_route = route[: pos - 1] + segment + route[pos + 1 :]
                cand_routes = {truck: list(r) for truck, r in sol.routes.items()}
                cand_routes[int(v)] = cand_route
                if not valid_h_normal_routes(self.data, cand_routes):
                    continue
                add_time = self.segment_time(segment) - self.data.truck_time.get((p, s), 0.0)
                add_energy = self.segment_energy(segment) - self.data.truck_energy.get((p, s), 0.0)
                if add_time < -1e-9 or add_energy < -1e-9:
                    continue
                options.append({"truck": v, "pos": pos, "segment": segment, "add_time": add_time, "add_energy": add_energy})
        return options

    def apply_insertion(self, sol: Solution, option: dict) -> None:
        v = int(option["truck"])
        pos = int(option["pos"])
        route = sol.routes[v]
        segment = list(option["segment"])
        sol.routes[v] = route[: pos - 1] + segment + route[pos + 1 :]

    def _routes_after_option(self, sol: Solution, option: dict) -> dict[int, list[str]]:
        routes = {int(v): list(route) for v, route in sol.routes.items()}
        v = int(option["truck"])
        pos = int(option["pos"])
        segment = list(option["segment"])
        routes[v] = routes[v][: pos - 1] + segment + routes[v][pos + 1 :]
        return self.normalized_routes(routes)

    def edge_chains(self, p: str, s: str, top_nodes: list[str]) -> list[list[str]]:
        out = []
        partials = [([], p)]
        for _depth in range(1, self.edge_chain_max_len + 1):
            new_partials = []
            for chain, last in partials:
                for c in top_nodes:
                    if c in chain:
                        continue
                    if (last, c) not in self.data.truck_arcs:
                        continue
                    new_chain = chain + [c]
                    if (c, s) in self.data.truck_arcs:
                        out.append(new_chain)
                    new_partials.append((new_chain, c))
            new_partials.sort(key=lambda item: self.chain_potential(item[0]), reverse=True)
            partials = new_partials[:20]
        return out

    def rank_ct_nodes(self) -> list[str]:
        return sorted(self.data.c_truck, key=self.ct_score, reverse=True)

    def ct_score(self, node: str) -> float:
        direct = BETA * self.data.population(node) if self.data.material_demand(node) > 1e-9 else 0.0
        reachable = sum(BETA * self.data.population(i) for i in self.route_ops._cover(node))
        return direct + reachable

    def chain_potential(self, chain: list[str]) -> float:
        covered = set()
        direct = 0.0
        for c in chain:
            direct += BETA * self.data.population(c) if self.data.material_demand(c) > 1e-9 else 0.0
            covered.update(self.route_ops._cover(c))
        return direct + sum(BETA * self.data.population(i) for i in covered)

    def h_duplicate_proxy(self, h: str, routes: dict[int, list[str]]) -> float:
        direct = BETA * self.data.population(h) if self.data.material_demand(h) > 1e-9 else 0.0
        covered = set()
        for route in routes.values():
            for anchor in route[1:-1]:
                if anchor in self.data.launch_nodes:
                    covered.update(self.route_ops._cover(anchor))
        star = sum(BETA * self.data.population(i) for i in (self.route_ops._cover(h) - covered))
        h_node = getattr(self.data, "nodes", {}).get(h)
        restoration = float(getattr(h_node, "R", 0.0) or 0.0) if h_node is not None else 0.0
        charging = 0.05 * self.data.population(h) * max(restoration, 0.0)
        return direct + star + charging

    def segment_time(self, segment: list[str]) -> float:
        return sum(self.data.truck_time[(a, b)] for a, b in zip(segment, segment[1:]))

    def segment_energy(self, segment: list[str]) -> float:
        return sum(self.data.truck_energy[(a, b)] for a, b in zip(segment, segment[1:]))

    def visited_anchors(self, sol: Solution) -> set[str]:
        return {node for route in sol.routes.values() for node in route[1:-1] if node in self.data.launch_nodes}

    def normalized_routes(self, routes: dict[int, list[str]]) -> dict[int, list[str]]:
        out = {}
        for v in self.data.truck_ids:
            route = list(routes.get(v, [self.data.depot]))
            if route == [self.data.depot, self.data.depot]:
                route = [self.data.depot]
            out[v] = route
        return out

    def route_signature(self, routes: dict[int, list[str]]) -> tuple[tuple[int, tuple[str, ...]], ...]:
        normalized = self.normalized_routes(routes)
        return tuple((int(v), tuple(normalized.get(v, [self.data.depot]))) for v in sorted(self.data.truck_ids))
