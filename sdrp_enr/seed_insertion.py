from __future__ import annotations
import heapq
import math
import random
from dataclasses import dataclass, field
from .data import ProblemData, microgrid_utility
from .solution import Solution

@dataclass
class RouteContext:
    rng: random.Random
    quota: int
    removed_anchors: list[str] = field(default_factory=list)
    insert_score_mode: str = "total_potential"
    insert_potential_gamma: float = 1.0
    max_destroy_fragment_len: int = 1
    fragment_destroy_log: list[dict] = field(default_factory=list)

class RouteOperators:
    def __init__(self, data: ProblemData) -> None:
        self.data = data
        self.adj: dict[str, list[tuple[str, float]]] = {}
        for a, b in data.truck_arcs:
            self.adj.setdefault(a, []).append((b, data.truck_time[(a, b)]))

    def r_route_anchor(self, sol: Solution, ctx: RouteContext) -> Solution:
        return self._best_insertion(sol, ctx, mode="anchor")

    def _best_insertion(self, sol: Solution, ctx: RouteContext, mode: str) -> Solution:
        cand = sol.copy()
        for _ in range(ctx.quota):
            candidates = self._insertion_candidates(cand, ctx)
            if not candidates:
                break
            if mode == "regret":
                by_target: dict[str, list[dict]] = {}
                for item in candidates:
                    by_target.setdefault(item["target"], []).append(item)
                choices = []
                for items in by_target.values():
                    items.sort(key=lambda x: x["score"], reverse=True)
                    regret = items[0]["score"] - (items[1]["score"] if len(items) > 1 else 0.0)
                    choices.append((regret, items[0]))
                choice = max(choices, key=lambda x: x[0])[1]
            elif mode == "energy":
                choice = max(candidates, key=lambda x: x["phi"] - 0.5 * x["delta_energy"])
            elif mode == "anchor":
                choice = max(candidates, key=lambda x: x["phi"])
            else:
                choice = max(candidates, key=lambda x: x["score"])
            self._apply_insertion(cand, choice)
        return cand

    def _insertion_candidates(self, sol: Solution, ctx: RouteContext) -> list[dict]:
        targets = self._repair_targets(sol, ctx)
        current_anchors = self._visited_service_anchors(sol)
        candidates = []
        for target in targets:
            if target in current_anchors:
                continue
            for v, route in sol.routes.items():
                for pos in range(1, len(route)):
                    p, s = route[pos - 1], route[pos]
                    segment = self._expanded_segment(p, target, s)
                    if not segment:
                        continue
                    new_anchors = [
                        n for n in segment[1:-1]
                        if n in self.data.launch_nodes and n not in current_anchors
                    ]
                    if not new_anchors or target not in new_anchors:
                        continue
                    segment_existing = [n for n in segment[1:-1] if n in self.data.launch_nodes and n in current_anchors]
                    if segment_existing:
                        continue
                    delta_time = self._segment_time(segment) - self.data.truck_time.get((p, s), 0.0)
                    delta_energy = self._segment_energy(segment) - self.data.truck_energy.get((p, s), 0.0)
                    phi_total = sum(self._phi_anchor(sol, n, v) for n in new_anchors)
                    phi = self._normalized_segment_potential(phi_total, len(new_anchors), ctx)
                    score = phi - 2.0 * delta_time - 0.05 * delta_energy
                    if score <= 1e-9:
                        continue
                    candidates.append(
                        {
                            "truck": v,
                            "pos": pos,
                            "segment": segment,
                            "target": target,
                            "new_anchors": new_anchors,
                            "phi_total": phi_total,
                            "phi": phi,
                            "delta_time": delta_time,
                            "delta_energy": delta_energy,
                            "score": score,
                        }
                    )
        return candidates

    def _normalized_segment_potential(self, phi_total: float, new_anchor_count: int, ctx: RouteContext) -> float:
        if ctx.insert_score_mode != "avg_potential":
            return phi_total
        return phi_total / (1e-6 + max(0, new_anchor_count) ** ctx.insert_potential_gamma)

    def _repair_targets(self, sol: Solution, ctx: RouteContext) -> list[str]:
        unvisited = [n for n in self.data.launch_nodes if n not in self._visited_service_anchors(sol)]
        high = sorted(
            unvisited,
            key=lambda n: max(self._phi_anchor(sol, n, v) for v in self.data.truck_ids),
            reverse=True,
        )[: max(5, ctx.quota * 3)]
        out = []
        for node in ctx.removed_anchors + high:
            if node in self.data.launch_nodes and node not in out:
                out.append(node)
        return out

    def _expanded_segment(self, p: str, x: str, s: str) -> list[str] | None:
        p1 = self._shortest_path(p, x)
        p2 = self._shortest_path(x, s)
        if not p1 or not p2:
            return None
        segment = p1 + p2[1:]
        if any((a, b) not in self.data.truck_arcs for a, b in zip(segment, segment[1:])):
            return None
        return segment

    def _shortest_path(self, source: str, target: str) -> list[str] | None:
        heap = [(0.0, source, [source])]
        best = {source: 0.0}
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node == target:
                return path
            if cost > best.get(node, float("inf")) + 1e-12:
                continue
            for nxt, edge_cost in self.adj.get(node, []):
                new_cost = cost + edge_cost
                if new_cost < best.get(nxt, float("inf")):
                    best[nxt] = new_cost
                    heapq.heappush(heap, (new_cost, nxt, path + [nxt]))
        return None

    def _apply_insertion(self, sol: Solution, item: dict) -> None:
        v = int(item["truck"])
        pos = int(item["pos"])
        segment = list(item["segment"])
        route = sol.routes[v]
        sol.routes[v] = route[: pos - 1] + segment + route[pos + 1 :]

    def _segment_time(self, segment: list[str]) -> float:
        return sum(self.data.truck_time[(a, b)] for a, b in zip(segment, segment[1:]))

    def _segment_energy(self, segment: list[str]) -> float:
        return sum(self.data.truck_energy[(a, b)] for a, b in zip(segment, segment[1:]))

    def _visited_service_anchors(self, sol: Solution) -> set[str]:
        return {node for route in sol.routes.values() for node in route[1:-1] if node in self.data.launch_nodes}

    def _served_material(self, sol: Solution, node: str) -> float:
        served = sum(x.quantity for x in sol.truck_deliveries if x.node == node)
        if node in self.data.h_nodes or node in self.data.c_truck:
            if any(node in route[1:-1] for route in sol.routes.values()):
                served = max(served, self.data.material_demand(node))
        served += sum(self.data.drone_payload * x.sorties for x in sol.star_tasks if x.service == node)
        served += sum(self.data.drone_payload for x in sol.rendezvous_tasks if x.service == node)
        return min(served, self.data.material_demand(node))

    def _phi_anchor(self, sol: Solution, node: str, truck: int) -> float:
        phi = self._self_potential(sol, node)
        for i in self.data.c_nodes:
            if i == node:
                continue
            rem = max(0.0, self.data.material_demand(i) - self._served_material(sol, i))
            if rem <= 1e-9:
                continue
            reachable = self.data.drone_time(node, i, node) <= self.data.drone_tmax_hours + 1e-9
            if reachable:
                phi += self.data.population(i) * min(1.0, rem / self.data.material_demand(i))
        if node in self.data.h_nodes:
            denom = self.data.h_energy_demand.get(node, 0.0)
            if denom > 1e-9:
                current_energy = sum(float(self.data.nodes[node].P_o) * tau for (h, _v), tau in sol.tau.items() if h == node)
                c0 = min(1.0, current_energy / denom)
                add = max(0.0, denom - current_energy)
                phi += float(self.data.nodes[node].p_m) * (microgrid_utility(min(1.0, c0 + add / denom)) - microgrid_utility(c0))
        return phi

    def _self_potential(self, sol: Solution, node: str) -> float:
        if node not in self.data.h_nodes and node not in self.data.c_truck:
            return 0.0
        demand = self.data.material_demand(node)
        if demand <= 1e-9:
            return 0.0
        rem = max(0.0, demand - self._served_material(sol, node))
        return self.data.population(node) * min(1.0, rem / demand)
