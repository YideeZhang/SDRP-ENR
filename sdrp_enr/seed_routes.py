"""Deterministic H-seed construction used by the accepted route pool."""
from __future__ import annotations
import random
from .solution import Solution
from .seed_insertion import RouteContext, RouteOperators


class HSeedConstructor:
    def __init__(self, data, seed):
        self.data = data
        self.rng = random.Random(seed)
        self.route_operators = RouteOperators(data)

    def construct(self):
        sol = Solution(routes={v: [self.data.depot, self.data.depot] for v in self.data.truck_ids})
        for idx, h in enumerate(self.data.h_nodes):
            self._insert_initial_anchor(sol, h, self.data.truck_ids[idx % len(self.data.truck_ids)])
        remaining = [h for h in self.data.h_nodes if h not in {n for r in sol.routes.values() for n in r}]
        if remaining:
            ctx = RouteContext(rng=self.rng, quota=len(remaining), removed_anchors=remaining,
                               insert_score_mode="avg_potential", insert_potential_gamma=1.0)
            sol = self.route_operators.r_route_anchor(sol, ctx)
        return sol

    def _insert_initial_anchor(self, sol: Solution, anchor: str, truck: int) -> None:
        route = sol.routes[truck]
        current = {n for r in sol.routes.values() for n in r[1:-1] if n in self.data.launch_nodes}
        if anchor in current:
            return
        best = None
        for pos in range(1, len(route)):
            p, s = route[pos - 1], route[pos]
            segment = self.route_operators._expanded_segment(p, anchor, s)
            if not segment:
                continue
            if any(n in current for n in segment[1:-1] if n in self.data.launch_nodes):
                continue
            cost = sum(self.data.truck_time[(a, b)] for a, b in zip(segment, segment[1:]))
            if best is None or cost < best[0]:
                best = (cost, pos, segment)
        if best is not None:
            _cost, pos, segment = best
            sol.routes[truck] = route[: pos - 1] + segment + route[pos + 1 :]
