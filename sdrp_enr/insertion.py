from __future__ import annotations
import heapq

class InsertionPaths:
    def __init__(self, data):
        self.data = data
        self.adj = {}
        for a, b in data.truck_arcs:
            self.adj.setdefault(a, []).append((b, data.truck_time[(a, b)]))

    def _expanded_segment(self, p: str, x: str, s: str) -> list[str] | None:
        p1 = self._shortest_path(p, x)
        p2 = self._shortest_path(x, s)
        if not p1 or not p2:
            return None
        segment = p1 + p2[1:]
        if any(node == self.data.depot for node in segment[1:-1]):
            return None
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

    def _cover(self, anchor: str) -> set[str]:
        if anchor not in self.data.launch_nodes:
            return set()
        return {
            i for i in self.data.c_nodes
            if i != anchor and self.data.drone_time(anchor, i, anchor) <= self.data.drone_tmax_hours + 1e-9
        }
