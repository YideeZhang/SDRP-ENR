from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TruckDelivery:
    node: str
    truck: int
    quantity: float


@dataclass
class StarTask:
    launch: str
    service: str
    truck: int
    drone: int
    sorties: int = 1


@dataclass
class RendezvousTask:
    launch: str
    service: str
    recover: str
    truck: int
    drone: int


@dataclass
class TruckSchedule:
    arrival: dict[str, float] = field(default_factory=dict)
    departure: dict[str, float] = field(default_factory=dict)
    sigma: dict[str, float] = field(default_factory=dict)
    waiting: dict[str, float] = field(default_factory=dict)
    return_time: float = 0.0


@dataclass
class Solution:
    routes: dict[int, list[str]]
    truck_deliveries: list[TruckDelivery] = field(default_factory=list)
    tau: dict[tuple[str, int], float] = field(default_factory=dict)
    star_tasks: list[StarTask] = field(default_factory=list)
    rendezvous_tasks: list[RendezvousTask] = field(default_factory=list)
    schedules: dict[int, TruckSchedule] = field(default_factory=dict)
    objective: float = 0.0
    served_material_score: float = 0.0
    microgrid_score: float = 0.0
    feasible: bool = False
    feasibility_status: str = "not_evaluated"
    notes: list[str] = field(default_factory=list)
    validation_metrics: dict[str, float] = field(default_factory=dict)

    def copy(self) -> "Solution":
        return copy.deepcopy(self)

    def visited_by_truck(self, truck: int) -> set[str]:
        return set(self.routes.get(truck, []))

    def route_position(self, truck: int, node: str) -> int | None:
        route = self.routes.get(truck, [])
        try:
            return route.index(node)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "routes": self.routes,
            "truck_deliveries": [asdict(x) for x in self.truck_deliveries],
            "tau": [{"h": h, "truck": v, "tau": value} for (h, v), value in self.tau.items() if value > 1e-9],
            "star_tasks": [asdict(x) for x in self.star_tasks],
            "rendezvous_tasks": [asdict(x) for x in self.rendezvous_tasks],
            "schedules": {
                str(v): {
                    "arrival": sch.arrival,
                    "departure": sch.departure,
                    "sigma": sch.sigma,
                    "waiting": sch.waiting,
                    "return_time": sch.return_time,
                }
                for v, sch in self.schedules.items()
            },
            "objective": self.objective,
            "served_material_score": self.served_material_score,
            "microgrid_score": self.microgrid_score,
            "feasible": self.feasible,
            "feasibility_status": self.feasibility_status,
            "notes": self.notes,
            "validation_metrics": self.validation_metrics,
        }

    def counts(self) -> dict[str, int]:
        return {
            "truck_delivery_rows": len(self.truck_deliveries),
            "star_rows": sum(max(0, int(t.sorties)) for t in self.star_tasks),
            "rendezvous_rows": len(self.rendezvous_tasks),
        }
