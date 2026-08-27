from __future__ import annotations
from dataclasses import dataclass, field
from .solution import Solution

BIG_M = 1e4

@dataclass
class ServiceMILPResult:
    solution: Solution
    status: str
    status_code: int
    objective: float
    runtime_sec: float
    gap: float
    notes: list[str] = field(default_factory=list)
    num_vars: int = 0
    num_constrs: int = 0
