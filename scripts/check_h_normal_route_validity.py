from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdrp_enr.data import load_data  # noqa: E402
from sdrp_enr.evaluator import Evaluator  # noqa: E402
from sdrp_enr.solution import Solution  # noqa: E402
from sdrp_enr.service_decoder import FixedRouteServiceMILP  # noqa: E402


OUT_DIR = ROOT / "results" / "sdrp_enr_h_normal_route_validity_smoke"
SCENARIO = ROOT / "data" / "benchmark" / "suite_v2_n10_i2_t2.json"


def validation_sum(sol: Solution) -> float:
    return sum(float(v) for v in (sol.validation_metrics or {}).values())


def duplicate_h_metrics(data, routes: dict[int, list[str]], result) -> dict:
    h_occurrences = [node for route in routes.values() for node in route[1:-1] if node in data.h_nodes]
    tau = result.solution.tau if result.solution is not None else {}
    tau_positive = [(h, v) for (h, v), value in tau.items() if float(value) > 1e-9]
    tau_counts: dict[str, int] = {}
    for h, _v in tau_positive:
        tau_counts[str(h)] = tau_counts.get(str(h), 0) + 1
    return {
        "h_visit_count": len(h_occurrences),
        "unique_h_visit_count": len(set(h_occurrences)),
        "duplicate_h_visit_count": len(h_occurrences) - len(set(h_occurrences)),
        "positive_tau_count": len(tau_positive),
        "duplicate_positive_tau_count": sum(max(0, count - 1) for count in tau_counts.values()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(SCENARIO)
    decoder = FixedRouteServiceMILP(data, time_limit_sec=10, output_flag=0, allow_star=True, allow_rendezvous=True)
    evaluator = Evaluator(data)
    h_nodes = list(data.h_nodes)
    if len(h_nodes) < 2 or len(data.truck_ids) < 2:
        raise RuntimeError("Smoke scenario needs at least two H nodes and two trucks")
    h0, h1 = h_nodes[0], h_nodes[1]
    truck0, truck1 = data.truck_ids[0], data.truck_ids[1]
    base_empty = {v: [data.depot] for v in data.truck_ids}

    tests = []
    internal = {**base_empty, truck0: [data.depot, h0, data.depot, h1, data.depot]}
    tests.append(("internal_depot_invalid", internal, "route_invalid"))

    duplicate_across = {**base_empty, truck0: [data.depot, h0, data.depot], truck1: [data.depot, h0, data.depot]}
    tests.append(("duplicate_h_across_trucks_valid", duplicate_across, "OPTIMAL"))

    same_truck_duplicate = {**base_empty, truck0: [data.depot, h0, h1, h0, data.depot]}
    tests.append(("same_truck_duplicate_h_invalid", same_truck_duplicate, "route_invalid"))

    rows = []
    details = {}
    for name, routes, expected in tests:
        result = decoder.solve(routes)
        evaluated = evaluator.evaluate(Solution(routes=routes))
        metrics = duplicate_h_metrics(data, routes, result)
        row = {
            "test": name,
            "expected_status": expected,
            "service_status": result.status,
            "service_objective": result.objective,
            "service_validation_metric_sum": validation_sum(result.solution) if result.solution is not None else math.inf,
            "evaluator_validation_metric_sum": validation_sum(evaluated),
            "internal_depot_count": evaluated.validation_metrics.get("internal_depot_count", math.nan),
            **metrics,
            "passed": False,
        }
        if name == "internal_depot_invalid":
            row["passed"] = result.status == "route_invalid" and row["evaluator_validation_metric_sum"] > 0
        elif name == "duplicate_h_across_trucks_valid":
            row["passed"] = (
                result.status == "OPTIMAL"
                and row["service_validation_metric_sum"] <= 1e-9
                and row["duplicate_h_visit_count"] > 0
                and row["duplicate_positive_tau_count"] == 0
            )
        elif name == "same_truck_duplicate_h_invalid":
            row["passed"] = result.status == "route_invalid"
        rows.append(row)
        details[name] = {
            "routes": routes,
            "service_notes": result.notes,
            "evaluator_notes": evaluated.notes,
            "validation_metrics": evaluated.validation_metrics,
        }
    pd.DataFrame(rows).to_csv(OUT_DIR / "route_validity_smoke.csv", index=False)
    (OUT_DIR / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    passed = all(bool(row["passed"]) for row in rows)
    lines = [
        "# H-normal route validity smoke",
        "",
        f"- Scenario: `{SCENARIO.name}`.",
        f"- Passed: {passed}.",
        "- Internal depot route must be rejected by ServiceMILP-v2 and evaluator validation.",
        "- Duplicate H across trucks must be accepted while duplicate positive tau remains zero.",
        "- Same-truck duplicate H must be rejected.",
    ]
    (OUT_DIR / "brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
