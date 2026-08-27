"""Check a structural cleanup against recorded, pre-cleanup algorithm outputs.

Run with PYTHONHASHSEED=0 because the archived implementation iterates sets.
The snapshot intentionally excludes solver timing and alternate-optimum tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def collect() -> dict:
    from sdrp_enr.data import load_data
    from sdrp_enr.evaluation import generate_seed_scenario_pool, evaluate_pool_with_decoder
    from sdrp_enr.service_decoder import FixedRouteServiceMILP as Decoder
    from sdrp_enr.elite_refinement import EliteRefiner as Refiner, select_elites

    args = SimpleNamespace(
        service_milp_time_limit_sec=10.0, route_pool_max_size_per_scenario=500,
        original_random_count=20, balanced_random_count=20,
        edge_replacement_chain_max_len=3, rcl_size=15, ct_target_ratio=0.25,
        max_ct_chain_len_preference=3, target_route_time_utilization=0.85,
        target_energy_utilization=0.85, insertion_temperature=0.6,
        enable_integrated_h_bridge=True,
    )
    selected = [
        "suite_v2_n5_i1_t1", "suite_v2_n10_i2_t2", "suite_v2_n20_i2_t2",
        "suite_v2_n30_i2_t3", "suite_v2_n50_i2_t4",
    ]
    output = {}
    for scenario in selected:
        data = load_data(ROOT / "data" / "benchmark" / (scenario + ".json"))
        _, pool, generated = generate_seed_scenario_pool(data, scenario, 20260427, args)
        serial = [{"generator": c.generator, "routes": c.routes} for c in pool]
        fingerprint = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
        row = {"candidate_count": len(pool), "generated": generated, "pool_sha256": fingerprint}
        if len(data.nodes) <= 20:
            decoder = Decoder(data, time_limit_sec=10, no_truck_visited_ct_drone_service=True)
            candidates, _ = evaluate_pool_with_decoder(data, scenario, 20260427, pool[:8], decoder, math.nan, "service_milp_v2", True, True)
            row["decoder"] = [{"status": r["service_milp_status"], "objective": round(float(r["service_milp_objective"]), 6), "validation": r["validation_metric_sum"]} for r in candidates.to_dict("records")]
            refiner = Refiner(data, scenario, seed=20260427, service_milp_time_limit_sec=10, max_new_candidates=8, allowed_operators={"Drop-and-reinsert-LNS", "Rebalance-LNS"}, enable_integrated_h_bridge=True)
            refiner.service = decoder
            refiner.preload(candidates)
            best, moves, _ = refiner.refine(select_elites(candidates, 2), 2)
            row["refinement_objective"] = round(float(best["objective"]), 6)
            row["operators"] = [r["operator"] for r in moves]
        output[scenario] = row
        print(scenario, row["candidate_count"], fingerprint, flush=True)
    return output


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 for the migration regression.")
    path = ROOT / "tests" / "fixtures" / "algorithm_snapshot.json"
    actual = collect()
    expected = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        diagnostic = ROOT / "tmp" / "algorithm_snapshot_after.json"
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(json.dumps(actual, indent=2), encoding="utf-8")
        raise AssertionError(f"Migration mismatch; inspect {diagnostic}")
    print("PASS: all pool signatures, decoded objectives, and refinement results match.")


if __name__ == "__main__":
    main()
