from __future__ import annotations
import math
import pandas as pd
from .route_pool import RoutePoolGenerator
from .pool_base import route_metrics, routes_to_string, signature_to_string, validation_metric_sum
from .elite_refinement import normalize_routes, route_signature

def rel_gap(reference: float, value: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or abs(reference) <= 1e-9:
        return math.nan
    return (reference - value) / abs(reference)


def sol_counts(result) -> dict:
    if result.solution is None:
        return {"star_count": 0, "rendezvous_count": 0, "positive_tau_count": 0, "material_objective": math.nan, "microgrid_objective": math.nan}
    counts = result.solution.counts()
    return {
        "star_count": int(counts.get("star_rows", 0)),
        "rendezvous_count": int(counts.get("rendezvous_rows", 0)),
        "positive_tau_count": int(len(result.solution.tau)),
        "material_objective": float(result.solution.served_material_score),
        "microgrid_objective": float(result.solution.microgrid_score),
    }


def generate_seed_scenario_pool(data, scenario: str, seed: int, args):
    diag = RoutePoolGenerator(
        data=data,
        scenario=scenario,
        seed=seed,
        service_milp_time_limit_sec=args.service_milp_time_limit_sec,
        max_pool_size=args.route_pool_max_size_per_scenario,
        random_route_count=args.original_random_count,
        edge_chain_max_len=args.edge_replacement_chain_max_len,
        balanced_random_count=args.balanced_random_count,
        rcl_size=args.rcl_size,
        ct_target_ratio=args.ct_target_ratio,
        max_ct_chain_len_preference=args.max_ct_chain_len_preference,
        target_route_time_utilization=args.target_route_time_utilization,
        target_energy_utilization=args.target_energy_utilization,
        insertion_temperature=args.insertion_temperature,
        h_duplicate_top_k=getattr(args, "h_duplicate_top_k", 10),
        max_h_duplicate_candidates=getattr(args, "max_h_duplicate_candidates", 100),
        h_duplicate_lambda_t=getattr(args, "h_duplicate_lambda_t", 0.1),
        h_duplicate_lambda_e=getattr(args, "h_duplicate_lambda_e", 0.01),
        enable_integrated_h_bridge=getattr(args, "enable_integrated_h_bridge", True),
        bridge_h_choice_probability=getattr(args, "bridge_h_choice_probability", 0.25),
    )
    pool = diag.generate_pool()
    return diag, pool, dict(diag.generated_by_generator)


def evaluate_pool_with_decoder(data, scenario: str, seed: int, pool, decoder_obj, gurobi_objective: float, decoder: str, allow_star: bool, allow_rendezvous: bool):
    rows = []
    service_rows = []
    cache = {}
    for cand in pool:
        routes = normalize_routes(data, cand.routes)
        sig = route_signature(data, routes)
        if sig in cache:
            result = cache[sig]
            cache_hit = True
        else:
            result = decoder_obj.solve(routes)
            cache[sig] = result
            cache_hit = False
        metrics = route_metrics(data, routes)
        counts = sol_counts(result)
        validation_sum = validation_metric_sum(result)
        route_sig = signature_to_string(sig)
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "generator": cand.generator,
                "route_signature": route_sig,
                "route_nodes_by_truck": routes_to_string(routes),
                **metrics,
                "service_milp_status": result.status,
                "service_milp_objective": result.objective,
                "relative_gap_to_gurobi": rel_gap(gurobi_objective, result.objective),
                "runtime_sec": result.runtime_sec,
                "validation_metric_sum": validation_sum,
                "decoder": decoder,
                "allow_star": allow_star,
                "allow_rendezvous": allow_rendezvous,
                **counts,
            }
        )
        service_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "variant": "POOL_EVAL",
                "decoder": decoder,
                "route_signature": route_sig,
                "status": result.status,
                "objective": result.objective,
                "runtime_sec": result.runtime_sec,
                "gap": result.gap,
                "validation_metric_sum": validation_sum,
                "cache_hit": cache_hit,
                **counts,
            }
        )
    return pd.DataFrame(rows), service_rows
