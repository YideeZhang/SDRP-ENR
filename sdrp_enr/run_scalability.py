from __future__ import annotations
import argparse
import math
import time
from dataclasses import replace
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from .data import ALPHA, BETA, SUCCESSOR_GAP_K, load_data
from .paths import ROOT
from .experiment_io import load_experiment_table, read_manifest
from .pool_base import signature_to_string, validation_metric_sum, route_metrics, routes_to_string
from .elite_refinement import (EliteRefiner, select_elites, extended_route_metrics,
    normalize_routes, parse_routes, route_signature)
from .service_decoder import FixedRouteServiceMILP
from .evaluation import evaluate_pool_with_decoder, generate_seed_scenario_pool, rel_gap

OUT_DIR = ROOT / "results" / "sdrp_enr_scalability"
MANIFEST = ROOT / "data" / "scalability" / "manifest.csv"
SEEDS = [20260427]
ALLOWED_OPERATORS = {"Drop-and-reinsert-LNS", "Rebalance-LNS"}
CSV_ENCODING = "utf-8-sig"




def theoretical_full_coverage_objective(data) -> float:
    material = sum(BETA * data.population(n) for n in data.h_nodes + data.c_nodes)
    microgrid = sum(ALPHA * float(data.nodes[h].p_m) for h in data.h_nodes)
    return float(material + microgrid)


def service_denominators(data) -> tuple[float, float]:
    material = sum(float(data.population(n)) for n in data.h_nodes + data.c_nodes)
    microgrid = sum(float(data.nodes[h].p_m) for h in data.h_nodes)
    return max(material, 1e-9), max(microgrid, 1e-9)


def make_decoder(data, args) -> FixedRouteServiceMILP:
    return FixedRouteServiceMILP(
        data,
        time_limit_sec=args.service_milp_time_limit_sec,
        output_flag=0,
        successor_gap=args.successor_gap,
        allow_star=True,
        allow_rendezvous=True,
        allow_microgrid_charging=True,
        no_truck_visited_ct_drone_service=True,
    )


def best_from_candidates(candidates: pd.DataFrame) -> dict | None:
    valid = candidates[
        candidates["service_milp_status"].eq("OPTIMAL")
        & candidates["validation_metric_sum"].le(1e-9)
        & candidates["service_milp_objective"].notna()
    ].copy()
    if valid.empty:
        return None
    return valid.loc[valid["service_milp_objective"].idxmax()].to_dict()


def refine_routes(data, scenario: str, seed: int, candidates: pd.DataFrame, args, decoder):
    best_before = best_from_candidates(candidates)
    if best_before is None:
        return None, [], [], {"cache_hit_count": 0, "cache_miss_count": 0, "accepted_move_count": 0, "improved_move_count": 0}
    refiner = EliteRefiner(
        data,
        scenario,
        seed=seed,
        service_milp_time_limit_sec=args.service_milp_time_limit_sec,
        max_new_candidates=args.max_new_candidates_per_scenario,
        allowed_operators=ALLOWED_OPERATORS,
        enable_integrated_h_bridge=True,
        enable_activation_rebalance=bool(getattr(args, "enable_activation_rebalance", False)),
        max_activation_rebalance_candidates=int(getattr(args, "max_activation_rebalance_candidates", 20)),
    )
    refiner.service = decoder
    refiner.preload(candidates)
    elites = select_elites(candidates, args.elite_k)
    best_lns, moves, summaries = refiner.refine(elites, args.lns_rounds_per_elite)
    if best_lns and math.isfinite(float(best_lns.get("objective", math.nan))) and float(best_lns["objective"]) >= float(best_before["service_milp_objective"]) - 1e-9:
        routes = normalize_routes(data, best_lns["routes"])
        result = {
            **best_before,
            "service_milp_objective": float(best_lns["objective"]),
            "route_nodes_by_truck": routes_to_string(routes),
            "best_lns_operator": best_lns.get("operator", ""),
            "_routes": routes,
        }
    else:
        result = {**best_before, "best_lns_operator": ""}
    return result, moves, refiner.service_rows, summaries[0] if summaries else {}


def h_metrics(data, routes: dict[int, list[str]]) -> dict:
    h_visits = [node for route in routes.values() for node in route[1:-1] if node in data.h_nodes]
    return {
        "h_visit_count": int(len(h_visits)),
        "unique_h_visit_count": int(len(set(h_visits))),
        "duplicate_h_visit_count": int(len(h_visits) - len(set(h_visits))),
    }


def ct_drone_service_count(data, routes: dict[int, list[str]], result) -> int:
    truck_visited_ct = {node for route in routes.values() for node in route[1:-1] if node in data.c_truck}
    if result.solution is None:
        return 0
    count = 0
    for task in result.solution.star_tasks:
        if task.service in truck_visited_ct:
            count += max(0, int(task.sorties))
    for task in result.solution.rendezvous_tasks:
        if task.service in truck_visited_ct:
            count += 1
    return int(count)


def used_truck_metrics(routes: dict[int, list[str]]) -> dict:
    used = sum(1 for route in routes.values() if len(route) > 2)
    return {"used_truck_count": int(used), "empty_truck_count": int(len(routes) - used)}


def completed_keys(out_dir: Path) -> set[tuple[str, int]]:
    path = out_dir / "all_runs.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return {(str(r["scenario"]), int(r["seed"])) for r in df.to_dict("records")}


def read_records(out_dir: Path, name: str) -> list[dict]:
    path = out_dir / name
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def write_outputs(out_dir: Path, run_rows: list[dict], move_rows: list[dict], service_rows: list[dict]) -> None:
    runs = pd.DataFrame(run_rows)
    moves = pd.DataFrame(move_rows)
    services = pd.DataFrame(service_rows)
    runs.to_csv(out_dir / "all_runs.csv", index=False, encoding=CSV_ENCODING)
    moves.to_csv(out_dir / "operator_moves_raw.csv", index=False, encoding=CSV_ENCODING)
    write_service_status(out_dir, services)
    write_summaries(out_dir, runs)
    write_operator_contribution(out_dir, moves)
    write_brief(out_dir, runs, moves)


def write_service_status(out_dir: Path, services: pd.DataFrame) -> None:
    required = ["scenario", "seed", "route_signature", "status", "objective", "runtime_sec", "gap", "validation_metric_sum", "star_count", "rendezvous_count", "positive_tau_count"]
    for col in required:
        if col not in services.columns:
            services[col] = math.nan
    services[required + [c for c in services.columns if c not in required]].to_csv(out_dir / "service_milp_status.csv", index=False, encoding=CSV_ENCODING)


def valid_rate(series: pd.Series) -> float:
    return float((pd.to_numeric(series, errors="coerce") <= 1e-9).mean()) if len(series) else math.nan


def write_summaries(out_dir: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        return
    runs.groupby("total_nodes").agg(
        scenario_count=("scenario", "nunique"),
        run_count=("scenario", "count"),
        valid_rate=("validation_metric_sum", valid_rate),
        mean_objective=("objective", "mean"),
        std_objective=("objective", "std"),
        mean_normalized_objective=("normalized_objective", "mean"),
        std_normalized_objective=("normalized_objective", "std"),
        mean_material_coverage_ratio=("material_coverage_ratio", "mean"),
        mean_microgrid_utility_ratio=("microgrid_utility_ratio", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
        median_runtime_sec=("runtime_sec", "median"),
        max_runtime_sec=("runtime_sec", "max"),
        mean_star_count=("star_count", "mean"),
        mean_rendezvous_count=("rendezvous_count", "mean"),
        mean_positive_tau_count=("positive_tau_count", "mean"),
        mean_used_truck_count=("used_truck_count", "mean"),
        mean_empty_truck_count=("empty_truck_count", "mean"),
        mean_route_anchor_count=("route_anchor_count", "mean"),
        mean_ct_anchor_count=("ct_anchor_count", "mean"),
        mean_duplicate_h_visit_count=("duplicate_h_visit_count", "mean"),
        mean_validation_metric_sum=("validation_metric_sum", "mean"),
        max_validation_metric_sum=("validation_metric_sum", "max"),
        max_drone_to_truck_visited_ct_count=("drone_to_truck_visited_ct_count", "max"),
    ).reset_index().to_csv(out_dir / "summary_by_nodes.csv", index=False, encoding=CSV_ENCODING)
    runs.groupby(["scenario", "total_nodes", "truck_count"]).agg(
        run_count=("scenario", "count"),
        valid_rate=("validation_metric_sum", valid_rate),
        mean_objective=("objective", "mean"),
        best_objective=("objective", "max"),
        worst_objective=("objective", "min"),
        std_objective=("objective", "std"),
        mean_normalized_objective=("normalized_objective", "mean"),
        best_normalized_objective=("normalized_objective", "max"),
        worst_normalized_objective=("normalized_objective", "min"),
        mean_runtime_sec=("runtime_sec", "mean"),
        max_runtime_sec=("runtime_sec", "max"),
        mean_material_coverage_ratio=("material_coverage_ratio", "mean"),
        mean_microgrid_utility_ratio=("microgrid_utility_ratio", "mean"),
        mean_star_count=("star_count", "mean"),
        mean_rendezvous_count=("rendezvous_count", "mean"),
        mean_positive_tau_count=("positive_tau_count", "mean"),
        mean_used_truck_count=("used_truck_count", "mean"),
        mean_empty_truck_count=("empty_truck_count", "mean"),
    ).reset_index().to_csv(out_dir / "summary_by_scenario.csv", index=False, encoding=CSV_ENCODING)
    runs[[
        "scenario",
        "total_nodes",
        "seed",
        "route_anchor_count",
        "h_anchor_count",
        "h_visit_count",
        "unique_h_visit_count",
        "duplicate_h_visit_count",
        "ct_anchor_count",
        "ct_chain_count",
        "max_ct_chain_length",
        "route_travel_time",
        "route_driving_energy",
        "route_time_balance_std",
        "route_energy_balance_std",
        "used_truck_count",
        "empty_truck_count",
    ]].to_csv(out_dir / "route_structure_summary.csv", index=False, encoding=CSV_ENCODING)
    runs[[
        "scenario",
        "total_nodes",
        "seed",
        "star_count",
        "rendezvous_count",
        "positive_tau_count",
        "material_objective_component",
        "microgrid_objective_component",
        "material_coverage_ratio",
        "microgrid_utility_ratio",
    ]].to_csv(out_dir / "service_mode_summary.csv", index=False, encoding=CSV_ENCODING)
    runs.groupby("total_nodes").agg(
        mean_normalized_objective=("normalized_objective", "mean"),
        std_normalized_objective=("normalized_objective", "std"),
        min_normalized_objective=("normalized_objective", "min"),
        max_normalized_objective=("normalized_objective", "max"),
    ).reset_index().to_csv(out_dir / "normalized_objective_summary.csv", index=False, encoding=CSV_ENCODING)


def write_operator_contribution(out_dir: Path, moves: pd.DataFrame) -> None:
    cols = ["operator", "selected_count", "candidate_count", "accepted_count", "improved_count", "mean_objective_delta", "mean_candidate_count", "best_improvement", "scenarios_improved_count"]
    if moves.empty:
        pd.DataFrame(columns=cols).to_csv(out_dir / "operator_contribution.csv", index=False, encoding=CSV_ENCODING)
        return
    moves = moves.copy()
    moves["objective_delta"] = moves["objective_after"] - moves["current_objective_before"]
    rows = []
    for operator, sub in moves.groupby("operator"):
        rows.append(
            {
                "operator": operator,
                "selected_count": len(sub),
                "candidate_count": sub["candidate_count"].sum(),
                "accepted_count": int(sub["accepted"].sum()),
                "improved_count": int(sub["improved"].sum()),
                "mean_objective_delta": sub["objective_delta"].mean(),
                "mean_candidate_count": sub["candidate_count"].mean(),
                "best_improvement": sub["objective_delta"].max(),
                "scenarios_improved_count": sub[sub["improved"] == True]["scenario"].nunique(),
            }
        )
    pd.DataFrame(rows)[cols].to_csv(out_dir / "operator_contribution.csv", index=False, encoding=CSV_ENCODING)




def write_brief(out_dir: Path, runs: pd.DataFrame, moves: pd.DataFrame) -> None:
    if runs.empty:
        (out_dir / "brief.md").write_text("# SDRP-ENR scalability\n\nNo runs completed.\n", encoding="utf-8")
        return
    summary = pd.read_csv(out_dir / "summary_by_nodes.csv")
    valid = valid_rate(runs["validation_metric_sum"])
    max_bad_ct = pd.to_numeric(runs["drone_to_truck_visited_ct_count"], errors="coerce").max()
    op_bad = int((moves.get("operator", pd.Series(dtype=str)).astype(str).str.contains("CT-chain|Segment", regex=True)).sum()) if not moves.empty else 0
    lines = [
        "# SDRP-ENR scalability",
        "",
        "## 结论",
        "",
        f"- 已完成 {len(runs)} runs，覆盖规模 {sorted(runs['total_nodes'].unique())}。",
        f"- valid rate = {valid:.3f}，max validation_metric_sum = {pd.to_numeric(runs['validation_metric_sum'], errors='coerce').max():.6g}。",
        f"- max drone_to_truck_visited_ct_count = {max_bad_ct:.6g}。",
        f"- operator log 中 CT-chain/Segment-exchange 记录数 = {op_bad}。",
        "",
        "## 按规模汇总",
        "",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"- {int(row['total_nodes'])} nodes: mean runtime={float(row['mean_runtime_sec']):.2f}s, "
            f"mean normalized objective={float(row['mean_normalized_objective']):.4f}, "
            f"valid_rate={float(row['valid_rate']):.3f}, empty trucks={float(row['mean_empty_truck_count']):.2f}."
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 本轮使用 final drone18 参数，直接读取已冻结参数的可移植算例。",
            "- 最终 elite refinement 只包含 Drop-reinsert 与 Truck rebalance；CT-chain 和 Segment-exchange 均未启用。",
            "- 大规模实验不报告 exact optimality gap，而通过 normalized objective、runtime、validity、route/service structure 评估可扩展性。",
        ]
    )
    (out_dir / "brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest).sort_values(["total_nodes", "scenario"])
    if args.max_scenarios > 0:
        manifest = manifest.head(args.max_scenarios)
    run_rows = read_records(out_dir, "all_runs.csv") if args.resume else []
    move_rows = read_records(out_dir, "operator_moves_raw.csv") if args.resume else []
    service_rows = read_records(out_dir, "service_milp_status.csv") if args.resume else []
    done = completed_keys(out_dir) if args.resume else set()

    for seed in args.seeds:
        for item in manifest.to_dict("records"):
            scenario = str(item["scenario"])
            if (scenario, int(seed)) in done:
                print(f"skip completed seed={seed} {scenario}", flush=True)
                continue
            data = load_data(Path(str(item["json"])))
            full_obj = theoretical_full_coverage_objective(data)
            material_total, microgrid_total = service_denominators(data)
            pool_start = time.perf_counter()
            _diag, pool, _generated = generate_seed_scenario_pool(data, scenario, int(seed), args)
            pool_runtime = time.perf_counter() - pool_start
            decoder = make_decoder(data, args)
            candidates, pool_services = evaluate_pool_with_decoder(data, scenario, int(seed), pool, decoder, math.nan, "service_milp_v2", True, True)
            service_rows.extend(pool_services)
            result, moves, services, lns_summary = refine_routes(data, scenario, int(seed), candidates, args, decoder)
            for move in moves:
                move["seed"] = int(seed)
                move_rows.append(move)
            service_rows.extend(services)
            base = {
                "scenario": scenario,
                "total_nodes": int(item["total_nodes"]),
                "seed": int(seed),
                "truck_count": int(item["truck_count"]),
                "drone_count": int(sum(len(v) for v in data.drones_by_truck.values())),
                "theoretical_full_coverage_objective": full_obj,
            }
            if result is None:
                run_rows.append({**base, "objective": math.nan, "normalized_objective": math.nan, "runtime_sec": time.perf_counter() - pool_start, "status": "NO_VALID_ROUTE", "service_milp_status": "NO_VALID_ROUTE", "validation_metric_sum": math.inf})
                write_outputs(out_dir, run_rows, move_rows, service_rows)
                continue
            routes = normalize_routes(data, result.get("_routes") if "_routes" in result else parse_routes(str(result["route_nodes_by_truck"])))
            final_result = decoder.solve(routes)
            metrics = extended_route_metrics(data, routes)
            hm = h_metrics(data, routes)
            used = used_truck_metrics(routes)
            counts = final_result.solution.counts() if final_result.solution is not None else {}
            material = float(final_result.solution.served_material_score) if final_result.solution is not None else math.nan
            microgrid = float(final_result.solution.microgrid_score) if final_result.solution is not None else math.nan
            obj = float(final_result.objective)
            row = {
                **base,
                "objective": obj,
                "normalized_objective": obj / full_obj if full_obj > 1e-9 else math.nan,
                "material_coverage_ratio": material / material_total if math.isfinite(material) else math.nan,
                "microgrid_utility_ratio": microgrid / microgrid_total if math.isfinite(microgrid) else math.nan,
                "runtime_sec": time.perf_counter() - pool_start,
                "status": final_result.status,
                "service_milp_status": final_result.status,
                "validation_metric_sum": validation_metric_sum(final_result),
                "drone_to_truck_visited_ct_count": ct_drone_service_count(data, routes, final_result),
                "route_signature": signature_to_string(route_signature(data, routes)),
                **metrics,
                **hm,
                "star_count": int(counts.get("star_rows", 0)),
                "rendezvous_count": int(counts.get("rendezvous_rows", 0)),
                "positive_tau_count": int(len(final_result.solution.tau)) if final_result.solution is not None else 0,
                "material_objective_component": material,
                "microgrid_objective_component": microgrid,
                **used,
            }
            run_rows.append(row)
            service_rows.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "route_signature": row["route_signature"],
                    "status": final_result.status,
                    "objective": final_result.objective,
                    "runtime_sec": final_result.runtime_sec,
                    "gap": final_result.gap,
                    "validation_metric_sum": row["validation_metric_sum"],
                    "star_count": row["star_count"],
                    "rendezvous_count": row["rendezvous_count"],
                    "positive_tau_count": row["positive_tau_count"],
                }
            )
            write_outputs(out_dir, run_rows, move_rows, service_rows)
            print(f"seed={seed} {scenario}: pool={len(pool)} obj={obj:.2f} norm={row['normalized_objective']:.4f}", flush=True)
    write_outputs(out_dir, run_rows, move_rows, service_rows)
    print(f"output_folder={out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--elite-k", type=int, default=10)
    parser.add_argument("--lns-rounds-per-elite", type=int, default=20)
    parser.add_argument("--max-new-candidates-per-scenario", type=int, default=200)
    parser.add_argument("--service-milp-time-limit-sec", type=float, default=10.0)
    parser.add_argument("--successor-gap", type=int, default=SUCCESSOR_GAP_K)
    parser.add_argument("--route-pool-max-size-per-scenario", type=int, default=500)
    parser.add_argument("--original-random-count", type=int, default=100)
    parser.add_argument("--balanced-random-count", type=int, default=100)
    parser.add_argument("--edge-replacement-chain-max-len", type=int, default=3)
    parser.add_argument("--rcl-size", type=int, default=15)
    parser.add_argument("--ct-target-ratio", type=float, default=0.25)
    parser.add_argument("--max-ct-chain-len-preference", type=int, default=3)
    parser.add_argument("--target-route-time-utilization", type=float, default=0.85)
    parser.add_argument("--target-energy-utilization", type=float, default=0.85)
    parser.add_argument("--insertion-temperature", type=float, default=0.6)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-activation-rebalance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-activation-rebalance-candidates", type=int, default=20)
    args = parser.parse_args()
    args.enable_integrated_h_bridge = True
    run(args)


if __name__ == "__main__":
    main()
