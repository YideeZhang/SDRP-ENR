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

OUT_DIR = ROOT / "results" / "sdrp_enr_ablation"
MANIFEST = ROOT / "data" / "benchmark" / "manifest.csv"
BASELINE = ROOT / "data" / "reference" / "baseline_summary.csv"
SEEDS = [20260427, 20260428, 20260429]
CSV_ENCODING = "utf-8-sig"

FULL_GENERATORS = {"H_only_current_initial", "EdgeReplacement", "RandomizedInsertion", "BalancedBiasedRandomizedInsertion"}
FULL_LNS = {"Drop-and-reinsert-LNS", "Rebalance-LNS"}

VARIANTS = [
    ("FULL_SDRP_ENR", "service"),
    ("NO_DRONE", "service"),
    ("NO_STAR", "service"),
    ("NO_RENDEZVOUS", "service"),
    ("NO_MICROGRID_CHARGING", "service"),
    ("H_SEED_ONLY", "generator"),
    ("GEN_NO_H_SEED", "generator"),
    ("GEN_NO_EDGE_ENRICHMENT", "generator"),
    ("GEN_NO_RANDOM_INSERTION", "generator"),
    ("GEN_NO_BALANCED_INSERTION", "generator"),
    ("NO_ELITE_REFINEMENT", "elite"),
    ("LNS_NO_DROP_REINSERT", "elite"),
    ("LNS_NO_REBALANCE", "elite"),
]


def service_switches(variant: str) -> tuple[bool, bool, bool]:
    allow_star = variant != "NO_DRONE"
    allow_rendezvous = variant not in {"NO_DRONE", "NO_RENDEZVOUS"}
    allow_microgrid = variant != "NO_MICROGRID_CHARGING"
    if variant == "NO_STAR":
        allow_star = False
        allow_rendezvous = True
    return allow_star, allow_rendezvous, allow_microgrid


def decoder_key(variant: str) -> str:
    allow_star, allow_rendezvous, allow_microgrid = service_switches(variant)
    return f"star{int(allow_star)}_rv{int(allow_rendezvous)}_grid{int(allow_microgrid)}"


def generator_filter(variant: str) -> set[str]:
    if variant == "H_SEED_ONLY":
        return {"H_only_current_initial"}
    gens = set(FULL_GENERATORS)
    if variant == "GEN_NO_H_SEED":
        gens.discard("H_only_current_initial")
    elif variant == "GEN_NO_EDGE_ENRICHMENT":
        gens.discard("EdgeReplacement")
    elif variant == "GEN_NO_RANDOM_INSERTION":
        gens.discard("RandomizedInsertion")
    elif variant == "GEN_NO_BALANCED_INSERTION":
        gens.discard("BalancedBiasedRandomizedInsertion")
    return gens


def lns_filter(variant: str) -> set[str]:
    if variant in {"H_SEED_ONLY", "NO_ELITE_REFINEMENT"}:
        return set()
    ops = set(FULL_LNS)
    if variant == "LNS_NO_DROP_REINSERT":
        ops.discard("Drop-and-reinsert-LNS")
    elif variant == "LNS_NO_REBALANCE":
        ops.discard("Rebalance-LNS")
    return ops


def make_decoder(data, args, variant: str) -> FixedRouteServiceMILP:
    allow_star, allow_rendezvous, allow_microgrid = service_switches(variant)
    return FixedRouteServiceMILP(
        data,
        time_limit_sec=args.service_milp_time_limit_sec,
        output_flag=0,
        successor_gap=args.successor_gap,
        allow_star=allow_star,
        allow_rendezvous=allow_rendezvous,
        allow_microgrid_charging=allow_microgrid,
        no_truck_visited_ct_drone_service=True,
    )


def best_from_candidates(candidates: pd.DataFrame) -> dict | None:
    if candidates.empty:
        return None
    valid = candidates[
        candidates["service_milp_status"].eq("OPTIMAL")
        & candidates["validation_metric_sum"].le(1e-9)
        & candidates["service_milp_objective"].notna()
    ].copy()
    if valid.empty:
        return None
    return valid.loc[valid["service_milp_objective"].idxmax()].to_dict()


def run_variant(data, scenario: str, seed: int, variant: str, candidates: pd.DataFrame, args, decoder_obj):
    sub = candidates[candidates["generator"].isin(generator_filter(variant))].copy()
    best_before = best_from_candidates(sub)
    if best_before is None:
        return None, [], [], {"cache_hit_count": 0, "cache_miss_count": 0, "accepted_move_count": 0, "improved_move_count": 0}
    allowed_ops = lns_filter(variant)
    if not allowed_ops:
        return best_before, [], [], {"cache_hit_count": 0, "cache_miss_count": 0, "accepted_move_count": 0, "improved_move_count": 0}
    refiner = EliteRefiner(
        data,
        scenario,
        seed=seed,
        service_milp_time_limit_sec=args.service_milp_time_limit_sec,
        max_new_candidates=args.max_new_candidates_per_scenario,
        allowed_operators=allowed_ops,
        enable_integrated_h_bridge=True,
    )
    refiner.service = decoder_obj
    refiner.preload(sub)
    elites = select_elites(sub, args.elite_k)
    best_lns, moves, summaries = refiner.refine(elites, args.lns_rounds_per_elite)
    if best_lns and math.isfinite(float(best_lns.get("objective", math.nan))) and float(best_lns["objective"]) >= float(best_before["service_milp_objective"]) - 1e-9:
        routes = normalize_routes(data, best_lns["routes"])
        result = {
            **best_before,
            "service_milp_objective": float(best_lns["objective"]),
            "route_nodes_by_truck": routes_to_string(routes),
            **route_metrics(data, routes),
            "best_lns_operator": best_lns.get("operator", ""),
            "_routes": routes,
        }
    else:
        result = {**best_before, "best_lns_operator": ""}
    return result, moves, refiner.service_rows, summaries[0] if summaries else {}


def sol_counts(result) -> dict:
    if result.solution is None:
        return {"star_count": 0, "rendezvous_count": 0, "positive_tau_count": 0, "material_objective_component": math.nan, "microgrid_objective_component": math.nan}
    counts = result.solution.counts()
    return {
        "star_count": int(counts.get("star_rows", 0)),
        "rendezvous_count": int(counts.get("rendezvous_rows", 0)),
        "positive_tau_count": int(len(result.solution.tau)),
        "material_objective_component": float(result.solution.served_material_score),
        "microgrid_objective_component": float(result.solution.microgrid_score),
    }


def service_denominators(data) -> tuple[float, float]:
    material = sum(float(data.population(n)) for n in data.h_nodes + data.c_nodes)
    microgrid = sum(float(data.nodes[h].p_m) for h in data.h_nodes)
    return max(material, 1e-9), max(microgrid, 1e-9)


def ct_metrics(data, routes: dict[int, list[str]], result) -> dict:
    ct_trucks: dict[str, set[int]] = {}
    for v, route in routes.items():
        seen = set()
        for node in route[1:-1]:
            if node in data.c_truck and node not in seen:
                ct_trucks.setdefault(node, set()).add(int(v))
                seen.add(node)
    drone_to_ct = 0
    if result.solution is not None:
        for task in result.solution.star_tasks:
            if task.service in ct_trucks:
                drone_to_ct += max(0, int(task.sorties))
        for task in result.solution.rendezvous_tasks:
            if task.service in ct_trucks:
                drone_to_ct += 1
    return {
        "cross_truck_duplicate_ct_visit_count": int(sum(max(0, len(vs) - 1) for vs in ct_trucks.values())),
        "drone_to_truck_visited_ct_count": int(drone_to_ct),
    }


def h_metrics(data, routes: dict[int, list[str]]) -> dict:
    h_visits = [node for route in routes.values() for node in route[1:-1] if node in data.h_nodes]
    return {
        "h_visit_count": int(len(h_visits)),
        "unique_h_visit_count": int(len(set(h_visits))),
        "duplicate_h_visit_count": int(len(h_visits) - len(set(h_visits))),
    }


def theoretical_upper(row: dict) -> float:
    obj = float(row.get("baseline_objective", math.nan))
    gap = float(row.get("baseline_gap", 0.0) or 0.0)
    if not math.isfinite(obj):
        return math.nan
    if str(row.get("baseline_status", "")).upper() == "OPTIMAL":
        return obj
    return obj * (1.0 + max(gap, 0.0))


def load_table(args):
    return load_experiment_table(args)


def completed_keys(out_dir: Path) -> set[tuple[str, int]]:
    path = out_dir / "all_runs.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return {(str(r["scenario"]), int(r["seed"])) for r in df.to_dict("records")}


def write_csv(out_dir: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(out_dir / name, index=False, encoding=CSV_ENCODING)


def write_outputs(out_dir: Path, run_rows: list[dict], move_rows: list[dict], service_rows: list[dict], candidate_rows: list[dict], generator_rows: list[dict], structure_rows: list[dict]) -> None:
    runs = pd.DataFrame(run_rows)
    moves = pd.DataFrame(move_rows)
    services = pd.DataFrame(service_rows)
    candidates = pd.DataFrame(candidate_rows)
    generators = pd.DataFrame(generator_rows)
    structures = pd.DataFrame(structure_rows)
    runs.to_csv(out_dir / "all_runs.csv", index=False, encoding=CSV_ENCODING)
    write_summary_tables(out_dir, runs)
    write_ablation_tables(out_dir, runs)
    write_operator_contribution(out_dir, moves)
    write_generator_contribution(out_dir, candidates, generators, runs)
    write_structure_tables(out_dir, runs, structures)
    write_service_status(out_dir, services)
    write_paper_suite_compat_outputs(out_dir, runs)
    write_brief(out_dir, runs)


def valid_rate(series: pd.Series) -> float:
    return float((pd.to_numeric(series, errors="coerce") <= 1e-9).mean()) if len(series) else math.nan


def write_summary_tables(out_dir: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        return
    runs.groupby(["variant", "ablation_group"]).agg(
        scenario_count=("scenario", "nunique"),
        run_count=("scenario", "count"),
        mean_gap_to_exact_if_available=("gap_to_exact_if_available", "mean"),
        median_gap_to_exact_if_available=("gap_to_exact_if_available", "median"),
        mean_objective=("objective", "mean"),
        mean_material_objective_component=("material_objective_component", "mean"),
        mean_microgrid_objective_component=("microgrid_objective_component", "mean"),
        mean_material_coverage_ratio=("material_coverage_ratio", "mean"),
        mean_microgrid_utility_ratio=("microgrid_utility_ratio", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
        mean_star_count=("star_count", "mean"),
        mean_rendezvous_count=("rendezvous_count", "mean"),
        mean_positive_tau_count=("positive_tau_count", "mean"),
        valid_rate=("validation_metric_sum", valid_rate),
        ).reset_index().to_csv(out_dir / "summary_by_variant.csv", index=False, encoding=CSV_ENCODING)
    runs.groupby(["variant", "ablation_group", "total_nodes"]).agg(
        scenario_count=("scenario", "nunique"),
        run_count=("scenario", "count"),
        mean_gap_to_exact_if_available=("gap_to_exact_if_available", "mean"),
        median_gap_to_exact_if_available=("gap_to_exact_if_available", "median"),
        mean_objective=("objective", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
        mean_star_count=("star_count", "mean"),
        mean_rendezvous_count=("rendezvous_count", "mean"),
        mean_positive_tau_count=("positive_tau_count", "mean"),
        valid_rate=("validation_metric_sum", valid_rate),
    ).reset_index().to_csv(out_dir / "summary_by_variant_nodes.csv", index=False, encoding=CSV_ENCODING)
    if "instance_group" in runs.columns:
        runs.groupby(["variant", "instance_group"]).agg(
            scenario_count=("scenario", "nunique"),
            run_count=("scenario", "count"),
            mean_gap_to_milp_incumbent_percent=("gap_to_milp_incumbent_percent", "mean"),
            median_gap_to_milp_incumbent_percent=("gap_to_milp_incumbent_percent", "median"),
            worst_gap_to_milp_incumbent_percent=("gap_to_milp_incumbent_percent", "max"),
            mean_objective=("objective", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            valid_rate=("validation_metric_sum", valid_rate),
            mean_star_count=("star_count", "mean"),
            mean_rendezvous_count=("rendezvous_count", "mean"),
            mean_material_coverage_ratio=("material_coverage_ratio", "mean"),
            mean_microgrid_utility_ratio=("microgrid_utility_ratio", "mean"),
        ).reset_index().to_csv(out_dir / "summary_by_variant_group.csv", index=False, encoding=CSV_ENCODING)


def paired(runs: pd.DataFrame, other: str) -> pd.DataFrame:
    return runs[runs["variant"].eq("FULL_SDRP_ENR")].merge(
        runs[runs["variant"].eq(other)],
        on=["scenario", "seed"],
        suffixes=("_full", "_ablation"),
    )


def effect_rows(runs: pd.DataFrame, variants: list[str], group_name: str) -> pd.DataFrame:
    rows = []
    for variant in variants:
        m = paired(runs, variant)
        if m.empty:
            continue
        delta_obj = m["objective_ablation"].mean() - m["objective_full"].mean()
        delta_gap = m["gap_to_exact_if_available_ablation"].mean() - m["gap_to_exact_if_available_full"].mean()
        rows.append(
            {
                "ablation_variant": variant,
                "baseline_variant": "FULL_SDRP_ENR",
                "ablation_group": group_name,
                "mean_gap_full": m["gap_to_exact_if_available_full"].mean(),
                "mean_gap_ablation": m["gap_to_exact_if_available_ablation"].mean(),
                "delta_gap": delta_gap,
                "mean_objective_full": m["objective_full"].mean(),
                "mean_objective_ablation": m["objective_ablation"].mean(),
                "delta_objective": delta_obj,
                "mean_runtime_full": m["runtime_sec_full"].mean(),
                "mean_runtime_ablation": m["runtime_sec_ablation"].mean(),
                "delta_runtime": m["runtime_sec_ablation"].mean() - m["runtime_sec_full"].mean(),
                "interpretation": "该组件有正贡献" if delta_gap > 0 else "该组件贡献不明显或存在随机噪声",
            }
        )
    return pd.DataFrame(rows)


def write_ablation_tables(out_dir: Path, runs: pd.DataFrame) -> None:
    service = ["NO_DRONE", "NO_STAR", "NO_RENDEZVOUS", "NO_MICROGRID_CHARGING"]
    generators = ["H_SEED_ONLY", "GEN_NO_H_SEED", "GEN_NO_EDGE_ENRICHMENT", "GEN_NO_RANDOM_INSERTION", "GEN_NO_BALANCED_INSERTION"]
    elite = ["NO_ELITE_REFINEMENT", "LNS_NO_DROP_REINSERT", "LNS_NO_REBALANCE"]
    service_df = effect_rows(runs, service, "service")
    gen_df = effect_rows(runs, generators, "generator")
    elite_df = effect_rows(runs, elite, "elite")
    pd.concat([service_df, gen_df, elite_df], ignore_index=True).to_csv(out_dir / "ablation_effect_table.csv", index=False, encoding=CSV_ENCODING)
    service_df.to_csv(out_dir / "service_mechanism_ablation_table.csv", index=False, encoding=CSV_ENCODING)
    gen_df.to_csv(out_dir / "route_pool_ablation_table.csv", index=False, encoding=CSV_ENCODING)
    elite_df.to_csv(out_dir / "elite_refinement_ablation_table.csv", index=False, encoding=CSV_ENCODING)
    service_df.to_csv(out_dir / "service_mechanism_ablation.csv", index=False, encoding=CSV_ENCODING)
    gen_df.to_csv(out_dir / "route_pool_ablation.csv", index=False, encoding=CSV_ENCODING)
    elite_df.to_csv(out_dir / "elite_refinement_ablation.csv", index=False, encoding=CSV_ENCODING)


def write_operator_contribution(out_dir: Path, moves: pd.DataFrame) -> None:
    cols = ["variant", "operator", "selected_count", "candidate_count", "accepted_count", "improved_count", "mean_objective_delta", "mean_candidate_count", "best_improvement", "scenarios_improved_count"]
    if moves.empty:
        pd.DataFrame(columns=cols).to_csv(out_dir / "operator_contribution.csv", index=False, encoding=CSV_ENCODING)
        return
    moves = moves.copy()
    moves["objective_delta"] = moves["objective_after"] - moves["current_objective_before"]
    rows = []
    for (variant, operator), sub in moves.groupby(["variant", "operator"]):
        rows.append(
            {
                "variant": variant,
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


def write_generator_contribution(out_dir: Path, candidates: pd.DataFrame, raw: pd.DataFrame, runs: pd.DataFrame) -> None:
    rows = []
    for variant, sub_runs in runs.groupby("variant"):
        dkey = decoder_key(variant)
        sub_cand = candidates[candidates["decoder_key"].eq(dkey)] if not candidates.empty else pd.DataFrame()
        for gen in sorted(FULL_GENERATORS):
            g = sub_cand[sub_cand["generator"].eq(gen)] if not sub_cand.empty else pd.DataFrame()
            valid = g[g["validation_metric_sum"].le(1e-9) & g["service_milp_status"].eq("OPTIMAL")] if not g.empty else pd.DataFrame()
            related = sub_runs[sub_runs["best_generator"].eq(gen)]
            rows.append(
                {
                    "variant": variant,
                    "generator": gen,
                    "raw_generated_count": int(raw[raw["generator"].eq(gen)]["raw_generated_count"].sum()) if not raw.empty else 0,
                    "unique_count": g["route_signature"].nunique() if not g.empty else 0,
                    "optimal_service_count": len(valid),
                    "best_count_by_scenario": related.groupby(["scenario", "seed"]).ngroups,
                    "mean_gap_to_exact_if_available": related["gap_to_exact_if_available"].mean(),
                    "mean_ct_anchor_count": related["ct_anchor_count"].mean(),
                    "mean_ct_chain_count": related["ct_chain_count"].mean(),
                    "mean_runtime_sec": related["runtime_sec"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "generator_contribution.csv", index=False, encoding=CSV_ENCODING)


def write_structure_tables(out_dir: Path, runs: pd.DataFrame, structures: pd.DataFrame) -> None:
    if not structures.empty:
        structures.groupby(["variant", "total_nodes"]).agg(
            mean_route_anchor_count=("route_anchor_count", "mean"),
            mean_h_anchor_count=("h_anchor_count", "mean"),
            mean_ct_anchor_count=("ct_anchor_count", "mean"),
            mean_ct_chain_count=("ct_chain_count", "mean"),
            mean_route_travel_time=("route_travel_time", "mean"),
            mean_route_driving_energy=("route_driving_energy", "mean"),
        ).reset_index().to_csv(out_dir / "route_structure_by_variant.csv", index=False, encoding=CSV_ENCODING)
    if not runs.empty:
        runs.groupby(["variant", "total_nodes"]).agg(
            mean_star_count=("star_count", "mean"),
            mean_rendezvous_count=("rendezvous_count", "mean"),
            mean_positive_tau_count=("positive_tau_count", "mean"),
            mean_material_objective_component=("material_objective_component", "mean"),
            mean_microgrid_objective_component=("microgrid_objective_component", "mean"),
        ).reset_index().to_csv(out_dir / "service_mode_by_variant.csv", index=False, encoding=CSV_ENCODING)


def write_service_status(out_dir: Path, services: pd.DataFrame) -> None:
    required = ["scenario", "seed", "variant", "decoder", "route_signature", "status", "objective", "runtime_sec", "gap", "validation_metric_sum", "star_count", "rendezvous_count", "positive_tau_count"]
    for col in required:
        if col not in services.columns:
            services[col] = math.nan
    services[required + [c for c in services.columns if c not in required]].to_csv(out_dir / "service_milp_status.csv", index=False, encoding=CSV_ENCODING)


def write_paper_suite_compat_outputs(out_dir: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        return
    full = runs[runs["variant"].eq("FULL_SDRP_ENR")].copy()
    comparison_cols = [
        "scenario",
        "instance_group",
        "total_nodes",
        "seed",
        "baseline_objective",
        "baseline_status",
        "baseline_gap",
        "method_objective",
        "gap_to_milp_incumbent_percent",
        "runtime_sec",
        "star_count",
        "rendezvous_count",
        "positive_tau_count",
        "validation_metric_sum",
    ]
    for col in comparison_cols:
        if col not in full.columns:
            full[col] = math.nan
    full.rename(columns={"method_objective": "sdrp_enr_objective"})[
        [c if c != "method_objective" else "sdrp_enr_objective" for c in comparison_cols]
    ].to_csv(out_dir / "comparison_to_milp.csv", index=False, encoding=CSV_ENCODING)

    if {"variant", "instance_group", "total_nodes"}.issubset(runs.columns):
        runs.groupby(["variant", "instance_group", "total_nodes"]).agg(
            run_count=("scenario", "count"),
            mean_rendezvous_count=("rendezvous_count", "mean"),
            positive_rendezvous_run_count=("rendezvous_count", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            positive_rendezvous_rate=("rendezvous_count", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_star_count=("star_count", "mean"),
            mean_material_coverage_ratio=("material_coverage_ratio", "mean"),
            mean_microgrid_utility_ratio=("microgrid_utility_ratio", "mean"),
            mean_objective=("objective", "mean"),
            mean_gap_to_milp_incumbent_percent=("gap_to_milp_incumbent_percent", "mean"),
        ).reset_index().to_csv(out_dir / "rendezvous_usage_by_variant.csv", index=False, encoding=CSV_ENCODING)

    param = {
        "material_unit_kg": 25,
        "truck_payload_Qt_units": 800,
        "drone_package_qD_units": 2,
        "drone_body_weight_wD_units": 2,
        "drone_tmax_sec": 1200,
        "drone_Bv_kwh": 4.0,
        "drone_max_sortie_path_km": 18,
        "drones_per_truck": 3,
        "successor_gap_K": 3,
        "alpha": 1,
        "beta": 1,
        "elite_k": 10,
        "lns_rounds_per_elite": 20,
        "max_new_candidates_per_scenario": 200,
        "service_milp_time_limit_sec": 10,
        "final_elite_refinement": "Drop-and-reinsert-LNS;Rebalance-LNS",
        "excluded_refinement": "CT-chain-LNS;Segment-exchange-LNS",
    }
    pd.DataFrame([param]).to_csv(out_dir / "parameter_summary.csv", index=False, encoding=CSV_ENCODING)


def write_brief(out_dir: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        (out_dir / "brief.md").write_text("# SDRP-ENR component ablation\n\nNo completed runs yet.\n", encoding="utf-8")
        return
    full = runs[runs["variant"].eq("FULL_SDRP_ENR")]
    lines = [
        "# SDRP-ENR component ablation",
        "",
        "## 当前完成情况",
        "",
        f"- 已写入 {len(runs)} 条 scenario-seed-variant run，其中 FULL_SDRP_ENR={len(full)} 条。",
        f"- FULL_SDRP_ENR mean objective={full['objective'].mean():.4f}，mean gap={full['gap_to_exact_if_available'].mean():.4%}，mean runtime={full['runtime_sec'].mean():.2f}s。",
        f"- validation_metric_sum 最大值={runs['validation_metric_sum'].max():.6g}，drone_to_truck_visited_ct_count 总和={int(runs['drone_to_truck_visited_ct_count'].sum())}。",
        "",
        "## 初步解释",
        "",
        "- 本实验使用 final drone18 参数：tmax=1200s、Bv=4.0kWh、最大 sortie path length=18km。",
        "- Segment-exchange 未进入任何 variant；elite refinement 只使用 Drop-reinsert、Rebalance。",
        "- NO_MICROGRID_CHARGING 通过默认关闭的 decoder 开关实现；默认 FULL_SDRP_ENR 语义没有改变。",
    ]
    (out_dir / "brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing(out_dir: Path):
    def read(name: str) -> list[dict]:
        path = out_dir / name
        return pd.read_csv(path).to_dict("records") if path.exists() else []

    return read("all_runs.csv"), read("operator_moves_raw.csv"), read("service_milp_status.csv"), read("route_pool_candidates_raw.csv"), read("generator_raw_counts.csv"), read("route_structure_by_variant_raw.csv")


def run(args) -> None:
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows, move_rows, service_rows, candidate_rows, generator_rows, structure_rows = load_existing(out_dir) if args.resume else ([], [], [], [], [], [])
    active_variants = [v for v in VARIANTS if not getattr(args, "variants", None) or v[0] in args.variants]
    expected = {v for v, _ in active_variants}
    existing = pd.DataFrame(run_rows)
    done = set()
    if not existing.empty:
        done = {key for key, group in existing.groupby(["scenario", "seed"])
                if expected.issubset(set(group["variant"]))}
    table = load_table(args)

    for seed in args.seeds:
        for item in table.to_dict("records"):
            scenario = str(item["scenario"])
            if (scenario, int(seed)) in done:
                print(f"skip completed seed={seed} {scenario}", flush=True)
                continue
            data = load_data(Path(str(item["json"])))
            baseline_obj = float(item["baseline_objective"])
            upper = theoretical_upper(item)
            material_total, microgrid_total = service_denominators(data)
            pool_start = time.perf_counter()
            _diag, pool, generated_counts = generate_seed_scenario_pool(data, scenario, int(seed), args)
            pool_runtime = time.perf_counter() - pool_start
            evaluated_by_decoder: dict[str, pd.DataFrame] = {}
            decoder_objs: dict[str, object] = {}
            decoder_elapsed = {}

            for variant, group in active_variants:
                dkey = decoder_key(variant)
                if dkey not in evaluated_by_decoder:
                    decoder = make_decoder(data, args, variant)
                    decoder_objs[dkey] = decoder
                    allow_star, allow_rendezvous, _allow_grid = service_switches(variant)
                    decode_start = time.perf_counter()
                    eval_df, pool_services = evaluate_pool_with_decoder(
                        data,
                        scenario,
                        int(seed),
                        pool,
                        decoder,
                        baseline_obj,
                        "service_milp_v2",
                        allow_star,
                        allow_rendezvous,
                    )
                    decoder_elapsed[dkey] = time.perf_counter() - decode_start
                    evaluated_by_decoder[dkey] = eval_df
                    service_rows.extend({**r, "variant": variant, "decoder": "service_milp_v2"} for r in pool_services)
                    candidate_rows.extend(eval_df.assign(total_nodes=int(item["total_nodes"]), decoder_key=dkey).to_dict("records"))
                candidates = evaluated_by_decoder[dkey]
                decoder = decoder_objs[dkey]
                variant_start = time.perf_counter()
                result, moves, services, lns_summary = run_variant(data, scenario, int(seed), variant, candidates, args, decoder)
                runtime = pool_runtime + decoder_elapsed[dkey] + (time.perf_counter() - variant_start)
                for move in moves:
                    move.update({"seed": int(seed), "variant": variant})
                    move_rows.append(move)
                for service in services:
                    service.update({"seed": int(seed), "variant": variant, "decoder": "service_milp_v2"})
                    service_rows.append(service)

                base = {
                    "scenario": scenario,
                    "instance_group": str(item.get("instance_group", "")),
                    "total_nodes": int(item["total_nodes"]),
                    "truck_count": int(item["truck_count"]) if "truck_count" in item and pd.notna(item["truck_count"]) else math.nan,
                    "seed": int(seed),
                    "variant": variant,
                    "ablation_group": group,
                    "baseline_objective": baseline_obj,
                    "baseline_status": str(item.get("baseline_status", "")),
                    "baseline_gap": float(item.get("baseline_gap", math.nan)),
                }
                if result is None:
                    run_rows.append({
                        **base,
                        "objective": math.nan,
                        "method_objective": math.nan,
                        "gap_to_exact_if_available": math.nan,
                        "gap_to_milp_incumbent": math.nan,
                        "gap_to_milp_incumbent_percent": math.nan,
                        "objective_loss_to_milp_incumbent": math.nan,
                        "runtime_sec": runtime,
                        "status": "NO_VALID_ROUTE",
                        "validation_metric_sum": math.inf,
                    })
                    continue
                routes = normalize_routes(data, result.get("_routes") if "_routes" in result else parse_routes(str(result["route_nodes_by_truck"])))
                final_result = decoder.solve(routes)
                counts = sol_counts(final_result)
                metrics = extended_route_metrics(data, routes)
                hm = h_metrics(data, routes)
                ct = ct_metrics(data, routes, final_result)
                obj = float(final_result.objective)
                route_sig = signature_to_string(route_signature(data, routes))
                row = {
                    **base,
                    "objective": obj,
                    "method_objective": obj,
                    "gap_to_exact_if_available": rel_gap(baseline_obj, obj) if str(item.get("baseline_status", "")).upper() == "OPTIMAL" else math.nan,
                    "gap_to_milp_incumbent": rel_gap(baseline_obj, obj),
                    "gap_to_milp_incumbent_percent": 100.0 * rel_gap(baseline_obj, obj),
                    "objective_loss_to_milp_incumbent": baseline_obj - obj,
                    "gap_to_upper_bound": rel_gap(upper, obj),
                    "valid": validation_metric_sum(final_result) <= 1e-9,
                    "material_coverage_ratio": counts["material_objective_component"] / material_total if math.isfinite(counts["material_objective_component"]) else math.nan,
                    "microgrid_utility_ratio": counts["microgrid_objective_component"] / microgrid_total if math.isfinite(counts["microgrid_objective_component"]) else math.nan,
                    "runtime_sec": runtime + final_result.runtime_sec,
                    "pool_generation_runtime_sec": pool_runtime,
                    "pool_decode_runtime_sec": decoder_elapsed[dkey],
                    "status": final_result.status,
                    "service_milp_status": final_result.status,
                    "validation_metric_sum": validation_metric_sum(final_result),
                    "best_generator": str(result.get("generator", "")),
                    "best_lns_operator": str(result.get("best_lns_operator", "")),
                    "service_milp_status": final_result.status,
                    "route_signature": route_sig,
                    **metrics,
                    **hm,
                    **ct,
                    **counts,
                    "cache_hit_count": lns_summary.get("cache_hit_count", 0),
                    "cache_miss_count": lns_summary.get("cache_miss_count", 0),
                    "accepted_move_count": lns_summary.get("accepted_move_count", 0),
                    "improved_move_count": lns_summary.get("improved_move_count", 0),
                }
                run_rows.append(row)
                structure_rows.append({"scenario": scenario, "total_nodes": int(item["total_nodes"]), "seed": int(seed), "variant": variant, **metrics})
            for gen, count in generated_counts.items():
                generator_rows.append({"scenario": scenario, "total_nodes": int(item["total_nodes"]), "seed": int(seed), "generator": gen, "raw_generated_count": count})
            write_outputs(out_dir, run_rows, move_rows, service_rows, candidate_rows, generator_rows, structure_rows)
            print(f"seed={seed} {scenario}: pool={len(pool)} variants={len(active_variants)}", flush=True)

    write_outputs(out_dir, run_rows, move_rows, service_rows, candidate_rows, generator_rows, structure_rows)
    print(f"output_folder={out_dir}", flush=True)


def main(argv=None, full_only=False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE)
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
    if not full_only:
        parser.add_argument("--variants", nargs="+", choices=[v for v, _ in VARIANTS])
    args = parser.parse_args(argv)
    if full_only:
        args.variants = ["FULL_SDRP_ENR"]
        if args.output_dir == OUT_DIR:
            args.output_dir = ROOT / "results" / "sdrp_enr_final"
    args.enable_integrated_h_bridge = True
    run(args)


if __name__ == "__main__":
    main()
