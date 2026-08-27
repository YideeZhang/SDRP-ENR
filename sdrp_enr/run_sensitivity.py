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

OUT_DIR = ROOT / "results" / "sdrp_enr_sensitivity"
MANIFEST = ROOT / "data" / "benchmark" / "manifest.csv"
BASELINE = ROOT / "data" / "reference" / "baseline_summary.csv"
SEEDS = [20260427]
CSV_ENCODING = "utf-8-sig"
METHOD_NAME = "SDRP-ENR"
DECODER_NAME = "fixed-route service decoder"
ALLOWED_OPERATORS = {"Drop-and-reinsert-LNS", "Rebalance-LNS"}

VARIANTS = [
    {"variant": "WEIGHT_MICROGRID_STRONG", "parameter_changed": "weights", "alpha": 3.0, "beta": 1.0, "successor_gap_K": 3, "drones_per_truck": 3},
    {"variant": "WEIGHT_MATERIAL_STRONG", "parameter_changed": "weights", "alpha": 1.0, "beta": 3.0, "successor_gap_K": 3, "drones_per_truck": 3},
    {"variant": "BASE", "parameter_changed": "base", "alpha": 1.0, "beta": 1.0, "successor_gap_K": 3, "drones_per_truck": 3},
    {"variant": "WEIGHT_MICROGRID_HIGH", "parameter_changed": "weights", "alpha": 2.0, "beta": 1.0, "successor_gap_K": 3, "drones_per_truck": 3},
    {"variant": "WEIGHT_MATERIAL_HIGH", "parameter_changed": "weights", "alpha": 1.0, "beta": 2.0, "successor_gap_K": 3, "drones_per_truck": 3},
    {"variant": "K_LOW", "parameter_changed": "successor_gap_K", "alpha": 1.0, "beta": 1.0, "successor_gap_K": 2, "drones_per_truck": 3},
    {"variant": "K_HIGH", "parameter_changed": "successor_gap_K", "alpha": 1.0, "beta": 1.0, "successor_gap_K": 4, "drones_per_truck": 3},
    {"variant": "DRONES_LOW", "parameter_changed": "drones_per_truck", "alpha": 1.0, "beta": 1.0, "successor_gap_K": 3, "drones_per_truck": 2},
    {"variant": "DRONES_HIGH", "parameter_changed": "drones_per_truck", "alpha": 1.0, "beta": 1.0, "successor_gap_K": 3, "drones_per_truck": 4},
]


def load_table(args):
    return load_experiment_table(args)


def with_drone_count(data, drones_per_truck: int):
    drones_by_truck = {v: list(range(int(drones_per_truck))) for v in data.truck_ids}
    truck_capacity = float(data.graph.e_truck.Qt - data.graph.drone.w * int(drones_per_truck))
    return replace(data, drones_by_truck=drones_by_truck, truck_capacity=truck_capacity)


def make_decoder(data, variant: dict, args) -> FixedRouteServiceMILP:
    return FixedRouteServiceMILP(
        data,
        time_limit_sec=args.service_milp_time_limit_sec,
        output_flag=0,
        successor_gap=int(variant["successor_gap_K"]),
        allow_star=True,
        allow_rendezvous=True,
        allow_microgrid_charging=True,
        no_truck_visited_ct_drone_service=True,
        alpha=float(variant["alpha"]),
        beta=float(variant["beta"]),
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


def refine_final(data, scenario: str, seed: int, candidates: pd.DataFrame, args, decoder):
    best_before = best_from_candidates(candidates)
    if best_before is None:
        return None, [], [], {}
    refiner = EliteRefiner(
        data,
        scenario,
        seed=seed,
        service_milp_time_limit_sec=args.service_milp_time_limit_sec,
        max_new_candidates=args.max_new_candidates_per_scenario,
        allowed_operators=ALLOWED_OPERATORS,
        enable_integrated_h_bridge=True,
        enable_activation_rebalance=bool(getattr(args, "enable_activation_rebalance", False)),
        max_activation_rebalance_candidates=args.max_activation_rebalance_candidates,
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
            **route_metrics(data, routes),
            "best_lns_operator": best_lns.get("operator", ""),
            "_routes": routes,
        }
    else:
        result = {**best_before, "best_lns_operator": ""}
    return result, moves, refiner.service_rows, summaries[0] if summaries else {}


def service_denominators(data) -> tuple[float, float]:
    material = sum(float(data.population(n)) for n in data.h_nodes + data.c_nodes)
    microgrid = sum(float(data.nodes[h].p_m) for h in data.h_nodes)
    return max(material, 1e-9), max(microgrid, 1e-9)


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


def drone_to_truck_visited_ct_count(data, routes: dict[int, list[str]], result) -> int:
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


def rel_gap(reference: float, value: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or abs(reference) <= 1e-9:
        return math.nan
    return (reference - value) / abs(reference)


def completed_keys(out_dir: Path) -> set[tuple[str, int, str]]:
    path = out_dir / "all_runs.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return {(str(r["scenario"]), int(r["seed"]), str(r["variant"])) for r in df.to_dict("records")}


def read_records(out_dir: Path, name: str) -> list[dict]:
    path = out_dir / name
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def write_outputs(out_dir: Path, run_rows: list[dict], move_rows: list[dict], service_rows: list[dict]) -> None:
    runs = pd.DataFrame(run_rows)
    moves = pd.DataFrame(move_rows)
    services = pd.DataFrame(service_rows)
    runs.to_csv(out_dir / "all_runs.csv", index=False, encoding=CSV_ENCODING)
    moves.to_csv(out_dir / "operator_moves_raw.csv", index=False, encoding=CSV_ENCODING)
    write_summary_tables(out_dir, runs)
    write_effect_tables(out_dir, runs)
    write_route_service_tables(out_dir, runs)
    write_operator_contribution(out_dir, moves)
    write_service_status(out_dir, services)
    write_brief(out_dir, runs)


def valid_rate(series: pd.Series) -> float:
    return float((pd.to_numeric(series, errors="coerce") <= 1e-9).mean()) if len(series) else math.nan


def write_summary_tables(out_dir: Path, runs: pd.DataFrame) -> None:
    variant_cols = ["variant", "parameter_changed", "alpha", "beta", "successor_gap_K", "drones_per_truck"]
    summary_cols = variant_cols + [
        "scenario_count",
        "run_count",
        "valid_rate",
        "mean_objective_weighted",
        "median_objective_weighted",
        "std_objective_weighted",
        "mean_material_objective_component",
        "mean_microgrid_objective_component",
        "mean_material_coverage_ratio",
        "mean_microgrid_utility_ratio",
        "mean_runtime_sec",
        "mean_star_count",
        "mean_rendezvous_count",
        "mean_positive_tau_count",
        "mean_ct_anchor_count",
        "mean_drone_to_truck_visited_ct_count",
    ]
    if runs.empty:
        pd.DataFrame(columns=summary_cols).to_csv(out_dir / "summary_by_variant.csv", index=False, encoding=CSV_ENCODING)
        pd.DataFrame().to_csv(out_dir / "summary_by_variant_nodes.csv", index=False, encoding=CSV_ENCODING)
        return
    runs.groupby(variant_cols).agg(
        scenario_count=("scenario", "nunique"),
        run_count=("scenario", "count"),
        valid_rate=("validation_metric_sum", valid_rate),
        mean_objective_weighted=("objective_weighted", "mean"),
        median_objective_weighted=("objective_weighted", "median"),
        std_objective_weighted=("objective_weighted", "std"),
        mean_material_objective_component=("material_objective_component", "mean"),
        mean_microgrid_objective_component=("microgrid_objective_component", "mean"),
        mean_material_coverage_ratio=("material_coverage_ratio", "mean"),
        mean_microgrid_utility_ratio=("microgrid_utility_ratio", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
        mean_star_count=("star_count", "mean"),
        mean_rendezvous_count=("rendezvous_count", "mean"),
        mean_positive_tau_count=("positive_tau_count", "mean"),
        mean_ct_anchor_count=("ct_anchor_count", "mean"),
        mean_drone_to_truck_visited_ct_count=("drone_to_truck_visited_ct_count", "mean"),
    ).reset_index()[summary_cols].to_csv(out_dir / "summary_by_variant.csv", index=False, encoding=CSV_ENCODING)
    runs.groupby(variant_cols + ["total_nodes"]).agg(
        scenario_count=("scenario", "nunique"),
        run_count=("scenario", "count"),
        valid_rate=("validation_metric_sum", valid_rate),
        mean_objective_weighted=("objective_weighted", "mean"),
        median_objective_weighted=("objective_weighted", "median"),
        std_objective_weighted=("objective_weighted", "std"),
        mean_material_objective_component=("material_objective_component", "mean"),
        mean_microgrid_objective_component=("microgrid_objective_component", "mean"),
        mean_material_coverage_ratio=("material_coverage_ratio", "mean"),
        mean_microgrid_utility_ratio=("microgrid_utility_ratio", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
        mean_star_count=("star_count", "mean"),
        mean_rendezvous_count=("rendezvous_count", "mean"),
        mean_positive_tau_count=("positive_tau_count", "mean"),
        mean_ct_anchor_count=("ct_anchor_count", "mean"),
        mean_drone_to_truck_visited_ct_count=("drone_to_truck_visited_ct_count", "mean"),
    ).reset_index().to_csv(out_dir / "summary_by_variant_nodes.csv", index=False, encoding=CSV_ENCODING)


def paired_effects(runs: pd.DataFrame) -> pd.DataFrame:
    base = runs[runs["variant"].eq("BASE")]
    rows = []
    for meta in VARIANTS:
        variant = meta["variant"]
        if variant == "BASE":
            continue
        sub = runs[runs["variant"].eq(variant)]
        merged = sub.merge(base, on=["scenario", "seed"], suffixes=("", "_base"))
        if merged.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "baseline_variant": "BASE",
                "parameter_changed": meta["parameter_changed"],
                "paired_run_count": len(merged),
                "mean_delta_weighted_objective": (merged["objective_weighted"] - merged["objective_weighted_base"]).mean(),
                "mean_delta_material_component": (merged["material_objective_component"] - merged["material_objective_component_base"]).mean(),
                "mean_delta_microgrid_component": (merged["microgrid_objective_component"] - merged["microgrid_objective_component_base"]).mean(),
                "mean_delta_material_coverage_ratio": (merged["material_coverage_ratio"] - merged["material_coverage_ratio_base"]).mean(),
                "mean_delta_microgrid_utility_ratio": (merged["microgrid_utility_ratio"] - merged["microgrid_utility_ratio_base"]).mean(),
                "mean_delta_runtime_sec": (merged["runtime_sec"] - merged["runtime_sec_base"]).mean(),
                "mean_delta_star_count": (merged["star_count"] - merged["star_count_base"]).mean(),
                "mean_delta_rendezvous_count": (merged["rendezvous_count"] - merged["rendezvous_count_base"]).mean(),
                "mean_delta_positive_tau_count": (merged["positive_tau_count"] - merged["positive_tau_count_base"]).mean(),
                "interpretation": interpretation(meta["parameter_changed"]),
            }
        )
    return pd.DataFrame(rows)


def interpretation(kind: str) -> str:
    if kind == "weights":
        return "Weight sensitivity is interpreted as a service-priority shift, not as better/worse by weighted objective."
    if kind == "successor_gap_K":
        return "Compare service gain and runtime; K=3 remains preferred unless K changes give clear benefit."
    if kind == "drones_per_truck":
        return "Compare marginal service gain against runtime and fleet resource expansion/contraction."
    return "Base setting."


def write_effect_tables(out_dir: Path, runs: pd.DataFrame) -> None:
    effects = paired_effects(runs)
    effects.to_csv(out_dir / "sensitivity_effect_table.csv", index=False, encoding=CSV_ENCODING)
    summary = pd.read_csv(out_dir / "summary_by_variant.csv") if (out_dir / "summary_by_variant.csv").exists() else pd.DataFrame()
    if summary.empty:
        pd.DataFrame().to_csv(out_dir / "weight_sensitivity_table.csv", index=False, encoding=CSV_ENCODING)
        pd.DataFrame().to_csv(out_dir / "k_sensitivity_table.csv", index=False, encoding=CSV_ENCODING)
        pd.DataFrame().to_csv(out_dir / "drone_count_sensitivity_table.csv", index=False, encoding=CSV_ENCODING)
        return
    base = summary[summary["variant"].eq("BASE")].iloc[0].to_dict() if not summary[summary["variant"].eq("BASE")].empty else {}
    weight = summary[summary["variant"].eq("BASE") | summary["variant"].str.startswith("WEIGHT_")].copy()
    weight["delta_material_coverage_ratio_vs_base"] = weight["mean_material_coverage_ratio"] - float(base.get("mean_material_coverage_ratio", math.nan))
    weight["delta_microgrid_utility_ratio_vs_base"] = weight["mean_microgrid_utility_ratio"] - float(base.get("mean_microgrid_utility_ratio", math.nan))
    weight["interpretation"] = "Service allocation shift; weighted objectives across alpha/beta are not directly comparable."
    weight.to_csv(out_dir / "weight_sensitivity_table.csv", index=False, encoding=CSV_ENCODING)
    k = summary[summary["variant"].isin(["BASE", "K_LOW", "K_HIGH"])].copy()
    k["K"] = k["successor_gap_K"]
    k["delta_objective_vs_base"] = k["mean_objective_weighted"] - float(base.get("mean_objective_weighted", math.nan))
    k["delta_runtime_vs_base"] = k["mean_runtime_sec"] - float(base.get("mean_runtime_sec", math.nan))
    k["interpretation"] = "K=3 is the balanced main setting unless K=2/4 shows clear gain."
    k.to_csv(out_dir / "k_sensitivity_table.csv", index=False, encoding=CSV_ENCODING)
    drones = summary[summary["variant"].isin(["BASE", "DRONES_LOW", "DRONES_HIGH"])].copy()
    drones["delta_objective_vs_base"] = drones["mean_objective_weighted"] - float(base.get("mean_objective_weighted", math.nan))
    drones["interpretation"] = "Three drones per truck is preferred unless the fourth drone gives meaningful marginal gain."
    drones.to_csv(out_dir / "drone_count_sensitivity_table.csv", index=False, encoding=CSV_ENCODING)


def write_route_service_tables(out_dir: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        pd.DataFrame().to_csv(out_dir / "service_mode_by_variant.csv", index=False, encoding=CSV_ENCODING)
        pd.DataFrame().to_csv(out_dir / "route_structure_by_variant.csv", index=False, encoding=CSV_ENCODING)
        return
    service_cols = [
        "scenario",
        "total_nodes",
        "seed",
        "variant",
        "star_count",
        "rendezvous_count",
        "positive_tau_count",
        "material_objective_component",
        "microgrid_objective_component",
        "material_coverage_ratio",
        "microgrid_utility_ratio",
        "drone_to_truck_visited_ct_count",
    ]
    runs[service_cols].to_csv(out_dir / "service_mode_by_variant.csv", index=False, encoding=CSV_ENCODING)
    route_cols = [
        "scenario",
        "total_nodes",
        "seed",
        "variant",
        "route_anchor_count",
        "h_anchor_count",
        "ct_anchor_count",
        "ct_chain_count",
        "route_travel_time",
        "route_driving_energy",
        "route_time_balance_std",
        "route_energy_balance_std",
    ]
    for col in route_cols:
        if col not in runs.columns:
            runs[col] = math.nan
    runs[route_cols].to_csv(out_dir / "route_structure_by_variant.csv", index=False, encoding=CSV_ENCODING)


def write_operator_contribution(out_dir: Path, moves: pd.DataFrame) -> None:
    cols = ["variant", "operator", "selected_count", "candidate_count", "accepted_count", "improved_count", "mean_objective_delta", "best_improvement"]
    if moves.empty:
        pd.DataFrame(columns=cols).to_csv(out_dir / "operator_contribution.csv", index=False, encoding=CSV_ENCODING)
        return
    moves = moves[~moves["operator"].astype(str).str.contains("CT-chain|Segment-exchange", regex=True, na=False)].copy()
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
                "best_improvement": sub["objective_delta"].max(),
            }
        )
    pd.DataFrame(rows, columns=cols).to_csv(out_dir / "operator_contribution.csv", index=False, encoding=CSV_ENCODING)


def write_service_status(out_dir: Path, services: pd.DataFrame) -> None:
    if services.empty:
        pd.DataFrame().to_csv(out_dir / "service_milp_status.csv", index=False, encoding=CSV_ENCODING)
        return
    services.to_csv(out_dir / "service_milp_status.csv", index=False, encoding=CSV_ENCODING)


def write_brief(out_dir: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        (out_dir / "brief.md").write_text("# SDRP-ENR Final Sensitivity\n\nNo runs completed.\n", encoding="utf-8")
        return
    summary = pd.read_csv(out_dir / "summary_by_variant.csv")
    try:
        effects = pd.read_csv(out_dir / "sensitivity_effect_table.csv") if (out_dir / "sensitivity_effect_table.csv").exists() else pd.DataFrame()
    except EmptyDataError:
        effects = pd.DataFrame()
    invalid = int((pd.to_numeric(runs["validation_metric_sum"], errors="coerce").fillna(math.inf) > 1e-9).sum())
    bad_ct = int((pd.to_numeric(runs["drone_to_truck_visited_ct_count"], errors="coerce").fillna(0) > 0).sum())
    forbidden = 0
    op_path = out_dir / "operator_moves_raw.csv"
    if op_path.exists():
        ops = pd.read_csv(op_path, usecols=["operator"])
        forbidden = int(ops["operator"].astype(str).str.contains("CT-chain|Segment-exchange", regex=True, na=False).sum())
    lines = [
        "# SDRP-ENR Final Sensitivity Analysis",
        "",
        "## 中文总结",
        "",
        f"- 本轮完成 {len(runs)} 条 run，覆盖 {runs['scenario'].nunique()} 个场景、{runs['seed'].nunique()} 个随机种子、{runs['variant'].nunique()} 个 one-factor sensitivity variants。",
        f"- validation 非零 run 数：{invalid}；无人机服务已被卡车访问的 C^T 的 run 数：{bad_ct}；CT-chain/Segment-exchange operator 记录数：{forbidden}。",
        "- 权重变化主要用于观察 material 与 microgrid 服务分配迁移；不同 alpha/beta 下的 weighted objective 不直接解释为更好或更差。",
        "- 主设定继续采用 alpha=1, beta=1，因为它给出均衡的 material coverage 与 microgrid support，避免单一服务目标主导。",
        "- K=3 保留为主设定：K=2 用于检验较窄 rendezvous 窗口，K=4 用于检验更宽窗口的边际收益和 runtime tradeoff。",
        "- drones_per_truck=3 保留为主设定：2 架代表资源收缩，4 架代表资源扩张；是否值得增加第 4 架需看边际收益是否明显超过 runtime/容量代价。",
        "",
        "## Variant summary",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"- {row['variant']}: material ratio={row['mean_material_coverage_ratio']:.4f}, "
            f"microgrid ratio={row['mean_microgrid_utility_ratio']:.4f}, runtime={row['mean_runtime_sec']:.2f}s, "
            f"star={row['mean_star_count']:.2f}, rendezvous={row['mean_rendezvous_count']:.2f}, tau={row['mean_positive_tau_count']:.2f}."
        )
    if not effects.empty:
        lines.extend(["", "## Paired effects vs BASE"])
        for row in effects.to_dict("records"):
            lines.append(
                f"- {row['variant']}: delta material ratio={row['mean_delta_material_coverage_ratio']:.4f}, "
                f"delta microgrid ratio={row['mean_delta_microgrid_utility_ratio']:.4f}, "
                f"delta runtime={row['mean_delta_runtime_sec']:.2f}s."
            )
    lines.extend(
        [
            "",
            "## English Results Paragraph",
            "",
            "Interpret paired changes before drafting empirical conclusions. Weighted objectives with different weights are not directly comparable; use unweighted service components and coverage ratios.",
        ]
    )
    (out_dir / "brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = read_records(out_dir, "all_runs.csv") if args.resume else []
    move_rows = read_records(out_dir, "operator_moves_raw.csv") if args.resume else []
    service_rows = read_records(out_dir, "service_milp_status.csv") if args.resume else []
    done = completed_keys(out_dir) if args.resume else set()
    table = load_table(args)
    active_variants = [v for v in VARIANTS if not args.variants or v["variant"] in args.variants]

    for seed in args.seeds:
        for item in table.to_dict("records"):
            scenario = str(item["scenario"])
            scenario_keys = {(scenario, int(seed), meta["variant"]) for meta in active_variants}
            if scenario_keys.issubset(done):
                print(f"skip completed seed={seed} {scenario}", flush=True)
                continue
            base_data = load_data(Path(str(item["json"])))
            pool_args = args
            pool_start = time.perf_counter()
            _diag, pool, _generated = generate_seed_scenario_pool(base_data, scenario, int(seed), pool_args)
            pool_runtime = time.perf_counter() - pool_start
            for variant in active_variants:
                key = (scenario, int(seed), variant["variant"])
                if key in done:
                    print(f"skip completed {variant['variant']} seed={seed} {scenario}", flush=True)
                    continue
                data = with_drone_count(load_data(Path(str(item["json"]))), int(variant["drones_per_truck"]))
                decoder = make_decoder(data, variant, args)
                decode_start = time.perf_counter()
                eval_df, pool_services = evaluate_pool_with_decoder(data, scenario, int(seed), pool, decoder, math.nan, DECODER_NAME, True, True)
                service_rows.extend({**r, "variant": variant["variant"], "decoder": DECODER_NAME} for r in pool_services)
                decode_runtime = time.perf_counter() - decode_start
                variant_start = time.perf_counter()
                result, moves, services, summary = refine_final(data, scenario, int(seed), eval_df, args, decoder)
                runtime = pool_runtime + decode_runtime + (time.perf_counter() - variant_start)
                for move in moves:
                    move.update({"seed": int(seed), "variant": variant["variant"]})
                    move_rows.append(move)
                for service in services:
                    service.update({"seed": int(seed), "variant": variant["variant"], "decoder": DECODER_NAME})
                    service_rows.append(service)
                base_row = {
                    "scenario": scenario,
                    "total_nodes": int(item["total_nodes"]),
                    "seed": int(seed),
                    "variant": variant["variant"],
                    "parameter_changed": variant["parameter_changed"],
                    "alpha": float(variant["alpha"]),
                    "beta": float(variant["beta"]),
                    "successor_gap_K": int(variant["successor_gap_K"]),
                    "drones_per_truck": int(variant["drones_per_truck"]),
                }
                if result is None:
                    run_rows.append(
                        {
                            **base_row,
                            "objective_weighted": math.nan,
                            "unweighted_material_score": math.nan,
                            "unweighted_microgrid_score": math.nan,
                            "unweighted_total_score": math.nan,
                            "runtime_sec": runtime,
                            "valid": False,
                            "status": "NO_VALID_ROUTE",
                            "service_milp_status": "NO_VALID_ROUTE",
                            "validation_metric_sum": math.inf,
                        }
                    )
                    write_outputs(out_dir, run_rows, move_rows, service_rows)
                    continue
                routes = normalize_routes(data, result.get("_routes") if "_routes" in result else parse_routes(str(result["route_nodes_by_truck"])))
                final = decoder.solve(routes)
                counts = sol_counts(final)
                material_total, microgrid_total = service_denominators(data)
                metrics = extended_route_metrics(data, routes)
                baseline_obj = float(item.get("baseline_objective", math.nan))
                gap = rel_gap(baseline_obj, float(final.objective)) if variant["variant"] == "BASE" and str(item.get("baseline_status", "")).upper() == "OPTIMAL" else math.nan
                row = {
                    **base_row,
                    "objective_weighted": float(final.objective),
                    "material_objective_component": counts["material_objective_component"],
                    "microgrid_objective_component": counts["microgrid_objective_component"],
                    "unweighted_material_score": counts["material_objective_component"],
                    "unweighted_microgrid_score": counts["microgrid_objective_component"],
                    "unweighted_total_score": counts["material_objective_component"] + counts["microgrid_objective_component"],
                    "material_coverage_ratio": counts["material_objective_component"] / material_total if math.isfinite(counts["material_objective_component"]) else math.nan,
                    "microgrid_utility_ratio": counts["microgrid_objective_component"] / microgrid_total if math.isfinite(counts["microgrid_objective_component"]) else math.nan,
                    "runtime_sec": runtime + final.runtime_sec,
                    "valid": bool(validation_metric_sum(final) <= 1e-9 and final.status == "OPTIMAL"),
                    "status": final.status,
                    "service_milp_status": final.status,
                    "validation_metric_sum": validation_metric_sum(final),
                    "gap_to_exact_if_available": gap,
                    "best_generator": str(result.get("generator", "")),
                    "best_lns_operator": str(result.get("best_lns_operator", "")),
                    "route_signature": signature_to_string(route_signature(data, routes)),
                    "drone_to_truck_visited_ct_count": drone_to_truck_visited_ct_count(data, routes, final),
                    **metrics,
                    **used_truck_metrics(routes),
                    **counts,
                }
                run_rows.append(row)
                service_rows.append(
                    {
                        "scenario": scenario,
                        "seed": int(seed),
                        "variant": variant["variant"],
                        "decoder": DECODER_NAME,
                        "route_signature": row["route_signature"],
                        "status": final.status,
                        "objective": final.objective,
                        "runtime_sec": final.runtime_sec,
                        "gap": final.gap,
                        "validation_metric_sum": row["validation_metric_sum"],
                        "star_count": row["star_count"],
                        "rendezvous_count": row["rendezvous_count"],
                        "positive_tau_count": row["positive_tau_count"],
                    }
                )
                write_outputs(out_dir, run_rows, move_rows, service_rows)
                print(f"[final sensitivity] {variant['variant']} seed={seed} {scenario} obj={row['objective_weighted']:.2f}", flush=True)
    write_outputs(out_dir, run_rows, move_rows, service_rows)
    print(f"output_folder={out_dir}", flush=True)


def main() -> None:
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
    parser.add_argument("--max-activation-rebalance-candidates", type=int, default=20)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--variants", nargs="+", choices=[v["variant"] for v in VARIANTS])
    parser.add_argument("--enable-activation-rebalance", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.enable_integrated_h_bridge = True
    run(args)


if __name__ == "__main__":
    main()
