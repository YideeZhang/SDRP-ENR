from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sdrp_enr.experiment_io import read_manifest
from humanitarian_graph import load_scenario_json  # noqa: E402
from scripts.solve_gurobi_model import save_solution, solve_model  # noqa: E402
import scripts.exact_reporting as agg  # noqa: E402


MANIFEST = ROOT / "data/benchmark/manifest.csv"
OUT_DIR = ROOT / "results/milp_benchmark"
CSV_ENCODING = "utf-8-sig"


def status_name(status: int | float | str) -> str:
    try:
        code = int(status)
    except Exception:
        return str(status)
    return {2: "OPTIMAL", 9: "TIME_LIMIT", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED"}.get(code, str(code))


def write_extra_summaries(out_dir: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    group_cols = ["instance_group"] if "instance_group" in df.columns else []
    if group_cols:
        df.groupby("instance_group").agg(
            scenario_count=("scenario", "nunique"),
            optimal_count=("status", lambda s: int((s == "OPTIMAL").sum())),
            feasible_count=("objective", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            mean_objective=("objective", "mean"),
            mean_gap=("gap", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_validation_metric_sum=("validation_metric_sum", "mean"),
            mean_rendezvous_count=("rendezvous_count", "mean"),
            mean_star_count=("star_count", "mean"),
        ).reset_index().to_csv(out_dir / "summary_by_instance_group.csv", index=False, encoding=CSV_ENCODING)
    lines = [
        "# Paper Suite Final25 with Rendezvous Gurobi Baseline",
        "",
        "- This baseline is regenerated under final25 physical parameters.",
        "- Final service rule enabled: drones cannot serve truck-visited C^T nodes.",
        "- Self-rendezvous is excluded.",
        f"- Scenario count attempted: {df['scenario'].nunique()}.",
        f"- Feasible incumbent count: {int(pd.to_numeric(df['objective'], errors='coerce').notna().sum())}.",
        f"- Max validation_metric_sum: {pd.to_numeric(df['validation_metric_sum'], errors='coerce').max()}.",
    ]
    (out_dir / "brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    if args.max_scenarios > 0:
        manifest = manifest.head(args.max_scenarios).copy()
    rows: list[dict] = []
    completed: set[str] = set()
    if args.resume and (out_dir / "baseline_summary.csv").exists():
        old = pd.read_csv(out_dir / "baseline_summary.csv")
        rows = old.to_dict("records")
        completed = {str(r["scenario"]) for r in rows}
        print(f"resume loaded {len(completed)} scenarios", flush=True)

    agg.MANIFEST = args.manifest
    for item in manifest.to_dict("records"):
        scenario = str(item["scenario"])
        if scenario in completed:
            print(f"skip completed {scenario}", flush=True)
            continue
        start = time.perf_counter()
        scenario_dir = out_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        graph = load_scenario_json(Path(str(item["json"])))
        tl = float(item.get("time_limit_sec", args.time_limit_sec))
        if args.time_limit_sec > 0:
            tl = min(tl, float(args.time_limit_sec))
        solution = solve_model(
            graph,
            time_limit_sec=tl,
            successor_gap=int(item.get("successor_gap", args.successor_gap)),
            no_truck_visited_ct_drone_service=True,
            exclude_self_rendezvous=True,
            output_flag=int(args.output_flag),
            alpha=1.0,
            beta=1.0,
        )
        save_solution(solution, scenario_dir)
        row = agg.summarize_scenario(item, scenario_dir, solution)
        nodes = graph.node_index()
        if "material" in solution:
            row["material_objective"] = sum(float(nodes[str(r["node"])].p) * float(r["coverage"])
                for r in solution["material"].to_dict("records"))
        if "coverage" in solution:
            row["microgrid_objective"] = sum(float(nodes[str(r["h"])].p_m) * float(r["g"])
                for r in solution["coverage"].to_dict("records"))
        row["instance_group"] = str(item.get("instance_group", ""))
        row["wall_runtime_sec"] = time.perf_counter() - start
        rows.append(row)
        agg.write_outputs(out_dir, rows)
        write_extra_summaries(out_dir, rows)
        print(f"{scenario}: status={row['status']} obj={row['objective']:.4f} gap={row['gap']} group={row['instance_group']}", flush=True)
    agg.write_outputs(out_dir, rows)
    write_extra_summaries(out_dir, rows)
    print(f"output_folder={out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--time-limit-sec", type=float, default=1800.0)
    parser.add_argument("--successor-gap", type=int, default=3)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--output-flag", type=int, default=0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
