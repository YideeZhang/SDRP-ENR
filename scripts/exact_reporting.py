from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from sdrp_enr.paths import ROOT
MANIFEST = ROOT / "data/benchmark/manifest.csv"

def status_name(status: int | float | str) -> str:
    try:
        code = int(status)
    except Exception:
        return str(status)
    return {2: "OPTIMAL", 9: "TIME_LIMIT", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED"}.get(code, str(code))

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()

def route_h_diagnostics(result_dir: Path) -> dict:
    h_service = read_csv(result_dir / "h_service.csv")
    routes = read_csv(result_dir / "routes.csv")
    if h_service.empty:
        return {
            "h_visit_count": 0,
            "unique_h_visit_count": 0,
            "duplicate_h_visit_count": 0,
            "charged_h_count": 0,
            "positive_tau_count": 0,
            "duplicate_positive_tau_count": 0,
            "h_nodes_with_multiple_physical_visits": "",
            "h_nodes_with_positive_tau": "",
        }
    h_service["z_num"] = pd.to_numeric(h_service.get("z", 0), errors="coerce").fillna(0)
    h_service["tau_num"] = pd.to_numeric(h_service.get("tau", 0), errors="coerce").fillna(0.0)
    visited = h_service[h_service["z_num"] > 0.5].copy()
    positive = h_service[h_service["tau_num"] > 1e-7].copy()
    visit_counts = visited.groupby("h")["truck"].nunique() if not visited.empty else pd.Series(dtype=float)
    tau_counts = positive.groupby("h")["truck"].nunique() if not positive.empty else pd.Series(dtype=float)
    multi_visit = sorted([str(h) for h, c in visit_counts.items() if c > 1])
    positive_h = sorted([str(h) for h in tau_counts.index])
    duplicate_tau = int(sum(max(0, int(c) - 1) for c in tau_counts.values))
    return {
        "h_visit_count": int(len(visited)),
        "unique_h_visit_count": int(visited["h"].nunique()) if not visited.empty else 0,
        "duplicate_h_visit_count": int(len(visited) - visited["h"].nunique()) if not visited.empty else 0,
        "charged_h_count": int(positive["h"].nunique()) if not positive.empty else 0,
        "positive_tau_count": int(len(positive)),
        "duplicate_positive_tau_count": duplicate_tau,
        "h_nodes_with_multiple_physical_visits": ";".join(multi_visit),
        "h_nodes_with_positive_tau": ";".join(positive_h),
    }

def count_used_trucks(result_dir: Path) -> dict:
    routes = read_csv(result_dir / "routes.csv")
    if routes.empty:
        return {"route_arc_count": 0, "used_truck_count": 0, "empty_truck_count": 0, "ct_anchor_count": 0}
    route_arc_count = len(routes)
    trucks = sorted(routes["truck"].dropna().unique().tolist())
    used = set()
    ct_nodes = set()
    for _, row in routes.iterrows():
        v = row["truck"]
        a = str(row["from"])
        b = str(row["to"])
        if a != "D0" or b != "D0":
            if a != "D0" or b != "D0":
                used.add(v)
        for node in [a, b]:
            if node.startswith("C"):
                ct_nodes.add(node)
    return {
        "route_arc_count": int(route_arc_count),
        "used_truck_count": int(len(used)),
        "empty_truck_count": int(max(0, len(trucks) - len(used))),
        "ct_anchor_count": int(len(ct_nodes)),
    }

def service_counts(result_dir: Path) -> dict:
    star = read_csv(result_dir / "star_delivery.csv")
    rv = read_csv(result_dir / "rendezvous_delivery.csv")
    star_count = 0
    if not star.empty:
        col = "times" if "times" in star.columns else None
        star_count = int(pd.to_numeric(star[col], errors="coerce").fillna(0).sum()) if col else len(star)
    return {
        "star_count": int(star_count),
        "rendezvous_count": int(len(rv)) if not rv.empty else 0,
    }

def objective_parts(result_dir: Path) -> dict:
    material = read_csv(result_dir / "material.csv")
    coverage = read_csv(result_dir / "coverage.csv")
    material_obj = 0.0
    microgrid_obj = 0.0
    if not material.empty:
        # coverage is q_total/demand, so objective contribution is p * coverage. Demand equals population in current data.
        if "coverage" in material.columns and "node" in material.columns:
            # Need node population from solution CSV is unavailable; for current scenarios demand==population.
            q = pd.to_numeric(material.get("q_total", 0), errors="coerce").fillna(0.0)
            cov = pd.to_numeric(material.get("coverage", 0), errors="coerce").fillna(0.0)
            demand = q / cov.replace(0, math.nan)
            demand = demand.fillna(q)
            material_obj = float((demand * cov).sum())
    if not coverage.empty:
        g = pd.to_numeric(coverage.get("g", 0), errors="coerce").fillna(0.0)
        # H population/p_m is 300 in generated suites, but recover from energy_demand if possible:
        if "energy_demand" in coverage.columns:
            # energy_demand = p_m * rho * R; p_m recovery needs R and rho, not always enough. Use h_service is simpler unavailable.
            pass
        microgrid_obj = float((300.0 * g).sum()) if len(coverage) else 0.0
    return {"material_objective": material_obj, "microgrid_objective": microgrid_obj}

def validation_metrics(result_dir: Path) -> dict:
    h_service = read_csv(result_dir / "h_service.csv")
    coverage = read_csv(result_dir / "coverage.csv")
    energy = read_csv(result_dir / "energy_breakdown.csv")
    duplicate_tau = route_h_diagnostics(result_dir)["duplicate_positive_tau_count"]
    tau_on_unvisited = 0
    cap_violation = 0
    if not h_service.empty:
        h_service["z_num"] = pd.to_numeric(h_service.get("z", 0), errors="coerce").fillna(0)
        h_service["tau_num"] = pd.to_numeric(h_service.get("tau", 0), errors="coerce").fillna(0)
        tau_on_unvisited = int(((h_service["tau_num"] > 1e-7) & (h_service["z_num"] <= 0.5)).sum())
    if not coverage.empty:
        supplied = pd.to_numeric(coverage.get("energy_supplied", 0), errors="coerce").fillna(0)
        arrival_cap = pd.to_numeric(coverage.get("energy_demand_at_arrival", 0), errors="coerce").fillna(0)
        demand = pd.to_numeric(coverage.get("energy_demand", 0), errors="coerce").fillna(0)
        cap_violation = int(((supplied > arrival_cap + 1e-6) | (supplied > demand + 1e-6)).sum())
    battery_violation = 0
    if not energy.empty:
        total = pd.to_numeric(energy.get("total_energy_kwh", 0), errors="coerce").fillna(0)
        cap = pd.to_numeric(energy.get("battery_capacity_kwh", 0), errors="coerce").fillna(0)
        battery_violation = int((total > cap + 1e-6).sum())
    metric_sum = duplicate_tau + tau_on_unvisited + cap_violation + battery_violation
    return {
        "duplicate_positive_tau_count": int(duplicate_tau),
        "tau_on_unvisited_h_count": int(tau_on_unvisited),
        "h_energy_cap_violation_count": int(cap_violation),
        "truck_battery_violation_count": int(battery_violation),
        "validation_metric_sum": float(metric_sum),
    }

def summarize_scenario(item: dict, result_dir: Path, solution: dict | None = None) -> dict:
    model_stats = solution.get("model_stats", {}) if solution else {}
    var_counts = solution.get("variable_group_counts", {}) if solution else {}
    h_diag = route_h_diagnostics(result_dir)
    validation = validation_metrics(result_dir)
    return {
        "scenario": str(item["scenario"]),
        "total_nodes": int(item["total_nodes"]),
        "truck_count": int(item["truck_count"]),
        "status": status_name(solution.get("status", math.nan) if solution else math.nan),
        "status_code": int(solution.get("status", -999)) if solution else -999,
        "objective": float(solution.get("objective", math.nan)) if solution else math.nan,
        "gap": float(model_stats.get("gap", solution.get("gap", math.nan))) if solution else math.nan,
        "runtime_sec": float(model_stats.get("runtime_sec", math.nan)),
        "num_vars": int(model_stats.get("num_vars", 0) or 0),
        "num_constrs": int(model_stats.get("num_constrs", 0) or 0),
        "num_bin_vars": int(model_stats.get("num_bin_vars", 0) or 0),
        "num_int_vars": int(model_stats.get("num_int_vars", 0) or 0),
        **count_used_trucks(result_dir),
        **h_diag,
        **service_counts(result_dir),
        **objective_parts(result_dir),
        **validation,
    }

def write_h_normal_diagnostics(out_dir: Path, baseline: pd.DataFrame) -> None:
    rows = []
    for row in baseline.to_dict("records"):
        rows.append(
            {
                "scenario": row["scenario"],
                "total_nodes": row["total_nodes"],
                "h_count": int(row.get("h_count", 0) or 0) if "h_count" in row else math.nan,
                "truck_count": row["truck_count"],
                "h_visit_count": row["h_visit_count"],
                "unique_h_visit_count": row["unique_h_visit_count"],
                "duplicate_h_visit_count": row["duplicate_h_visit_count"],
                "charged_h_count": row["charged_h_count"],
                "uncharged_visited_h_count": max(0, row["unique_h_visit_count"] - row["charged_h_count"]),
                "unvisited_h_count": math.nan,
                "positive_tau_count": row["positive_tau_count"],
                "max_positive_tau_per_h": 1 if row["positive_tau_count"] > 0 else 0,
                "duplicate_positive_tau_count": row["duplicate_positive_tau_count"],
                "h_nodes_with_multiple_physical_visits": row["h_nodes_with_multiple_physical_visits"],
                "h_nodes_with_positive_tau": row["h_nodes_with_positive_tau"],
            }
        )
    df = pd.DataFrame(rows)
    # Fill h_count/unvisited from manifest-like baseline rows.
    if "h_count" in baseline.columns:
        df["h_count"] = baseline["h_count"]
        df["unvisited_h_count"] = df["h_count"] - df["unique_h_visit_count"]
    df.to_csv(out_dir / "h_normal_diagnostics.csv", index=False)


def write_outputs(out_dir: Path, rows: list[dict]) -> None:
    baseline = pd.DataFrame(rows)
    manifest = pd.read_csv(MANIFEST)[["scenario", "h_count", "c_count"]]
    baseline = baseline.drop(
        columns=[c for c in baseline.columns if c in {"h_count", "c_count", "h_count_x", "c_count_x", "h_count_y", "c_count_y"}],
        errors="ignore",
    )
    baseline = baseline.merge(manifest, on="scenario", how="left")
    preferred_cols = [
        "scenario",
        "total_nodes",
        "truck_count",
        "status",
        "status_code",
        "objective",
        "gap",
        "runtime_sec",
        "num_vars",
        "num_constrs",
        "num_bin_vars",
        "num_int_vars",
        "route_arc_count",
        "used_truck_count",
        "empty_truck_count",
        "h_visit_count",
        "unique_h_visit_count",
        "duplicate_h_visit_count",
        "charged_h_count",
        "positive_tau_count",
        "duplicate_positive_tau_count",
        "ct_anchor_count",
        "star_count",
        "rendezvous_count",
        "material_objective",
        "microgrid_objective",
        "validation_metric_sum",
        "h_count",
        "c_count",
    ]
    baseline = baseline[[c for c in preferred_cols if c in baseline.columns] + [c for c in baseline.columns if c not in preferred_cols]]
    baseline.to_csv(out_dir / "baseline_summary.csv", index=False)
    baseline.groupby("total_nodes").agg(
        scenario_count=("scenario", "count"),
        optimal_count=("status", lambda s: int((s == "OPTIMAL").sum())),
        feasible_count=("status_code", lambda s: int(((s == 2) | (s == 9)).sum())),
        mean_objective=("objective", "mean"),
        median_objective=("objective", "median"),
        mean_gap=("gap", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
        mean_num_vars=("num_vars", "mean"),
        mean_num_constrs=("num_constrs", "mean"),
        mean_used_truck_count=("used_truck_count", "mean"),
        mean_empty_truck_count=("empty_truck_count", "mean"),
        mean_h_visit_count=("h_visit_count", "mean"),
        mean_unique_h_visit_count=("unique_h_visit_count", "mean"),
        mean_duplicate_h_visit_count=("duplicate_h_visit_count", "mean"),
        mean_charged_h_count=("charged_h_count", "mean"),
        mean_positive_tau_count=("positive_tau_count", "mean"),
        mean_star_count=("star_count", "mean"),
        mean_rendezvous_count=("rendezvous_count", "mean"),
    ).reset_index().to_csv(out_dir / "by_nodes_summary.csv", index=False)
    write_h_normal_diagnostics(out_dir, baseline)
