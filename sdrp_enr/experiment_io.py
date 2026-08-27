"""Portable manifests and optional, separately identified MILP references."""
from pathlib import Path
import math
import pandas as pd
from .paths import ROOT


def read_manifest(path):
    path = Path(path).resolve()
    table = pd.read_csv(path)
    def resolve(value):
        candidate = Path(str(value))
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        for base in (path.parent, ROOT):
            resolved = base / candidate
            if resolved.exists():
                return str(resolved.resolve())
        raise FileNotFoundError(f"Scenario does not exist: {value} (manifest {path})")
    table["json"] = table["json"].map(resolve)
    if table["scenario"].duplicated().any():
        raise ValueError("Duplicate scenario keys in manifest")
    return table.sort_values(["total_nodes", "scenario"])


def load_experiment_table(args):
    table = read_manifest(args.manifest)
    baseline_path = getattr(args, "baseline_summary", None)
    if baseline_path and Path(baseline_path).is_file():
        baseline = pd.read_csv(baseline_path).rename(columns={
            "objective": "baseline_objective", "status": "baseline_status", "gap": "baseline_gap"})
        keep = [c for c in ["scenario", "baseline_objective", "baseline_status", "baseline_gap"] if c in baseline]
        table = table.merge(baseline[keep], on="scenario", how="left", validate="one_to_one")
    for col, default in [("baseline_objective", math.nan), ("baseline_gap", math.nan), ("baseline_status", "UNAVAILABLE")]:
        if col not in table:
            table[col] = default
    if args.max_scenarios > 0:
        table = table.head(args.max_scenarios)
    return table
