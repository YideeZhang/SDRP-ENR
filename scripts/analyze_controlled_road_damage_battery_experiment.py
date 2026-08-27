from __future__ import annotations

"""Validate the frozen controlled road-damage results and draw paper figures.

This script is intentionally analysis-only.  It reads the nine frozen CSV files
listed in ``REQUIRED_CSVS``, checks that their scenario keys and reported metrics
agree, and exports publication figures.  It does not generate scenarios, invoke
a solver, or rewrite any experiment table.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "results" / "controlled_road_damage_battery_n20_exact"
CSV_ENCODING = "utf-8-sig"
NUMERICAL_TOLERANCE = 1.0e-8
PNG_DPI = 600

REQUIRED_CSVS = (
    "summary_by_damage.csv",
    "battery_allocation_by_instance.csv",
    "service_response_by_damage.csv",
    "all_runs.csv",
    "truck_inaccessible_demand_by_instance.csv",
    "response_by_truck_inaccessible_demand.csv",
    "exploratory_accessibility_spearman.csv",
    "solver_status.csv",
    "validation.csv",
)

# Common TITS/OR visual language.
COLORS = {
    "truck": "#1F4E79",
    "drone": "#D97706",
    "microgrid": "#2E7D32",
    "material": "#4B5563",
    "unused": "#C9D2DC",
    "grid": "#D9DEE5",
    "axis": "#374151",
}
MARKERS = {
    "truck": "o",
    "drone": "s",
    "microgrid": "^",
    "material": "D",
    "unused": "v",
}


def configure_style() -> None:
    """Apply one visual system to every exported panel."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "font.size": 10.0,
            "mathtext.fontset": "stix",
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["axis"],
            "axes.linewidth": 0.75,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "legend.fontsize": 10.0,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": PNG_DPI,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight: bool = True,
) -> None:
    """Export an editable SVG and a 600-dpi PNG with identical bounds."""

    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = {"bbox_inches": "tight", "pad_inches": 0.04} if tight else {}
    fig.savefig(output_dir / f"{stem}.svg", **bounds)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=PNG_DPI,
        **bounds,
    )
    plt.close(fig)


def load_tables(result_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_CSVS if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen CSV files: {', '.join(missing)}")

    tables: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_CSVS:
        tables[name] = pd.read_csv(
            result_dir / name,
            encoding=CSV_ENCODING,
            dtype={"damage_realization": str},
        )
    return tables


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def assert_close(label: str, left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> None:
    left_values = pd.to_numeric(pd.Series(left), errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(pd.Series(right), errors="coerce").to_numpy(dtype=float)
    require(left_values.shape == right_values.shape, f"{label}: shape mismatch")
    if not np.allclose(left_values, right_values, rtol=0.0, atol=NUMERICAL_TOLERANCE, equal_nan=True):
        difference = np.nanmax(np.abs(left_values - right_values))
        raise RuntimeError(f"{label}: frozen CSV values disagree (max abs difference={difference:.3e})")


def validate_frozen_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Fail closed if the nine frozen evidence tables do not reconcile."""

    runs = tables["all_runs.csv"].copy()
    summary = tables["summary_by_damage.csv"].copy().sort_values("damage_q")
    battery = tables["battery_allocation_by_instance.csv"].copy()
    service = tables["service_response_by_damage.csv"].copy().sort_values("damage_q")
    inaccessible = tables["truck_inaccessible_demand_by_instance.csv"].copy()
    response = tables["response_by_truck_inaccessible_demand.csv"].copy()
    correlations = tables["exploratory_accessibility_spearman.csv"].copy()
    statuses = tables["solver_status.csv"].copy()
    validation = tables["validation.csv"].copy()

    expected_counts = {0: 1, 4: 5, 8: 5, 12: 5, 16: 5}
    observed_counts = {
        int(key): int(value)
        for key, value in runs["damage_q"].value_counts().sort_index().to_dict().items()
    }
    require(len(runs) == 21, f"Expected 21 unique exact cases, observed {len(runs)}")
    require(runs["scenario"].nunique() == 21, "all_runs.csv contains duplicate scenario keys")
    require(observed_counts == expected_counts, f"Unexpected damage-level counts: {observed_counts}")
    require(set(pd.to_numeric(runs["total_nodes"], errors="coerce")) == {20}, "Cases are not all 20-node instances")

    scenario_set = set(runs["scenario"].astype(str))
    for name, frame in (
        ("battery_allocation_by_instance.csv", battery),
        ("truck_inaccessible_demand_by_instance.csv", inaccessible),
        ("response_by_truck_inaccessible_demand.csv", response),
        ("solver_status.csv", statuses),
        ("validation.csv", validation),
    ):
        require(len(frame) == 21, f"{name} must contain 21 rows")
        require(frame["scenario"].nunique() == 21, f"{name} contains duplicate scenario keys")
        require(set(frame["scenario"].astype(str)) == scenario_set, f"{name} scenario keys do not match all_runs.csv")

    require(statuses["status"].astype(str).eq("OPTIMAL").all(), "Not all cases are marked OPTIMAL in solver_status.csv")
    require(pd.to_numeric(validation["validation_metric_sum"], errors="coerce").abs().le(NUMERICAL_TOLERANCE).all(), "At least one case fails validation.csv")
    violation_columns = [
        column
        for column in validation.columns
        if column.endswith("_count") or column.endswith("_violation_count")
    ]
    for column in violation_columns:
        require(
            pd.to_numeric(validation[column], errors="coerce").fillna(np.inf).abs().le(NUMERICAL_TOLERANCE).all(),
            f"Nonzero validation count in {column}",
        )
    for column in ("max_energy_reconciliation_error_kwh", "max_share_sum_residual", "objective_reconciliation_error"):
        require(
            pd.to_numeric(validation[column], errors="coerce").abs().le(NUMERICAL_TOLERANCE).all(),
            f"Validation tolerance exceeded in {column}",
        )

    battery_sorted = battery.sort_values("scenario").reset_index(drop=True)
    runs_sorted = runs.sort_values("scenario").reset_index(drop=True)
    response_sorted = response.sort_values("scenario").reset_index(drop=True)
    inaccessible_sorted = inaccessible.sort_values("scenario").reset_index(drop=True)
    shared_instance_metrics = (
        "phi_drive",
        "phi_drone",
        "phi_grid",
        "route_distance_km",
        "total_drone_sorties",
        "material_coverage_ratio",
        "microgrid_utility_ratio",
    )
    for metric in shared_instance_metrics:
        assert_close(f"all_runs vs battery instance: {metric}", runs_sorted[metric], battery_sorted[metric])
        if metric in response_sorted.columns:
            assert_close(f"all_runs vs accessibility response: {metric}", runs_sorted[metric], response_sorted[metric])
    assert_close(
        "inaccessible-demand share",
        inaccessible_sorted["truck_inaccessible_material_demand_share"],
        response_sorted["truck_inaccessible_material_demand_share"],
    )
    battery_share_sum = battery[["phi_drive", "phi_drone", "phi_grid", "phi_unused"]].sum(axis=1)
    assert_close("battery share sum", battery_share_sum, np.ones(len(battery_share_sum)))

    q_values = summary["damage_q"].astype(int).tolist()
    require(q_values == list(expected_counts), f"summary_by_damage.csv has unexpected q levels: {q_values}")
    require(
        summary["observation_count"].astype(int).tolist() == list(expected_counts.values()),
        "summary_by_damage.csv observation counts do not match the frozen design",
    )
    require(service["damage_q"].astype(int).tolist() == q_values, "service_response_by_damage.csv has inconsistent q levels")

    grouped = runs.groupby("damage_q", as_index=False).mean(numeric_only=True).sort_values("damage_q")
    for metric in (
        "phi_drive",
        "phi_drone",
        "phi_grid",
        "phi_unused",
        "route_distance_km",
        "total_drone_sorties",
        "material_coverage_ratio",
        "microgrid_utility_ratio",
    ):
        assert_close(
            f"summary mean: {metric}",
            grouped[metric],
            summary[f"{metric}_mean"],
        )
    for metric in ("material_coverage_ratio", "microgrid_utility_ratio", "total_drone_sorties"):
        assert_close(
            f"service summary: {metric}",
            service[f"{metric}_mean"],
            summary[f"{metric}_mean"],
        )

    require(len(correlations) == 7, "exploratory_accessibility_spearman.csv must contain seven audited associations")
    require(
        correlations["predictor"].astype(str).eq("truck_inaccessible_material_demand_share").all(),
        "Unexpected predictor in exploratory_accessibility_spearman.csv",
    )
    require(
        pd.to_numeric(correlations["n_unique_instances"], errors="coerce").eq(21).all(),
        "Accessibility associations are not based on the 21 unique cases",
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.0, width=0.7)


def damage_range_plot(
    summary: pd.DataFrame,
    *,
    metric: str,
    color: str,
    marker: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
    scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (3.42, 2.62),
) -> None:
    q = summary["damage_q"].to_numpy(dtype=float)
    mean = scale * summary[f"{metric}_mean"].to_numpy(dtype=float)
    low = scale * summary[f"{metric}_min"].to_numpy(dtype=float)
    high = scale * summary[f"{metric}_max"].to_numpy(dtype=float)
    mask = q > 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(q[mask], low[mask], high[mask], color=color, alpha=0.14, linewidth=0)
    ax.plot(q, mean, color=color, marker=marker, markerfacecolor="white", markeredgewidth=0.9)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(q)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.97)
    save_figure(fig, output_dir, stem)


def make_battery_stacked(summary: pd.DataFrame, output_dir: Path) -> None:
    q = summary["damage_q"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(3.42, 2.32))
    bottom = np.zeros(len(summary), dtype=float)
    series = (
        ("phi_drive_mean", "Truck driving", COLORS["truck"]),
        ("phi_drone_mean", "Drone operations", COLORS["drone"]),
        ("phi_grid_mean", "Microgrid support", COLORS["microgrid"]),
        ("phi_unused_mean", "Unused battery", COLORS["unused"]),
    )
    for column, label, color in series:
        values = np.clip(100.0 * summary[column].to_numpy(dtype=float), 0.0, None)
        ax.bar(
            q,
            values,
            bottom=bottom,
            width=2.7,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            label=label,
        )
        bottom += values
    ax.set_xlabel(r"$q$")
    ax.set_ylabel("Battery allocation (%)")
    ax.set_xticks(q)
    ax.set_ylim(0, 100)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.85,
        handlelength=1.25,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    style_axis(ax, grid_axis="y")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.75)
    save_figure(fig, output_dir, "battery_allocation_stacked_by_damage")


def make_battery_ribbon(summary: pd.DataFrame, output_dir: Path) -> None:
    q = summary["damage_q"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(3.42, 2.32))
    series = (
        ("phi_drive", "Truck driving", COLORS["truck"], MARKERS["truck"], "-"),
        ("phi_drone", "Drone operations", COLORS["drone"], MARKERS["drone"], "-"),
        ("phi_grid", "Microgrid support", COLORS["microgrid"], MARKERS["microgrid"], "-"),
        ("phi_unused", "Unused battery", "#8793A1", MARKERS["unused"], "--"),
    )
    mask = q > 0
    for metric, label, color, marker, linestyle in series:
        mean = np.clip(100.0 * summary[f"{metric}_mean"].to_numpy(dtype=float), 0.0, None)
        low = np.clip(100.0 * summary[f"{metric}_min"].to_numpy(dtype=float), 0.0, None)
        high = np.clip(100.0 * summary[f"{metric}_max"].to_numpy(dtype=float), 0.0, None)
        alpha = 0.10 if metric == "phi_unused" else 0.14
        ax.fill_between(q[mask], low[mask], high[mask], color=color, alpha=alpha, linewidth=0)
        ax.plot(
            q,
            mean,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.15 if metric == "phi_unused" else 1.8,
            alpha=0.78 if metric == "phi_unused" else 1.0,
            label=label,
        )
    ax.set_xlabel(r"$q$")
    ax.set_ylabel("Battery allocation (%)")
    ax.set_xticks(q)
    ax.set_ylim(0, 78)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.85,
        handlelength=1.25,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    style_axis(ax)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.75)
    save_figure(fig, output_dir, "battery_allocation_response_by_damage")


def make_truck_drone_substitution(summary: pd.DataFrame, output_dir: Path) -> None:
    """Draw route contraction and total drone use in one dual-axis figure."""

    q = summary["damage_q"].to_numpy(dtype=float)
    mask = q > 0
    route_mean = summary["route_distance_km_mean"].to_numpy(dtype=float)
    route_low = summary["route_distance_km_min"].to_numpy(dtype=float)
    route_high = summary["route_distance_km_max"].to_numpy(dtype=float)
    sortie_mean = summary["total_drone_sorties_mean"].to_numpy(dtype=float)
    sortie_low = summary["total_drone_sorties_min"].to_numpy(dtype=float)
    sortie_high = summary["total_drone_sorties_max"].to_numpy(dtype=float)

    fig, left = plt.subplots(figsize=(3.42, 2.32))
    left.fill_between(
        q[mask],
        route_low[mask],
        route_high[mask],
        color=COLORS["truck"],
        alpha=0.14,
        linewidth=0,
    )
    route_line, = left.plot(
        q,
        route_mean,
        color=COLORS["truck"],
        marker=MARKERS["truck"],
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="Route distance (left)",
    )
    left.set_xlabel(r"$q$")
    left.set_ylabel("Truck route distance (km)", color=COLORS["truck"])
    left.set_xticks(q)
    left.set_ylim(0, 240)
    left.tick_params(axis="y", colors=COLORS["truck"])
    left.spines["left"].set_color(COLORS["truck"])
    style_axis(left)

    right = left.twinx()
    right.fill_between(
        q[mask],
        sortie_low[mask],
        sortie_high[mask],
        color=COLORS["drone"],
        alpha=0.14,
        linewidth=0,
    )
    sortie_line, = right.plot(
        q,
        sortie_mean,
        color=COLORS["drone"],
        marker=MARKERS["drone"],
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="Drone sorties (right)",
    )
    right.set_ylabel("Total drone sorties", color=COLORS["drone"])
    right.set_ylim(0, 220)
    right.tick_params(axis="y", colors=COLORS["drone"], length=3.0, width=0.7)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color(COLORS["drone"])
    right.spines["top"].set_visible(False)

    fig.legend(
        handles=[route_line, sortie_line],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.25,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.20, right=0.80, bottom=0.20, top=0.78)
    # Preserve the fixed canvas so this panel and the paired service panel have
    # identical external dimensions when assembled in LaTeX.
    save_figure(fig, output_dir, "truck_drone_substitution_by_damage", tight=False)


def relative_to_baseline(values: np.ndarray, baseline: float) -> np.ndarray:
    """Return percentage change relative to one positive baseline value."""

    require(np.isfinite(baseline) and baseline > 0.0, "Relative-change baseline must be positive")
    return 100.0 * (values / baseline - 1.0)


def make_service_relative_change(service: pd.DataFrame, output_dir: Path) -> None:
    """Draw material coverage and microgrid utility relative to q=0."""

    q = service["damage_q"].to_numpy(dtype=float)
    baseline_rows = service.loc[service["damage_q"].eq(0)]
    require(len(baseline_rows) == 1, "Service response must contain one q=0 baseline")
    baseline_row = baseline_rows.iloc[0]
    mask = q > 0

    transformed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for metric in ("material_coverage_ratio", "microgrid_utility_ratio"):
        baseline = float(baseline_row[f"{metric}_mean"])
        transformed[metric] = tuple(
            relative_to_baseline(service[f"{metric}_{stat}"].to_numpy(dtype=float), baseline)
            for stat in ("mean", "min", "max")
        )

    fig, ax = plt.subplots(figsize=(3.42, 2.32))
    series = (
        ("material_coverage_ratio", "Material coverage", COLORS["material"], MARKERS["material"]),
        ("microgrid_utility_ratio", "Microgrid utility", COLORS["microgrid"], MARKERS["microgrid"]),
    )
    observed_bounds: list[float] = []
    for metric, label, color, marker in series:
        mean, low, high = transformed[metric]
        observed_bounds.extend(low[mask].tolist())
        observed_bounds.extend(high[mask].tolist())
        ax.fill_between(q[mask], low[mask], high[mask], color=color, alpha=0.14, linewidth=0)
        ax.plot(
            q,
            mean,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=label,
        )

    ax.axhline(0.0, color=COLORS["axis"], linewidth=0.8, linestyle="--", alpha=0.72)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel("Relative service change (%)")
    ax.set_xticks(q)
    lower = min(observed_bounds + [0.0])
    upper = max(observed_bounds + [0.0])
    padding = max(0.8, 0.08 * (upper - lower))
    ax.set_ylim(lower - padding, upper + padding)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.85,
        handlelength=1.25,
        borderaxespad=0.0,
    )
    style_axis(ax)
    # Match the mobility panel's axes rectangle as well as its canvas.  The
    # reserved right margin mirrors the space used by the mobility panel's
    # secondary y-axis and keeps the paired data regions visually equal.
    fig.subplots_adjust(left=0.20, right=0.80, bottom=0.20, top=0.78)
    # Preserve the fixed canvas so this panel and the paired mobility panel have
    # identical external dimensions when assembled in LaTeX.
    save_figure(fig, output_dir, "service_relative_change_by_damage", tight=False)


def nested_trajectories(response: pd.DataFrame) -> pd.DataFrame:
    common = response.loc[response["damage_q"].eq(0)].copy()
    require(len(common) == 1, "The accessibility response must contain one common q=0 case")
    realization_ids = sorted(
        response.loc[response["damage_q"].gt(0), "damage_realization"].astype(str).unique()
    )
    trajectories: list[pd.DataFrame] = []
    for realization in realization_ids:
        baseline = common.copy()
        baseline["trajectory_id"] = realization
        damaged = response.loc[
            response["damage_q"].gt(0)
            & response["damage_realization"].astype(str).eq(realization)
        ].copy()
        damaged["trajectory_id"] = realization
        trajectories.append(pd.concat([baseline, damaged], ignore_index=True))
    return pd.concat(trajectories, ignore_index=True).sort_values(["trajectory_id", "damage_q"])


def accessibility_panel(
    response: pd.DataFrame,
    *,
    metric: str,
    color: str,
    marker: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
    scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (2.16, 2.16),
) -> None:
    x_column = "truck_inaccessible_material_demand_share"
    trajectories = nested_trajectories(response)
    grouped = response.groupby("damage_q", as_index=False).mean(numeric_only=True).sort_values("damage_q")

    fig, ax = plt.subplots(figsize=figsize)
    for _, group in trajectories.groupby("trajectory_id", sort=True):
        ordered = group.sort_values("damage_q")
        ax.plot(
            100.0 * ordered[x_column].to_numpy(dtype=float),
            scale * ordered[metric].to_numpy(dtype=float),
            color=color,
            linewidth=0.75,
            alpha=0.16,
            zorder=1,
        )
    ax.scatter(
        100.0 * response[x_column].to_numpy(dtype=float),
        scale * response[metric].to_numpy(dtype=float),
        s=18,
        facecolor="white",
        edgecolor=color,
        linewidth=0.75,
        alpha=0.48,
        zorder=2,
    )
    ax.plot(
        100.0 * grouped[x_column].to_numpy(dtype=float),
        scale * grouped[metric].to_numpy(dtype=float),
        color=color,
        marker=marker,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.95,
        zorder=3,
    )
    ax.set_xlabel("Truck-inaccessible material\ndemand (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-3, 86)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.29, top=0.97)
    save_figure(fig, output_dir, stem)


def combined_accessibility_panel(
    response: pd.DataFrame,
    *,
    series: tuple[tuple[str, str, str, str, float], ...],
    ylabel: str,
    stem: str,
    output_dir: Path,
    ylim: tuple[float, float],
    legend_ncol: int,
    figsize: tuple[float, float] = (2.30, 2.30),
) -> None:
    """Draw one compact accessibility panel with one or more response series."""

    x_column = "truck_inaccessible_material_demand_share"
    trajectories = nested_trajectories(response)
    grouped = response.groupby("damage_q", as_index=False).mean(numeric_only=True).sort_values("damage_q")

    fig, ax = plt.subplots(figsize=figsize)
    mean_handles = []
    for metric, label, color, marker, scale in series:
        for _, group in trajectories.groupby("trajectory_id", sort=True):
            ordered = group.sort_values("damage_q")
            ax.plot(
                100.0 * ordered[x_column].to_numpy(dtype=float),
                scale * ordered[metric].to_numpy(dtype=float),
                color=color,
                linewidth=0.7,
                alpha=0.12,
                zorder=1,
            )
        ax.scatter(
            100.0 * response[x_column].to_numpy(dtype=float),
            scale * response[metric].to_numpy(dtype=float),
            s=16,
            facecolor="white",
            edgecolor=color,
            linewidth=0.7,
            alpha=0.38,
            zorder=2,
        )
        mean_line, = ax.plot(
            100.0 * grouped[x_column].to_numpy(dtype=float),
            scale * grouped[metric].to_numpy(dtype=float),
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.95,
            label=label,
            zorder=3,
        )
        mean_handles.append(mean_line)

    ax.set_xlabel("Truck-inaccessible\nmaterial demand (%)", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xlim(-3, 86)
    ax.set_ylim(*ylim)
    ax.legend(
        handles=mean_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=legend_ncol,
        columnspacing=0.65,
        handlelength=1.25,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    style_axis(ax)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.245, top=0.80)
    save_figure(fig, output_dir, stem, tight=False)


def generate_figures(tables: dict[str, pd.DataFrame], figure_dir: Path) -> list[str]:
    summary = tables["summary_by_damage.csv"].sort_values("damage_q").reset_index(drop=True)
    service = tables["service_response_by_damage.csv"].sort_values("damage_q").reset_index(drop=True)
    response = tables["response_by_truck_inaccessible_demand.csv"].copy()

    make_battery_stacked(summary, figure_dir)
    make_battery_ribbon(summary, figure_dir)
    make_truck_drone_substitution(summary, figure_dir)
    make_service_relative_change(service, figure_dir)

    for metric, color, marker, ylabel, stem in (
        ("phi_drive", COLORS["truck"], MARKERS["truck"], "Battery share (%)", "battery_driving_vs_truck_inaccessible_demand"),
        ("phi_drone", COLORS["drone"], MARKERS["drone"], "Battery share (%)", "battery_drone_vs_truck_inaccessible_demand"),
        ("phi_grid", COLORS["microgrid"], MARKERS["microgrid"], "Battery share (%)", "battery_microgrid_vs_truck_inaccessible_demand"),
    ):
        accessibility_panel(
            response,
            metric=metric,
            color=color,
            marker=marker,
            ylabel=ylabel,
            stem=stem,
            output_dir=figure_dir,
            scale=100.0,
            ylim=(0, 75),
        )

    accessibility_panel(
        response,
        metric="material_coverage_ratio",
        color=COLORS["material"],
        marker=MARKERS["material"],
        ylabel="Material coverage (%)",
        stem="material_coverage_vs_truck_inaccessible_demand",
        output_dir=figure_dir,
        scale=100.0,
        ylim=(70, 101),
    )
    accessibility_panel(
        response,
        metric="microgrid_utility_ratio",
        color=COLORS["microgrid"],
        marker=MARKERS["microgrid"],
        ylabel="Microgrid utility (%)",
        stem="microgrid_utility_vs_truck_inaccessible_demand",
        output_dir=figure_dir,
        scale=100.0,
        ylim=(70, 101),
    )
    accessibility_panel(
        response,
        metric="total_drone_sorties",
        color=COLORS["drone"],
        marker=MARKERS["drone"],
        ylabel="Total drone sorties",
        stem="total_drone_sorties_vs_truck_inaccessible_demand",
        output_dir=figure_dir,
        ylim=(0, 220),
    )

    combined_accessibility_panel(
        response,
        series=(
            ("phi_drive", "Driving", COLORS["truck"], MARKERS["truck"], 100.0),
            ("phi_grid", "Microgrid", COLORS["microgrid"], MARKERS["microgrid"], 100.0),
            ("phi_drone", "Drone", COLORS["drone"], MARKERS["drone"], 100.0),
        ),
        ylabel="Battery share (%)",
        stem="accessibility_battery_allocation",
        output_dir=figure_dir,
        ylim=(0, 75),
        legend_ncol=2,
    )
    combined_accessibility_panel(
        response,
        series=(
            (
                "material_coverage_ratio",
                "Material coverage",
                COLORS["material"],
                MARKERS["material"],
                100.0,
            ),
            (
                "microgrid_utility_ratio",
                "Microgrid utility",
                COLORS["microgrid"],
                MARKERS["microgrid"],
                100.0,
            ),
        ),
        ylabel="Service level (%)",
        stem="accessibility_service_levels",
        output_dir=figure_dir,
        ylim=(70, 101),
        legend_ncol=1,
    )
    combined_accessibility_panel(
        response,
        series=(
            (
                "total_drone_sorties",
                "Total drone sorties",
                COLORS["drone"],
                MARKERS["drone"],
                1.0,
            ),
        ),
        ylabel="Total drone sorties",
        stem="accessibility_drone_sorties",
        output_dir=figure_dir,
        ylim=(0, 220),
        legend_ncol=1,
    )

    stems = (
        "battery_allocation_stacked_by_damage",
        "battery_allocation_response_by_damage",
        "truck_drone_substitution_by_damage",
        "service_relative_change_by_damage",
        "battery_driving_vs_truck_inaccessible_demand",
        "battery_drone_vs_truck_inaccessible_demand",
        "battery_microgrid_vs_truck_inaccessible_demand",
        "material_coverage_vs_truck_inaccessible_demand",
        "microgrid_utility_vs_truck_inaccessible_demand",
        "total_drone_sorties_vs_truck_inaccessible_demand",
        "accessibility_battery_allocation",
        "accessibility_service_levels",
        "accessibility_drone_sorties",
    )
    return [f"{stem}.{suffix}" for stem in stems for suffix in ("png", "svg")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen controlled road-damage CSVs and draw paper figures only."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir if args.result_dir.is_absolute() else ROOT / args.result_dir
    figure_dir = args.figure_dir or (result_dir / "figures")
    if not figure_dir.is_absolute():
        figure_dir = ROOT / figure_dir

    tables = load_tables(result_dir)
    validate_frozen_tables(tables)
    configure_style()
    artifacts = generate_figures(tables, figure_dir)

    print("analysis_only=true solver_invoked=false")
    print("validated_unique_cases=21 all_optimal=true all_valid=true")
    print(f"figure_dir={figure_dir}")
    print(f"figure_artifacts={len(artifacts)}")


if __name__ == "__main__":
    main()
