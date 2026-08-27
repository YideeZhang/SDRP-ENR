from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.controlled_road_damage_battery_common import (  # noqa: E402
    BASE_INSTANCE,
    DAMAGE_LEVELS,
    DAMAGE_SEEDS,
    DATA_DIR,
    TARGET_MIP_GAP,
    TIME_LIMIT_SEC,
    apply_damage,
    canonical_invariant_hashes,
    canonicalize_undamaged,
    file_hash,
    read_json,
    require_base_instance,
    stable_hash,
    validate_graph,
    write_json,
)


def scenario_name(q: int, seed: int | None) -> str:
    if q == 0:
        return "controlled_n20_q00_common"
    if seed is None:
        raise ValueError("A nonzero damage level requires a realization seed")
    return f"controlled_n20_q{q:02d}_r{seed}"


def build_suite(base_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not base_path.exists():
        raise FileNotFoundError(f"Required base instance is missing: {base_path}")
    base = read_json(base_path)
    backbone, eligible, graph_rows = require_base_instance(base)
    canonical = canonicalize_undamaged(base, eligible)
    invariants = canonical_invariant_hashes(canonical)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    audit_rows: list[dict] = []
    damage_rows: list[dict] = []

    def save_case(q: int, seed: int | None, permutation: list[str], closed: list[str]) -> None:
        realization = "common" if seed is None else str(seed)
        name = scenario_name(q, seed)
        payload = apply_damage(
            canonical,
            scenario_name=name,
            q=q,
            realization=realization,
            closed_links=closed,
        )
        target = output_dir / f"{name}.json"
        write_json(target, payload)
        audit = validate_graph(
            payload,
            expected_q=q,
            expected_closed=closed,
            invariant_hashes=invariants,
            backbone=backbone,
            eligible=eligible,
        )
        audit_rows.append(
            {
                "scenario": name,
                "damage_q": q,
                "damage_realization": realization,
                "json_sha256": file_hash(target),
                "canonical_payload_sha256": stable_hash(payload),
                **audit,
            }
        )
        manifest_rows.append(
            {
                "scenario": name,
                "json": str(target.resolve()),
                "total_nodes": audit["total_nodes"],
                "h_count": audit["h_count"],
                "c_count": audit["ct_count"] + audit["drone_only_count"],
                "truck_count": audit["truck_count"],
                "damage_q": q,
                "damage_realization": realization,
                "closed_link_count": len(closed),
                "closed_link_ids": ";".join(sorted(closed)),
                "permutation_sha256": stable_hash(permutation),
                "json_sha256": file_hash(target),
                "time_limit_sec": TIME_LIMIT_SEC,
                "target_mip_gap": TARGET_MIP_GAP,
                "successor_gap": 3,
                "pilot_case": q == 0 or seed == DAMAGE_SEEDS[0],
                "generation_attempts": 1,
                "rejection_sampling_used": False,
            }
        )
        rank = {uid: idx + 1 for idx, uid in enumerate(permutation)}
        for uid in eligible:
            damage_rows.append(
                {
                    "scenario": name,
                    "damage_q": q,
                    "damage_realization": realization,
                    "link_id": uid,
                    "permutation_rank": rank.get(uid, 0),
                    "closed": uid in set(closed),
                    "protected_backbone": False,
                }
            )
        for uid in backbone:
            damage_rows.append(
                {
                    "scenario": name,
                    "damage_q": q,
                    "damage_realization": realization,
                    "link_id": uid,
                    "permutation_rank": 0,
                    "closed": False,
                    "protected_backbone": True,
                }
            )

    save_case(0, None, [], [])
    for seed in DAMAGE_SEEDS:
        permutation = list(eligible)
        random.Random(seed).shuffle(permutation)
        for q in DAMAGE_LEVELS[1:]:
            save_case(q, seed, permutation, permutation[:q])

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["damage_q", "damage_realization"], kind="stable"
    )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["damage_q", "damage_realization"], kind="stable"
    )
    damage = pd.DataFrame(damage_rows).sort_values(
        ["damage_q", "damage_realization", "protected_backbone", "permutation_rank", "link_id"],
        kind="stable",
    )
    graph_audit = pd.DataFrame(graph_rows)
    graph_audit.insert(0, "source_instance", str(base_path.resolve()))
    graph_audit.to_csv(output_dir / "base_graph_inventory.csv", index=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    audit.to_csv(output_dir / "graph_audit.csv", index=False)
    damage.to_csv(output_dir / "damage_links.csv", index=False)
    write_json(
        output_dir / "suite_config.json",
        {
            "source_instance": str(base_path.resolve()),
            "source_instance_sha256": file_hash(base_path),
            "canonical_invariant_hashes": invariants,
            "damage_levels": list(DAMAGE_LEVELS),
            "damage_realization_seeds": list(DAMAGE_SEEDS),
            "protected_backbone_links": backbone,
            "eligible_secondary_links": eligible,
            "unique_instance_count": len(manifest),
            "sampling_rule": "random.Random(seed).shuffle(sorted eligible link IDs); close the first q links",
            "nested_damage_sets": True,
            "outcome_adaptive_regeneration": False,
            "rejection_sampling_used": False,
        },
    )
    return manifest, audit, damage


def verify_nested_sets(damage: pd.DataFrame) -> None:
    closed_mask = (
        damage["closed"]
        if pd.api.types.is_bool_dtype(damage["closed"])
        else damage["closed"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )
    for seed in DAMAGE_SEEDS:
        previous: set[str] = set()
        for q in DAMAGE_LEVELS[1:]:
            current = set(
                damage.loc[
                    (damage["damage_realization"].astype(str) == str(seed))
                    & (damage["damage_q"] == q)
                    & closed_mask,
                    "link_id",
                ].astype(str)
            )
            if len(current) != q or not previous.issubset(current):
                raise RuntimeError(f"Nested damage verification failed for realization={seed}, q={q}")
            previous = current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-instance", type=Path, default=BASE_INSTANCE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    base = args.base_instance if args.base_instance.is_absolute() else ROOT / args.base_instance
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    manifest, audit, damage = build_suite(base, output)
    verify_nested_sets(damage)
    if len(manifest) != 21:
        raise RuntimeError(f"Expected 21 unique instances, generated {len(manifest)}")
    if not audit["structural_validation_passed"].astype(bool).all():
        failed = audit.loc[~audit["structural_validation_passed"].astype(bool), "scenario"].tolist()
        raise RuntimeError(f"Structural validation failed for: {failed}")
    print(f"generated_instances={len(manifest)}")
    print(f"all_structural_valid={audit['structural_validation_passed'].all()}")
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
