# Algorithm Cleanup (2026-08-27)

## Retained Algorithm

The implementation is now self-contained in `sdrp_enr/`. It no longer imports
`alns/` or `ALNS+MILP/`. The accepted four generators, joint service decoder,
integrated H bridge handling, and two elite neighborhoods were migrated without
changing their mathematical evaluation or candidate-generation rules.

Historical ALNS variants, the star-only decoder, sequential decoding, regional
generator experiments, CT-chain, segment exchange, and recombination experiment
entrypoints have been removed from the active source tree. The full MILP model,
frozen formulation, original scenarios, figures, manuscripts and experimental
results were not deleted. The old full-MILP experiment wrapper was consolidated
as `scripts/run_exact_benchmark.py`.

## Recovery

Before migration, the algorithm directories, scripts and historical design notes
were backed up locally as `tmp/algorithm_cleanup_before_20260827.zip`.
SHA-256: `BAFAFF9A9C80940BDF87556F311905AFBCB455F0CAF021053005A3391A31A132`.
This recovery archive is excluded from Git. Historical output files remain local
and are not overwritten by the new commands.

## Verification

- A static dependency audit covered 80 retained Python files and found no imports
  of the deleted modules.
- Five pre-migration route-pool signatures (5, 10, 20, 30 and 50 nodes) matched.
- Small-instance decoder objectives and refinement results matched the recorded
  pre-migration fixtures under `PYTHONHASHSEED=0`.
- Eight unit/integration tests passed, including duplicate physical H visits,
  charging uniqueness, route validity, global CT drone exclusion and frozen
  parameters in all 33 portable scenarios.
- Full-solver, 13-variant ablation, nine-variant sensitivity, scalability-runner
  and full-MILP entrypoints were exercised on a small instance.
- These are migration tests, not a rerun of every paper experiment or a new
  large-scale performance claim.

## Reporting Corrections

The migrated experiment runners include route-pool decoding in `runtime_sec`.
Some historical runners omitted it. New timing values must not be silently
compared with those old values; historical CSVs have not been edited.

`gap_to_exact_if_available` is reported only for an OPTIMAL full-MILP reference;
`gap_to_milp_incumbent` remains a separate comparison when the reference stopped
at its limit. Solver OPTIMAL status is subject to the configured tolerances.

The full-MILP reporting wrapper obtains service component populations from the
scenario, rather than assuming a fixed population at every microgrid. The
underlying optimization model and its objective remain unchanged.

Legacy source paths mentioned in the frozen mathematical specification refer to
their migrated counterparts: data/solution/evaluator are in `sdrp_enr/`, the
current decoder is `sdrp_enr/service_decoder.py`, and the two-neighborhood search
is `sdrp_enr/elite_refinement.py`. The formula document itself was left untouched.
