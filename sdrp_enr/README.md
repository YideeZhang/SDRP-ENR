# SDRP-ENR Implementation

The current implementation has four route generators, one joint service MILP
decoder, and two elite neighborhoods. No legacy algorithm directory is required.

## Components

| Module | Responsibility |
| --- | --- |
| `data.py`, `solution.py`, `evaluator.py` | Instance loading, solution records and feasibility checks |
| `seed_routes.py`, `seed_insertion.py` | Accepted round-robin H-seed construction and fallback insertion |
| `insertion.py`, `pool_base.py`, `route_pool.py` | Feasible shortest-path insertion and four-generator pool |
| `service_decoder.py` | Joint-fleet fixed-route service optimization |
| `elite_refinement.py` | Cached, state-aware drop-and-reinsert and rebalance |
| `evaluation.py` | Shared pool evaluation and reporting helpers |
| `run_final_solver.py` | Final full-method benchmark |
| `run_ablation.py` | Service and method-component ablations |
| `run_scalability.py` | Large-instance feasibility and service performance |
| `run_sensitivity.py` | Objective weights, successor window and drone count |

## Frozen Defaults

| Parameter | Value | Meaning |
| --- | --- | --- |
| Material unit | 25 kg | Physical interpretation of one payload unit |
| Truck battery | 600 kWh | Shared driving, drone and microgrid budget |
| Truck gross capacity | 800 units | Onboard drone weight is subtracted |
| Drone payload | 2 units | Fixed delivery package per sortie |
| Drone battery | 4 kWh | Per-sortie energy capacity |
| Drone endurance | 1200 s / 18 km | At the scenario drone speed |
| Drones per truck | 3 | Base fleet configuration |
| Mission duration | 24 h | Truck mission bound |
| Microgrid output | 60 kW | Energy supplied during support dwell |
| Restoration horizon | 10 h | Base microgrid demand horizon |
| Objective weights | alpha=beta=1 | Energy utility and goods service |
| Successor window | K=3 | Maximum rendezvous route-position gap |
| Random insertion attempts | 100 | Before deduplication |
| Balanced insertion attempts | 100 | Before deduplication |
| Unique pool cap | 500 | Upper bound, not guaranteed pool size |
| Edge-chain maximum length | 3 | Enrichment candidate length |
| Balanced RCL size | 15 | Restricted candidate list |
| CT target ratio | 0.25 | Balanced insertion preference |
| Time / energy utilization targets | 0.85 / 0.85 | Balanced insertion scoring targets |
| Insertion temperature | 0.6 | Randomized score sampling |
| Elite count | 10 | Best distinct, valid OPTIMAL route evaluations |
| Rounds per elite | 20 | Maximum neighborhood rounds |
| New refinement evaluations | 200 | Per-scenario cap with route-signature caching |
| Decoder time limit | 10 s | Per fixed-route MILP solve |
| Random neighborhood selection | 20% | Exploration; otherwise state-aware |
| Integrated H bridge handling | Enabled | Embedded within generators and neighborhoods |

Pool proportions are emergent after feasibility checks and deduplication; the
100/100 settings are attempt counts, not fixed final percentages. Identical
route skeletons are evaluated once within a pool/decoder configuration.

Refinement accepts non-worsening candidates, within numerical tolerance. The
final method does not use simulated annealing or adaptive ALNS weights.
CT-chain, segment exchange, standalone duplicate-H insertion, sequential
decoding and guarded recombination are not part of this source release.

The optional activation branch remains inside truck rebalance, not as a third
neighborhood. As in the accepted experiment entrypoints, the benchmark/ablation
and scalability commands leave it off by default; sensitivity retains its
previously accepted enabled setting, controllable with
`--no-enable-activation-rebalance`. Do not silently mix these settings in
cross-experiment claims.

## Output Interpretation

`objective` is the achieved population-weighted score.
`material_coverage_ratio` and `microgrid_utility_ratio` use the corresponding
population denominators. The microgrid ratio is concave utility, not raw kWh
coverage. `normalized_objective` divides by the theoretical full-service score.

`gap_to_exact_if_available` is only populated for an OPTIMAL MILP reference.
A time-limited MILP incumbent is not an exact optimum. New runner
`runtime_sec` includes pool generation, pool decoding, refinement and final
decoding; historical timings may use a narrower scope.

`star_count` counts sorties (including repeated sorties in a task row);
`rendezvous_count` counts activated rendezvous tasks. A zero validation metric
is a feasibility check, not an optimality certificate.

The mathematical fixed-route optimum can be written as F(R)=max_S Z(R,S).
With a time limit, a solver incumbent is only an achieved value unless the
fixed-route solve is certified OPTIMAL. No global route optimum is claimed.

## Programmatic Use

```python
from sdrp_enr.data import load_data
from sdrp_enr.service_decoder import FixedRouteServiceMILP

data = load_data("data/benchmark/suite_v2_n5_i1_t1.json")
routes = {0: ["D0", "H0", "D0"]}
result = FixedRouteServiceMILP(data, time_limit_sec=10).solve(routes)
print(result.status, result.objective)
```

Use real feasible route arcs for the chosen instance; the decoder rejects
invalid skeletons. All truck routes are supplied together.
