# SDRP-ENR

**Service-Decoded Route Pool Matheuristic with Elite Neighborhood Refinement**

Research implementation for post-disaster electric-truck and drone relief
logistics with temporary microgrid energy support.

## Method

```text
Instance -> Four route generators -> Joint fixed-route service MILP
         -> Top-10 elite route skeletons -> Two-neighborhood refinement
         -> Truck routes, goods deliveries, energy support and drone sorties
```

The route pool combines H-seed routes, edge enrichment, random insertion and
balanced insertion. Elite refinement uses drop-and-reinsert and truck rebalance.
The service decoder jointly considers all trucks, not independent per-truck
allocations. It allocates direct deliveries, microgrid support, star sorties and
rendezvous sorties under the original shared battery, payload and time budgets.

Physical H visits may be shared by trucks; at most one truck charges each H.
Drones cannot serve ordinary demand nodes physically visited by any truck.
Within a truck route, repeated non-depot nodes are prohibited.

The mathematical model is in [FROZEN_MODEL_SPEC.md](FROZEN_MODEL_SPEC.md).
Algorithm parameters and implementation notes are in
[sdrp_enr/README.md](sdrp_enr/README.md).

## Setup

Python 3.10 or newer and a valid Gurobi license are required. Run commands from
this repository's root. Installing the Python package does not supply a Gurobi
license.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[figures]"
$env:PYTHONHASHSEED = "0"
.\.venv\Scripts\python.exe -m sdrp_enr.run_final_solver --max-scenarios 1 --seeds 20260427
```

For the tested dependency versions, see `requirements-reproducible.txt`.
On Linux/macOS, use the equivalent Python environment and
`export PYTHONHASHSEED=0`.

## Experiment Entrypoints

```powershell
python -m sdrp_enr.run_final_solver --seeds 20260427 20260428 20260429
python -m sdrp_enr.run_ablation --seeds 20260427 20260428 20260429
python -m sdrp_enr.run_scalability --seeds 20260427
python -m sdrp_enr.run_sensitivity --seeds 20260427
python scripts/run_exact_benchmark.py --time-limit-sec 1800
```

Use `--max-scenarios 1` for a smoke run and `--help` for parameters. Output
folders default to `results/sdrp_enr_final/`, `results/sdrp_enr_ablation/`,
`results/sdrp_enr_scalability/`, `results/sdrp_enr_sensitivity/` and
`results/milp_benchmark/`. Use separate output folders for different settings;
do not resume one configuration into another.

The repository includes 25 benchmark instances (5-50 nodes), eight large
instances (75-200 nodes), and a compact, status-labelled existing MILP reference.
Missing exact references are reported as unavailable, not as zero optimality gap.

## Verification

```powershell
$env:PYTHONHASHSEED = "0"
python -m unittest discover -s tests -v
python scripts/check_algorithm_migration.py
```

SDRP-ENR is a heuristic: the route pool does not certify global optimality.
Fixed-route MILPs have time limits, and the accepted refinement currently uses
OPTIMAL, validation-passing route evaluations. Results depend on solver version,
tolerances, time limit and hash/random seeds. A migration check is not evidence
that all large-instance optima are known.

## Repository Layout

- `sdrp_enr/`: current solver and experiment entrypoints.
- `src/humanitarian_graph/`: network, vehicle and scenario representations.
- `scripts/solve_gurobi_model.py`: preserved full-MILP benchmark model.
- `data/benchmark/`, `data/scalability/`: portable final-parameter instances.
- `tests/`: structural and numerical migration regression checks.
- `docs/ALGORITHM_CLEANUP.md`: migration provenance and reporting corrections.

Manuscripts, third-party papers, complete historical results, virtual
environments, credentials and the local recovery archive are excluded from Git.
