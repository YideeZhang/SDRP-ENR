# MILP Reference

`baseline_summary.csv` is a compact copy of the existing final-parameter reference
from `results/paper_suite_final25_with_rendezvous_gurobi/baseline_summary.csv`.
It is not a newly solved benchmark. Each row retains its solver status and gap;
TIME_LIMIT rows are incumbents, not certified exact objective values.

The 25 corresponding instances are in `data/benchmark/`. Their JSON contents
were copied verbatim; the new manifest uses relative paths for portability.
The eight large instances are in `data/scalability/`; only their parent-source
path metadata is made relative, without altering modeled inputs. They have no exact-reference
optimality claims. Full logs, papers and historical experiment outputs are not
part of this source release.

Recompute references with `python scripts/run_exact_benchmark.py` from the
repository root. A valid Gurobi license is required.
