"""Structural and fixed-route regression tests for the published configuration."""
import importlib
import math
from pathlib import Path
import pkgutil
import unittest
from types import SimpleNamespace

from sdrp_enr.data import load_data, microgrid_utility
from sdrp_enr.experiment_io import read_manifest, load_experiment_table
from sdrp_enr.elite_refinement import EliteRefiner
from sdrp_enr.service_decoder import FixedRouteServiceMILP

ROOT = Path(__file__).resolve().parents[1]


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(ROOT / "data/benchmark/suite_v2_n10_i2_t2.json")

    def test_package_imports(self):
        import sdrp_enr
        for module in pkgutil.iter_modules(sdrp_enr.__path__):
            importlib.import_module("sdrp_enr." + module.name)

    def test_portable_manifests_and_parameters(self):
        for suite, count in [("benchmark", 25), ("scalability", 8)]:
            table = read_manifest(ROOT / "data" / suite / "manifest.csv")
            self.assertEqual(len(table), count)
            for path in table["json"]:
                d = load_data(path)
                self.assertEqual(d.drone_battery, 4)
                self.assertEqual(d.drone_payload, 2)
                self.assertAlmostEqual(d.drone_tmax_hours, 1 / 3)
                self.assertEqual(d.graph.e_truck.Qt, 800)
                self.assertEqual(d.truck_battery, 600)

    def test_missing_baseline_is_not_zero_gap(self):
        args = SimpleNamespace(manifest=ROOT / "data/benchmark/manifest.csv", baseline_summary=None, max_scenarios=1)
        row = load_experiment_table(args).iloc[0]
        self.assertTrue(math.isnan(row["baseline_objective"]))

    def test_utility_breakpoints(self):
        for c, g in [(0, 0), (0.5, 0.625), (1, 1)]:
            self.assertAlmostEqual(microgrid_utility(c), g)

    def test_final_neighborhoods_and_disabled_refinement(self):
        r = EliteRefiner(self.data, "test", 1, 10, 10)
        self.assertEqual(r.allowed_operators, {"Rebalance-LNS", "Drop-and-reinsert-LNS"})
        r = EliteRefiner(self.data, "test", 1, 10, 10, allowed_operators=set())
        self.assertEqual(r.allowed_operators, set())

    def test_shared_h_single_charge(self):
        d = self.data
        h = d.h_nodes[0]
        routes = {v: [d.depot, h, d.depot] for v in d.truck_ids}
        result = FixedRouteServiceMILP(d).solve(routes)
        self.assertEqual(result.status, "OPTIMAL")
        self.assertLessEqual(sum(result.solution.validation_metrics.values()), 1e-9)
        self.assertLessEqual(sum(t > 1e-8 for (node, v), t in result.solution.tau.items() if node == h), 1)

    def test_route_rejections(self):
        d = self.data
        h, h2 = d.h_nodes[:2]
        empty = {v: [d.depot] for v in d.truck_ids}
        for route in [[d.depot, h, d.depot, h2, d.depot], [d.depot, h, h2, h, d.depot]]:
            result = FixedRouteServiceMILP(d).solve({**empty, 0: route})
            self.assertEqual(result.status, "route_invalid")

    def test_global_ct_exclusion_and_objective(self):
        from sdrp_enr.evaluation import generate_seed_scenario_pool
        args = SimpleNamespace(service_milp_time_limit_sec=10, route_pool_max_size_per_scenario=40,
            original_random_count=10, balanced_random_count=10, edge_replacement_chain_max_len=3,
            rcl_size=15, ct_target_ratio=.25, max_ct_chain_len_preference=3,
            target_route_time_utilization=.85, target_energy_utilization=.85, insertion_temperature=.6)
        d = load_data(ROOT / "data/benchmark/rvct_final25_n10_t1.json")
        _, pool, _ = generate_seed_scenario_pool(d, d.scenario, 20260427, args)
        candidate = next(c for c in pool if any(n in d.c_truck for r in c.routes.values() for n in r))
        result = FixedRouteServiceMILP(d).solve(candidate.routes)
        self.assertEqual(result.status, "OPTIMAL")
        visited = {n for r in candidate.routes.values() for n in r}
        self.assertFalse(any(t.service in visited for t in result.solution.star_tasks + result.solution.rendezvous_tasks))
        self.assertAlmostEqual(result.objective, result.solution.served_material_score + result.solution.microgrid_score, places=5)


if __name__ == "__main__":
    unittest.main()
