"""Offline regression and adversarial checks. No network credentials needed."""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meridian_adapter import inspect  # noqa: E402
from research import (  # noqa: E402
    DataAnalyst,
    IndependentAuditor,
    Orchestrator,
    ScenarioSimulator,
    evaluate,
    load_weekly,
)


class CommerceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_weekly(ROOT / "data/public_weekly_2019.csv")

    def test_01_weeks(self):
        self.assertEqual(len(self.rows), 52)

    def test_02_no_future(self):
        with self.assertRaises(ValueError):
            DataAnalyst().run(self.rows[:13], 12)

    def test_03_min_history(self):
        with self.assertRaises(ValueError):
            DataAnalyst().run(self.rows[:10], 11)

    def test_04_window(self):
        with self.assertRaises(ValueError):
            DataAnalyst().run(self.rows[:15], 15, 99)

    def test_05_nan(self):
        with self.assertRaises(ValueError):
            ScenarioSimulator.budgets(
                {"facebook": 1, "google": 2},
                {"recipient": "facebook", "transfer_fraction": math.nan},
            )

    def test_06_channel(self):
        with self.assertRaises(ValueError):
            ScenarioSimulator.budgets(
                {"facebook": 1, "google": 2}, {"recipient": "invalid", "transfer_fraction": 0.1}
            )

    def test_07_negative_budget(self):
        with self.assertRaises(ValueError):
            ScenarioSimulator.budgets(
                {"facebook": 100, "google": 1}, {"recipient": "facebook", "transfer_fraction": 0.5}
            )

    def test_08_conservation(self):
        x = ScenarioSimulator.budgets(
            {"facebook": 50, "google": 80}, {"recipient": "facebook", "transfer_fraction": 0.1}
        )
        self.assertAlmostEqual(sum(x.values()), 130)

    def test_09_over_limit(self):
        past = self.rows[:12]
        a = DataAnalyst().run(past, 12)
        p = {"recipient": "facebook", "transfer_fraction": 0.2}
        self.assertIn(
            "over_limit",
            IndependentAuditor().evaluate(a, p, ScenarioSimulator().stress(past, p), past)[
                "violations"
            ],
        )

    def test_10_boolean(self):
        with self.assertRaises(ValueError):
            ScenarioSimulator.budgets(
                {"facebook": 50, "google": 80}, {"recipient": "facebook", "transfer_fraction": True}
            )

    def test_11_missing_simulation(self):
        past = self.rows[:12]
        a = DataAnalyst().run(past, 12)
        self.assertFalse(
            IndependentAuditor().evaluate(
                a, {"recipient": "facebook", "transfer_fraction": 0.1}, None, past
            )["research_pass"]
        )

    def test_12_feedback_loop(self):
        d = Orchestrator().decide(self.rows[:12], 12)
        self.assertEqual(len(d["attempts"]), 3)
        self.assertEqual(d["decision"], "research_only")
        self.assertFalse(d["production_write_allowed"])

    def test_13_holdout(self):
        r = evaluate(self.rows)
        self.assertEqual(r["holdout_weeks"], 40)
        self.assertTrue(all(w["history_end"] < w["week"] for w in r["decisions"]))

    def test_14_forecast_count(self):
        self.assertEqual(len(evaluate(self.rows)["predictions"]), 240)

    def test_15_nonfinite_audit(self):
        past = self.rows[:12]
        a = DataAnalyst().run(past, 12)
        self.assertFalse(
            IndependentAuditor().evaluate(
                a, {"recipient": "facebook", "transfer_fraction": float("inf")}, None, past
            )["research_pass"]
        )

    def test_16_meridian_header(self):
        cols = "time,conversions,revenue_per_conversion,Promo,competitor_sales_control," + ",".join(
            f"Channel{i}_spend" for i in range(5)
        )
        sample = cols + "\n2021-01-01,10,.02,0,0,1,2,3,4,5\n"
        self.assertTrue(inspect(sample)["synthetic"])

    def test_17_meridian_dupe(self):
        cols = "time,conversions,revenue_per_conversion,Promo,competitor_sales_control," + ",".join(
            f"Channel{i}_spend" for i in range(5)
        )
        sample = cols + "\n2021-01-01,10,.02,0,0,1,2,3,4,5\n2021-01-01,10,.02,0,0,1,2,3,4,5\n"
        with self.assertRaises(ValueError):
            inspect(sample)


if __name__ == "__main__":
    unittest.main()
