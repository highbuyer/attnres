from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sft import resolve_resume_state  # noqa: E402


class SftResumeTest(unittest.TestCase):
    def test_resume_uses_best_val_when_present(self) -> None:
        step, best = resolve_resume_state(
            {'optimizer_state': {'state': {}, 'param_groups': []}, 'step': 9000, 'val_bpt': 3.8, 'best_val_bpt': 3.7},
            True,
        )
        self.assertEqual(step, 9000)
        self.assertEqual(best, 3.7)

    def test_resume_falls_back_to_val_bpt(self) -> None:
        step, best = resolve_resume_state(
            {'optimizer_state': {'state': {}, 'param_groups': []}, 'step': 500, 'val_bpt': 4.2},
            True,
        )
        self.assertEqual(step, 500)
        self.assertEqual(best, 4.2)

    def test_no_resume_starts_fresh(self) -> None:
        step, best = resolve_resume_state({'step': 9000, 'val_bpt': 3.8}, False)
        self.assertEqual(step, 0)
        self.assertTrue(math.isinf(best))

    def test_resume_requires_optimizer_state(self) -> None:
        with self.assertRaises(ValueError):
            resolve_resume_state({'step': 9000, 'val_bpt': 3.8}, True)


if __name__ == '__main__':
    unittest.main()
