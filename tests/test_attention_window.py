from __future__ import annotations

import sys
import unittest
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attention_window import build_causal_window_mask  # noqa: E402


HAS_TORCH = importlib.util.find_spec("torch") is not None


class AttentionWindowTest(unittest.TestCase):
    def test_full_window_returns_none(self) -> None:
        self.assertIsNone(build_causal_window_mask(8, (-1, -1)))

    def test_sliding_window_is_causal_and_bounded(self) -> None:
        mask = build_causal_window_mask(5, (2, 0))
        self.assertEqual(
            mask,
            (
                (True, False, False, False, False),
                (True, True, False, False, False),
                (True, True, True, False, False),
                (False, True, True, True, False),
                (False, False, True, True, True),
            ),
        )

    def test_mask_result_is_cached(self) -> None:
        self.assertIs(
            build_causal_window_mask(16, (4, 0)),
            build_causal_window_mask(16, (4, 0)),
        )

    @unittest.skipUnless(HAS_TORCH, "torch is required for dtype alignment test")
    def test_value_residual_dtype_alignment(self) -> None:
        import torch

        v = torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16)
        ve = torch.ones((1, 2, 3, 4), dtype=torch.float32)
        gate = torch.ones((1, 2, 3), dtype=torch.float32)
        mixed = v + gate.to(v.dtype).unsqueeze(-1) * ve.to(v.dtype)
        self.assertEqual(mixed.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
