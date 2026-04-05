from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sft import compute_dtype_for_training, parameter_dtype_for_training  # noqa: E402


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is required for dtype tests")
class TrainingDtypeTest(unittest.TestCase):
    def test_sft_keeps_parameters_in_fp32_on_cpu(self) -> None:
        import torch

        device = torch.device("cpu")
        self.assertEqual(parameter_dtype_for_training(device), torch.float32)
        self.assertEqual(compute_dtype_for_training(device), torch.float32)

    def test_sft_uses_bf16_compute_on_cuda_but_fp32_params(self) -> None:
        import torch

        device = torch.device("cuda")
        self.assertEqual(parameter_dtype_for_training(device), torch.float32)
        self.assertEqual(compute_dtype_for_training(device), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
