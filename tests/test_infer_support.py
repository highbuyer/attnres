from __future__ import annotations

import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infer_support import ensure_inference_backend, has_flash_attention_backend  # noqa: E402


class InferSupportTest(unittest.TestCase):
    def test_has_flash_attention_backend_requires_callable_kernel(self) -> None:
        self.assertFalse(has_flash_attention_backend(None))
        self.assertFalse(has_flash_attention_backend(types.SimpleNamespace()))
        self.assertTrue(has_flash_attention_backend(types.SimpleNamespace(flash_attn_func=lambda *args, **kwargs: None)))

    def test_cpu_inference_is_allowed(self) -> None:
        backend = types.SimpleNamespace(flash_attn_func=lambda *args, **kwargs: None)
        ensure_inference_backend("cpu", backend)

    def test_missing_flash_attention_on_cuda_is_allowed_with_warning(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            ensure_inference_backend("cuda", None)
        self.assertIn("FlashAttention not found", stdout.getvalue())

    def test_unsupported_device_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            ensure_inference_backend("tpu", None)


if __name__ == "__main__":
    unittest.main()
