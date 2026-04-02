from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import project_paths  # noqa: E402


class ProjectPathsTest(unittest.TestCase):
    def test_resolve_default_checkpoint_prefers_repo_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            ckpt = repo_root / "checkpoints" / "best_checkpoint.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_text("x", encoding="utf-8")
            with mock.patch.object(project_paths, "REPO_ROOT", repo_root):
                resolved = project_paths.resolve_default_checkpoint(None)
            self.assertEqual(resolved, ckpt)

    def test_resolve_sft_data_uses_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            data_path = repo_root / "custom.jsonl"
            data_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(project_paths, "REPO_ROOT", repo_root):
                with mock.patch.dict(os.environ, {"ATTNRES_SFT_DATA": str(data_path)}, clear=False):
                    resolved = project_paths.resolve_sft_data_path(None)
            self.assertEqual(resolved, data_path)

    def test_resolve_sft_data_raises_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            with mock.patch.object(project_paths, "REPO_ROOT", repo_root):
                with mock.patch.dict(os.environ, {}, clear=True):
                    with mock.patch("pathlib.Path.cwd", return_value=repo_root):
                        with self.assertRaises(FileNotFoundError):
                            project_paths.resolve_sft_data_path(None)


if __name__ == "__main__":
    unittest.main()
