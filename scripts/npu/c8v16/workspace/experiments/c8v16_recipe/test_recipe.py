"""Exercise the actual base DAPO CLI builder on CPU with a temporary filesystem."""
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import aime_protocol
import train_c8v16 as recipe


class RecipeTests(unittest.TestCase):
    def test_dry_build_routes_both_roles_and_all_checkpoint_steps(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = workspace / "models/Qwen3-30B-A3B"
            model.mkdir(parents=True)
            (model / "config.json").write_text('{"torch_dtype":"bfloat16"}')
            for folder, filename, count in [("dapo-math-17k", "train.jsonl", 1), ("aime-2024", "eval.jsonl", 30)]:
                path = workspace / "data" / folder / filename
                path.parent.mkdir(parents=True)
                path.write_text('{"label":"2"}\n' * count)
            with patch.object(recipe, "dataset_identity"), patch.dict(os.environ, {"HCCL_IF_IP": "127.0.0.1"}), contextlib.redirect_stdout(io.StringIO()):
                recipe.main(["--workspace", directory, "--print-command"])
            [manifest] = (workspace / "runs").rglob("manifest.json")
            info = json.loads(manifest.read_text())
            argv = info["argv"]
            for name in ["--sglang-attention-backend", "--sglang-prefill-attention-backend", "--sglang-decode-attention-backend"]:
                self.assertEqual(argv.count(name), 1)
                self.assertEqual(argv[argv.index(name) + 1], recipe.BACKEND)
            self.assertEqual(argv[argv.index("--spec") + 1:argv.index("--spec") + 3], recipe.SPEC)
            train_env = json.loads(argv[argv.index("--train-env-vars") + 1])
            sg_env = info["sglang_worker_environment"]
            for key in ["HBY_C8V16_POLICY_SHA256", "HBY_C8V16_IMPLEMENTATION_SHA256"]:
                self.assertEqual(train_env[key], sg_env[key])
                self.assertEqual(len(train_env[key]), 64)
            self.assertIn("W16A16C8V16", info["precision"])
            self.assertEqual(info["target_new_optimizer_steps"], 100)
            self.assertEqual(info["validation_expected_global_steps"], list(range(0, 101, 5)))
            self.assertEqual(info["evaluation"]["questions"], 30)
            self.assertFalse(info["enable_thinking"])
            self.assertEqual(info["status"], "prepared")

    def test_unmatched_input_fails_before_launch(self):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "data/dapo-math-17k/train.jsonl"
            dataset.parent.mkdir(parents=True)
            dataset.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                recipe.main(["--workspace", directory, "--run"])

    def test_bf16_or_c4v8_evaluation_route_rejected(self):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.write_text("fixture\n")
            with patch.object(aime_protocol, "AIME_SHA", aime_protocol.file_sha(dataset)):
                for backend in ("ascend", "hby_c4v8_sim"):
                    with self.assertRaisesRegex(ValueError, "W16A16C8V16"):
                        aime_protocol.build_protocol(directory, dataset, {}, False,
                                                     aime_protocol.BACKEND | {"attention_backend": backend})


if __name__ == "__main__":
    unittest.main()
