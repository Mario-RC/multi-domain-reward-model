"""Synthetic training checks that require no private experiments or datasets."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file


def _training_module():
    script = Path(__file__).resolve().parents[1] / "stage-2_train.py"
    spec = importlib.util.spec_from_file_location("stage2_input_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_selection_preserves_every_aligned_tensor():
    module = _training_module()
    indices = torch.tensor([0, 2, 4])
    tensors = (torch.arange(30).reshape(5, 2, 3), torch.arange(15).reshape(5, 3),
               torch.arange(5), torch.tensor([0, 1, 2, 3, 0]), torch.arange(10, 15))
    selected = module.select_aligned_training_rows(*tensors, indices)
    for actual, source in zip(selected, tensors):
        torch.testing.assert_close(actual, source.index_select(0, indices))


def test_row_indices_reject_invalid_alignment():
    module = _training_module()
    invalid = [
        {"source_rows": 6, "indices": [0, 2]},
        {"indices": []}, {"indices": [0, 0]}, {"indices": [2, 0]},
        {"indices": [0, 5]}, {"indices": [-1, 0]}, {"indices": [True]},
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "indices.json"
        for payload in invalid:
            path.write_text(json.dumps(payload))
            try:
                module.load_training_row_indices(path, 5)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid row selection was accepted: {payload}")
        path.write_text(json.dumps({"source_rows": 5, "indices": [0, 2, 4]}))
        indices, _ = module.load_training_row_indices(path, 5)
        assert indices.tolist() == [0, 2, 4]


def test_public_training_modes_and_partitions_cpu_smoke():
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="public-training-") as directory:
        root = Path(directory)
        embeddings = root / "embeddings" / "fixture" / "pref-train"
        embeddings.mkdir(parents=True)
        torch.manual_seed(7)
        rows = 128
        save_file({
            "embeddings": torch.randn(rows, 2, 8),
            "prompt_embeddings": torch.randn(rows, 8),
            "group_ids": torch.arange(rows) // 4,
            "domains": (torch.arange(rows) % 4).short(),
            "difficulties": (torch.arange(rows) % 3).short(),
            "format_version": torch.tensor([2]),
        }, str(embeddings / "pref-train.safetensors"))
        (root / "regression_weights").mkdir()
        torch.save({"weight": torch.randn(23, 8)},
                   root / "regression_weights" / "fixture_score_100pct.pt")
        config = root / "config.yaml"
        config.write_text("{}\n")
        split = root / "split.json"
        split.write_text(json.dumps({"validation_group_ids": list(range(8))}))
        indices = root / "indices.json"
        indices.write_text(json.dumps({"source_rows": rows, "indices": list(range(96))}))
        modes = {
            "prompt": ["--validation_group_ids_path", str(split)],
            "global": ["--train_on_all", "--training_row_indices_path", str(indices),
                       "--exclude_training_domain", "multicultural"],
            "shuffled_prompt": ["--held_out_domain", "coherence"],
            "candidate_conditioned": [],
        }
        for mode, extra in modes.items():
            command = [sys.executable, str(project / "stage-2_train.py"),
                       "--config_path", str(config), "--base_data_dir", str(root),
                       "--model_path", "fixture", "--multi_objective_dataset_name", "score",
                       "--preference_dataset_name", "pref", "--reference_dataset_name", "null",
                       "--gate_input_mode", mode, "--seed", "13", "--n_steps", "2",
                       "--eval_every", "2", "--batch_size", "16", "--logit_scale", "4",
                       "--domain_loss_weight", "0", "--hidden_size", "8", "--n_hidden", "1",
                       "--no-curriculum", "--checkpoint_tag", mode, *extra]
            before = set((root / "gating_network").glob("*.pt"))
            result = subprocess.run(command, cwd=project, capture_output=True, text=True,
                                    timeout=90, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            assert result.returncode == 0, result.stdout + result.stderr
            added = set((root / "gating_network").glob("*.pt")) - before
            assert len(added) == 1
            saved = torch.load(added.pop(), map_location="cpu", weights_only=True)
            assert saved["training_config"]["gate_input_mode"] == mode
            assert saved["training_config"]["execution_device_type"] == "cpu"
            assert saved["training_config"]["steps_completed"] == 2
            predictions = saved["validation_predictions"]
            assert torch.isfinite(predictions["margin"]).all()
            if mode == "prompt":
                assert predictions["source_indices"].tolist() == list(range(32))
                assert saved["split"]["group_overlap"] == 0
            elif mode == "global":
                assert saved["training_config"]["training_selected_rows"] == 72
                assert 3 not in predictions["domains"].tolist()
            elif mode == "shuffled_prompt":
                assert predictions["domains"].unique().tolist() == [0]
