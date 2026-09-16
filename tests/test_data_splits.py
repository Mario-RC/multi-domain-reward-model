"""Public, synthetic checks for seed-independent validation partitions."""

import json
from pathlib import Path
import tempfile

from data_splits import frozen_group_split


def test_frozen_split_keeps_prompt_groups_together():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "split.json"
        path.write_text(json.dumps({
            "validation_group_ids": [2],
            "training_indices": [0, 3],
            "validation_indices": [1, 2],
        }))
        train, val = frozen_group_split([1, 2, 2, 3], path)
        assert train.tolist() == [0, 3]
        assert val.tolist() == [1, 2]


def test_frozen_split_rejects_missing_groups_and_changed_order():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "split.json"
        path.write_text(json.dumps({
            "validation_group_ids": [2],
            "training_indices": [0, 3],
            "validation_indices": [1, 2],
        }))
        for groups in ([1, 4, 4, 3], [1, 2, 3, 2]):
            try:
                frozen_group_split(groups, path)
            except ValueError:
                pass
            else:
                raise AssertionError("Changed source alignment was accepted.")


def test_frozen_split_rejects_empty_duplicate_and_all_validation_groups():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "split.json"
        for validation in ([], [2, 2], [1, 2, 3]):
            path.write_text(json.dumps({"validation_group_ids": validation}))
            try:
                frozen_group_split([1, 2, 2, 3], path)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid validation partition was accepted.")
