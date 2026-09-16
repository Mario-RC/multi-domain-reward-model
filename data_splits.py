"""Seed-independent prompt-group partitions defined by a frozen manifest."""

import json
from pathlib import Path

import numpy as np


def frozen_group_split(group_ids, manifest_path):
    """Restore a fixed split and reject missing groups or changed row ordering."""
    groups = np.asarray(group_ids, dtype=np.int64)
    manifest = json.loads(Path(manifest_path).read_text())
    validation = manifest.get("validation_group_ids", [])
    if not validation or len(set(validation)) != len(validation):
        raise ValueError("Validation groups must be nonempty and unique.")
    if not set(validation) <= set(groups.tolist()):
        raise ValueError("Frozen validation groups are missing from the source.")
    mask = np.isin(groups, validation)
    train, val = np.flatnonzero(~mask), np.flatnonzero(mask)
    if not len(train) or not len(val):
        raise ValueError("Frozen split requires nonempty training and validation.")
    for key, indices in (("training_indices", train), ("validation_indices", val)):
        if key in manifest and manifest[key] != indices.tolist():
            raise ValueError(f"Frozen {key} do not match the current source order.")
    return train, val
