# stage-3_package_model.py

import os
import torch
from transformers import AutoConfig, AutoTokenizer
from modeling_custom import LlamaForRewardModelWithGating

BASE_MODEL_ID = "sfairXC/FsfairX-LLaMA3-RM-v0.1"
STAGE_1_WEIGHTS_PATH = "model/regression_weights/FsfairX-LLaMA3-RM-v0.1_mdo.pt"

STAGE_2_WEIGHTS_PATH = "model/gating_network/gating_network_FsfairX-LLaMA3-RM-v0.1_mo_mdo_pref_stage_2-train_T10.0_N2000_seed0.pt"

OUTPUT_DIR = "./model/multi-domain-rm-llama-3-8b-it"


def _safe_torch_load(path: str):
    # Prefer safe weights-only loading when the installed PyTorch supports it.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_state_dict(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    raise TypeError(f"Unsupported checkpoint payload type: {type(obj)}")


def _extract_stage1_weight_tensor(obj) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "weight" in obj and isinstance(obj["weight"], torch.Tensor):
            return obj["weight"]
        if "regression_layer.weight" in obj and isinstance(obj["regression_layer.weight"], torch.Tensor):
            return obj["regression_layer.weight"]
    raise TypeError("Stage 1 checkpoint must contain a tensor under 'weight' or 'regression_layer.weight'.")


def main() -> None:
    print("Loading configuration and tokenizer...")
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    config.num_objectives = 23
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print("Instantiating custom architecture with base model weights...")
    model = LlamaForRewardModelWithGating.from_pretrained(
        BASE_MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
    )

    print(f"Loading Stage 1 regression weights from: {STAGE_1_WEIGHTS_PATH}")
    stage1_payload = _safe_torch_load(STAGE_1_WEIGHTS_PATH)
    stage1_weights = _extract_stage1_weight_tensor(stage1_payload)
    if tuple(stage1_weights.shape) != tuple(model.regression_layer.weight.shape):
        raise ValueError(
            f"Stage 1 tensor shape {tuple(stage1_weights.shape)} does not match "
            f"regression layer shape {tuple(model.regression_layer.weight.shape)}"
        )
    stage1_weights = stage1_weights.to(model.regression_layer.weight.dtype)
    model.regression_layer.weight.data.copy_(stage1_weights)

    print(f"Loading Stage 2 gating network weights from: {STAGE_2_WEIGHTS_PATH}")
    stage2_payload = _safe_torch_load(STAGE_2_WEIGHTS_PATH)
    stage2_state_dict = _resolve_state_dict(stage2_payload)
    if not isinstance(stage2_state_dict, dict):
        raise TypeError("Stage 2 checkpoint must resolve to a state_dict dictionary.")
    model.gating.load_state_dict(stage2_state_dict)

    print(f"Saving finalized model to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. The multidomain reward model is packaged at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()