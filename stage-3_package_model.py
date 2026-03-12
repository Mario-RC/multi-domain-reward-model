# stage-3_package_model.py

import os
import torch
from argparse import ArgumentParser
from transformers import AutoConfig, AutoTokenizer
from modeling_custom import RewardModelWithGating
from config_utils import load_yaml_config, resolve_model_from_config

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


def _build_defaults_from_config(config: dict, model_path: str):
    model_name = model_path.split("/")[-1]
    stage2_cfg = config.get("stage_2_train", {}) if isinstance(config, dict) else {}
    stage3_cfg = config.get("stage_3_package", {}) if isinstance(config, dict) else {}
    multi_objective_dataset_name = str(stage2_cfg.get("multi_objective_dataset", "mdo")).split("/")[-1]
    preference_dataset_name = str(stage2_cfg.get("preference_dataset", "data/stage_2")).split("/")[-1]
    prepared_split = str(stage2_cfg.get("prepared_split", "all"))

    stage1_weights_path = os.path.join(
        "model", "regression_weights", f"{model_name}_{multi_objective_dataset_name}.pt"
    )
    stage2_weights_path = os.path.join(
        "model",
        "gating_network",
        (
            f"gating_network_{model_name}_mo_{multi_objective_dataset_name}_"
            f"pref_{preference_dataset_name}-{prepared_split}_T10.0_N2000_seed0.pt"
        ),
    )
    model_parent_dir = str(stage3_cfg.get("model_parent_dir", stage3_cfg.get("output_parent_dir", "model")))
    final_model_name = str(
        stage3_cfg.get(
            "output_model_name",
            stage3_cfg.get("final_model_name", f"multi-domain-rm-{model_name.lower()}"),
        )
    )
    output_dir = os.path.join(model_parent_dir, final_model_name)
    return stage1_weights_path, stage2_weights_path, output_dir


def main() -> None:
    parser = ArgumentParser(description="Stage 3: package final reward model.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model:registry.")
    parser.add_argument("--model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1", help="Base model HF ID/path.")
    parser.add_argument("--stage_1_weights_path", type=str, default=None, help="Optional override for Stage 1 regression weights path.")
    parser.add_argument("--stage_2_weights_path", type=str, default=None, help="Optional override for Stage 2 gating network weights path.")
    parser.add_argument("--model_parent_dir", type=str, default=None, help="Optional output parent directory (e.g., model).")
    parser.add_argument("--output_model_name", type=str, default=None, help="Optional packaged model directory name.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for final packaged model output directory.")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    args = resolve_model_from_config(args, config, needs_family=False)

    inferred_stage1, inferred_stage2, inferred_output = _build_defaults_from_config(config, args.model_path)
    stage_1_weights_path = args.stage_1_weights_path or inferred_stage1
    stage_2_weights_path = args.stage_2_weights_path or inferred_stage2
    inferred_parent = os.path.dirname(inferred_output)
    inferred_name = os.path.basename(inferred_output)
    model_parent_dir = args.model_parent_dir or inferred_parent
    output_model_name = args.output_model_name or inferred_name
    output_dir = args.output_dir or os.path.join(model_parent_dir, output_model_name)

    print("Loading configuration and tokenizer...")
    model_config = AutoConfig.from_pretrained(args.model_path)
    model_config.num_objectives = 23
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Instantiating custom architecture with base model weights...")
    model = RewardModelWithGating.from_pretrained(
        args.model_path,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    print(f"Loading Stage 1 regression weights from: {stage_1_weights_path}")
    stage1_payload = _safe_torch_load(stage_1_weights_path)
    stage1_weights = _extract_stage1_weight_tensor(stage1_payload)
    if tuple(stage1_weights.shape) != tuple(model.regression_layer.weight.shape):
        raise ValueError(
            f"Stage 1 tensor shape {tuple(stage1_weights.shape)} does not match "
            f"regression layer shape {tuple(model.regression_layer.weight.shape)}"
        )
    stage1_weights = stage1_weights.to(model.regression_layer.weight.dtype)
    model.regression_layer.weight.data.copy_(stage1_weights)

    print(f"Loading Stage 2 gating network weights from: {stage_2_weights_path}")
    stage2_payload = _safe_torch_load(stage_2_weights_path)
    stage2_state_dict = _resolve_state_dict(stage2_payload)
    if not isinstance(stage2_state_dict, dict):
        raise TypeError("Stage 2 checkpoint must resolve to a state_dict dictionary.")
    model.gating.load_state_dict(stage2_state_dict)

    print(f"Saving finalized model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. The multidomain reward model is packaged at: {output_dir}")


if __name__ == "__main__":
    main()