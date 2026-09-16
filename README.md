# Multi-Domain Reward Model

This directory contains a multi-objective reward-model pipeline that evaluates responses across four complementary domains: **Coherence**, **Commonsense**, **Empathy**, and **Multicultural understanding**. Each model predicts 23 fine-grained attributes and combines them through a prompt-conditioned gating network to produce one preference score.

The current architecture uses a **shared prompt gate**: the gate is computed from the prompt alone and reused when scoring both the chosen and rejected responses. This keeps routing independent of candidate-specific content and makes pairwise comparisons consistent.

## Project Goal

The goal is to train a reward model in three stages:

- **Stage 1 — Multi-objective regression:** extract response representations and fit one regression head over the 23 attributes.
- **Stage 2 — Shared gating network:** learn how to combine the attributes for each prompt using preference pairs.
- **Stage 3 — Packaging:** combine the base reward model, Stage 1 regression weights, and Stage 2 gate into a Transformers-compatible model.

---

## Data Source

The multi-domain data (`Multi-Domain-Data-Scoring.jsonl` and `Multi-Domain-Data-Preference-Pairs.jsonl`) come from:

- [mestecha/multidomain_data_scoring](https://github.com/mestecha/multidomain_data_scoring)

### Datasets used

- **Multi-objective data:** [`Multi-Domain-Data-Scoring`](https://github.com/mestecha/multidomain_data_scoring/tree/main)
- **Preference data:** [`Multi-Domain-Data-Preference-Pairs`](https://github.com/mestecha/multidomain_data_scoring/tree/main)
- **Optional reference data:** [`RLHFlow/UltraFeedback-preference-standard`](https://huggingface.co/datasets/RLHFlow/UltraFeedback-preference-standard)
- **Optional evaluation data:** [`allenai/reward-bench`](https://huggingface.co/datasets/allenai/reward-bench)

Raw datasets, generated embeddings, checkpoints, and evaluation outputs are intentionally excluded from version control.

---

## Base Models

The pipeline supports the following base reward models:

- **FsfairX Llama 3:** [`sfairXC/FsfairX-LLaMA3-RM-v0.1`](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)
- **FsfairX Gemma 2:** [`sfairXC/FsfairX-Gemma2-RM-v0.1`](https://huggingface.co/sfairXC/FsfairX-Gemma2-RM-v0.1)
- **Qwen 3 Nemotron:** [`nvidia/Qwen3-Nemotron-8B-BRRM`](https://huggingface.co/nvidia/Qwen3-Nemotron-8B-BRRM)
- **Mistral:** [`weqweasdas/RM-Mistral-7B`](https://huggingface.co/weqweasdas/RM-Mistral-7B)
- **Skywork Llama 3.1:** [`Skywork/Skywork-Reward-V2-Llama-3.1-8B`](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)
- **Skywork Qwen 3:** [`Skywork/Skywork-Reward-V2-Qwen3-8B`](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)

The architecture family passed to the scripts is one of `llama3`, `gemma2`, `qwen3`, `mistral`, or `auto`.

---

## Working Attributes

The project uses **23 custom attributes** defined in `attributes.py`, which is the single source of truth imported by training, evaluation, and inference scripts.

### Coherence (`co_`)

- `co_discourse_structure`
- `co_logical_consistency`
- `co_mutual_grounding`
- `co_overall_coherence_score`
- `co_temporal_causal_coherence`
- `co_topic_coherence`

### Commonsense (`cs_`)

- `cs_causality`
- `cs_coherence`
- `cs_consistency`
- `cs_desire`
- `cs_empathy`
- `cs_reaction`

### Empathy (`em_`)

- `em_emotional_awareness`
- `em_emotional_validation`
- `em_helpful_response`
- `em_overall_empathy_score`
- `em_perspective_taking`
- `em_supportive_engagement`

### Multicultural (`mu_`)

- `mu_coherence`
- `mu_cultural_specificity`
- `mu_cultural_value`
- `mu_empathy`
- `mu_naturalness`

These attributes are always the 23 regression targets in Stage 1. Stage 2 can mask reversible subsets for controlled experiments without changing the Stage 1 representation.

---

## Installation

The project requires **Python 3.12**.

```bash
python -m pip install -r requirements.txt
```

For development and regression tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Installing `flash-attn` is optional and can accelerate supported attention implementations.

---

## Quickstart Execution Flow

Run the individual stages below to use the common full-refit training recipe. The examples use FsfairX Llama 3; the same Stage 2 settings apply to all six backbones. Backbone representations are frozen: Stage 1 fits attribute regressors and Stage 2 trains the gate, rather than fine-tuning the entire backbone.

The repository also includes a configurable baseline launcher:

```bash
./mdorm.sh
```

`mdorm.sh` and the shipped `config.yaml` contain baseline settings, not the full-refit recipe below. To use that recipe, run the explicit commands below or first transfer their settings into your configuration and launcher. CLI arguments override values read from the configuration file.

Local output names are your choice. The packaging and inference examples use `my-reward-model`; replace it with your own name. No campaign-specific directory is required.

### Stage 1: prepare multi-objective representations

```bash
python3 stage-1_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/dataset/Multi-Domain-Data-Scoring \
  --output_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train \
  --n_shards 1 \
  --shard_idx 1 \
  --device 0
```

Important options:

- `--dataset_path` accepts one or more local JSON/JSONL paths; the extension is optional.
- `--n_shards` and `--shard_idx` allow representation extraction to be distributed across GPUs.
- `--max_seq_len` overrides the base model's maximum position length when truncation is required.
- `--model_key` selects an entry from the six-backbone `model_registry` in `config.yaml`.

New scoring embeddings also store the domain and a stable conversation group ID alongside the attribute labels. Stage 1 training uses those group IDs when constructing its internal validation split.

### Stage 1: train the attribute regressors

```bash
python3 stage-1_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train \
  --split_seed 42 \
  --val_size 0.2
```

Stage 1 creates two regression-weight files:

- `_80pct.pt` contains the validation-best Ridge regressors fitted on the internal training portion. The default validation fraction is 0.2; the filename remains `_80pct.pt` if you change `--val_size`.
- `_100pct.pt` refits the selected regularization values on all available Stage 1 rows and is the default for later stages.

Stage 2 and Stage 3 automatically resolve the `_100pct.pt` file unless `--stage_1_weights_path` is supplied explicitly.

`--split_seed` controls the internal partition and `--val_size` controls its validation fraction. When complete group metadata is available, records with the same group ID stay in the same partition.

### Stage 2: prepare preference representations

```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/dataset/Multi-Domain-Data-Preference-Pairs \
  --output_dataset_name Multi-Domain-Data-Preference-Pairs-SharedGate \
  --dataset_split train \
  --prompt_batch_size 8 \
  --prompt_pooling generation_boundary \
  --n_shards 1 \
  --shard_idx 1 \
  --device 0
```

This stage stores:

- one prompt-only representation per preference pair;
- separate final-response representations for the chosen and rejected candidates;
- the pair domain plus stable prompt and pair identifiers. Prompt IDs drive grouped splitting; pair IDs support traceable downstream evaluation.

The prompt-only representation is the input to the shared gate. `--prompt_pooling generation_boundary` uses the last prompt token after adding the assistant generation boundary and matches the packaged shared-gate inference path. `mean_prompt` and `last_message_boundary` are additional preparation options; they require matching inference handling and are not interchangeable with the default packaged prompt gate. The chosen pooling mode is recorded in the embedding file.

Use `--reuse_candidate_embeddings_path` to reuse aligned candidate representations from the same, unmodified source data. `--selection_manifest` selects pairs without rewriting the source dataset.

### Optional Stage 2 preparation: reference data

```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path RLHFlow/UltraFeedback-preference-standard \
  --output_dataset_name UltraFeedback-preference-standard \
  --dataset_split train \
  --prompt_batch_size 8 \
  --n_shards 1 \
  --shard_idx 1 \
  --device 0
```

Reference representations are only required when Stage 2 debiasing is enabled.

### Optional Stage 2 preparation: RewardBench

```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path allenai/reward-bench \
  --output_dataset_name reward-bench \
  --dataset_split filtered \
  --prompt_batch_size 8 \
  --n_shards 1 \
  --shard_idx 1 \
  --device 0
```

### Stage 2: train the shared gate

This is the common final Stage 2 recipe: all 23 attributes, generation-boundary prompt embeddings, and all rows of the source-training split. Internal development rows may be included in this full refit, but test rows must not be added. Use CUDA with bfloat16 support to match the training execution setting.

```bash
python3 stage-2_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs-SharedGate \
  --reference_dataset_name null \
  --debiasing_dims -1 \
  --temperature 2.0 \
  --n_steps 2000 \
  --seed 20260908 \
  --n_hidden 1 \
  --hidden_size 64 \
  --learning_rate 0.0005 \
  --weight_decay 0.0 \
  --dropout 0.05 \
  --batch_size 2048 \
  --logit_scale 4.0 \
  --no-learnable_logit_scale \
  --gate_input_mode prompt \
  --attribute_subset full \
  --exclude_attributes \
  --domain_loss_weight 0.0 \
  --entropy_weight 0.02 \
  --entropy_floor_fraction 0.35 \
  --load_balance_weight 0.02 \
  --balance_domains \
  --no-balance_difficulties \
  --no-curriculum \
  --train_on_all \
  --eval_every 2000 \
  --dataset_split train \
  --device 0
```

This command saves the gate after exactly 2,000 updates. In `--train_on_all` mode, there is no validation-based checkpoint selection or early stopping: the saved diagnostics describe training data, not held-out performance. Keep the exact checkpoint path printed at the end for Stage 3. Matching this recipe does not guarantee bit-identical weights across different data, preprocessing, hardware, or library versions.

The shared-prompt training path has several important properties:

- Chosen and rejected candidates always use the same prompt-derived gate.
- In development mode (`--no-train_on_all`), train/validation partitions are grouped by prompt ID to keep the same stored ID out of both partitions.
- `--entropy_weight` and `--entropy_floor_fraction` control per-example routing concentration.
- `--load_balance_weight` controls batch-level use of the available attributes.
- `--balance_domains` samples training examples uniformly across non-empty domains.
- `--balance_difficulties` optionally balances non-empty domain-by-difficulty cells.
- `--train_on_all` refits a selected configuration on all training groups for a fixed number of steps.
- `--checkpoint_tag` gives experiment checkpoints an explicit, filesystem-safe identifier.

`--attribute_subset` and `--exclude_attributes` provide reversible masks over the 23 attributes. They do not retrain or alter the Stage 1 regressors.

#### Optional debiasing

Set `--debiasing_dims -1` to disable debiasing. When one or more non-negative indices are supplied, the reference dataset is used to construct a `reward_transform_matrix` that reduces correlations between selected dimensions and the remaining attributes. `--corr_threshold` controls the maximum target correlation. Signed penalties are considered, and training fails rather than saving a partial transform if the requested threshold cannot be reached.

The reference dataset is not loaded when debiasing is disabled, even if a reference dataset name is present in the configuration.

### Stage 3: package the final model

Pass the matching Stage 1 and Stage 2 checkpoints explicitly. Set `STAGE2_CHECKPOINT` to the actual path printed by training, and choose any unused output name:

```bash
MODEL_NAME="my-reward-model"
STAGE2_CHECKPOINT="model/gating_network/your-selected-checkpoint.pt"

python3 stage-3_package_model.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --stage_1_weights_path model/regression_weights/FsfairX-LLaMA3-RM-v0.1_Multi-Domain-Data-Scoring_100pct.pt \
  --stage_2_weights_path "${STAGE2_CHECKPOINT}" \
  --output_model_name "${MODEL_NAME}" \
  --output_dir "model/${MODEL_NAME}"
```

`--output_dir` selects the exact destination. Without it, the destination is `--model_parent_dir` plus `--output_model_name`. Neither the directory name nor its parent needs to match a published model name.

A packaged directory contains weights, tokenizer files, configuration, custom modeling code, and training metadata. A weight index is included when weights are sharded, and tokenizer-specific files depend on the base model. The configuration declares:

- `RewardModelWithGating` as its architecture;
- `modeling_custom.RewardModelWithGating` in `auto_map`;
- `shared_prompt_gating=true`;
- `num_objectives=23`.

Allow enough disk space for a complete package. Existing output or staging directories are not overwritten; inspect any leftover `.incomplete` directory before retrying.

---

## Evaluation and Analysis

### Evaluate a packaged model

```bash
MODEL_NAME="my-reward-model"

python3 evaluate.py \
  --model_path "model/${MODEL_NAME}" \
  --scoring_data_path data/dataset/Multi-Domain-Data-Scoring \
  --preference_data_path data/dataset/Multi-Domain-Data-Preference-Pairs
```

The evaluator reports scoring metrics, overall preference accuracy, domain-level results, gate diagnostics, and difficulty slices. By default it also saves per-pair margins that can be used for paired statistical tests. Use `--skip_pair_predictions` when those records are not required.

Results are saved under `<package-directory>/results/`. `--output_json` additionally saves the combined report to your chosen path.

Optional switches include `--skip_scoring`, `--skip_preference`, `--max_samples`, `--max_length`, `--eval`, and `--output_json`.

### Run a quick prediction comparison

```bash
MODEL_NAME="my-reward-model"

python3 predict.py \
  --model_path "model/${MODEL_NAME}"
```

`predict.py` renders one prompt and two candidate conversations, computes the prompt gate once, and reuses it for both scores.

### Analyze attribute correlations

```bash
python3 analyze_correlations.py \
  --dataset_path data/dataset/Multi-Domain-Data-Scoring.jsonl \
  --split train \
  --threshold 0.3
```

The report includes:

- per-attribute range, variance, and low-variance flags;
- correlations between attributes and response length;
- within-domain pairwise Spearman correlations;
- correlation matrices and high-correlation markers;
- PCA-based dimensionality summaries;
- attribute dominance and debiasing suggestions.

### Evaluate a base-model baseline

Scalar reward model:

```bash
python3 evaluate_baseline.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --scoring_data_path data/dataset/Multi-Domain-Data-Scoring \
  --preference_data_path data/dataset/Multi-Domain-Data-Preference-Pairs \
  --model_name my-scalar-baseline
```

Generative-judge mode for Qwen 3 Nemotron:

```bash
python3 evaluate_baseline.py \
  --model_path nvidia/Qwen3-Nemotron-8B-BRRM \
  --preference_data_path data/dataset/Multi-Domain-Data-Preference-Pairs \
  --generative_judge \
  --skip_scoring \
  --model_name my-generative-baseline
```

### Compare packaged models

Use the directory names you chose when packaging. For example, after evaluating two models named `my-reward-model` and `another-reward-model`:

```bash
python3 compare_models.py \
  --model_parent_dir model \
  --no_baselines \
  --models my-reward-model another-reward-model
```

The comparison utility produces side-by-side tables, CSV files, and optional plots for all discovered evaluation results.

---

## Configuration-Driven Flow

Every main script accepts `--config_path config.yaml`. The configuration contains a six-backbone `model_registry` plus flat sections for Stage 1 preparation/training, Stage 2 preparation/training, Stage 3 packaging, inference, correlations, model comparison, and baseline evaluation.

For a configuration-driven run, first set the desired recipe in your YAML. The following commands use the file's values; they do not automatically inherit the explicit full-refit settings shown above:

```bash
python3 stage-1_prepare.py --config_path config.yaml --model_key fsfair_gemma2
python3 stage-1_train.py --config_path config.yaml --model_key fsfair_gemma2
python3 stage-2_prepare.py --config_path config.yaml --model_key fsfair_gemma2
python3 stage-2_train.py --config_path config.yaml --model_key fsfair_gemma2
python3 stage-3_package_model.py --config_path config.yaml --model_key fsfair_gemma2 --output_model_name my-reward-model --output_dir model/my-reward-model
python3 analyze_correlations.py --config_path config.yaml
python3 compare_models.py --config_path config.yaml
```

CLI values explicitly supplied by the user take precedence over their `config.yaml` counterparts. Use the same `--model_key` in each training stage; the registry resolves the base path and architecture family and supplies a default package name. Override that name with `--output_model_name`, and use `--output_dir` for an explicit destination. Stage 3 naming parameters must match Stage 2, unless you pass the checkpoint paths explicitly.

---

## Released Hugging Face Models

| Model | Base reward model | Test accuracy (%) | Scoring Spearman |
| :--- | :--- | :---: | :---: |
| [**`multi-domain-rm-fsfairx-gemma-2-9b-it`**](https://huggingface.co/mario-rc/multi-domain-rm-fsfairx-gemma-2-9b-it) | [sfairXC/FsfairX-Gemma2-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-Gemma2-RM-v0.1) | **88.01** | 0.7346 |
| [**`multi-domain-rm-skywork-qwen-3-8b-it`**](https://huggingface.co/mario-rc/multi-domain-rm-skywork-qwen-3-8b-it) | [Skywork/Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B) | **87.82** | 0.7156 |
| [**`multi-domain-rm-fsfairx-llama-3-8b-it`**](https://huggingface.co/mario-rc/multi-domain-rm-fsfairx-llama-3-8b-it) | [sfairXC/FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) | **86.86** | 0.7108 |
| [**`multi-domain-rm-skywork-llama-3.1-8b-it`**](https://huggingface.co/mario-rc/multi-domain-rm-skywork-llama-3.1-8b-it) | [Skywork/Skywork-Reward-V2-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B) | **86.82** | 0.7264 |
| [**`multi-domain-rm-mistral-7b-it`**](https://huggingface.co/mario-rc/multi-domain-rm-mistral-7b-it) | [weqweasdas/RM-Mistral-7B](https://huggingface.co/weqweasdas/RM-Mistral-7B) | **84.41** | 0.6710 |
| [**`multi-domain-rm-qwen-3-nemotron-8b-it`**](https://huggingface.co/mario-rc/multi-domain-rm-qwen-3-nemotron-8b-it) | [nvidia/Qwen3-Nemotron-8B-BRRM](https://huggingface.co/nvidia/Qwen3-Nemotron-8B-BRRM) | **83.65** | 0.6704 |

---

## Loading a Released Model

Released packages use custom Transformers code, so `trust_remote_code=True` is required. For a preference pair, render the prompt separately, compute its gate once, and pass the same tensor as `gating_output_override` for both candidates.

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "mario-rc/multi-domain-rm-fsfairx-gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()

prompt = [{"role": "user", "content": "How can I support a friend who feels excluded?"}]
chosen = prompt + [{
    "role": "assistant",
    "content": "Listen without judging, validate how they feel, and ask what support would help.",
}]
rejected = prompt + [{"role": "assistant", "content": "Tell them to ignore it."}]

prompt_inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)
chosen_inputs = tokenizer.apply_chat_template(
    chosen,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
    return_dict=True,
).to(model.device)
rejected_inputs = tokenizer.apply_chat_template(
    rejected,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.inference_mode():
    gate = model.compute_gating(
        input_ids=prompt_inputs["input_ids"],
        attention_mask=prompt_inputs["attention_mask"],
    )
    chosen_score = model(
        input_ids=chosen_inputs["input_ids"],
        attention_mask=chosen_inputs["attention_mask"],
        gating_output_override=gate,
    ).score
    rejected_score = model(
        input_ids=rejected_inputs["input_ids"],
        attention_mask=rejected_inputs["attention_mask"],
        gating_output_override=gate,
    ).score

print({"chosen": chosen_score.item(), "rejected": rejected_score.item()})
```

`return_dict=True` makes the tokenizer output explicit: pass its `input_ids` tensor and `attention_mask` to the model, not the whole dictionary as `input_ids`. This also preserves padding masks for batched inputs. Always supply `gating_output_override` for shared-gate scoring, using the pattern above or the helpers in `utils.py`; candidate-conditioned routing would change this scoring protocol. Scores are intended for comparison within a prompt; they are not calibrated probabilities or universal utility values.

For a local package, set `model_id = "model/my-reward-model"` or use your own package path. The same loading and scoring pattern applies.

---

## Model Directory Tree

```text
model/
├── embeddings/
│   └── <model_name>/
│       ├── <multi_objective_dataset_name>-<split>/
│       │   └── <multi_objective_dataset_name>-<split>.safetensors
│       ├── <preference_dataset_name>-<split>/
│       │   └── <preference_dataset_name>-<split>.safetensors
│       ├── <reference_dataset_name>-<split>/
│       │   └── <reference_dataset_name>-<split>.safetensors
│       └── reward-bench-filtered/
│           └── reward-bench-filtered.safetensors
│
├── regression_weights/
│   ├── <model_name>_<multi_objective_dataset_name>_80pct.pt
│   └── <model_name>_<multi_objective_dataset_name>_100pct.pt
│
├── gating_network/
│   └── <checkpoint-filename>.pt
│
└── <your-model-name>/
    ├── chat_template.jinja           # when supplied by the tokenizer
    ├── config.json
    ├── model*.safetensors
    ├── model.safetensors.index.json  # for sharded weights
    ├── modeling_custom.py
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── training_metadata.json
    └── utils.py
```

### Artifact paths

- Stage 1/2 representations: `model/embeddings/<model_name>/<dataset_name>-<split>/*.safetensors`
- Stage 1 regressors: `model/regression_weights/<model_name>_<dataset_name>_{80pct,100pct}.pt`
- Stage 2 gates: `model/gating_network/<checkpoint-filename>.pt`
- Packaged models: `model/<your-model-name>/`, or any explicit `--output_dir`
- Evaluation results: `<package-directory>/results/`; `--output_json` additionally saves the combined report to your chosen path

Generated artifacts can be large and are ignored by Git.

---

## Repository Structure

```text
multidomain_model/
├── attributes.py                # canonical 23-attribute definition
├── config.yaml                  # configuration for all pipeline stages
├── data/                        # dataset loaders and small templates
├── data_splits.py                # frozen validation group partitions
├── mdorm.sh                     # baseline end-to-end entry point
├── modeling_custom.py           # reward model with shared prompt gate
├── requirements.txt             # runtime/training dependencies
├── requirements-dev.txt         # development and test dependencies
├── stage-1_prepare.py           # Stage 1 representation extraction
├── stage-1_train.py             # multi-objective Ridge regression
├── stage-2_prepare.py           # prompt and candidate representation extraction
├── stage-2_train.py             # grouped shared-gate training
├── stage-3_package_model.py     # Transformers package creation
├── evaluate.py                  # packaged-model evaluation
├── evaluate_baseline.py         # base-model evaluation
├── predict.py                   # quick pairwise inference
├── compare_models.py            # result aggregation and comparison
├── analyze_correlations.py      # attribute-correlation analysis
├── tests/                       # focused regression tests
└── utils.py                     # shared data and scoring utilities
```

Generated experiment outputs, logs, intermediate checkpoints, and internal research documentation are kept locally and excluded from the public repository.

---

## Credits

- **Reward-modeling foundation:** [ArmoRM / RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling)
- **Multi-domain data:** [`mestecha/multidomain_data_scoring`](https://github.com/mestecha/multidomain_data_scoring)

## License

The project code is released under the [Apache License 2.0](LICENSE). Released checkpoints are also subject to the licenses and usage conditions of their respective base models and training datasets.
