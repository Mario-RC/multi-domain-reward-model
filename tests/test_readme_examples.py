"""Keep the public README examples executable without downloading model weights."""

import ast
from pathlib import Path
import re
import shlex
import subprocess
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, AutoTokenizer, LlamaConfig, PreTrainedTokenizerFast

from modeling_custom import RewardModelWithGating


ROOT = Path(__file__).resolve().parents[1]


def _blocks(language):
    return re.findall(r"```" + language + r"\n(.*?)\n```", (ROOT / "README.md").read_text(), re.S)


def test_readme_inference_preserves_tensor_inputs_masks_and_one_shared_gate():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "friend": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, pad_token_id=0)
    config.num_objectives = 4
    config.gating_hidden_dim = 4
    config.gating_n_hidden = 1
    config.gating_temperature = 2
    config.gating_logit_scale = 4
    config.shared_prompt_gating = True
    config._attn_implementation = "eager"
    model = RewardModelWithGating(config).eval()
    examples = [block for block in _blocks("python") if "chosen_score" in block]
    assert len(examples) == 1
    namespace = {}
    with patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer), \
            patch.object(AutoModel, "from_pretrained", return_value=model), \
            patch.object(model, "compute_gating", wraps=model.compute_gating) as gate_call, \
            patch.object(model, "forward", wraps=model.forward) as score_calls:
        exec(compile(examples[0], "README.md inference example", "exec"), namespace)
    assert gate_call.call_count == 1
    assert score_calls.call_count == 2
    for call in [gate_call.call_args, *score_calls.call_args_list]:
        assert isinstance(call.kwargs["input_ids"], torch.Tensor)
        assert call.kwargs["input_ids"].shape == call.kwargs["attention_mask"].shape
    for call in score_calls.call_args_list:
        assert call.kwargs["gating_output_override"] is namespace["gate"]
    assert torch.isfinite(namespace["chosen_score"]).all()
    assert torch.isfinite(namespace["rejected_score"]).all()


def test_readme_shell_examples_parse_and_only_use_supported_script_flags():
    for block in _blocks("bash"):
        subprocess.run(["bash", "-n"], input=block, text=True, check=True, capture_output=True)
        for line in block.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line, comments=True)
            if len(tokens) < 2 or tokens[0] != "python3" or not tokens[1].endswith(".py"):
                continue
            source = ast.parse((ROOT / tokens[1]).read_text())
            flags = set()
            for node in ast.walk(source):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "add_argument":
                    continue
                names = [arg.value for arg in node.args
                         if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                         and arg.value.startswith("--")]
                flags.update(names)
                if any(keyword.arg == "action" and isinstance(keyword.value, ast.Name)
                       and keyword.value.id == "BooleanOptionalAction" for keyword in node.keywords):
                    flags.update("--no-" + name[2:] for name in names)
            supplied = {token.split("=", 1)[0] for token in tokens[2:] if token.startswith("--")}
            assert supplied <= flags, (tokens[1], supplied - flags)
