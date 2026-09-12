"""Production model loads must initialize derived routing masks correctly."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModel, LlamaConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_custom import GatingNetwork, RewardModelWithGating


class InferenceMaskLoadingTests(unittest.TestCase):
    def model(self, indices=None):
        torch.set_num_threads(1)
        torch.manual_seed(107)
        config=LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0)
        config.num_objectives=4
        config.gating_hidden_dim=4
        config.gating_n_hidden=1
        config.gating_temperature=2
        config.gating_logit_scale=4
        config.gating_active_attribute_indices=indices
        config.shared_prompt_gating=True
        config._attn_implementation='eager'
        return RewardModelWithGating(config).to(dtype=torch.bfloat16).eval()

    def test_monolithic_and_sharded_low_memory_loads_restore_mask_and_exact_scores(self):
        device=os.environ.get('MASK_TEST_DEVICE','cpu')
        for indices in (None,[0,2]):
            for sharded in (False,True):
                with self.subTest(indices=indices,sharded=sharded), tempfile.TemporaryDirectory() as folder:
                    original=self.model(indices).to(device)
                    tokens=torch.tensor([[1,2,3]],device=device)
                    with torch.inference_mode():
                        gate=original.compute_gating(tokens)
                        score=original(tokens,gating_output_override=gate).score
                    original.save_pretrained(folder,max_shard_size=1024 if sharded else '1GB')
                    for repeat in range(2):
                        loaded,info=RewardModelWithGating.from_pretrained(folder,
                            local_files_only=True,dtype=torch.bfloat16,device_map={'':device},
                            low_cpu_mem_usage=True,attn_implementation='eager',output_loading_info=True)
                        self.assertFalse(any(info.values()))
                        expected=torch.tensor([True,indices is None,True,indices is None],device=device)
                        self.assertTrue(torch.equal(loaded.gating.active_attribute_mask,expected))
                        self.assertNotIn('gating.active_attribute_mask',loaded.state_dict())
                        for key,value in original.state_dict().items():
                            self.assertTrue(torch.equal(value,loaded.state_dict()[key]),key)
                        with torch.inference_mode():
                            actual_gate=loaded.compute_gating(tokens)
                            actual_score=loaded(tokens,gating_output_override=actual_gate).score
                        self.assertTrue(torch.equal(gate,actual_gate))
                        self.assertTrue(torch.equal(score,actual_score))
                        del loaded

    def test_loader_reinitializes_an_explicitly_corrupted_materialized_buffer(self):
        original_move=RewardModelWithGating._move_missing_keys_from_meta_to_device
        def corrupt_after_materialization(model,*args,**kwargs):
            original_move(model,*args,**kwargs)
            model.gating.active_attribute_mask.fill_(False)
        with tempfile.TemporaryDirectory() as folder:
            original=self.model([0,2])
            original.save_pretrained(folder)
            with patch.object(RewardModelWithGating,'_move_missing_keys_from_meta_to_device',corrupt_after_materialization):
                loaded=RewardModelWithGating.from_pretrained(folder,local_files_only=True,
                    dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='eager')
            self.assertEqual(loaded.gating.active_attribute_mask.tolist(),[True,False,True,False])
            for key,value in original.state_dict().items():
                self.assertTrue(torch.equal(value,loaded.state_dict()[key]),key)

    def test_auto_model_remote_code_path_uses_the_fix(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            original=self.model([0,2])
            original.config.auto_map={'AutoModel':'modeling_custom.RewardModelWithGating'}
            original.save_pretrained(folder)
            root=Path(__file__).resolve().parents[1]
            for name in ('modeling_custom.py','utils.py'):
                shutil.copy2(root/name,Path(folder)/name)
            loaded=AutoModel.from_pretrained(folder,trust_remote_code=True,local_files_only=True,
                dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='eager')
            self.assertEqual(loaded.gating.active_attribute_mask.tolist(),[True,False,True,False])
            for key,value in original.state_dict().items():
                self.assertTrue(torch.equal(value,loaded.state_dict()[key]),key)

    def test_invalid_attribute_lists_fail_before_loading(self):
        for indices in ([],[-1],[4],[True],[0,0],'all'):
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                self.model(indices)


if __name__=='__main__':
    unittest.main()
