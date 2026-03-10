from typing import Dict, List
import torch
from transformers import AutoTokenizer
from modeling_custom import LlamaForRewardModelWithGating

class MultiDomainRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=None, truncation=True, max_length=4096):
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = LlamaForRewardModelWithGating.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = next(self.model.parameters()).device
        self.max_length = max_length
        self.model.eval()

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}


def main() -> None:
    # Initialize the local model.
    rm = MultiDomainRMPipeline("./model/multi-domain-rm-llama-3-8b-it")

    prompt = "I just moved to Japan for work and I feel overwhelmed and lonely."

    # Response 1: high empathy and naturalness.
    response1 = "It makes complete sense that you feel this way. Relocating is a major life transition. Give yourself time to adapt."
    score1 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}])
    print(f"Response 1 (Empathetic) - Score: {score1['score']:.5f}")

    # Response 2: colder and more mechanical style.
    response2 = "To solve loneliness, join expat groups and study the language for two hours every day."
    score2 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}])
    print(f"Response 2 (Robotic) - Score: {score2['score']:.5f}")


if __name__ == "__main__":
    main()