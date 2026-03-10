import torch
from transformers import AutoTokenizer
from modeling_custom import LlamaForRewardModelWithGating

def main() -> None:
    path = "./model/multi-domain-rm-llama-3-8b-it"
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model = LlamaForRewardModelWithGating.from_pretrained(
        path,
        device_map="auto" if use_cuda else None,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model.eval()

    device = next(model.parameters()).device

    # Example prompt/response to test empathy and multicultural sensitivity.
    prompt = "I just moved to Japan for work and I feel overwhelmed and lonely. Today I accidentally offended my manager because I did not know a local custom."
    response = "I am sorry you are going through this. Feeling overwhelmed after moving to a different culture is completely normal. Etiquette mistakes happen often, especially early on. If you want, we can walk through what happened with your manager and draft a respectful way to address it."

    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(input_ids)
        # Raw rewards for each of the 23 objectives.
        multi_obj_rewards = output.rewards.cpu().float()
        # Gating output (objective importance conditioned on the prompt).
        gating_output = output.gating_output.cpu().float()
        # Final preference score.
        preference_score = output.score.cpu().float()

    obj_transform = model.reward_transform_matrix.data.cpu().float()
    multi_obj_coeffs = gating_output @ obj_transform.T

    # Ensure the decomposition is numerically consistent (allowing small mixed-precision error).
    reconstructed_score = torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1)
    if not torch.allclose(reconstructed_score, preference_score, atol=1e-2, rtol=1e-3):
        max_abs_diff = torch.max(torch.abs(reconstructed_score - preference_score)).item()
        print(f"Warning: score decomposition mismatch (max abs diff = {max_abs_diff:.6f}).")

    K = 3
    top_obj_dims = torch.argsort(torch.abs(multi_obj_coeffs), dim=1, descending=True)[:, :K]
    top_obj_coeffs = torch.gather(multi_obj_coeffs, dim=1, index=top_obj_dims)

    # The order must match Stage 1 training exactly.
    attributes = [
        'co_discourse_structure', 'co_logical_consistency', 'co_mutual_grounding',
        'co_overall_coherence_score', 'co_temporal_causal_coherence', 'co_topic_coherence',
        'cs_causality', 'cs_coherence', 'cs_consistency', 'cs_desire', 'cs_empathy', 'cs_reaction',
        'em_emotional_awareness', 'em_emotional_validation', 'em_helpful_response',
        'em_overall_empathy_score', 'em_perspective_taking', 'em_supportive_engagement',
        'mu_coherence', 'mu_cultural_specificity', 'mu_cultural_value', 'mu_empathy', 'mu_naturalness'
    ]

    print("\n--- MULTI-DOMAIN EVALUATION ---")
    print(f"Global Preference Score: {preference_score.item():.5f}")
    print(f"\nTop {K} attributes driving this decision:")

    example_index = 0
    for i in range(K):
        attribute_idx = int(top_obj_dims[example_index, i].item())
        attribute = attributes[attribute_idx]
        coeff = top_obj_coeffs[example_index, i].item()
        print(f" - {attribute}: {round(coeff, 5)}")


if __name__ == "__main__":
    main()