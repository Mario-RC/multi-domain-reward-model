"""Shared attribute and domain definitions for the multi-domain reward model pipeline."""

ATTRIBUTES = [
    "co_discourse_structure", "co_logical_consistency", "co_mutual_grounding",
    "co_overall_coherence_score", "co_temporal_causal_coherence", "co_topic_coherence",
    "cs_causality", "cs_coherence", "cs_consistency", "cs_desire", "cs_empathy", "cs_reaction",
    "em_emotional_awareness", "em_emotional_validation", "em_helpful_response",
    "em_overall_empathy_score", "em_perspective_taking", "em_supportive_engagement",
    "mu_coherence", "mu_cultural_specificity", "mu_cultural_value", "mu_empathy", "mu_naturalness",
]

DOMAIN_PREFIXES = {
    "coherence": "co_",
    "commonsense": "cs_",
    "empathy": "em_",
    "multicultural": "mu_",
}
