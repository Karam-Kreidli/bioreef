"""
Hierarchical Distance (HD) — an ordinal ERROR-SEVERITY score for a prediction,
graded by how far the mistake is in the Linnaean hierarchy (Family/Genus/Species).

    Same species   -> 0      Same family    -> 2
    Same genus     -> 1      Different fam. -> 3

NOTE: these are ORDINAL SEVERITY levels (0/1/2/3), not a literal count of tree
edges (a true species->genus->family graph-hop distance would give different
numbers). "Hierarchical distance" here means "how severe is the taxonomic error",
defined exactly as the table above. This is deliberately distinct from MATANet's
own training tree-distance matrix (0/2/4/6) — both models are SCORED by this same
severity score at evaluation, but it is not the same objective they train on.

This is the per-prediction penalty; eval.metrics aggregates it (mean HD, mistake
severity, per-level accuracy). Define the rule once here so every metric uses the
same severity scale (paper 5.1/D).
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("bioreef.eval.hd")

# Taxonomic level penalties.
DEFAULT_LEVEL_WEIGHTS = {
    "species": 0,  # correct — no penalty
    "genus": 1,    # within-genus error — minor
    "family": 2,   # within-family error — moderate
    "root": 3,     # cross-family error — major
}


def hierarchical_distance(
    predicted_species: str,
    true_species: str,
    taxonomy_tree: Dict[str, Dict[str, str]],
    level_weights: Optional[Dict[str, int]] = None,
) -> Tuple[float, str]:
    """HD penalty + deepest matching level. Species missing from the tree get the
    maximum (root) penalty with level 'unknown'."""
    weights = level_weights or DEFAULT_LEVEL_WEIGHTS

    if predicted_species == true_species:
        return 0.0, "species"

    pred_tax = taxonomy_tree.get(predicted_species)
    true_tax = taxonomy_tree.get(true_species)
    if pred_tax is None or true_tax is None:
        return float(weights["root"]), "unknown"

    if pred_tax["genus"] == true_tax["genus"]:
        return float(weights["genus"]), "genus"
    if pred_tax["family"] == true_tax["family"]:
        return float(weights["family"]), "family"
    return float(weights["root"]), "root"
