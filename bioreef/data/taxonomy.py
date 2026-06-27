"""
Taxonomy helpers: placeholder-species filtering and the family/genus/species
maps used by the HSLM marginalization loss and the HD metric.

The benchmark threshold (>=20 samples/species) and placeholder filter live in
split.py; this module only builds the taxonomy structures from the metadata CSV.
"""

import logging
import re as _re

logger = logging.getLogger("bioreef.data.taxonomy")

_PLACEHOLDER_SPECIES = {
    "unidentified", "fish", "unknown", "unidentifiable", "other", "spp",
}
_SP_PATTERN = _re.compile(r"^sp\d+$", _re.IGNORECASE)


def is_placeholder_species(name) -> bool:
    """True if the species label is a placeholder (sp1, sp3, unidentified, ...)."""
    if not isinstance(name, str):
        return True
    s = name.strip().lower()
    return s in _PLACEHOLDER_SPECIES or bool(_SP_PATTERN.match(s))


def get_taxonomy_tree(csv_path: str) -> dict:
    """{species: {'genus', 'family', 'species'}} from the metadata CSV."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    tree = {}
    for _, row in df.dropna(subset=["species", "genus", "family"]).iterrows():
        tree[row["species"]] = {
            "genus": row["genus"], "family": row["family"], "species": row["species"]
        }
    return tree


def build_taxonomy_maps(idx_to_sp, taxonomy_tree):
    """species-idx -> genus-idx / family-idx maps for HSLMLoss. Returns
    (species_to_genus, species_to_family, num_genera, num_families, n_missing);
    species absent from the taxonomy go to shared "__unknown__" buckets (counted
    in n_missing) so training never crashes."""
    num_species = len(idx_to_sp)
    genus_names, family_names = [], []
    n_missing = 0
    for i in range(num_species):
        tax = taxonomy_tree.get(idx_to_sp[i])
        if tax is None:
            genus_names.append("__unknown_genus__")
            family_names.append("__unknown_family__")
            n_missing += 1
        else:
            genus_names.append(tax["genus"])
            family_names.append(tax["family"])

    genus_to_idx = {g: i for i, g in enumerate(sorted(set(genus_names)))}
    family_to_idx = {f: i for i, f in enumerate(sorted(set(family_names)))}
    species_to_genus = [genus_to_idx[g] for g in genus_names]
    species_to_family = [family_to_idx[f] for f in family_names]
    return (species_to_genus, species_to_family,
            len(genus_to_idx), len(family_to_idx), n_missing)
