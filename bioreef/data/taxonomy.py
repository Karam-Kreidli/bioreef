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
    """{binomial: {'genus', 'family', 'species'}} from the metadata CSV.

    Keyed by the full binomial ('Genus epithet') so it joins to the class labels
    produced by split.py (which key classes on the binomial, not the bare
    epithet). HD and the HSLM maps look species up by this key, so the two must
    agree; keying on the epithet alone would break the join for any epithet that
    appears under more than one genus."""
    import pandas as pd
    from bioreef.data.split import binomial
    # Deliberately NOT wrapped in try/except. An unreadable taxonomy CSV used to
    # return {}, which sends every species to __unknown_genus__/__unknown_family__:
    # the genus and family terms of the HSLM loss then become constants, training
    # runs to completion looking healthy, and the whole campaign is invalid. A
    # missing/corrupt taxonomy must stop the run, not degrade it silently.
    from bioreef.data.split import canonical_genus, canonical_family
    df = pd.read_csv(csv_path)
    tree = {}
    for _, row in df.dropna(subset=["species", "genus", "family"]).iterrows():
        name = binomial(row["genus"], row["species"])
        # Canonicalize the stored parent with the SAME functions that built the
        # key. binomial() already canonicalizes the genus, so a raw parent here
        # (e.g. 'Epinephalis' vs the corrected 'Epinephelus') would create a
        # phantom genus node; canonical_family() fixes known-wrong families
        # (e.g. Epinephelus faveatus mislabelled Percichthyidae).
        entry = {"genus": canonical_genus(row["genus"]),
                 "family": canonical_family(name, row["family"]), "species": name}
        # A binomial mapping to two different genera/families means the metadata
        # is inconsistent; silently keeping the last row would hide that.
        if name in tree and tree[name] != entry:
            raise ValueError(f"conflicting taxonomy for '{name}': "
                             f"{tree[name]} vs {entry}. Fix the metadata CSV.")
        tree[name] = entry
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
