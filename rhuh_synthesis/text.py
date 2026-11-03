"""Deterministic image/mask prompt templates for EXP-2026-001.

Prompts follow the released MedSegFactory loader: separate image and mask
prompts ``a photo of {organ} image, with {kind}.`` / ``... label, with ...``,
where ``kind`` joins the present class names with "," in an order that the
reference shuffles every epoch. To keep that variation with cached embeddings,
every valid class order of a slice is returned. An unannotated slice uses an
explicit "no annotated compartment" kind; it is not described as normal brain.

Two arms are kept apart:

- TARGET_INFORMED_RECONSTRUCTION: classes present on the actual post-op target
  slice. These are oracle-text diagnostics, not pre-operative knowledge.
- SUPPLIED_SCENARIO: classes supplied independently of any held-out target.

Names describe annotations; an ID-2 annotation is not a claim that the T1ce
image shows that compartment. Clinical GTR/NTR, residual size or laterality
must never be inserted here.
"""

from itertools import permutations

import numpy as np

TARGET_INFORMED = "TARGET_INFORMED_RECONSTRUCTION"
SUPPLIED_SCENARIO = "SUPPLIED_SCENARIO"


def slice_classes(label_slice):
    """Sorted foreground IDs present on a 2D label slice."""
    ids = np.unique(np.asarray(label_slice))
    return tuple(int(value) for value in ids if value != 0)


def kind_orders(ids, text_config):
    """All valid "," joined class-name orders, or the empty-slice kind."""
    names = text_config["class_names"]
    unknown = [value for value in ids if str(value) not in names]
    if unknown:
        raise ValueError(f"No template name for IDs {unknown}")
    if not ids:
        return [text_config["empty_kind"]]
    return [",".join(names[str(value)] for value in order) for order in permutations(sorted(ids))]


def prompts_for_classes(ids, text_config):
    """``[(image_prompt, mask_prompt), ...]`` for every valid class order."""
    pairs = []
    for kind in kind_orders(tuple(ids), text_config):
        fields = {"organ": text_config["organ"], "kind": kind}
        pairs.append((text_config["image_template"].format(**fields), text_config["mask_template"].format(**fields)))
    return pairs


def target_informed_prompts(label_slice, text_config):
    """Oracle-text prompts from the post-op TARGET slice (diagnostic arm only)."""
    return {"arm": TARGET_INFORMED, "prompts": prompts_for_classes(slice_classes(label_slice), text_config)}


def scenario_prompts(ids, text_config):
    """Prompts for an independently supplied desired class set."""
    return {"arm": SUPPLIED_SCENARIO, "prompts": prompts_for_classes(tuple(sorted(set(ids))), text_config)}


def check_token_lengths(prompts, tokenizer):
    """Return the longest token count; raise instead of silently truncating.

    ``tokenizer`` follows the transformers interface (``__call__`` returning
    ``input_ids`` and a ``model_max_length`` attribute).
    """
    limit = int(tokenizer.model_max_length)
    longest = 0
    for prompt in prompts:
        length = len(tokenizer(prompt, truncation=False)["input_ids"])
        if length > limit:
            raise ValueError(f"Prompt exceeds the tokenizer limit ({length} > {limit}): {prompt!r}")
        longest = max(longest, length)
    return longest
