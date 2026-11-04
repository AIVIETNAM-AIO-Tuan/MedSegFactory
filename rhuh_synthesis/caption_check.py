"""VLM caption feasibility check (config block "text_control_captions_v1", Test B).

Open vision-language models (VLMs), run locally on the approved GPU host, describe
PRE-op slices only, which are available at inference. A caption of a post-op
slice would describe the answer, so it is never produced here. Nothing is
trained on these captions. The record answers three questions:

- what the models say, in short terms and in sentences;
- how many CLIP tokens each caption needs, against the 77-token limit of the
  Stable Diffusion text encoder (the pipeline refuses to truncate);
- how often an automatic keyword flag disagrees with the slice's pre-op labels.
  The flags are crude, so the raw captions are kept for reading.

RHUH images must never be sent to an external VLM service.
"""

import re

import numpy as np
import torch

_ABNORMAL = re.compile(
    r"\b(tumou?r\w*|mass(es)?|lesions?|glioma\w*|neoplas\w*|enhanc\w*|o?edema\w*|necro\w*|abnormal\w*|"
    r"hyperintens\w*|hypointens\w*|cysts?|cystic|ha?emorrhag\w*|midline shift)\b",
    re.IGNORECASE,
)
_NEGATED = re.compile(
    r"\b(no|without)\s+(\w+\s+){0,2}(abnormalit\w*|lesions?|mass(es)?|tumou?rs?|enhancement|o?edema)\b",
    re.IGNORECASE,
)
_NORMAL = re.compile(r"\b(normal|unremarkable)\b", re.IGNORECASE)


def slice_to_rgb(image):
    """A [H, W] slice in [0, 1] as an anterior-up RGB PIL image (the grids draw row 0 at the bottom)."""
    from PIL import Image

    pixels = (np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(np.flipud(pixels))).convert("RGB")


@torch.no_grad()
def vlm_caption(model, processor, image, prompt, *, max_new_tokens):
    """Greedy caption of one image by a chat VLM with a transformers processor."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(model.device)
            if value.is_floating_point():
                value = value.to(model.dtype)
        moved[key] = value
    output = model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0, moved["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def caption_flags(text):
    """Rough keyword flags: ``mentions_abnormality`` (ignoring negated phrases) and ``says_normal``."""
    return {
        "mentions_abnormality": bool(_ABNORMAL.search(_NEGATED.sub(" ", text))),
        "says_normal": bool(_NORMAL.search(text) or _NEGATED.search(text)),
    }


def token_count(tokenizer, text):
    """CLIP tokens of ``text`` including the start and end tokens, never truncated."""
    return len(tokenizer(text, truncation=False)["input_ids"])


def agreement(rows):
    """Keyword flag against labels: counts of abnormality mentioned on labelled / label-free slices."""
    labelled = [r for r in rows if r["has_pre_label"]]
    free = [r for r in rows if not r["has_pre_label"]]
    return {
        "labelled_slices": len(labelled),
        "labelled_mentioned": sum(r["mentions_abnormality"] for r in labelled),
        "label_free_slices": len(free),
        "label_free_mentioned": sum(r["mentions_abnormality"] for r in free),
        "label_free_said_normal": sum(r["says_normal"] for r in free),
    }
