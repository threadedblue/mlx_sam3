"""Prompt-template expansion for SegForge (templating v1).

A template arrives once from DoubleNaught at /initSession and is stored in
registry.parquet alongside `mode`. This module turns one template plus one
mask's facts into the draft text that pre-fills the caption box.

Deliberately NOT `str.format`. A template is user-authored prose that will
routinely contain stray single braces and JSON-ish fragments ("output {"a":
1}"), and `str.format` raises on every one of them — KeyError, IndexError or
ValueError — turning a typo into a 500. A regex over `{{name}}` cannot raise
on anything the user types.

SINGLE PASS, BY CONSTRUCTION. `re.sub` with a replacement FUNCTION scans the
template exactly once and never re-examines what it substituted, so a
text_tag whose own text is "{{pass}}" comes out literally. A naive
str.replace loop would expand it a second time and leak one mask's data into
another field's slot.

Nothing here touches SAM3, a MaskRecord, or any session state: it takes
values and returns a string. The expansion is computed on demand and never
persisted — it becomes persistent only if the user saves it as the caption
through the existing Save Caption path.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

# `{{name}}` with optional inner whitespace. Names are restricted to word
# characters so an unmatched or malformed brace run can never match: "{{",
# "{{ }}" and "{{a b}}" all fall through untouched rather than raising.
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# Exactly the four names v1 supports. Anything else is left literal, so a
# template written against a later version degrades to visible text rather
# than silently losing content.
KNOWN_PLACEHOLDERS = ("text_tag", "pass", "bbox", "panel")

# What `panel` renders when assign_panel found no enclosing panel. The
# caller uses the full page in that case, and the draft has to say so —
# "panel 1" and an unlabelled page are not interchangeable to whoever reads
# the caption later.
FULL_PAGE_LABEL = "full page"


def format_bbox(bbox: Optional[Sequence[float]]) -> str:
    """`"x0,y0,x1,y1"` in integer page pixels, or "" when there is none.

    Ints, not floats: the bbox comes from pixel indices
    (`_geometry_bbox`/`_bbox_from_geometry`), so "12,30,88,96" is the honest
    rendering and "12.0,30.0,88.0,96.0" is noise in a human-read prompt.
    """
    if bbox is None:
        return ""
    return ",".join(str(int(v)) for v in bbox)


def panel_label(index: Optional[int]) -> str:
    """`"panel N"` for a 1-BASED authoring index, or "full page" for None.

    1-based because the Panels card numbers its rows from 1 for display; a
    draft saying "panel 0" would not match anything the user can see.
    """
    return FULL_PAGE_LABEL if index is None else f"panel {index}"


def expand_prompt_template(
    template: str,
    *,
    text_tag: Optional[str] = None,
    pass_: Optional[int] = None,
    bbox: Optional[Sequence[float]] = None,
    panel_label: Optional[str] = None,
) -> str:
    """Expand `{{...}}` placeholders in *template*.

    A KNOWN placeholder with no value renders as "" — an empty slot, never
    the literal `{{text_tag}}`, because a mask with no text_tag genuinely
    has nothing to say there and showing the raw placeholder would look like
    a bug to the user editing the caption.

    An UNKNOWN name is left exactly as written. Malformed input never raises.
    """
    if not template:
        return ""

    values = {
        "text_tag": text_tag or "",
        "pass": "" if pass_ is None else str(pass_),
        "bbox": format_bbox(bbox),
        "panel": panel_label or "",
    }

    def _replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        # `match.group(0)` is the original text, so an unknown name is
        # returned byte-for-byte rather than reconstructed.
        return values[name] if name in values else match.group(0)

    return _PLACEHOLDER.sub(_replace, template)
