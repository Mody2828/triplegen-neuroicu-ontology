"""Strip control characters from corpus/chunk text (pipeline-wide).

PDF extraction and copy-paste can introduce form-feed, vertical tab, and other
ASCII control chars. Used at multiple points in the pipeline (not just prompt-time):

- After reading raw text (ingest)
- During normalization (normalize.py)
- Before chunking / when building chunks (chunk.py)
- Before prompting (run.py)
- Before JSON parsing of LLM output (parse.py)

This keeps JSON and prompts safe and avoids duplicating control-char logic.
"""

from __future__ import annotations

import re

# Unsafe in JSON strings: form-feed \f, vertical tab, and other ASCII control chars.
# We keep \t (0x09), \n (0x0A), \r (0x0D) as normal whitespace; strip the rest.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control_chars_for_prompt(text: str) -> str:
    """Replace control characters (except tab, newline, CR) with space so chunk text is safe for prompting.

    Use on corpus/chunk text before sending to the LLM. Stops the model from
    copying form-feed or other invalid chars into JSON evidence fields.
    """
    if not text:
        return text or ""
    return _CONTROL_CHARS.sub(" ", text)
