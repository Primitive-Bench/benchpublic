"""Query-form construction for the web_search primitive.

Ported from arlenk2021/GoldenEvalsWebSearch `src/probe/queries.py`.

The search analog of the extraction depth stratification. A registry-delta row
tempts you toward a query that *contains* the truth token ("CVE-2026-12345
details") — but that is a navigational lookup every vendor saturates, exactly
like title-zone extraction rows. The honest discriminator is finding the
document from a description that does NOT contain its unique identifier.

Forms (easy -> hard):
  - token_in_query: the identifier appears verbatim (navigational; saturates)
  - descriptive:    natural-language description with the identifier removed
                    (the honest discriminator)
"""

from __future__ import annotations

import re

QUERY_FORMS = ("descriptive", "token_in_query")
DEFAULT_FORM = "descriptive"


def strip_token(text: str, token: str) -> str:
    """Remove the truth token (and trivial variants) from descriptive text so the
    descriptive form cannot be answered by string-matching the identifier."""
    if not text:
        return text
    out = text.replace(token, "")
    # also strip the token with internal punctuation removed (e.g. CVE 2026 12345)
    loose = re.escape(token).replace(r"\-", r"[\s\-]?")
    out = re.sub(loose, "", out)
    return re.sub(r"\s{2,}", " ", out).strip(" :;,-")


def build_variants(
    *, descriptive: str, token: str, token_phrase: str | None = None
) -> dict[str, str]:
    """Build the form->query map.

    `descriptive` is the no-identifier ask (already token-free, or cleaned here).
    `token_phrase` is how the identifier is named in the easy form; defaults to
    appending the bare token.
    """
    desc = strip_token(descriptive, token).strip()
    tphrase = token_phrase or token
    return {
        "descriptive": desc,
        "token_in_query": f"{desc} ({tphrase})".strip(),
    }
