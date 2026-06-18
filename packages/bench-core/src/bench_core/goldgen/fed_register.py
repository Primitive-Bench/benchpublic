"""Federal Register documents — the cleanest delta source (no key, JSON).

GET /api/v1/documents.json?conditions[publication_date][gte]=YYYY-MM-DD
Authoritative timestamp: publication_date. Canonical: html_url.
Truth token: document_number (e.g. "2026-26402"), printed on the document page.
"""

from __future__ import annotations

from datetime import datetime

from bench_core.config import get_settings
from bench_core.http import RateLimiter, get_with_retry, make_client
from bench_core.domain import Candidate
from bench_core.goldgen.base import build_variants

# Document types most likely to have a single defensible answer URL.
_TYPE_PHRASE = {
    "Rule": "final rule",
    "Proposed Rule": "proposed rule",
    "Notice": "notice",
    "Presidential Document": "presidential document",
}


class FederalRegisterAdapter:
    name = "fed_register"

    async def fetch(self, since: datetime, limit: int) -> list[Candidate]:
        s = get_settings()
        limiter = RateLimiter(rate=5, per=1.0)
        url = f"{s.fr_api_base}/documents.json"
        params = {
            "per_page": min(limit, 100),
            "order": "newest",
            "conditions[publication_date][gte]": since.date().isoformat(),
            "fields[]": [
                "document_number",
                "title",
                "html_url",
                "publication_date",
                "type",
                "agencies",
            ],
        }
        out: list[Candidate] = []
        async with make_client() as client:
            resp = await get_with_retry(client, url, params=params, limiter=limiter)
            resp.raise_for_status()
            for doc in resp.json().get("results", [])[:limit]:
                agency = ""
                if doc.get("agencies"):
                    agency = (doc["agencies"][0] or {}).get("name", "")
                phrase = _TYPE_PHRASE.get(doc.get("type", ""), "document")
                title = (doc.get("title") or "").strip()
                docnum = doc["document_number"]
                # descriptive form has no identifier; token form names the doc number.
                descriptive = (
                    f"Official Federal Register page for the {agency} {phrase} “{title}”"
                ).strip()
                variants = build_variants(
                    descriptive=descriptive,
                    token=docnum,
                    token_phrase=f"Federal Register document number {docnum}",
                )
                out.append(
                    Candidate(
                        query=variants["descriptive"],
                        golden_url=doc["html_url"],
                        truth_token=docnum,
                        authoritative_timestamp=doc["publication_date"],
                        stratum="fresh",
                        source=self.name,
                        query_variants=variants,
                    )
                )
        return out
