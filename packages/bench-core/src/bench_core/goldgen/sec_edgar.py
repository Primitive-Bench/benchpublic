"""SEC EDGAR delta pump via the "current events" Atom feed.

The getcurrent action returns the most recently received filings, which is a
true delta without needing a CIK list. SEC mandates a descriptive User-Agent
("name email") and a 10 req/s ceiling across sec.gov hosts.

Authoritative timestamp: <updated> on each entry. Canonical: the filing-index
URL. Truth token: the accession number (########-YY-######), printed on the
filing index page.
"""

from __future__ import annotations

import re
from datetime import datetime
from xml.etree import ElementTree as ET

from bench_core.config import get_settings
from bench_core.http import RateLimiter, get_with_retry, make_client
from bench_core.domain import Candidate
from bench_core.goldgen.base import build_variants

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
# Form types where one official filing-index URL is the single defensible answer.
FORM_TYPES = ("8-K", "10-K", "10-Q", "S-1", "DEF 14A")
_ACCESSION_RE = re.compile(r"accession[_-]?number=([0-9-]+)", re.I)
_ACCESSION_PATH_RE = re.compile(r"/(\d{10}-\d{2}-\d{6})")


def _accession_from(url: str, summary: str) -> str | None:
    for src in (url, summary):
        m = _ACCESSION_RE.search(src) or _ACCESSION_PATH_RE.search(src)
        if m:
            return m.group(1).replace("-", "") if "-" not in m.group(1)[:11] else m.group(1)
    return None


class SecEdgarAdapter:
    name = "sec_edgar"

    async def fetch(self, since: datetime, limit: int) -> list[Candidate]:
        s = get_settings()
        headers = {"User-Agent": s.sec_edgar_user_agent}
        limiter = RateLimiter(rate=8, per=1.0)  # under the 10 req/s ceiling
        out: list[Candidate] = []
        async with make_client(headers) as client:
            for form in FORM_TYPES:
                url = "https://www.sec.gov/cgi-bin/browse-edgar"
                params = {
                    "action": "getcurrent",
                    "type": form,
                    "company": "",
                    "dateb": "",
                    "owner": "include",
                    "count": "40",
                    "output": "atom",
                }
                resp = await get_with_retry(client, url, params=params, limiter=limiter)
                if resp.status_code != 200:
                    continue
                try:
                    root = ET.fromstring(resp.text)
                except ET.ParseError:
                    continue
                for entry in root.findall("a:entry", ATOM_NS):
                    title = (
                        entry.findtext("a:title", default="", namespaces=ATOM_NS) or ""
                    ).strip()
                    updated = entry.findtext("a:updated", default="", namespaces=ATOM_NS) or ""
                    link_el = entry.find("a:link", ATOM_NS)
                    href = link_el.get("href") if link_el is not None else ""
                    summary = entry.findtext("a:summary", default="", namespaces=ATOM_NS) or ""
                    if not href:
                        continue
                    accession = _accession_from(href, summary)
                    if not accession:
                        continue
                    if updated:
                        try:
                            if datetime.fromisoformat(updated.replace("Z", "+00:00")) < since:
                                continue
                        except ValueError:
                            pass
                    descriptive = f"Official SEC EDGAR filing index for {form}: {title}"
                    variants = build_variants(
                        descriptive=descriptive,
                        token=accession,
                        token_phrase=f"accession number {accession}",
                    )
                    out.append(
                        Candidate(
                            query=variants["descriptive"],
                            golden_url=href,
                            truth_token=accession,
                            authoritative_timestamp=updated,
                            stratum="fresh",
                            source=self.name,
                            query_variants=variants,
                        )
                    )
                    if len(out) >= limit:
                        return out
        return out
