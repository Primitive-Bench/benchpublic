"""NVD CVE API 2.0 delta pump.

GET /rest/json/cves/2.0?pubStartDate=...&pubEndDate=...
Header `apiKey` raises the limit from 5 to 50 req/30s.
Authoritative timestamp: published. Canonical: nvd.nist.gov/vuln/detail/{CVE}.
Truth token: the CVE ID itself, which appears in the page body.
"""

from __future__ import annotations

from datetime import datetime

from bench_core.config import get_settings
from bench_core.http import RateLimiter, get_with_retry, make_client
from bench_core.domain import Candidate
from bench_core.goldgen.base import build_variants

ENDPOINT = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _iso(dt: datetime) -> str:
    # NVD wants e.g. 2026-06-10T00:00:00.000
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000")


def _short_desc(cve: dict) -> str:
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            text = d.get("value", "")
            return (text[:140] + "…") if len(text) > 140 else text
    return ""


class NvdAdapter:
    name = "nvd_cve"

    async def fetch(self, since: datetime, limit: int) -> list[Candidate]:
        s = get_settings()
        headers = {"apiKey": s.nvd_api_key} if s.nvd_api_key else {}
        # With a key NVD allows 50/30s; without, 5/30s. Stay well under.
        limiter = RateLimiter(rate=45 if s.nvd_api_key else 4, per=30.0)
        now = datetime.now(since.tzinfo)
        params = {
            "pubStartDate": _iso(since),
            "pubEndDate": _iso(now),
            "resultsPerPage": min(limit, 2000),
        }
        out: list[Candidate] = []
        async with make_client(headers) as client:
            resp = await get_with_retry(client, ENDPOINT, params=params, limiter=limiter)
            resp.raise_for_status()
            for item in resp.json().get("vulnerabilities", [])[:limit]:
                cve = item.get("cve", {})
                cid = cve.get("id")
                if not cid:
                    continue
                desc = _short_desc(cve)
                # descriptive form must NOT contain the CVE id; build_variants strips it.
                descriptive = (
                    f"NVD vulnerability record: {desc}" if desc else "NVD vulnerability record"
                )
                variants = build_variants(descriptive=descriptive, token=cid, token_phrase=cid)
                published = cve.get("published") or ""
                out.append(
                    Candidate(
                        query=variants["descriptive"],
                        golden_url=f"https://nvd.nist.gov/vuln/detail/{cid}",
                        truth_token=cid,
                        authoritative_timestamp=published,
                        stratum="fresh",
                        source=self.name,
                        query_variants=variants,
                    )
                )
        return out
