"""GitHub Releases delta pump.

GET /repos/{owner}/{repo}/releases — no server-side date filter, so we filter
client-side on published_at. PAT raises the limit 60 -> 5000 req/hr.
Authoritative timestamp: published_at. Canonical: html_url. Truth token: tag_name.
"""

from __future__ import annotations

from datetime import datetime

from bench_core.config import get_settings
from bench_core.http import RateLimiter, get_with_retry, make_client
from bench_core.domain import Candidate
from bench_core.goldgen.base import build_variants


class GithubReleasesAdapter:
    name = "github_releases"

    async def fetch(self, since: datetime, limit: int) -> list[Candidate]:
        s = get_settings()
        repos = s.repos()
        if not repos:
            return []
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if s.github_token:
            headers["Authorization"] = f"Bearer {s.github_token}"
        limiter = RateLimiter(rate=10, per=1.0)
        out: list[Candidate] = []
        async with make_client(headers) as client:
            for repo in repos:
                url = f"https://api.github.com/repos/{repo}/releases"
                resp = await get_with_retry(client, url, params={"per_page": 30}, limiter=limiter)
                if resp.status_code != 200:
                    continue
                for rel in resp.json():
                    if rel.get("draft") or rel.get("prerelease"):
                        continue
                    published = rel.get("published_at")
                    if not published:
                        continue
                    if datetime.fromisoformat(published.replace("Z", "+00:00")) < since:
                        continue
                    tag = rel.get("tag_name")
                    name = rel.get("name") or tag
                    descriptive = f"Official GitHub release page for {repo} {name}"
                    variants = build_variants(
                        descriptive=descriptive,
                        token=tag,
                        token_phrase=f"release {tag}",
                    )
                    out.append(
                        Candidate(
                            query=variants["descriptive"],
                            golden_url=rel["html_url"],
                            truth_token=tag,
                            authoritative_timestamp=published,
                            stratum="fresh",
                            source=self.name,
                            query_variants=variants,
                        )
                    )
                    if len(out) >= limit:
                        return out
        return out
