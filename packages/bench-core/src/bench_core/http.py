"""Robust async HTTP client factory, a polite rate limiter, and the
main-content `fetch` used for truth-token checking.

Centralises redirect handling, retries, timeouts and the per-host courtesy
limits the registries require (SEC: 10 req/s; NVD: 5 or 50 req/30s).

The truth token must appear in MAIN CONTENT, not site-wide boilerplate
(plan must-fix #4): `fetch` strips nav/header/footer/script/style before the
caller matches, so a date or id living only in a footer does not count as a
verified hit.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx
from selectolax.parser import HTMLParser

from bench_core.config import get_settings
from bench_core.urls import registrable_domain

# Let httpx negotiate Accept-Encoding from the codecs actually installed
# (gzip/deflate always; br/zstd only if their libs are present). Forcing "br"
# here without a brotli decoder yields undecodable bytes -> UnicodeDecodeError.
DEFAULT_HEADERS: dict[str, str] = {}

# Browser-like UA for verification/liveness fetches. Many authoritative sites
# (e.g. federalregister.gov) serve an anti-bot challenge to non-browser agents,
# which would otherwise read as a false "truth token missing" exclusion. Registry
# *API* calls override this with their own required UA (e.g. SEC's name+email).
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_BOILERPLATE = ("script", "style", "nav", "header", "footer", "aside", "noscript")


@dataclass
class RateLimiter:
    """Simple token-bucket: `rate` requests per `per` seconds."""

    rate: float
    per: float = 1.0
    _allowance: float = field(default=0.0, init=False)
    _last: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._allowance = self.rate

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._allowance += (now - self._last) * (self.rate / self.per)
            self._last = now
            if self._allowance > self.rate:
                self._allowance = self.rate
            if self._allowance < 1.0:
                wait = (1.0 - self._allowance) * (self.per / self.rate)
                await asyncio.sleep(wait)
                self._allowance = 0.0
            else:
                self._allowance -= 1.0


def make_client(headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    s = get_settings()
    merged = {**DEFAULT_HEADERS, **(headers or {})}
    return httpx.AsyncClient(
        timeout=s.http_timeout_seconds,
        follow_redirects=True,
        http2=True,
        headers=merged,
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    )


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 3,
    backoff: float = 1.5,
    **kwargs,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        if limiter:
            await limiter.acquire()
        try:
            resp = await client.get(url, **kwargs)
            # Transient: rate-limit + server-side 5xx (federalregister.gov 500s
            # under rapid access). Back off and retry; non-transient codes return.
            if resp.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(backoff ** (attempt + 1))
                continue
            return resp
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            await asyncio.sleep(backoff ** (attempt + 1))
    if last_exc:
        raise last_exc
    raise httpx.HTTPError(f"exhausted retries for {url}")


# ---------------------------------------------------------------------------
# Main-content fetch (ported from verify/fetcher.py)
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    final_url: str
    status: int
    main_text: str
    canonical_link: str | None
    redirect_chain: list[str]
    soft_404: bool


def extract_main_text(html: str) -> str:
    tree = HTMLParser(html)
    for tag in _BOILERPLATE:
        for node in tree.css(tag):
            node.decompose()
    body = tree.body or tree.root
    return body.text(separator=" ", strip=True) if body else ""


def _canonical_link(html: str) -> str | None:
    tree = HTMLParser(html)
    node = tree.css_first('link[rel="canonical"]')
    if node:
        return node.attributes.get("href")
    return None


def _looks_soft_404(text: str, status: int) -> bool:
    if status == 200 and len(text) < 200:
        lowered = text.lower()
        return any(s in lowered for s in ("not found", "page doesn't exist", "404"))
    return False


def _user_agent_for(url: str) -> str:
    # SEC mandates its descriptive "name email" UA and 403s a browser UA; other
    # registries (e.g. federalregister.gov) 403 a non-browser UA. Pick per host.
    if registrable_domain(url) == "sec.gov":
        return get_settings().sec_edgar_user_agent
    return BROWSER_UA


async def fetch(url: str) -> FetchResult:
    async with make_client({"User-Agent": _user_agent_for(url)}) as client:
        try:
            resp = await get_with_retry(client, url)
        except httpx.HTTPError:
            return FetchResult(url, 0, "", None, [url], soft_404=False)
        chain = [str(r.url) for r in resp.history] + [str(resp.url)]
        html = resp.text if "html" in resp.headers.get("content-type", "") else ""
        text = extract_main_text(html) if html else resp.text
        return FetchResult(
            final_url=str(resp.url),
            status=resp.status_code,
            main_text=text,
            canonical_link=_canonical_link(html) if html else None,
            redirect_chain=chain,
            soft_404=_looks_soft_404(text, resp.status_code),
        )
