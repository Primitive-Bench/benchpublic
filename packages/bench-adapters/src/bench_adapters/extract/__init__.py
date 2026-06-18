"""Web-EXTRACTION vendor adapters (URL -> clean content).

Importing this subpackage auto-registers every extraction adapter via the
`@register("name")` decorators in `adapters`.

Registered names: firecrawl, jina, exa_live, exa_cached, tavily_extract, apify.
"""

from __future__ import annotations

from bench_adapters.extract import adapters as adapters  # noqa: F401  (import for side effects)

__all__ = ["adapters"]
