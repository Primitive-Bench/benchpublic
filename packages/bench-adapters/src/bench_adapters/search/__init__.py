"""Web-SEARCH vendor adapters (query -> ranked URLs).

Importing this subpackage auto-registers every search adapter via the
`@register("name")` decorators in `adapters`.

Registered names: exa, brave, tavily, google_cse, bing, serpapi, perplexity, you.
"""

from __future__ import annotations

from bench_adapters.search import adapters as adapters  # noqa: F401  (import for side effects)

__all__ = ["adapters"]
