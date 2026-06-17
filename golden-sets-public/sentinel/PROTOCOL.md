# Sentinel protocol — truth by construction (index-freshness instrument)

Ported from arlenk2021/GoldenEvalsWebSearch (`sentinel/publish.py`).

A **sentinel** is a page the benchmark itself publishes, so its content, golden
URL, and publish time are known with certainty **before the world has ever seen
it**. Sentinels are the `sentinel` stratum (`GroundTruthTier.SENTINEL_PLANTED`,
D-09). They exist to do one thing the public web cannot: split a search
`not_found` miss into its two real causes — *not indexed yet* vs *ranked below k*.

## Why it is a prerequisite for a public search board

Without a page the bench owns, a search miss is ambiguous: did the vendor fail to
rank a page it has indexed, or has it simply not crawled it yet (freshness lag)?
Conflating the two misreports freshness as accuracy. Because a sentinel carries a
globally-unique truth token, one extra "indexing probe" with the `token_in_query`
variant settles it:

- the unique-token query surfaces the canonical URL → **ranked_below_k** (indexed; the descriptive query just didn't rank it in top-k)
- even the unique-token query cannot surface it → **not_indexed** (true index lag)

## Minting protocol

1. **Mint a cohort** (default 5 pages per nightly run). Each page gets a fresh
   random truth token `SENT-<16 hex>` and a canonical `<link rel="canonical">`.
2. **Vary the URL shape** per page (`notes/{slug}`, `ref/{slug}`, `docs/a/{slug}`,
   `k/{slug}/index`, …) so vendors cannot fingerprint a single "bench-owned" URL
   pattern and special-case it.
3. **The publish timestamp is authoritative**: the git commit time of the page
   file (recorded by the publishing CI job) is T0 for the lag clock. Deploy the
   sentinel cohort **before** anything else in a measurement run — any delay
   between minting and publish is lost freshness signal.
4. Emit a matching `Candidate` row (stratum `sentinel`) into the candidate stream;
   it flows through the same verify → split pipeline as registry rows.

## Trust notes

- The sentinel base URL and the salt are configured via env (`SENTINEL_BASE_URL`,
  `HMAC_SALT`); the salt is never published.
- Sentinels measure index freshness, not an intent slice — they carry no intent
  slice tag.
