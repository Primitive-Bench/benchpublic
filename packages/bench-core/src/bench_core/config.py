"""Typed configuration loaded from the environment (.env)."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # registry pump
    sec_edgar_user_agent: str = Field(default="arlen-bench arlen1788@berkeley.edu")
    nvd_api_key: str = Field(default="")
    github_token: str = Field(default="")
    github_release_repos: str = Field(default="")
    fr_api_base: str = Field(default="https://www.federalregister.gov/api/v1")

    # search vendors (web_search primitive).
    # Accept the wrodium-bench WRODIUM_<VENDOR>_API_KEY convention as the primary
    # env name, falling back to the bare name.
    exa_api_key: str = Field(
        default="", validation_alias=AliasChoices("WRODIUM_EXA_API_KEY", "EXA_API_KEY")
    )
    brave_search_api_key: str = Field(
        default="", validation_alias=AliasChoices("WRODIUM_BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY")
    )
    tavily_api_key: str = Field(
        default="", validation_alias=AliasChoices("WRODIUM_TAVILY_API_KEY", "TAVILY_API_KEY")
    )
    serpapi_key: str = Field(
        default="", validation_alias=AliasChoices("WRODIUM_SERP_API_KEY", "SERPAPI_KEY")
    )
    perplexity_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("WRODIUM_PERPLEXITY_API_KEY", "PERPLEXITY_API_KEY"),
    )
    google_cse_key: str = ""
    google_cse_engine_id: str = ""
    bing_search_key: str = ""
    you_api_key: str = ""

    # extraction vendors (web_extraction primitive).
    firecrawl_api_key: str = Field(
        default="", validation_alias=AliasChoices("WRODIUM_FIRECRAWL_API_KEY", "FIRECRAWL_API_KEY")
    )
    jina_api_key: str = Field(
        default="", validation_alias=AliasChoices("WRODIUM_JINA_API_KEY", "JINA_API_KEY")
    )
    apify_api_key: str = Field(
        default="", validation_alias=AliasChoices("WRODIUM_APIFY_API_KEY", "APIFY_API_KEY")
    )
    apify_actor: str = Field(
        default="apify/website-content-crawler",
        validation_alias=AliasChoices("WRODIUM_APIFY_ACTOR", "APIFY_ACTOR"),
    )
    brightdata_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("WRODIUM_BRIGHTDATA_API_KEY", "BRIGHTDATA_API_KEY"),
    )

    # split secret — never serialized into any artifact
    hmac_salt: str = Field(default="")

    # sentinel
    sentinel_base_url: str = "https://sentinel.arlenkumar.com"
    sentinel_publish_branch: str = "gh-pages"

    # knobs
    split_public_fraction: float = 0.70
    probe_repetitions: int = 3
    liveness_window_hours: int = 6
    verify_min_gap_hours: int = 6
    http_timeout_seconds: float = 30.0
    duckdb_path: str = "data/eval.duckdb"

    def repos(self) -> list[str]:
        return [r.strip() for r in self.github_release_repos.split(",") if r.strip()]

    def require_salt(self) -> str:
        if not self.hmac_salt:
            raise RuntimeError("HMAC_SALT is unset. Generate with `openssl rand -hex 32`.")
        return self.hmac_salt


@lru_cache
def get_settings() -> Settings:
    return Settings()
