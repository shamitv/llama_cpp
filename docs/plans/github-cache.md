# GitHub API Caching Plan

Goal
- Avoid hitting GitHub API rate limits during changelog generation and package rebuilds by adding an optional on-disk cache for GitHub REST responses and PyGithub queries.

Scope
- Cache release list and release-by-tag JSON responses fetched by `build_package.py` (`_http_get_json`, `fetch_releases_paginated`, `fetch_release_by_tag`).
- Cache supplemental queries used by `scripts/changelog/github_client.py` (release notes, PR info, comparisons) when possible.
- Do not change binary asset download behavior (existing file-exists check remains). Cache is for API metadata only.

Design Overview
- Simple file-based cache, configurable directory (default: `.cache/github`).
- Cache entries keyed by a stable key derived from request: for REST endpoints use URL (normalized) hashed (sha256) to filename; for PyGithub objects (e.g., PR, release) use deterministic keys like `release-{tag}`.
- Each cache file stores an envelope JSON:
  - `url` or `key`
  - `status` (HTTP status)
  - `headers` (selected response headers, including `ETag`, `Last-Modified`)
  - `body` (JSON-serializable content)
  - `fetched_at` (ISO timestamp)

- TTL and validation:
  - Default TTL: 24 hours (configurable).
  - If cache is stale but has `ETag` or `Last-Modified`, make a conditional request using `If-None-Match` / `If-Modified-Since` to reduce payloads; on 304, update `fetched_at` and reuse body.
  - If no validator headers available and cache stale, perform full GET and update cache.
  - If cache read fails or JSON is corrupt, fall back to network fetch and overwrite the cache entry.

- API:
  - `github_cache.get(url_or_key, ttl_seconds=..., validate=True) -> (cached_body, from_cache_bool)`
  - `github_cache.set(url_or_key, response_json, headers)`
  - `github_cache.invalidate(key_or_prefix)` and `github_cache.clear()`
  - Optional logging hooks for cache hit/miss/refresh events (debug level).

- Integration points:
  - Replace `_http_get_json(url)` in `build_package.py` with a wrapper that consults `github_cache.get(...)` before performing a network request. Respect `GITHUB_NO_CACHE` or CLI flags.
  - For `scripts/changelog/github_client.py`, add an argument to `GitHubClient(token, cache=None)` where `cache` is an optional cache object; use `requests` session for conditional requests (or use PyGithub but fall back to REST for conditional requests). Note: PyGithub doesn’t expose response headers easily, so conditional requests may need REST endpoints for proper validators. For parts using PyGithub objects (which are in-memory), consider caching by converting relevant objects to plain dicts and storing them in the cache.
  - Normalize query param order for URL-based keys to avoid duplicate cache entries for the same request.
  - Pagination: cache each page response separately (one file per page URL) for `fetch_releases_paginated`.

- Config:
  - Default cache dir: `.cache/github` under project root.
  - Configurable TTL via `scripts/changelog/config.yaml` (add `cache.ttl_seconds`, `cache.dir`, `cache.enabled`).
  - Environment overrides: `GITHUB_CACHE_DIR`, `GITHUB_CACHE_TTL`, `GITHUB_NO_CACHE` / `GITHUB_FORCE_REFRESH`.

- CLI flags and behavior:
  - Add optional CLI flags to changelog generator and `build_package.py` integration: `--no-cache` (always fetch), `--force-refresh` (invalidate and fetch), `--cache-ttl <seconds>`.
  - Define `--force-refresh` semantics: clear only keys touched in the current run (preferred) or clear entire cache directory (fallback).

- Security and correctness:
  - Cache only public GitHub API responses; avoid caching any sensitive tokens. Stored cache files are JSON only—no tokens saved.
  - Ensure cache respects conditional headers and updates on 200 with new ETag.
  - Optionally store a `.cache/.gitignore` to avoid committing cache.

Implementation Steps (approx effort)
1. Create `llama_cpp/github_cache.py` (or `scripts/changelog/github_cache.py`) implementing the file cache API, TTL, hashing key, and conditional request helpers. (~2-3 hours)
   - Use `requests` for conditional requests.
   - Implement safe atomic writes (write temp file then rename).
2. Add config support and environment variable parsing. (~30m)
3. Integrate into `build_package.py` by wrapping `_http_get_json` to consult the cache and perform conditional requests. (~1 hour)
4. Integrate into `scripts/changelog/github_client.py`—either by passing a cache instance to `GitHubClient` or by wrapping REST calls that the client uses. Add caching for `fetch_release_notes`, `fetch_pr_info`, and `fetch_releases_in_range`. (~1.5-2 hours)
5. Add CLI flags and environment variables to control cache behavior in `generate_changelog.py` and `build_package.py`. (~30m)
6. Write unit tests for cache behavior (cache hit, TTL expiry, 304 handling, invalidation, corrupt-cache fallback). (~2 hours)
7. Add docs in `docs/CHANGELOG_TOOLS_USAGE.md` and the new plan file with rollout instructions and examples. (~30m)
8. Optional: Add GitHub Actions job to warm the cache on schedule (if desired). (~1 hour)

Acceptance Criteria
- Re-running changelog generation or `build_package.py` within TTL should not perform full API requests for cached endpoints (verified via tests or logging).
- Conditional requests should result in 304 when upstream unchanged and cause no JSON payload downloads.
- `--no-cache` and `--force-refresh` flags must work as documented.
- Cache files created in `.cache/github` and not committed to repo.
- Cache should tolerate corrupted cache entries by refetching.

Example usage (developer):
- Default: uses cache with 24h TTL.
- Force refresh during development:

```bash
export GITHUB_NO_CACHE=1
python3 build_package.py
# or
python3 scripts/generate_changelog.py --no-cache
```

Rollout notes
- Start with read-only cache usage (only consult and write) and monitor logs for unexpected 304/200 patterns.
- After initial landing, monitor rate-limit usage.


Questions / Decisions to make
- TTL default (24h suggested) — change if you want more/less aggressive caching.
- Where to place cache module: top-level `llama_cpp/github_cache.py` vs `scripts/changelog/github_cache.py` (I recommend top-level `llama_cpp/` so both build and changelog code can import it).
- Whether to cache more aggressive objects (PR body, commit comparisons) — some endpoints may be large; conditional requests preferred.
- Whether to use REST endpoints for cached calls even when PyGithub is available (to access ETag/Last-Modified headers).

Next steps
- Implement `github_cache.py` and update `_http_get_json` and `GitHubClient` to use it. Would you like me to implement the cache module now and integrate it into `build_package.py` and `scripts/changelog/github_client.py`?