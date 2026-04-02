"""GitHub API client for fetching llama.cpp release information."""

import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import requests
from github import Github, GithubException
import logging

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for interacting with GitHub API to fetch llama.cpp information."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token. If None, uses GITHUB_TOKEN env var.
        """
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.github = Github(self.token) if self.token else Github()
        self.repo_name = "ggml-org/llama.cpp"
        self.repo = None
        self.session = requests.Session()

        # Caching
        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = project_root / ".cache" / "github_client"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting / retry config
        self.last_request_time = 0.0
        self.min_request_interval = 0.2  # polite default seconds between requests
        self.max_retries = 3
        self.backoff_factor = 2.0
        
    def _get_repo(self):
        """Get repository object, cached."""
        if self.repo is None:
            self.repo = self.github.get_repo(self.repo_name)
        return self.repo
    
    def _rate_limit_wait(self):
        """Wait if necessary to avoid rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _cache_path_for(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _cache_get(self, key: str, ttl: int = 3600) -> Optional[Any]:
        p = self._cache_path_for(key)
        try:
            if not p.exists():
                return None
            raw = json.loads(p.read_text(encoding="utf-8"))
            ts = raw.get("ts", 0)
            if time.time() - ts > ttl:
                try:
                    p.unlink()
                except Exception:
                    pass
                return None
            return raw.get("data")
        except Exception:
            return None

    def _cache_set(self, key: str, data: Any):
        p = self._cache_path_for(key)
        try:
            p.write_text(json.dumps({"ts": time.time(), "data": data}, default=str), encoding="utf-8")
        except Exception:
            pass

    def _extract_rate_limit_resources(self, rate_limit: Any) -> Dict[str, Dict[str, Any]]:
        """Normalize PyGithub rate-limit responses across supported versions."""
        core = getattr(rate_limit, 'core', None)
        search = getattr(rate_limit, 'search', None)
        if core and search:
            return {
                'core': {
                    'limit': getattr(core, 'limit', None),
                    'remaining': getattr(core, 'remaining', None),
                    'reset': getattr(core, 'reset', None),
                },
                'search': {
                    'limit': getattr(search, 'limit', None),
                    'remaining': getattr(search, 'remaining', None),
                    'reset': getattr(search, 'reset', None),
                },
            }

        raw = getattr(rate_limit, 'raw_data', None)
        if isinstance(raw, dict):
            resources = raw.get('resources', {})
            return {
                'core': resources.get('core', {}),
                'search': resources.get('search', {}),
            }

        if isinstance(rate_limit, dict):
            resources = rate_limit.get('resources', {})
            return {
                'core': resources.get('core', {}),
                'search': resources.get('search', {}),
            }

        return {}

    def _check_rate_limit_and_wait(self, threshold: int = 10, buffer_seconds: int = 5):
        """Check current API rate limit and sleep until reset if below threshold."""
        try:
            rl = self.github.get_rate_limit()
            core = self._extract_rate_limit_resources(rl).get('core', {})
            remaining = core.get("remaining")
            reset = core.get("reset")
            if remaining is None:
                return
            if remaining <= threshold:
                # reset may be datetime
                if isinstance(reset, datetime):
                    now = datetime.now(timezone.utc)
                    sleep_for = (reset - now).total_seconds() + buffer_seconds
                    if sleep_for > 0:
                        logger.warning(f"GitHub API low on requests ({remaining} remaining). Sleeping {int(sleep_for)}s until reset.")
                        time.sleep(sleep_for)
                else:
                    # unknown reset format - be conservative
                    logger.warning("GitHub API low on requests; sleeping briefly to avoid hitting limits.")
                    time.sleep(buffer_seconds)
        except Exception:
            # If we can't fetch rate limit, don't block but remain polite
            time.sleep(0.1)
    
    def fetch_release_notes(self, tag: str) -> Optional[Dict[str, Any]]:
        """
        Fetch release notes for a specific tag.
        
        Args:
            tag: Git tag (e.g., 'b6499')
            
        Returns:
            Dict with release information or None if not found
        """
        cache_key = f"release:{tag}"
        cached = self._cache_get(cache_key, ttl=24 * 3600)
        if cached:
            return cached

        for attempt in range(1, self.max_retries + 1):
            try:
                self._check_rate_limit_and_wait()
                self._rate_limit_wait()
                repo = self._get_repo()
                release = repo.get_release(tag)

                data = {
                    'tag': tag,
                    'name': release.title,
                    'body': release.body,
                    'published_at': release.published_at.isoformat() if release.published_at else None,
                    'url': release.html_url,
                    'author': release.author.login if release.author else None
                }
                self._cache_set(cache_key, data)
                return data
            except GithubException as e:
                if getattr(e, "status", None) == 404:
                    logger.warning(f"Release not found for tag {tag}")
                    return None
                # Rate limit or abuse detection
                msg = str(e).lower()
                if getattr(e, "status", None) in (403, 429) or "rate limit" in msg:
                    logger.warning(f"Rate-limited when fetching release {tag} (attempt {attempt}/{self.max_retries}).")
                    self._check_rate_limit_and_wait(threshold=0)
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                logger.error(f"Error fetching release {tag}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching release {tag}: {e}")
                return None
    
    def fetch_pr_info(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """
        Fetch pull request information.
        
        Args:
            pr_number: PR number
            
        Returns:
            Dict with PR information or None if not found
        """
        cache_key = f"pr:{pr_number}"
        cached = self._cache_get(cache_key, ttl=6 * 3600)
        if cached:
            return cached

        for attempt in range(1, self.max_retries + 1):
            try:
                self._check_rate_limit_and_wait()
                self._rate_limit_wait()
                repo = self._get_repo()
                pr = repo.get_pull(pr_number)

                data = {
                    'number': pr_number,
                    'title': pr.title,
                    'body': pr.body,
                    'state': pr.state,
                    'merged': pr.merged,
                    'url': pr.html_url,
                    'labels': [label.name for label in pr.labels],
                    'author': pr.user.login if pr.user else None,
                    'created_at': pr.created_at.isoformat() if pr.created_at else None,
                    'merged_at': pr.merged_at.isoformat() if pr.merged_at else None
                }
                self._cache_set(cache_key, data)
                return data
            except GithubException as e:
                if getattr(e, "status", None) == 404:
                    logger.warning(f"PR #{pr_number} not found")
                    return None
                msg = str(e).lower()
                if getattr(e, "status", None) in (403, 429) or "rate limit" in msg:
                    logger.warning(f"Rate-limited when fetching PR #{pr_number} (attempt {attempt}/{self.max_retries}).")
                    self._check_rate_limit_and_wait(threshold=0)
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                logger.error(f"Error fetching PR #{pr_number}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching PR #{pr_number}: {e}")
                return None
    
    def fetch_commit_range(self, from_tag: str, to_tag: str) -> List[Dict[str, Any]]:
        """
        Fetch all commits between two tags.
        
        Args:
            from_tag: Starting tag
            to_tag: Ending tag
            
        Returns:
            List of commit information dicts
        """
        try:
            self._rate_limit_wait()
            repo = self._get_repo()
            comparison = repo.compare(from_tag, to_tag)
            
            commits = []
            for commit in comparison.commits:
                commits.append({
                    'sha': commit.sha,
                    'message': commit.commit.message,
                    'author': commit.commit.author.name,
                    'date': commit.commit.author.date,
                    'url': commit.html_url
                })
            
            return commits
        except GithubException as e:
            logger.error(f"Error fetching commits {from_tag}..{to_tag}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching commits: {e}")
            return []
    
    def fetch_releases_in_range(self, from_tag: str, to_tag: str) -> List[Dict[str, Any]]:
        """
        Fetch all releases between two tags (inclusive).
        
        Args:
            from_tag: Starting tag (e.g., 'b6666')
            to_tag: Ending tag (e.g., 'b6792')
            
        Returns:
            List of release information dicts
        """
        try:
            # Extract build numbers
            from_num = int(from_tag.replace('b', ''))
            to_num = int(to_tag.replace('b', ''))

            # Try cache first (range key)
            cache_key = f"releases:{from_num}:{to_num}"
            cached = self._cache_get(cache_key, ttl=6 * 3600)
            if cached:
                return cached

            releases: List[Dict[str, Any]] = []

            # Use REST API with pagination to reduce per-object API calls
            per_page = 100
            page = 1
            headers = {"Accept": "application/vnd.github+json", "User-Agent": "llama_cpp_pydist/1.0"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            retry_count = 0
            while True:
                self._check_rate_limit_and_wait()
                self._rate_limit_wait()
                url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page={per_page}&page={page}"
                try:
                    resp = self.session.get(url, headers=headers, timeout=30)
                    if resp.status_code == 403 and "rate limit" in (resp.text or "").lower():
                        # Wait until reset then retry
                        retry_count += 1
                        if retry_count > self.max_retries:
                            logger.error(f"Exceeded retry budget fetching releases page {page} after rate limiting.")
                            break
                        logger.warning(f"Rate limited fetching releases page {page}; sleeping before retry.")
                        self._check_rate_limit_and_wait(threshold=0)
                        retry_delay = self.backoff_factor ** retry_count
                        time.sleep(retry_delay)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    retry_count = 0
                except requests.RequestException as e:
                    logger.error(f"HTTP error fetching releases page {page}: {e}")
                    break

                if not isinstance(data, list) or not data:
                    break

                for rel in data:
                    tag = rel.get("tag_name") or ""
                    if tag.startswith('b'):
                        num = -1
                        try:
                            num = int(tag.replace('b', ''))
                        except Exception:
                            continue
                        if from_num <= num <= to_num:
                            releases.append({
                                'tag': tag,
                                'name': rel.get('name') or tag,
                                'body': rel.get('body') or "",
                                'published_at': rel.get('published_at'),
                                'url': rel.get('html_url'),
                                'author': (rel.get('author') or {}).get('login') if rel.get('author') else None
                            })

                if len(data) < per_page:
                    break
                page += 1

            releases.sort(key=lambda x: int(x['tag'].replace('b', '')))
            self._cache_set(cache_key, releases)
            return releases
        except Exception as e:
            logger.error(f"Error fetching releases in range: {e}")
            return []
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get current rate limit information.
        
        Returns:
            Dict with rate limit details
        """
        try:
            rate_limit = self.github.get_rate_limit()
            return self._extract_rate_limit_resources(rate_limit)
        except Exception as e:
            logger.error(f"Error fetching rate limit: {e}")
            return {}
    
    def fetch_tags_in_range(self, from_tag: str, to_tag: str) -> List[str]:
        """
        Get list of all tags between two versions.
        
        Args:
            from_tag: Starting tag (e.g., 'b6666')
            to_tag: Ending tag (e.g., 'b6792')
            
        Returns:
            List of tag names
        """
        try:
            from_num = int(from_tag.replace('b', ''))
            to_num = int(to_tag.replace('b', ''))
            
            # Generate all possible tags in range
            tags = []
            for num in range(from_num, to_num + 1):
                tags.append(f'b{num}')
            
            return tags
        except ValueError as e:
            logger.error(f"Invalid tag format: {e}")
            return []
