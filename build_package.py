import os
import argparse
import subprocess
import re # For regex operations on setup.py
import urllib.request
import shutil
import glob # For finding .egg-info directories
import logging  # Added for logging
import json
import sys
from datetime import datetime, timezone

SCRIPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from generate_changelog import ChangelogManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in project root
LLAMA_CPP_SUBMODULE_PATH = os.path.join(PROJECT_ROOT, "vendor_llama_cpp_pydist", "llama.cpp")  # Updated to actual submodule path
LLAMA_CPP_PACKAGE_BINARIES_PATH = os.path.join(PROJECT_ROOT, "llama_cpp", "binaries") # For storing downloaded binaries
LLAMA_CPP_EXAMPLES_PATH = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "examples")
LLAMA_CPP_MODELS_PATH = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "models")
LLAMA_CPP_CMAKE_FILE = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "CMakeLists.txt")
SETUP_PY_PATH = os.path.join(PROJECT_ROOT, "setup.py")
CHANGELOG_PATH = os.path.join(PROJECT_ROOT, "CHANGELOG.md")
GITHUB_API_BASE = "https://api.github.com"

# Repo to collect release notes from (llama.cpp)
LLAMA_CPP_REPO_OWNER = "ggml-org"
LLAMA_CPP_REPO_NAME = "llama.cpp"

def run_command(command, cwd=None, check=True, shell=False):
    """Helper function to run a shell command."""
    logging.info(f"Running command: {command if shell else ' '.join(command)} (in {cwd or PROJECT_ROOT})")
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, shell=shell)
    if check and process.returncode != 0:
        logging.error(f"Error running command: {command if shell else ' '.join(command)}")
        logging.error(f"Stdout: {process.stdout}")
        logging.error(f"Stderr: {process.stderr}")
        raise subprocess.CalledProcessError(process.returncode, command, output=process.stdout, stderr=process.stderr)
    return process

def download_and_place_windows_binary(tag_name):
    """Downloads the Windows binary for the given tag and places it in the package."""
    logging.info(f"\n--- Downloading Windows binary for tag: {tag_name} ---")
    os.makedirs(LLAMA_CPP_PACKAGE_BINARIES_PATH, exist_ok=True)

    # Determine the binary filename for the current tag_name first
    # This is used to avoid deleting it if it already exists and matches the current tag.
    match_current_tag = re.search(r'(b\\d+[a-fA-F0-9]*)$', tag_name)
    if match_current_tag:
        processed_tag_name_for_current_file = match_current_tag.group(1)
    else:
        processed_tag_name_for_current_file = tag_name
    binary_filename_for_current_tag = f"llama-{processed_tag_name_for_current_file}-bin-win-cpu-x64.zip"

    # Clear previous binaries in llama_cpp/binaries, but not the one for the current tag
    logging.info(f"Cleaning previous binaries from {LLAMA_CPP_PACKAGE_BINARIES_PATH} (excluding {binary_filename_for_current_tag} if present)...")
    for item in os.listdir(LLAMA_CPP_PACKAGE_BINARIES_PATH):
        if item == binary_filename_for_current_tag:
            logging.info(f"Skipping deletion of existing binary for current tag: {item}")
            continue # Skip deleting the binary for the current tag

        item_path = os.path.join(LLAMA_CPP_PACKAGE_BINARIES_PATH, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            logging.warning(f"Failed to delete {item_path}. Reason: {e}")


    # Skip download if binary for this tag already exists
    # The filename construction logic is similar to what's done above for the exclusion.
    # Re-using binary_filename_for_current_tag and processed_tag_name_for_current_file
    # for clarity, though the values would be the same as derived from tag_name.
    binary_dest_path = os.path.join(LLAMA_CPP_PACKAGE_BINARIES_PATH, binary_filename_for_current_tag)
    if os.path.exists(binary_dest_path):
        logging.info(f"Binary {binary_filename_for_current_tag} already exists at {binary_dest_path}. Skipping download.")
        return

    # Construct binary filename and URL
    # Example tag from user: b5479. Releases use full tags like 'master-b5479' or 'b5479'
    # We need to handle if the tag from `git describe` has a prefix or not.
    # The release asset seems to use the plain tag number.
    # Let's assume the tag_name from `git describe --tags --abbrev=0` is what's needed for the URL path,
    # but the filename might need processing if it contains prefixes not in the release asset name.
    
    # Use the already processed tag name for the filename
    # processed_tag_name_for_file = processed_tag_name_for_current_file (from above)

    # The binary_filename is already determined as binary_filename_for_current_tag
    # binary_filename = binary_filename_for_current_tag
    # The download URL uses the full tag name as it appears in the releases page
    download_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag_name}/{binary_filename_for_current_tag}"
    # binary_dest_path is already defined above

    logging.info(f"Attempting to download: {download_url}")
    try:
        # Create a request object to disable SSL verification if needed, or set headers
        req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(binary_dest_path, 'wb') as out_file:
            if response.status == 200:
                shutil.copyfileobj(response, out_file)
                logging.info(f"Successfully downloaded {binary_filename_for_current_tag} to {binary_dest_path}")
            else:
                logging.error(f"Failed to download {binary_filename_for_current_tag} from {download_url}. Status: {response.status}")
                if os.path.exists(binary_dest_path):
                    try:
                        os.remove(binary_dest_path)
                        logging.info(f"Removed partially downloaded file: {binary_dest_path}")
                    except Exception as rm_err:
                        logging.error(f"Failed to remove partially downloaded file {binary_dest_path}: {rm_err}")
                raise RuntimeError(f"Failed to download {binary_filename_for_current_tag} from {download_url}. Status: {response.status}")
    except urllib.error.HTTPError as e:
        logging.error(f"HTTPError {e.code} when trying to download {binary_filename_for_current_tag} from {download_url}: {e.reason}")
        logging.error("This could mean the binary for this specific tag/format doesn't exist or the URL is incorrect.")
        if os.path.exists(binary_dest_path):
            try:
                os.remove(binary_dest_path)
                logging.info(f"Removed partially downloaded file: {binary_dest_path}")
            except Exception as rm_err:
                logging.error(f"Failed to remove partially downloaded file {binary_dest_path}: {rm_err}")
        raise RuntimeError(f"HTTPError {e.code} downloading {binary_filename_for_current_tag} from {download_url}: {e.reason}") from e
    except urllib.error.URLError as e:
        logging.error(f"URLError when trying to download {binary_filename_for_current_tag} from {download_url}: {e.reason}")
        logging.error("This could be a network issue or an invalid URL.")
        if os.path.exists(binary_dest_path):
            try:
                os.remove(binary_dest_path)
                logging.info(f"Removed partially downloaded file: {binary_dest_path}")
            except Exception as rm_err:
                logging.error(f"Failed to remove partially downloaded file {binary_dest_path}: {rm_err}")
        raise RuntimeError(f"URLError downloading {binary_filename_for_current_tag} from {download_url}: {e.reason}") from e
    except Exception as e:
        logging.error(f"An unexpected error occurred while downloading {binary_filename_for_current_tag} from {download_url}: {e}")
        if os.path.exists(binary_dest_path):
            try:
                os.remove(binary_dest_path)
                logging.info(f"Removed partially downloaded file: {binary_dest_path}")
            except Exception as rm_err:
                logging.error(f"Failed to remove partially downloaded file {binary_dest_path}: {rm_err}")
        raise RuntimeError(f"Unexpected error downloading {binary_filename_for_current_tag} from {download_url}: {e}") from e

def modify_cmake_config():
    """
    Modifies the CMakeLists.txt in vendor_llama_cpp_pydist/llama.cpp to:
    1. Set LLAMA_BUILD_EXAMPLES to OFF.
    2. Comment out `add_subdirectory(examples)`.
    """
    logging.info(f"Modifying CMake config: {LLAMA_CPP_CMAKE_FILE}")
    if not os.path.exists(LLAMA_CPP_CMAKE_FILE):
        logging.error(f"Error: CMake file not found at {LLAMA_CPP_CMAKE_FILE}")
        return

    with open(LLAMA_CPP_CMAKE_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    modified = False
    for line in lines:
        if "option(LLAMA_BUILD_EXAMPLES" in line:
            if "OFF" not in line:
                new_lines.append('option(LLAMA_BUILD_EXAMPLES "llama: build examples" OFF) # Modified by build script\n')
                logging.info(" - Set LLAMA_BUILD_EXAMPLES to OFF")
                modified = True
            else:
                new_lines.append(line) # Already off
        elif "add_subdirectory(examples)" in line and not line.strip().startswith("#"):
            new_lines.append("# " + line) # Comment out adding the examples subdirectory
            logging.info(" - Commented out add_subdirectory(examples)")
            modified = True
        else:
            new_lines.append(line)

    if modified:
        with open(LLAMA_CPP_CMAKE_FILE, 'w') as f:
            f.writelines(new_lines)
        logging.info("CMake config modified.")
    else:
        logging.info("CMake config already up-to-date.")

def get_current_version(file_path):
    """Reads setup.py and extracts the version, returning a fallback if unavailable."""
    fallback_version = "0.0.0"
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            version = match.group(1)
            logging.info(f"--- Current package version from setup.py: {version} ---")
            return version
        logging.warning(f"Warning: Could not find version in {file_path}. Using fallback '{fallback_version}'.")
        return fallback_version
    except Exception as e:
        logging.warning(f"Warning: Could not read version from {file_path}: {e}. Using fallback '{fallback_version}'.")
        return fallback_version

def update_version_in_setup_py(file_path, new_version, old_version):
    """Writes the new version to setup.py."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        old_version_pattern = r"version\s*=\s*['\"]" + re.escape(old_version) + r"['\"]"
        new_version_string = f"version='{new_version}'" # Standardize on single quotes for the new version

        if not re.search(old_version_pattern, content):
            logging.warning(f"Warning: Old version '{old_version}' not found in {file_path} as expected using pattern '{old_version_pattern}'. Version not updated.")
            return

        new_content, num_replacements = re.subn(old_version_pattern, new_version_string, content, count=1)
        
        if num_replacements == 0:
            logging.warning(f"Warning: Old version '{old_version}' was not replaced in {file_path}. Check the version string and pattern.")
            return

        with open(file_path, 'w') as f:
            f.write(new_content)
        logging.info(f"Updated version in {file_path} from {old_version} to {new_version}")
    except FileNotFoundError:
        logging.error(f"Error: {file_path} not found during version update.")
    except Exception as e:
        logging.error(f"Error updating version in {file_path}: {e}")


def get_submodule_tag():
    """Gets the current tag or short commit hash for the llama.cpp submodule, logging status."""
    try:
        # Try getting the latest tag
        describe_cmd = ["git", "describe", "--tags", "--abbrev=0", "--always"]
        result = run_command(describe_cmd, cwd=LLAMA_CPP_SUBMODULE_PATH)
        tag = result.stdout.strip()
        if not tag:
            logging.warning("Could not determine a tag for the submodule. Falling back to commit hash.")
            result = run_command(["git", "rev-parse", "--short", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH)
            tag = result.stdout.strip()
        logging.info(f"Submodule vendor_llama_cpp_pydist/llama.cpp is at ref: {tag}")
        # Warn if dirty
        status = run_command(["git", "status", "--porcelain"], cwd=LLAMA_CPP_SUBMODULE_PATH, check=False)
        if status.stdout.strip():
            logging.warning(f"Submodule {LLAMA_CPP_SUBMODULE_PATH} is dirty. This might affect the build.")
        return tag
    except Exception as e:
        logging.error(f"Error determining submodule tag: {e}")
        return None

def init_and_update_submodule():
    """Initializes and updates the llama.cpp submodule to latest origin/master."""
    run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=PROJECT_ROOT)
    run_command(["git", "fetch"], cwd=LLAMA_CPP_SUBMODULE_PATH)
    run_command(["git", "checkout", "master"], cwd=LLAMA_CPP_SUBMODULE_PATH)
    run_command(["git", "reset", "--hard", "origin/master"], cwd=LLAMA_CPP_SUBMODULE_PATH)
    run_command(["git", "pull"], cwd=LLAMA_CPP_SUBMODULE_PATH)
    run_command(["git", "fetch", "--tags"], cwd=LLAMA_CPP_SUBMODULE_PATH)

def handle_version_increment_on_tag_change(old_tag, new_tag, current_version_str, setup_py_path):
    """
    Compares submodule tags and increments package version if they differ.
    Returns the effective version for build and the (potentially updated) current package version.
    """
    effective_build_version = current_version_str
    updated_current_version = current_version_str

    if old_tag and new_tag and old_tag != new_tag:
        logging.info(f"Submodule tag changed from {old_tag} to {new_tag}. Incrementing package version.")
        try:
            parts = current_version_str.split('.')
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                major, minor, patch = map(int, parts)
                minor += 1
                patch = 0 # Reset patch version
                new_package_version_str = f"{major}.{minor}.{patch}"
            else:
                logging.warning(f"Current version '{current_version_str}' is not in expected major.minor.patch format. Setting to 0.1.0.")
                new_package_version_str = "0.1.0"
            
            update_version_in_setup_py(setup_py_path, new_package_version_str, current_version_str)
            effective_build_version = new_package_version_str
            updated_current_version = new_package_version_str
        except Exception as e:
            logging.error(f"Failed to update version: {e}. Using original version {current_version_str} for build.")
            # effective_build_version remains current_version_str
            # updated_current_version remains current_version_str
    elif old_tag and new_tag and old_tag == new_tag:
        logging.info(f"Submodule tag '{new_tag}' unchanged. Package version remains {current_version_str}.")
    else:
        logging.warning("Could not reliably determine submodule tag change. Package version remains the same.")
    
    return effective_build_version, updated_current_version


# --------------------------- Changelog Helpers ---------------------------

def _normalize_llama_tag(tag: str) -> str:
    """Normalize tag to a form used in llama.cpp releases, e.g., 'b6497'.
    Accepts variants like 'master-b6497' and returns 'b6497'.
    """
    if not tag:
        return tag
    m = re.search(r"(b\d+[a-fA-F0-9]*)$", tag)
    return m.group(1) if m else tag


def _parse_llama_tag_number(tag: str) -> int:
    """Return the numeric component of a llama.cpp tag like 'b6497' -> 6497.
    Returns -1 if not parseable.
    """
    if not tag:
        return -1
    m = re.search(r"b(\d+)", tag)
    try:
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def _http_get_json(url: str):
    headers = {"User-Agent": "llama_cpp_pydist/1.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        data = resp.read()
        return json.loads(data.decode("utf-8"))


def fetch_release_by_tag(tag: str):
    """Fetch a single GitHub release by tag. Returns dict or None if not found.
    Uses the public GitHub API without auth (subject to rate limits)."""
    api = f"{GITHUB_API_BASE}/repos/{LLAMA_CPP_REPO_OWNER}/{LLAMA_CPP_REPO_NAME}/releases/tags/{tag}"
    try:
        return _http_get_json(api)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logging.warning(f"Release for tag {tag} not found via API.")
            return None
        logging.error(f"HTTPError fetching release for tag {tag}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error fetching release for tag {tag}: {e}")
        return None


def fetch_releases_paginated(max_pages: int = 5, per_page: int = 100):
    """Fetch recent releases list, paginated. Returns list (most recent first)."""
    releases = []
    for page in range(1, max_pages + 1):
        url = f"{GITHUB_API_BASE}/repos/{LLAMA_CPP_REPO_OWNER}/{LLAMA_CPP_REPO_NAME}/releases?per_page={per_page}&page={page}"
        try:
            data = _http_get_json(url)
            if not isinstance(data, list) or not data:
                break
            releases.extend(data)
            if len(data) < per_page:
                break
        except Exception as e:
            logging.error(f"Error fetching releases page {page}: {e}")
            break
    return releases


def collect_llama_changes_between_tags(old_tag: str, new_tag: str):
    """Collect release notes for llama.cpp tags between old_tag (exclusive) and new_tag (inclusive).
    If old_tag is None or can't be parsed, only includes new_tag if found.
    Returns a list of dicts with keys: tag_name, name, body, html_url, published_at.
    """
    new_norm = _normalize_llama_tag(new_tag) if new_tag else None
    old_norm = _normalize_llama_tag(old_tag) if old_tag else None
    new_num = _parse_llama_tag_number(new_norm)
    old_num = _parse_llama_tag_number(old_norm)

    if new_num < 0:
        logging.warning(f"New submodule tag '{new_tag}' not parseable; skipping changelog collection.")
        return []

    releases = fetch_releases_paginated()
    if not releases:
        logging.warning("No releases fetched from GitHub; skipping changelog collection.")
        return []

    selected = []
    for rel in releases:
        tag = rel.get("tag_name") or ""
        tag_norm = _normalize_llama_tag(tag)
        num = _parse_llama_tag_number(tag_norm)
        if num < 0:
            continue
        # Include those strictly greater than old_num and <= new_num
        if (old_num < 0 and num == new_num) or (old_num >= 0 and old_num < num <= new_num):
            selected.append({
                "tag_name": tag_norm,
                "name": rel.get("name") or tag_norm,
                "body": rel.get("body") or "",
                "html_url": rel.get("html_url") or f"https://github.com/{LLAMA_CPP_REPO_OWNER}/{LLAMA_CPP_REPO_NAME}/releases/tag/{tag_norm}",
                "published_at": rel.get("published_at") or "",
            })

    # Sort ascending by numeric tag for chronological order
    selected.sort(key=lambda r: _parse_llama_tag_number(r.get("tag_name", "")))
    return selected


def _extract_bullets_from_markdown(md: str):
    """Extract bullet-like lines from markdown body.
    - Handles -, *, +, •, –, — and numeric lists (1. ...)
    - Strips GitHub checkbox markers like [ ] or [x]
    - Skips headings and code fences
    Returns list[str].
    """
    if not md:
        return []

    bullets = []
    in_code_block = False
    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        s = line.strip()
        if not s:
            continue
        # Code block fences
        if s.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        # Skip headings and quote banners
        if s.startswith(('#', '>')):
            continue

        # Recognize bullet markers
        if re.match(r"^[\-\*\+•–—]\s+", s):
            content = re.sub(r"^[\-\*\+•–—]\s+", "", s)
            content = re.sub(r"^\[([ xX])\]\s+", "", content)  # strip checkboxes
            bullets.append(content.strip())
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", s):
            content = re.sub(r"^\d+\.\s+", "", s)
            bullets.append(content.strip())
            continue

    return bullets


def _extract_summary_lines(md: str, max_lines: int = 10):
    """Fallback: extract up to max_lines of plain text lines from markdown, excluding
    headings, images, links-only lines, and code blocks. Useful when no bullets exist."""
    if not md:
        return []
    lines = []
    in_code_block = False
    for raw_line in md.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if line.startswith(('#', '>', '![', '<')):
            continue
        # skip lines that are just links
        if re.fullmatch(r"\[.*?\]\(.*?\)", line):
            continue
        # keep short reasonable lines
        text = re.sub(r"\s+", " ", line)
        lines.append(text)
        if len(lines) >= max_lines:
            break
    return lines


def update_changelog(old_tag: str, new_tag: str):
    """Prepend a new section to CHANGELOG.md using the newer changelog generator."""
    new_norm = _normalize_llama_tag(new_tag) if new_tag else None
    if not new_norm:
        logging.info("No new tag resolved; skipping changelog update.")
        return

    from_norm = _normalize_llama_tag(old_tag) if old_tag else new_norm
    if not from_norm:
        from_norm = new_norm

    if _parse_llama_tag_number(new_norm) < 0:
        logging.warning(f"New submodule tag '{new_tag}' not parseable; skipping changelog update.")
        return

    if _parse_llama_tag_number(from_norm) < 0:
        logging.warning(f"Old submodule tag '{old_tag}' not parseable; using {new_norm} only.")
        from_norm = new_norm

    config_path = os.path.join(PROJECT_ROOT, "scripts", "changelog", "config.yaml")
    manager = ChangelogManager(config_path=config_path)

    today = datetime.now(timezone.utc).date().isoformat()
    section = manager.generate_changelog_section(from_norm, new_norm, date=today)

    if not section.strip():
        logging.info("Generated changelog section is empty; skipping changelog update.")
        return

    try:
        updated = manager.generator.update_changelog(CHANGELOG_PATH, section)
        if updated:
            logging.info(f"CHANGELOG.md updated with llama.cpp changes up to {new_norm}.")
        else:
            logging.warning("Changelog generator did not update CHANGELOG.md.")
    except Exception as e:
        logging.error(f"Failed to write CHANGELOG.md: {e}")

def main():
    parser = argparse.ArgumentParser(description="Build llama_cpp_pydist package")
    parser.add_argument(
        "--regenerate-current-version",
        action="store_true",
        help=(
            "Rewrite setup.py with the current version and use it for the build "
            "(no version bump even if submodule tag changes)."
        ),
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT) # Ensure commands run from project root

    current_package_version = get_current_version(SETUP_PY_PATH)
    # effective_version_for_build = current_package_version # Initialized by handle_version_increment

    # --- Step 1: Submodule Operations (Fetch, Checkout, Clean) ---
    logging.info("\\\\n--- Step 1: Processing submodule (Fetch, Checkout, Clean) ---")
    
    logging.info("Getting submodule tag before update...")
    old_submodule_tag = get_submodule_tag()

    # Initialize and update submodule
    init_and_update_submodule()
    
    logging.info("Getting submodule tag after update...")
    new_submodule_tag = get_submodule_tag()

    if args.regenerate_current_version:
        logging.info("Regenerating current package version in setup.py (no bump).")
        update_version_in_setup_py(SETUP_PY_PATH, current_package_version, current_package_version)
        effective_version_for_build = current_package_version
    else:
        effective_version_for_build, current_package_version = handle_version_increment_on_tag_change(
            old_submodule_tag, new_submodule_tag, current_package_version, SETUP_PY_PATH
        )

    # Determine submodule tag (this is now effectively new_submodule_tag)
    submodule_tag = new_submodule_tag 

    # --- Step 1.5: Download Windows Binary ---
    if submodule_tag:
        download_and_place_windows_binary(submodule_tag)
    else:
        logging.error("Critical: Submodule tag/commit not determined. Skipping download of Windows binary.")
        # Ensure binaries directory exists for setup.py even if download fails, but it should be there from download_and_place_windows_binary
        os.makedirs(LLAMA_CPP_PACKAGE_BINARIES_PATH, exist_ok=True)

    # --- Step 2: Update CHANGELOG with upstream changes ---
    logging.info("\n--- Step 2: Updating CHANGELOG.md with llama.cpp release notes ---")
    try:
        update_changelog(old_submodule_tag, new_submodule_tag)
    except Exception as e:
        logging.error(f"Changelog update failed (non-fatal): {e}")


    # --- Step 3: Modify CMake Configuration in submodule (if needed) ---
    logging.info("\n--- Step 3: Modifying CMake config in submodule (if applicable) ---")
    modify_cmake_config()  # Call the CMake modification function

    # --- Step 4: Versioning and Committing to Parent Repository (Simplified) ---
    # This section is simplified. Original script had more complex logic for version bumping
    # and committing submodule changes. For now, we assume the parent repo is managed manually
    # regarding submodule pointer updates.
    logging.info("\n--- Step 4: Versioning and Committing (Simplified) ---")
    logging.info(f"Parent repository operations (like committing submodule changes) are assumed to be handled manually or by a separate process for now.")


    logging.info(f"\n--- Step 5: Building the wheel (using version: {effective_version_for_build}) ---")
    # --- 4. Generate source and binary wheel ---
    # Clean previous builds
    logging.info("Cleaning previous build artifacts...")
    if os.path.exists(os.path.join(PROJECT_ROOT, "dist")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "dist"))
    if os.path.exists(os.path.join(PROJECT_ROOT, "build")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "build"))
    
    # Use glob to find and remove .egg-info directories
    egg_info_dirs = glob.glob(os.path.join(PROJECT_ROOT, "*.egg-info"))
    for egg_dir in egg_info_dirs:
        logging.info(f"Removing old .egg-info directory: {egg_dir}")
        shutil.rmtree(egg_dir)

    # Build the source distribution and wheel
    try:
        logging.info("Building source distribution (sdist)...")
        run_command(["python3", SETUP_PY_PATH, "sdist"], cwd=PROJECT_ROOT)
        logging.info("Building wheel (bdist_wheel)...")
        run_command(["python3", SETUP_PY_PATH, "bdist_wheel"], cwd=PROJECT_ROOT)
        logging.info("Wheel build process completed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during wheel build: {e}")
        return # Or raise
    
    logging.info("\n--- Automation Script Finished ---")
    logging.info(f"Check the '{os.path.join(PROJECT_ROOT, 'dist')}' directory for the generated wheel and source distribution.")

if __name__ == "__main__":
    main()
