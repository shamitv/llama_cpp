# Changelog Tools Usage Guide

## Quick Start

The changelog generation tools have been implemented and are ready to use. They automatically fetch release information from llama.cpp and generate enhanced changelog entries.

## Installation

Before using the tools, install the required dependencies:

```bash
# Option 1: Using virtual environment (recommended)
python3 -m venv venv-changelog
source venv-changelog/bin/activate
pip install -r requirements-changelog.txt

# Option 2: Using system packages with --break-system-packages (not recommended)
pip install -r requirements-changelog.txt --break-system-packages

# Option 3: Using pipx for isolated installation
pipx install --spec requirements-changelog.txt generate-changelog
```

## Dependencies

The following packages are required (see `requirements-changelog.txt`):
- `requests>=2.31.0` - HTTP requests
- `PyGithub>=2.1.1` - GitHub API client
- `pyyaml>=6.0` - YAML configuration
- `jinja2>=3.1.2` - Template engine
- `python-dateutil>=2.8.2` - Date handling
- `click>=8.1.0` - CLI framework

## Usage Examples

### 1. Generate Enhanced Changelog

```bash
# Generate changelog for b6666 to b6792 (example from screenshot)
python scripts/generate_changelog.py generate \
    --from-tag b6666 \
    --to-tag b6792 \
    --date 2025-10-18

# Preview before updating
python scripts/generate_changelog.py generate \
    --from-tag b6666 \
    --to-tag b6792 \
    --preview

# Use custom date
python scripts/generate_changelog.py generate \
    --from-tag b6499 \
    --to-tag b6511 \
    --date 2025-09-18 \
    --preview
```

### 2. Check Release Information

```bash
# View details for b6499 (Llama4ForCausalLM example)
python scripts/generate_changelog.py info --tag b6499
```

### 3. Monitor API Rate Limits

```bash
# Check current GitHub API rate limit
python scripts/generate_changelog.py rate-limit
```

## Enhanced Output Format

The tool transforms simple version lists into detailed, categorized changelogs:

### Current Format (Before)
```markdown
- b6499 (b6499) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6499
```

### Enhanced Format (After)
```markdown
#### 🆕 New Features
- **b6499**: convert: add Llama4ForCausalLM ([#16042](https://github.com/ggml-org/llama.cpp/pull/16042))
  - Added support for Llama 4 causal language models
  - Implemented SWA (Sliding Window Attention) handling
  - Fixed use_kq_norm configuration
```

## Configuration

Edit `scripts/changelog/config.yaml` to customize:

- **Categories**: Define change types (features, bugs, performance, etc.)
- **Formatting**: Control output style, link inclusion, length limits
- **GitHub Settings**: Repository and token configuration

## GitHub Token (Optional but Recommended)

For better rate limits (5000 vs 60 requests/hour):

```bash
# 1. Create token at https://github.com/settings/tokens
# 2. Select 'public_repo' scope
# 3. Set environment variable
export GITHUB_TOKEN='your_github_token_here'
```

## Workflow Integration

### When Updating llama.cpp Version

```bash
# 1. Note current and new versions
OLD_VERSION=b6666
NEW_VERSION=b6792

# 2. Generate enhanced changelog
python scripts/generate_changelog.py generate \
    --from-tag $OLD_VERSION \
    --to-tag $NEW_VERSION \
    --date $(date +%Y-%m-%d)

# 3. Review changes
git diff CHANGELOG.md

# 4. Commit
git add CHANGELOG.md
git commit -m "docs: enhance changelog for llama.cpp $NEW_VERSION"
```

## Features Implemented

✅ **GitHub API Integration**
- Fetch release notes and PR information
- Handle rate limiting automatically
- Support for both authenticated and unauthenticated access

✅ **Smart Parsing**
- Extract PR references (#16042 format)
- Detect breaking changes
- Parse conventional commit messages
- Extract structured descriptions

✅ **Intelligent Categorization**
- 7 categories: Breaking Changes, Features, Performance, Bugs, Docs, Examples, Maintenance
- Keyword matching
- Label-based classification
- Path-based detection

✅ **Rich Formatting**
- Markdown with emojis (🆕 🐛 🚀 ⚠️)
- PR links and references
- Nested descriptions
- Summary generation

✅ **Configuration**
- YAML-based settings
- Customizable categories
- Formatting controls
- Cache settings

## Architecture

```
scripts/
├── generate_changelog.py       # Main CLI (375 lines)
└── changelog/
    ├── __init__.py
    ├── config.yaml            # Configuration (140 lines)
    ├── github_client.py       # GitHub API (240 lines)
    ├── parser.py              # Text parsing (280 lines)
    ├── categorizer.py         # Categorization (200 lines)
    ├── generator.py           # Markdown generation (290 lines)
    └── README.md              # Detailed documentation
```

## Troubleshooting

### ModuleNotFoundError
```bash
# Install dependencies first
pip install -r requirements-changelog.txt
```

### Rate Limit Exceeded
```bash
# Check status
python scripts/generate_changelog.py rate-limit

# Use GitHub token
export GITHUB_TOKEN='your_token'
```

### Missing Releases
Some tags may not have releases - the tool logs warnings but continues processing.

## Next Steps

1. **Install dependencies** in a virtual environment
2. **Set GitHub token** for better rate limits
3. **Test with recent versions** using --preview flag
4. **Customize config.yaml** if needed
5. **Integrate into workflow** for future updates

## References

- Tool documentation: `scripts/changelog/README.md`
- Implementation plan: `docs/changelog-improvement-plan.md`
- llama.cpp releases: https://github.com/ggml-org/llama.cpp/releases
