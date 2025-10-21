# Changelog Generation Tools

Automated tools for generating enhanced changelog entries from llama.cpp releases.

## Features

- 🔍 **Automatic Release Fetching**: Retrieves release notes and PR information from GitHub
- 🏷️ **Smart Categorization**: Groups changes into Features, Bugs, Performance, etc.
- 📝 **Rich Formatting**: Generates well-formatted markdown with links and descriptions
- ⚡ **Rate Limit Handling**: Manages GitHub API rate limits automatically
- 🎨 **Customizable**: Configure categories, formatting, and output via YAML

## Installation

### Prerequisites

```bash
# Python 3.7+
python --version

# Install dependencies
pip install -r requirements-changelog.txt
```

### Optional: GitHub Token

For higher API rate limits (5000 requests/hour vs 60/hour), set up a GitHub token:

```bash
# Create a token at https://github.com/settings/tokens
# Only needs 'public_repo' scope

export GITHUB_TOKEN='your_token_here'
```

## Usage

### Generate Changelog for Version Range

```bash
# Basic usage
python scripts/generate_changelog.py generate --from-tag b6666 --to-tag b6792

# Preview without updating file
python scripts/generate_changelog.py generate --from-tag b6666 --to-tag b6792 --preview

# Specify custom date
python scripts/generate_changelog.py generate \
    --from-tag b6666 \
    --to-tag b6792 \
    --date 2025-10-18

# Custom output file
python scripts/generate_changelog.py generate \
    --from-tag b6666 \
    --to-tag b6792 \
    --output CHANGELOG_TEST.md
```

### Check Release Information

```bash
# View details for a specific release
python scripts/generate_changelog.py info --tag b6499
```

### Check API Rate Limit

```bash
# View current GitHub API rate limit status
python scripts/generate_changelog.py rate-limit
```

## Configuration

Edit `scripts/changelog/config.yaml` to customize:

### Categories

Define how changes are categorized:

```yaml
categories:
  feature:
    name: "New Features"
    icon: "🆕"
    priority: 2
    keywords:
      - add
      - new
      - implement
    labels:
      - enhancement
      - feature
```

### Formatting

Control output format:

```yaml
formatting:
  max_pr_title_length: 100
  include_pr_links: true
  group_minor_changes: true
  minor_change_threshold: 5
```

## Architecture

```
scripts/
├── generate_changelog.py    # Main CLI tool
└── changelog/
    ├── github_client.py     # GitHub API integration
    ├── parser.py            # Text parsing & extraction
    ├── categorizer.py       # Change categorization
    ├── generator.py         # Markdown generation
    └── config.yaml          # Configuration file
```

### Components

1. **GitHubClient**: Fetches release notes, PR info, and commits from llama.cpp repository
2. **ChangeParser**: Extracts PR references, detects breaking changes, parses structured data
3. **ChangeCategorizer**: Classifies changes using keywords, labels, and patterns
4. **ChangelogGenerator**: Formats changes into markdown with templates

## Example Output

### Before (Current)
```markdown
## 2025-10-02: Update to llama.cpp b6666

- b6499 (b6499) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6499
- b6500 (b6500) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6500
```

### After (Enhanced)
```markdown
## 2025-10-02: Update to llama.cpp b6666

### Summary
Updated llama.cpp from b6490 to b6666, incorporating 177 upstream commits with new features and bug fixes.

### Notable Changes

#### 🆕 New Features
- **b6499**: convert: add Llama4ForCausalLM ([#16042](https://github.com/ggml-org/llama.cpp/pull/16042))
  - Added support for Llama 4 causal language models
  - Implemented SWA (Sliding Window Attention) handling
  - Fixed use_kq_norm configuration

#### 🐛 Bug Fixes
- **b6512**: fix: memory leak in batch processing ([#16078](https://github.com/ggml-org/llama.cpp/pull/16078))

### Full Commit Range
- b6490 to b6666 (177 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b6490...b6666
```

## Workflow Integration

### Manual Updates

When updating llama.cpp version:

```bash
# 1. Update submodule or vendored code
cd vendor_llama_cpp_pydist/llama.cpp
git pull origin master

# 2. Note the old and new tags
OLD_TAG=b6666
NEW_TAG=b6792

# 3. Generate changelog
cd ../..
python scripts/generate_changelog.py generate \
    --from-tag $OLD_TAG \
    --to-tag $NEW_TAG \
    --date $(date +%Y-%m-%d)

# 4. Review CHANGELOG.md
git diff CHANGELOG.md

# 5. Commit changes
git add CHANGELOG.md
git commit -m "docs: update changelog for llama.cpp $NEW_TAG"
```

### Future: Automated Updates

Could be integrated into CI/CD:

```yaml
# .github/workflows/update-changelog.yml
- name: Generate Changelog
  run: |
    python scripts/generate_changelog.py generate \
      --from-tag ${{ env.OLD_TAG }} \
      --to-tag ${{ env.NEW_TAG }}
```

## Troubleshooting

### Rate Limit Errors

```bash
# Check rate limit status
python scripts/generate_changelog.py rate-limit

# Solution: Use GitHub token
export GITHUB_TOKEN='your_token'
```

### Missing Releases

Some build tags might not have GitHub releases. The tool handles this gracefully by:
- Skipping tags without releases
- Using commit messages as fallback
- Logging warnings for missing data

### Categorization Issues

If changes are miscategorized:

1. Review `config.yaml` keywords and labels
2. Add more specific patterns
3. Manually edit generated changelog if needed

## Development

### Running Tests

```bash
# TODO: Add unit tests
python -m pytest tests/
```

### Adding New Categories

Edit `config.yaml`:

```yaml
categories:
  security:
    name: "Security Fixes"
    icon: "🔒"
    priority: 1
    keywords:
      - security
      - vulnerability
      - cve
    labels:
      - security
```

## License

Same as llama_cpp_pydist project.

## References

- [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
- [Keep a Changelog](https://keepachangelog.com/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
