# Changelog Improvement Plan

**Created:** October 20, 2025  
**Status:** Draft  
**Purpose:** Enhance CHANGELOG.md to include detailed change information from llama.cpp releases

---

## Problem Statement

The current `CHANGELOG.md` only lists llama.cpp version numbers (build tags) with dates and URLs. It lacks:

- **Actual change descriptions** from each llama.cpp release
- **PR titles and numbers** that explain what was modified
- **Categorization** of changes (features, fixes, performance, etc.)
- **User-facing impact** information
- **Breaking changes** highlights

**Example of current format:**
```markdown
## 2025-10-18: Update to llama.cpp b6792

- b6670 (b6670) – 2025-10-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6670
- b6671 (b6671) – 2025-10-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6671
...
```

**Example of desired format (based on llama.cpp release notes):**
```markdown
## 2025-10-18: Update to llama.cpp b6792

### Summary
Updated llama.cpp from b6666 to b6792, incorporating 127 upstream commits with new model support, performance improvements, and bug fixes.

### Notable Changes

#### New Features
- **b6499**: convert: add Llama4ForCausalLM (#16042)
  - Added support for Llama 4 causal language models
  - Implemented SWA (Sliding Window Attention) handling
  - Fixed use_kq_norm configuration
  
- **b6501**: server: add multimodal chat template support (#15998)
  - Enhanced server to handle multimodal inputs in chat templates

#### Performance Improvements
- **b6523**: cuda: optimize matrix multiplication kernels
- **b6545**: ggml: improve quantization performance for large models

#### Bug Fixes
- **b6512**: fix: memory leak in context handling
- **b6587**: fix: incorrect token probabilities in sampling

#### Breaking Changes
- **b6600**: BREAKING: change default context size from 512 to 2048

### Full List of Versions
- b6670 – b6792 (127 commits)
- See individual release notes: https://github.com/ggml-org/llama.cpp/releases
```

---

## Requirements

### R1: Fetch Change Descriptions
- Automatically fetch release notes from llama.cpp GitHub releases
- Extract PR titles, numbers, and descriptions
- Parse commit messages for each build tag

### R2: Categorize Changes
- Group changes into categories:
  - 🆕 New Features
  - 🚀 Performance Improvements
  - 🐛 Bug Fixes
  - ⚠️ Breaking Changes
  - 📚 Documentation
  - 🔧 Maintenance
  - 🎨 Examples

### R3: Highlight Important Changes
- Flag breaking changes prominently
- Summarize key features that impact users
- Link to relevant PRs and issues

### R4: Maintain Readability
- Keep changelog concise but informative
- Use consistent formatting
- Group minor changes together
- Provide drill-down links for details

### R5: Automation
- Script to generate enhanced changelog entries
- Ability to update existing entries
- Integration with version update workflow

---

## Proposed Solution

### Architecture

```
┌─────────────────────────────────────────────┐
│  Changelog Generation Tool                  │
│  (scripts/generate_changelog.py)            │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ GitHub API   │        │ Local Git    │
│ - Releases   │        │ - Commits    │
│ - PR Info    │        │ - Tags       │
└──────────────┘        └──────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  Parser & Categorizer │
        │  - Extract PR info    │
        │  - Categorize changes │
        │  - Detect breaking    │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Template Engine      │
        │  - Format markdown    │
        │  - Apply styling      │
        │  - Generate summary   │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   CHANGELOG.md        │
        │   (Updated)           │
        └───────────────────────┘
```

### Components

#### 1. GitHub API Client
- **File:** `scripts/changelog/github_client.py`
- **Purpose:** Fetch release notes and PR information
- **Functions:**
  - `fetch_release_notes(tag)` - Get release description from GitHub
  - `fetch_pr_info(pr_number)` - Get PR title, labels, description
  - `fetch_commit_range(from_tag, to_tag)` - Get all commits between versions

#### 2. Change Parser
- **File:** `scripts/changelog/parser.py`
- **Purpose:** Extract and structure change information
- **Functions:**
  - `parse_release_notes(notes)` - Extract structured data from release text
  - `extract_pr_references(text)` - Find PR numbers in text
  - `categorize_change(pr_info)` - Assign category based on labels/content
  - `detect_breaking_changes(text)` - Identify breaking changes

#### 3. Changelog Generator
- **File:** `scripts/changelog/generator.py`
- **Purpose:** Generate formatted changelog entries
- **Functions:**
  - `generate_section(from_tag, to_tag)` - Create section for version range
  - `format_change_item(change)` - Format individual change entry
  - `generate_summary(changes)` - Create high-level summary
  - `update_changelog(new_section)` - Insert into CHANGELOG.md

#### 4. CLI Tool
- **File:** `scripts/generate_changelog.py`
- **Purpose:** Main entry point for changelog generation
- **Usage:**
  ```bash
  # Generate changelog for latest update
  python scripts/generate_changelog.py --latest
  
  # Generate for specific version range
  python scripts/generate_changelog.py --from b6666 --to b6792
  
  # Update existing entry
  python scripts/generate_changelog.py --update "2025-10-18"
  
  # Fetch and show changes without updating
  python scripts/generate_changelog.py --preview --from b6666 --to b6792
  ```

### Data Sources

1. **GitHub Release API**
   - Endpoint: `https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}`
   - Contains: Release title, body (description), author, date
   
2. **GitHub PR API**
   - Endpoint: `https://api.github.com/repos/ggml-org/llama.cpp/pulls/{pr_number}`
   - Contains: PR title, labels, description, files changed
   
3. **Local Git Repository**
   - Location: `vendor_llama_cpp_pydist/llama.cpp/`
   - Usage: Fallback for commit messages if API fails

### Change Categorization Logic

```python
CATEGORY_RULES = {
    'New Features': {
        'keywords': ['add', 'new', 'implement', 'support'],
        'labels': ['enhancement', 'feature'],
        'prefixes': ['feat:', 'feature:']
    },
    'Performance Improvements': {
        'keywords': ['optimize', 'performance', 'faster', 'speed'],
        'labels': ['performance', 'optimization'],
        'prefixes': ['perf:', 'opt:']
    },
    'Bug Fixes': {
        'keywords': ['fix', 'bug', 'issue', 'crash', 'error'],
        'labels': ['bug', 'bugfix'],
        'prefixes': ['fix:', 'bugfix:']
    },
    'Breaking Changes': {
        'keywords': ['breaking', 'break', 'remove', 'deprecate'],
        'labels': ['breaking-change', 'breaking'],
        'prefixes': ['breaking:', 'BREAKING:'],
        'markers': ['BREAKING CHANGE:', 'BREAKING:']
    },
    'Documentation': {
        'keywords': ['doc', 'docs', 'documentation', 'readme'],
        'labels': ['documentation'],
        'prefixes': ['docs:', 'doc:']
    },
    'Examples': {
        'keywords': ['example', 'demo', 'sample'],
        'labels': ['examples'],
        'paths': ['examples/']
    }
}
```

### Template Format

```markdown
## {date}: Update to llama.cpp {to_version}

### Summary
Updated llama.cpp from {from_version} to {to_version}, incorporating {commit_count} upstream commits.

{high_level_summary}

### Notable Changes

{categorized_changes}

### Full Commit Range
- {from_version} to {to_version} ({commit_count} commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/{from_version}...{to_version}

---
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up project structure under `scripts/changelog/`
- [ ] Implement GitHub API client with rate limiting
- [ ] Create configuration file for API tokens and settings
- [ ] Add error handling and retry logic
- [ ] Write unit tests for API client

### Phase 2: Parser & Categorization (Week 2)
- [ ] Implement release notes parser
- [ ] Build PR information extractor
- [ ] Create change categorization logic
- [ ] Implement breaking change detection
- [ ] Add tests for parser with sample data

### Phase 3: Generator & Templates (Week 2-3)
- [ ] Design markdown templates
- [ ] Implement changelog generator
- [ ] Add summary generation logic
- [ ] Create CHANGELOG.md updater
- [ ] Test with historical data

### Phase 4: CLI & Integration (Week 3)
- [ ] Build command-line interface
- [ ] Add preview mode
- [ ] Implement update mode for existing entries
- [ ] Create usage documentation
- [ ] Add to version update workflow

### Phase 5: Enhancement & Automation (Week 4)
- [ ] Add caching for API responses
- [ ] Implement batch processing for multiple versions
- [ ] Create GitHub Actions workflow for automation
- [ ] Add changelog validation
- [ ] Generate initial enhanced changelog for existing entries

---

## Configuration

### `scripts/changelog/config.yaml`
```yaml
github:
  repo: "ggml-org/llama.cpp"
  token_env: "GITHUB_TOKEN"  # Optional, for higher rate limits
  
changelog:
  file: "CHANGELOG.md"
  categories:
    - name: "Breaking Changes"
      icon: "⚠️"
      priority: 1
    - name: "New Features"
      icon: "🆕"
      priority: 2
    - name: "Performance Improvements"
      icon: "🚀"
      priority: 3
    - name: "Bug Fixes"
      icon: "🐛"
      priority: 4
    - name: "Documentation"
      icon: "📚"
      priority: 5
    - name: "Examples"
      icon: "🎨"
      priority: 6
      
  formatting:
    max_pr_title_length: 100
    include_pr_links: true
    include_commit_links: false
    group_minor_changes: true
    minor_change_threshold: 5
    
cache:
  enabled: true
  directory: ".cache/changelog"
  ttl_hours: 24
```

---

## Example Enhanced Entry

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
Updated llama.cpp from b6490 to b6666, incorporating 177 upstream commits with major improvements including Llama 4 support, server multimodal enhancements, and CUDA optimizations.

### Notable Changes

#### 🆕 New Features
- **b6499**: convert: add Llama4ForCausalLM ([#16042](https://github.com/ggml-org/llama.cpp/pull/16042))
  - Added support for Llama 4 causal language models
  - Implemented SWA (Sliding Window Attention) handling
  - Fixed use_kq_norm configuration

- **b6501**: server: add multimodal chat template support ([#15998](https://github.com/ggml-org/llama.cpp/pull/15998))

- **b6523**: llama: add support for MiniCPM-V 2.6 model ([#16089](https://github.com/ggml-org/llama.cpp/pull/16089))

#### 🚀 Performance Improvements
- **b6545**: cuda: optimize MMVQ kernel for faster inference
- **b6587**: ggml: improve quantization speed for Q4_K models

#### 🐛 Bug Fixes
- **b6512**: fix: memory leak in batch processing
- **b6556**: fix: incorrect token probabilities in beam search
- **b6598**: server: fix CORS headers for OPTIONS requests

#### 📚 Documentation
- **b6534**: docs: update build instructions for macOS
- **b6621**: readme: add Llama 4 to supported models list

#### 🎨 Examples
- **b6545**: examples: add multimodal chat example
- **b6600**: examples: demonstrate custom stopping criteria

### Additional Changes
34 minor improvements, dependency updates, and maintenance tasks. See [full commit range](https://github.com/ggml-org/llama.cpp/compare/b6490...b6666).

### Full Version List
b6490, b6491, b6492, ... b6664, b6665, b6666 (177 commits)

---
```

---

## Testing Strategy

### Unit Tests
- API client with mocked responses
- Parser with sample release notes
- Categorization logic with test cases
- Template rendering

### Integration Tests
- End-to-end changelog generation
- CHANGELOG.md update without corruption
- Multi-version range processing

### Manual Testing
- Generate changelog for recent updates
- Verify GitHub links work correctly
- Check markdown rendering in GitHub
- Validate categorization accuracy

---

## Rollout Plan

### Stage 1: Script Development
1. Develop and test scripts locally
2. Generate enhanced entries for recent versions
3. Review and refine categorization

### Stage 2: Historical Update
1. Run script on all existing changelog entries
2. Review generated content
3. Manual curation of important changes
4. Commit updated CHANGELOG.md

### Stage 3: Process Integration
1. Add to version update workflow
2. Document usage in CONTRIBUTING.md
3. Create pre-commit hook (optional)

### Stage 4: Automation
1. Set up GitHub Actions workflow
2. Automatic PR creation for updates
3. Scheduled checks for new llama.cpp releases

---

## Maintenance

### Ongoing Tasks
- Update categorization rules as patterns emerge
- Refine templates based on feedback
- Monitor GitHub API rate limits
- Cache management

### Quarterly Reviews
- Review changelog quality
- Update documentation
- Optimize performance
- Gather user feedback

---

## Success Metrics

1. **Completeness**: ≥90% of changes have descriptions
2. **Accuracy**: ≥95% correct categorization
3. **Readability**: User feedback positive
4. **Automation**: ≥80% entries generated automatically
5. **Timeliness**: Changelog updated within 24h of version bump

---

## Dependencies

### Python Libraries
```
requirements-changelog.txt:
  - requests>=2.31.0
  - PyGithub>=2.1.1
  - pyyaml>=6.0
  - jinja2>=3.1.2
  - python-dateutil>=2.8.2
  - click>=8.1.0  # for CLI
```

### External Services
- GitHub API (authenticated for higher rate limits)
- Optional: GitHub Actions for automation

### File Dependencies
- `vendor_llama_cpp_pydist/llama.cpp/` - Git repository
- `CHANGELOG.md` - Target file
- `.git/` - For git operations

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GitHub API rate limits | Script fails | Use authentication, implement caching |
| Release notes format changes | Parser breaks | Robust parsing, fallback to commit messages |
| Manual edits overwritten | Data loss | Preserve manual sections, use markers |
| Inaccurate categorization | Poor UX | Manual review, refinement, override file |
| Large changelog file | Performance | Archive old entries, pagination |

---

## Future Enhancements

### Phase 2 Features
- Interactive web viewer for changelog
- Searchable/filterable change history
- Diff viewer integration
- Release notes email generator

### Advanced Categorization
- ML-based categorization
- Impact scoring (minor/major/critical)
- User-facing vs internal changes

### Integration
- Auto-generate release notes for this package
- Slack/Discord notifications
- Blog post generation

---

## TODO List

### Immediate (This Week)
- [ ] Review and approve this plan
- [ ] Set up GitHub token for API access
- [ ] Create `scripts/changelog/` directory structure
- [ ] Initialize Git branch for implementation

### Short-term (Next 2 Weeks)
- [ ] Implement GitHub API client
- [ ] Build parser for release notes
- [ ] Create categorization engine
- [ ] Design and test markdown templates
- [ ] Develop CLI tool

### Medium-term (Next Month)
- [ ] Generate enhanced entries for recent versions
- [ ] Manual review and curation
- [ ] Update historical entries (optional)
- [ ] Document usage and workflow
- [ ] Integrate into version update process

### Long-term (Future)
- [ ] Automate with GitHub Actions
- [ ] Add advanced features (search, filter, export)
- [ ] Consider ML-based improvements
- [ ] Build web viewer (optional)

---

## References

### Helpful Resources
- [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

### Similar Projects
- [conventional-changelog](https://github.com/conventional-changelog/conventional-changelog)
- [release-please](https://github.com/googleapis/release-please)
- [auto-changelog](https://github.com/cookpete/auto-changelog)

---

**Document Version:** 1.0  
**Last Updated:** October 20, 2025  
**Next Review:** After Phase 1 completion
