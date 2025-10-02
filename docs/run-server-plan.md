# Run Server Feature – Plan & TODOs

This document outlines the plan to add a "quickly run servers" feature to this repo. The feature prepares model assets (download/convert/quantize), ensures binaries are available (compile on Linux/macOS; use prebuilt on Windows), and launches the llama.cpp server with sensible defaults. The flow must be idempotent and use directories relative to the repo for models and bins.

## Goals (from requirements)

1. If a Hugging Face model name is provided, download a GGUF model from HF.
2. If a GGUF file path is provided, use it directly.
3. If a safetensors directory is provided, convert to GGUF and quantize first.
4. Only do the needed step(s); don’t re-download or re-convert when artifacts already exist.
5. Install binaries if needed (Windows) or compile if on Linux/macOS.
6. Use directories relative to current repo for storing models and bins.

## User Experience (CLI)

We’ll add a single entrypoint that orchestrates preparation and runs the server.

- Python entrypoint: `python -m llama_cpp.run_server` (new module)
- Optional wrapper script: `scripts/run_server.sh` for convenience

Proposed arguments:

- Exactly one of the following model sources is required:
  - `--hf-model <org/model[:revision]>`
  - `--gguf-path </path/to/model.gguf>`
  - `--safetensors-dir </path/to/hf_dir_or_safetensors_dir>`

- Common server options (forwarded to llama.cpp server):
  - `--port 8080` (default 8080)
  - `--host 127.0.0.1`
  - `--threads <N>` (default: CPU count)
  - `--ctx-size <tokens>` (default: 4096)
  - `--gpu-layers <N>` (default: 0 on CPU)
  - `--device {cpu|cuda|metal|vulkan}` (auto-detect default)

- Conversion/quantization options (when converting safetensors):
  - `--quantization {q2_k|q3_k_m|q4_k_m|q5_k_m|q6_k|q8_0|f16}` (default: `q4_k_m`)
  - `--keep-f16` or explicit dtype flags as needed

## High-level Architecture

1. Resolve model source and target GGUF file path (decision layer):
   - If `--gguf-path`: verify file exists and is GGUF.
   - If `--hf-model`: query HF for GGUF assets; download if missing.
   - If `--safetensors-dir`: convert → produce GGUF; then quantize if requested.

2. Ensure llama.cpp binaries are available:
   - Linux/macOS: build from vendored `vendor_llama_cpp_pydist/llama.cpp` with CMake.
   - Windows: use prebuilt binaries from `llama_cpp/binaries/` (fallback to download if needed in the future).

3. Launch server binary with resolved GGUF, passing through relevant runtime flags.

4. Idempotency: all steps check for existing artifacts with markers and integrity checks; skip repeat work when nothing changed.

## Directory Layout (relative to repo root)

```
./models/
  gguf/
    <org>__<model>__<rev>/
      model.gguf            # downloaded or converted
      model.<quant>.gguf    # quantized variants
      .state.json           # metadata: source, sha256, timestamps, quant params
  safetensors/
    <org>__<model>__<rev>/  # optional local cache for source tensors

./bins/
  linux-x86_64/
    llama-server            # built from vendor
    quantize                # quantization tool
  macos-universal/
    llama-server
    quantize
  windows-x64/
    llama-server.exe        # when available, or main.exe fallback

./logs/
  run-server-<date>.log
```

Notes:
- Use normalized identifiers for `<org>__<model>__<rev>` to keep paths safe.
- If HF artifact already exists in `models/gguf/...`, re-use it.
- If converting from safetensors, place outputs in the same `models/gguf/...` bucket.

## Implementation Plan

### 1) CLI Orchestrator: `llama_cpp/run_server.py`

- Parse args and determine the source mode (HF vs GGUF path vs safetensors dir).
- Compute target model directory under `models/gguf/...` and planned output filenames.
- Compose a work plan:
  - Ensure binaries exist (build or use prebuilt)
  - Prepare GGUF (download or convert+quantize)
  - Run server
- Write `.state.json` in the model dir with details sufficient for idempotency.

### 2) Binaries – Build/Install

- Linux/macOS:
  - Use CMake in `vendor_llama_cpp_pydist/llama.cpp`
  - Prefer presets from `CMakePresets.json` if suitable; else use standard out-of-source build under `build/vendor/llama.cpp/<platform>`
  - Build targets: `llama-server` and `quantize` (and any required helper like `llama-cli` as fallback)
  - Detect CUDA/Metal/Vulkan availability to adjust build flags (basic heuristic; allow overrides via CLI env/flags)

- Windows:
  - Use prebuilt zip already in `llama_cpp/binaries/`
  - Extract into `./bins/windows-x64/` on first run; skip if already extracted

### 3) Model Preparation

- Case A: `--gguf-path`
  - Validate the file is present and readable; compute checksum; copy or symlink into `models/gguf/...` (optional) and record in `.state.json`.

- Case B: `--hf-model`
  - Use `huggingface_hub` to discover GGUF assets for the specified model/revision.
  - Select a file by heuristic (prefer quantized if user requested exact quant; else highest-quality within resource constraints or default policy)
  - Download to `models/gguf/<org>__<model>__<rev>/` with integrity checks, resume support, and progress.

- Case C: `--safetensors-dir`
  - Run the official convert script from vendor llama.cpp (e.g., `convert_hf_to_gguf.py`) or reuse project’s `convert_model.py` if compatible.
  - Produce a base FP16 or BF16 GGUF.
  - Quantize using the `quantize` tool to the requested format (e.g., `q4_k_m`).

### 4) Idempotency Strategy

- Before any work:
  - If `models/gguf/.../model.<quant>.gguf` exists and `.state.json` matches input config, skip prepare steps.
  - If `.state.json` shows prior source and checksum/revision unchanged, skip download/convert.
- After each step:
  - Update `.state.json` with:
    - `source`: {type: hf|gguf|safetensors, ref/path}
    - `revision` (HF) or source checksum(s)
    - `gguf_outputs`: list of produced files with size, sha256
    - `quant`: parameters
    - `bin_build`: platform, versions, build flags

### 5) Launch Server

- Use the resolved GGUF path and invoke `llama-server` (or fallback to `main` if needed) with user-specified runtime flags.
- Stream logs to stdout and a log file under `./logs/`.
- Provide clear startup summary (paths, device, threads, ctx size, listening URL).

## Acceptance Criteria

- Passes all 6 requirements:
  1) Given `--hf-model`, downloads GGUF to `./models/gguf/...` and runs server.
  2) Given `--gguf-path`, uses that file directly and runs server.
  3) Given `--safetensors-dir`, converts + quantizes to GGUF and runs server.
  4) Re-running with same inputs does not redo work; only starts server.
  5) On Linux/macOS, compiles bins if missing; on Windows, uses prebuilt.
  6) All artifacts stored under `./models` and `./bins`.

## Risks & Mitigations

- Model selection ambiguity on HF repos with many GGUFs → add `--pattern` or `--filename` override.
- GPU backends diversity → start with CPU default; detect or allow `--device` to switch.
- Conversion script compatibility → prefer vendored scripts; keep a thin shim for our CLI.

## Open Questions

- Which default quantization for conversion? Proposed: `q4_k_m`.
- Minimum required llama.cpp version? Use vendored commit; record build info in state.

---

## TODOs

- [ ] Create CLI orchestrator `llama_cpp/run_server.py` with arg parsing and routing.
- [ ] Add shared paths helper: resolve `./models`, `./bins`, `./logs` per platform.
- [ ] Implement `.state.json` read/write and integrity checks (sha256 helper).
- [ ] Implement binaries preparation:
  - [ ] Linux/macOS: CMake configure + build `llama-server` and `quantize`.
  - [ ] Windows: unzip prebuilt into `./bins/windows-x64/` if missing.
- [ ] Implement model prep:
  - [ ] GGUF path validation and optional import to cache.
  - [ ] HF GGUF discovery + download using `huggingface_hub`.
  - [ ] Safetensors → GGUF conversion via vendored scripts.
  - [ ] Quantization via `quantize` tool.
- [ ] Add server launch wrapper with process management and logfile writing.
- [ ] Add `scripts/run_server.sh` convenience wrapper (optional).
- [ ] Write minimal docs in `README.md` linking to this plan and showing quickstart.
- [ ] Add smoke test script for CI: prepare tiny model and run `--help` + short start.
- [ ] Gate errors with helpful messages and remediation tips.

## Nice-to-haves (later)

- [ ] Add environment variable overrides (e.g., `LLAMA_SERVER_BINS_DIR`).
- [ ] Cache busting flag `--force` to rebuild/re-download.
- [ ] Support containerized run (Dockerfile or devcontainer) for reproducible builds.
- [ ] Auto-pick smallest GGUF by RAM if none specified.
- [ ] Add telemetry-free, anonymized diagnostics flag `--print-diagnostics`.
