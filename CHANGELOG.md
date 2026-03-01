# Changelog

## 2026-03-01: Update to llama.cpp b8182

### Summary
Updated llama.cpp from b8087 to b8182, incorporating 76 upstream commits with breaking changes, new features, and performance improvements.

### Notable Changes

#### ⚠️ Breaking Changes
- **b8098**: models : dedup qwen35 graphs ([#19660](https://github.com/ggml-org/llama.cpp/pull/19660))
  - cont #19597
  - Use the new `struct llm_build_delta_net_base` to deduplicate the delta net graphs from Qwen35 models.
  - TODO:
- **b8101**: llama : use output_resolve_row() in get_logits_ith/get_embeddings_ith ([#19663](https://github.com/ggml-org/llama.cpp/pull/19663))
  - This commit updates get_logits_ith(), and get_embeddings_ith() to use output_resolve_row() to resolve the batch index to output row index.
  - The motivation for this is to remove some code duplication between these functions.
- **b8140**: hexagon refactor all Ops to use local context struct ([#19819](https://github.com/ggml-org/llama.cpp/pull/19819))
  - This PR completes the refactoring of all Hexagon Ops to use a local context structure. This allows each Op to precompute and cache more state. The refactoring also removes redundant function wrappers and unnecessary boilerplate.
  - Most Ops now use DMA for fetching inputs and writing back outputs.
  - The main loops of RoPE and Unary Ops have been completely rewritten for better DMA pipelining.
- **b8146**: ggml/gguf : prevent integer overflows ([#19856](https://github.com/ggml-org/llama.cpp/pull/19856))
  - Strengthen integer overflow validation in ggml/gguf
  - Impose max limits for string length and array elements of GGUF metadata
  - Remove deprecated `ggml_type_sizef()`

#### 🆕 New Features
- **b8091**: ggml webgpu: shader library organization ([#19530](https://github.com/ggml-org/llama.cpp/pull/19530))
  - We've been converting many of the existing WGSL shaders into a format that allows for efficient just-in-time compilation of variants used in specific model graphs, as well as sets them up for better performance tuning down the road. This PR makes a pretty large organizational change, moving the shader preprocessing, compilation, and caching into a new `ggml_webgpu_shader_lib` structure. As part of this, the existing matrix multiplication shaders were also converted in to the JIT compilation format (using the wgsl preprocessor), along with get_rows and scale.
  - This new shader library class also opens up the opportunity for tons of interesting specialization in the WebGPU backend. For example, if you have a shader specialized for a particular GPU vendor/architecture in WGSL, it should be pretty easy to hook it into the logic for choosing the right shader/pipeline.
  - It's always nice to have a PR that removes more lines of code than it adds too :)
- **b8091**: Add oneliner for batch quantization ([#17](https://github.com/ggml-org/llama.cpp/pull/17))
- **b8100**: full modern bert support ([#18330](https://github.com/ggml-org/llama.cpp/pull/18330))
  - Made support for conversion from hf->gguf and execution on llama.cpp after my recent (granite-embd-support)[https://github.com/ggml-org/llama.cpp/pull/15641] which is a modern bert based model, this pr continues off of that and has some tweaks. I have ran cosine similarity tests with this script
  - ```
  - from sentence_transformers import SentenceTransformer
- **b8102**: model : Add tokenizer from LFM2.5-Audio-1.5B ([#19687](https://github.com/ggml-org/llama.cpp/pull/19687))
  - [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) introduced a lightweight audio tokenizer.
  - It is based on the LFM2 architecture and serves as an embedding model with different input `n_embd` and output `n_embd_out`.
  - To be used in https://github.com/ggml-org/llama.cpp/pull/18641.
- **b8106**: model: add JAIS-2 architecture support ([#19488](https://github.com/ggml-org/llama.cpp/pull/19488))
  - Add support for the JAIS-2 family of Arabic-English bilingual models from Inception AI (https://huggingface.co/inceptionai/Jais-2-8B-Chat).
  - Architecture characteristics:
  - LayerNorm (not RMSNorm) with biases
- **b8106**: CUDA: fix padding of GQA to power of 2 in FA ([#19115](https://github.com/ggml-org/llama.cpp/pull/19115))
  - Fixes https://github.com/ggml-org/llama.cpp/issues/19112 , the issue was introduced with https://github.com/ggml-org/llama.cpp/pull/19092 .
  - The MMA CUDA FlashAttention kernel uses a stream-k decomposition to treat the four-dimensional input tensors as one continuous dimension to split across streaming multiprocessors. However, in conjunction with the GQA-specific optimizations in the MMA kernel this is only correct if the number of Q columns per CUDA block exactly divide `n_gqa`. Otherwise the wrong Q and K/V heads will be associated and the result will be wrong (if there is only a single K/V head this doesn't matter so it was not detected in testing).
  - This PR extends the 4D space on master to a 5D space by splitting the "z" dimension with the number of Q heads into one dimension for the number of K/V heads and another dimension for the number of Q heads per K/V head. This then makes it possible to simply pad the Q columns per CUDA block to a power of 2.
- **b8116**: ggml-quants : weighted rounding algorithms with cumulative search ([#12557](https://github.com/ggml-org/llama.cpp/pull/12557))
  - This adds proper `imatrix` support to `TQ1_0` and `TQ2_0`, in addition to improving the rounding algorithm used for `Q3_K`, `IQ4_NL`, `IQ4_XS` (both with and without `imatrix`), as well as when using `imatrix` with `Q4_0` and `Q5_0`.
  - **This is backward *and* forward compatible with other versions of `llama.cpp`**.
  - Since this doesn't change the format of the types, only how the values are rounded when quantized, even previous (or current) versions of `llama.cpp` can use quants made with this PR.
- **b8117**: ggml-cpu: add RVV vec dot kernels for quantization types ([#18784](https://github.com/ggml-org/llama.cpp/pull/18784))
  - This PR adds RVV vector dot kernels for a number of quantization types.
  - Added the following RVV kernels:
  - | Kernel | VLEN |
- **b8118**: common : merge qwen3-coder and nemotron nano 3 parsers ([#19765](https://github.com/ggml-org/llama.cpp/pull/19765))
  - Users are experiencing several issues with Qwen3-Coder-Next. Until #18675 is merged in, this PR serves as a stop-gap by replacing the existing Qwen3-Coder parsing with the Nemotron Nano 3 PEG parsing variant already present.
  - This PR also adds parallel tool calling and fixes JSON schema support.
  - fixes #19382
- **b8123**: Add a build target to generate ROCm artifacts using ROCm 7.2 ([#19433](https://github.com/ggml-org/llama.cpp/pull/19433))
  - This builds the following targets:
  - gfx1151
  - gfx1150
- **b8128**: model: Add Kanana-2 model support ([#19803](https://github.com/ggml-org/llama.cpp/pull/19803))
  - *Make sure to read the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) before submitting a PR*
  - This PR adds support for following Kanana-2 model family:
  - [kakaocorp/kanana-2-30b-a3b-instruct-2601](https://huggingface.co/kakaocorp/kanana-2-30b-a3b-instruct-2601)
- **b8131**: jinja: correct stats for tojson and string filters ([#19785](https://github.com/ggml-org/llama.cpp/pull/19785))
  - Target fix https://github.com/ggml-org/llama.cpp/pull/18675
  - @pwilkin please give this a try (see the added test case for more info)
- **b8142**: vulkan: fix coopmat1 without bf16 support ([#19793](https://github.com/ggml-org/llama.cpp/pull/19793))
  - This should fix the CI failure on lavapipe. lavapipe added coopmat1 support recently, but does not have bf16 support, so it falls back to the scalar path. This fallback didn't have quite the same tile size logic for subgroupsize=8 as when going through the scalar path directly.
- **b8143**: Vulkan Scalar Flash Attention Refactor ([#19625](https://github.com/ggml-org/llama.cpp/pull/19625))
  - This started out as an attempt to go through the scalar FA version and add proper float16 support to improve AMD and Intel performance and went quite a bit further. @jeffbolznv Sorry about the amount of changes, let me know if there's something I can do to make the review easier. Please also let me know if you have architectural concerns. Flash Attention has so many dimensions and making it work well on so much hardware and models is pretty hard. I had to spend quite a lot of time figuring out and fixing regressions on specific configurations.
  - <details>
  - <summary>AI-generated summary of changes</summary>
- **b8149**: gguf : fix ftell/fseek for Windows ([#19870](https://github.com/ggml-org/llama.cpp/pull/19870))
  - Regression introduced in #19856.
  - This changes the `ftell/fseek` calls to use `_ftelli64/_fseeki64` on Windows, and `ftello/fseeko` for POSIX systems.
  - `long` on Windows is always 32-bit. Since that would cause an overflow on large files, `ftell/fseek` fails and `nbytes_remain()` returns `0`.
- **b8155**: common : add more aliases for sampler CLI params ([#19797](https://github.com/ggml-org/llama.cpp/pull/19797))
  - Adds two CLI argument aliases for sampler parameters:
  - `--top-n-sigma` (for existing `--top-nsigma`)
  - `--temperature` (for existing `--temp`)
- **b8161**: jinja : correct default size for string slices ([#19913](https://github.com/ggml-org/llama.cpp/pull/19913))
  - *Make sure to read the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) before submitting a PR*
  - As of b8157, when trying to use string slices in a chat template, and the slice does not specify end index (e.g. `content[1 : ]`), no output will be emitted since the default end index is calculated only for arrays, and remains 0 for strings. This PR adds handling for strings, and should be complete for currently supported data types.
  - <details>
- **b8164**: llama: Add option to merge gate and exp weights ([#19139](https://github.com/ggml-org/llama.cpp/pull/19139))
  - Continuing on #18740 and #18866, add option `--fuse_gate_up_exps` to `convert_hf_to_gguf.py`.
  - I've just added the gate_up tracking for deepseek2 (GLM 4.7 flash) and gpt-oss - although for gpt-oss we need even more changes (it goes through the `generate_extra_tensors` for generating expert weights). This PR is not complete as we would need to add this check in all MoE models and their tensors, but putting it out there in any case.
  - on 5090:
- **b8165**: kv-cache : fix can_shift() check to take into account M-RoPE ([#19928](https://github.com/ggml-org/llama.cpp/pull/19928))
  - fix #19915
  - KV cache shift is not supported with M-RoPE (yet).
- **b8169**: ggml : fix AMX and add batched support ([#19925](https://github.com/ggml-org/llama.cpp/pull/19925))
  - llama-perplexity -hf ggml-org/Qwen3-0.6B-GGUF:Q4_0 -f wikitext-2-raw/wiki.test.raw -c 2048 -b 2048 --chunks 2
  - before this commit:
  - ```
- **b8175**: ggml-cpu: add repack for mxfp4 ([#19738](https://github.com/ggml-org/llama.cpp/pull/19738))
  - This is just a faithful copy of the `iq4_nl` quant to mxfp4 with just the scale loading changed. Tested on AVX2 only, would appreciate tests on ARM and AVX512. Perplexity is already high for gpt-oss-20b but I see it is the same between master and this branch
  - | Model                 | Test   |   t/s master |   t/s mxfp4-repack-cpu |   Speedup |
  - |:----------------------|:-------|-------------:|-----------------------:|----------:|
- **b8179**: CUDA: add CDNA3 MFMA support for flash attention MMA kernel ([#19806](https://github.com/ggml-org/llama.cpp/pull/19806))
  - Adds MI300X (gfx942) MFMA tensor core flash attention to `fattn-mma-f16.cuh`. MI300X now routes to `BEST_FATTN_KERNEL_MMA_F16` instead of the tile-based fallback.
  - Uses `v_mfma_f32_16x16x16_f16` (FP16 inputs, FP32 accumulate) with wavefront64
  - Supports head sizes 64, 80, 96, 112, 128 via MMA; others fall back to VEC
- **b8180**: Add model metadata loading from huggingface for use with tests requiring real model data ([#19796](https://github.com/ggml-org/llama.cpp/pull/19796))
  - This is based on the work from huggingface here:
  - https://github.com/huggingface/huggingface.js/tree/main/packages/gguf
  - Idea is to partially load GGUF models from huggingface, just enough to get the metadata

#### 🚀 Performance Improvements
- **b8087**: opencl: refactor expm1 and softplus ([#19404](https://github.com/ggml-org/llama.cpp/pull/19404))
  - This PR refactors the EXPM1 and Softplus OpenCL operators to improve code clarity and reduce duplication.
- **b8099**: powerpc: add FP16 MMA path for Q4/Q8 matmul ([#19709](https://github.com/ggml-org/llama.cpp/pull/19709))
  - Avoid xvi8ger4pp signed→unsigned bias correction by dequantizing Q4/Q8 inputs to FP16 and using FP16×FP16→FP32 MMA. This removes post-processing overhead and improves performance.
  - Performance Impact:
  - 1.5 ~ 2x improvement in PP_Speed for Q4 and Q8 Models, measured with llama-bench and llama-batched-bench. Q8 Model: granite-4.0-h-micro-Q8_0.gguf (from huggingface) Q4 Model: Meta-Llama3-8b Q4 model (generated with llama-quantize from f32 model)
- **b8121**: Improve CUDA graph capture ([#19754](https://github.com/ggml-org/llama.cpp/pull/19754))
  - Currently, CUDA graphs are eagerly enabled on the first call to ggml_backend_cuda_graph_compute. If the graph properties keep changing (4+ consecutive updates), the graph is permanently disabled. This is suboptimal because:
  - The first call always incurs CUDA graph capture overhead even if the graph is unstable
  - Once permanently disabled, CUDA graphs never re-enable even after the graph stabilizes (e.g., switching from prompt processing to decode)

#### 🐛 Bug Fixes
- **b8088**: common : make small string helpers as inline functions ([#19693](https://github.com/ggml-org/llama.cpp/pull/19693))
  - Also use string_view when it make sense and fix some corner cases.
- **b8089**: vulkan: split mul_mat into multiple dispatches to avoid overflow ([#19509](https://github.com/ggml-org/llama.cpp/pull/19509))
  - The batch dimensions can be greater than the max workgroup count limit, in which case we need to split into multiple dispatches and pass the base index through a push constant.
  - Fall back for the less common p021 and nc variants.
  - Fixes #19471.
- **b8095**: ggml webgpu: Fix bug in dispatching large matrix-vector multiplication ([#19535](https://github.com/ggml-org/llama.cpp/pull/19535))
  - Bug fix for calculating overflowing workgroup sizes for large matrix-vector multiplication batches. Should fix failures from new tests in https://github.com/ggml-org/llama.cpp/pull/19519.
  - This approach isn't ideal because it may over-provision workgroups by quite a bit, a better strategy is the one proposed for Vulkan in https://github.com/ggml-org/llama.cpp/pull/19509, but this will work for now.
- **b8105**: CUDA: fix kernel selection logic for tile FA ([#19686](https://github.com/ggml-org/llama.cpp/pull/19686))
  - Fixes https://github.com/ggml-org/llama.cpp/issues/19652 .
  - The problem is that the kernel selection logic is slightly wrong so the host code runs into an abort.
- **b8109**: vulkan: fix MMQ shader push constants and multi-dispatch ([#19732](https://github.com/ggml-org/llama.cpp/pull/19732))
  - We forgot to update the mul_mmq shader in #19509. This should fix #19710.
- **b8112**: common : fix gpt-oss Jinja error with content and thinking on tool-call messages ([#19704](https://github.com/ggml-org/llama.cpp/pull/19704))
  - Erase the `content` from the adjusted message after copying `reasoning_content` to `thinking`.
  - Regression from #16937
  - Fixes #19703.
- **b8113**: common : fix Step-3.5-Flash format detection and thinking support ([#19635](https://github.com/ggml-org/llama.cpp/pull/19635))
  - Step-3.5-Flash (196B MoE) uses the same XML tool call output format as Qwen3-Coder and Nemotron 3 Nano (\`<tool_call><function=...><parameter=...>\`), but its template lacks the bare \`<function>\` and plural \`<parameters>\` markers in the tool enumeration section. The previous detection logic required all five XML markers, so Step-3.5-Flash fell through to Hermes 2 Pro, which doesn't call \`func_args_not_string()\`. Tool arguments stayed as JSON strings and templates using \`arguments|items\` crashed.
  - Reported by multiple users in #19283:
  - [Leaked tool tokens with Codex](https://github.com/ggml-org/llama.cpp/pull/19283#issuecomment-3839920985) (@tarruda)
- **b8115**: test: mul_mat tests with huge batch size ([#19519](https://github.com/ggml-org/llama.cpp/pull/19519))
  - tests for #19471.
  - vulkan fix is in #19509.
- **b8119**: hexagon : fix build release (#19444) ([#19587](https://github.com/ggml-org/llama.cpp/pull/19587))
  - fixes: #19444
  - cc: @max-krasnyansky
- **b8130**: common : fix improper trimming in XML parser on complete message ([#19805](https://github.com/ggml-org/llama.cpp/pull/19805))
  - Fix courtesy of @julio75012. Although his use case has already been fixed, I'm submitting this PR to address other models that exhibit similar behavior.
  - The issue is that the XML parser trims partially matched tags. The reason `>` was trimmed from Seed-OSS is because `tool_sep = >`, and the reason a trailing `"` is trimmed from MiniMax/Kimi-K2 is because `tool_sep = ">`. This trimming should only happen when the message is still partial. Once the full message has been received, no trimming should occur.
  - Fixes #19795
- **b8141**: vulkan: fix data race in mul_mat_id shader ([#19790](https://github.com/ggml-org/llama.cpp/pull/19790))
  - I've been working on automated data race detection (see https://github.com/KhronosGroup/Vulkan-ValidationLayers/pull/11717), and it found a data race in the mul_mat_id shaders. All invocations in a subgroup were storing the same value to shared memory, but this is still technically a data race. Just store on the first invocation.
- **b8148**: models : fix graph splits ([#19866](https://github.com/ggml-org/llama.cpp/pull/19866))
  - fix #19860
  - fix #19864
  - Ensure the node order of Qwen 3.5 graphs is suitable for multi-GPU systems.
- **b8156**: vulkan: check for memory overlap before doing fusion ([#19768](https://github.com/ggml-org/llama.cpp/pull/19768))
  - This fixes a class of potential fusion bugs where the destination could overwrite a source tensor while other elements of the same op still need those source values. Add some logic to compare the memory ranges and disable fusion if the bad case is detected. Some operations contribute to the destination in an elementwise fashion and can do a more relaxed check where exact overlap is allowed.
  - In practice, I see this disabling TOPK_MOE fusion in some models (gpt-oss, qwen3) when there's more than one row, and this does appear to be a latent bug.
- **b8157**: [SYCL] Fix binbcast.cpp:200: GGML_ASSERT(s10 == 1) failed of Qwen3-Coder-Next-Q3_K_M.gguf ([#19889](https://github.com/ggml-org/llama.cpp/pull/19889))
  - Fix issue: https://github.com/ggml-org/llama.cpp/issues/19779
  - The PR (1725e316c models : optimize qwen3next graph) lead to the OP shape is changed and lead to assert.
  - In binbcast ops:
- **b8159**: gguf : avoid too many file size calls ([#19919](https://github.com/ggml-org/llama.cpp/pull/19919))
  - cont #19856
  - fix #19912
  - No need to do file calls on each read. Instead, determine the remaining bytes once at the start and after that update the value on each read.
- **b8168**: vulkan: fix fp16 Flash Attention on Windows AMD RDNA2 and below ([#19921](https://github.com/ggml-org/llama.cpp/pull/19921))
  - For some reason a f16vec4 subgroupShuffleXor is broken on RDNA2 and lower. I found a workaround by shuffling vec4 instead. This also fixes fp16 Flash Attention on AMD GCN, so I removed the fp32 fallback.
  - Fixes #19881 and also the issue reported here: https://github.com/ggml-org/llama.cpp/pull/19625#issuecomment-3940674420
  - @masamaru-san @DeryabinIvan Please try this fix and let me know if it works for you.
- **b8171**: [SYCL] Replace the magic nunber 768 by max work group size to support iGPU ([#19920](https://github.com/ggml-org/llama.cpp/pull/19920))
  - Fix issue: https://github.com/ggml-org/llama.cpp/issues/19886
- **b8172**: [CMake] Enable test-chat out of tree build ([#19558](https://github.com/ggml-org/llama.cpp/pull/19558))
  - The test-chat binary relies on model files that it tries to find. However, when configuring the build directory to be parallel to the source tree those heuristics fail.
  - This sets the working directory for the test executable to be the source-tree which resolves this issue.
  - I validated locally with a build parallel to the source tree and nested inside the source tree.
- **b8182**: vendors: update miniaudio library to 0.11.24 ([#19914](https://github.com/ggml-org/llama.cpp/pull/19914))
  - https://github.com/mackron/miniaudio/releases/tag/0.11.24.
  -   Fixed a possible glitch when processing the audio of a `ma_sound` when doing resampling.
  -   Fixed a possible crash in the node graph relating to scheduled starts and stops.


### Additional Changes
27 minor improvements: 3 documentation, 19 examples, 5 maintenance.

### Full Commit Range
- b8087 to b8182 (76 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b8087...b8182

---

## 2026-02-18: Update to llama.cpp b8087

### Summary
Updated llama.cpp from b8053 to b8087, incorporating 28 upstream commits with breaking changes, new features, and performance improvements.

### Notable Changes

#### ⚠️ Breaking Changes
- **b8057**: ggml-cpu: FA add GEMM microkernel ([#19422](https://github.com/ggml-org/llama.cpp/pull/19422))
  - This PR contains the following improvements for the tiled FA kernel
  - Add a simd gemm for float32 in the tiled FA kernel.
  - Tune tile sizes for larger context
- **b8075**: Remove annoying warnings (unused functions) ([#18639](https://github.com/ggml-org/llama.cpp/pull/18639))
  - When using common.h as a library, these function produce annoying warnings about not being used.
  - Using "static" linking for these also doesn't make much sense because it potentially increases executable size with no gains.

#### 🆕 New Features
- **b8059**: ggml : avoid UB in gemm ukernel + tests ([#19642](https://github.com/ggml-org/llama.cpp/pull/19642))
  - cont #19422
  - Reword the GEMM ukernel to not trip the compiler's aggressive loop optimization warnings. It's better to avoid the global pragma as it might be useful for other static analysis
  - Add `test-backend-ops` with BS=75 to exercise the new tiled SIMD implementation
- **b8061**: cmake : check if KleidiAI API has been fetched ([#19640](https://github.com/ggml-org/llama.cpp/pull/19640))
  - This commit addresses a build issue with the KleidiAI backend when building multiple cpu backends. Commmit
  - 3a00c98584e42a20675b6569d81beadb282b0952 ("cmake : fix KleidiAI install target failure with EXCLUDE_FROM_ALL") introduced a change where FetchContent_Populate is called instead of FetchContent_MakeAvailable, where the latter does handle this case (it is idempotent but FetchContent_Populate is not).
  - I missed this during my review and I should not have commited without verifying the CI failure, sorry about that.
- **b8068**: ggml: aarch64: Implement SVE in Gemm q4_k 8x8 q8_k Kernel  ([#19132](https://github.com/ggml-org/llama.cpp/pull/19132))
  - This PR introduces support for SVE (Scalable Vector Extensions) kernels for the q4_K_q8_K gemm using i8mm and vector instructions. ARM Neon support for this kernel added in PR [#16739](https://github.com/ggml-org/llama.cpp/pull/16739)
  - **Verifying Feature**
  - `----------------------------------------------------------------------------`
- **b8070**: models : deduplicate delta-net graphs for Qwen family ([#19597](https://github.com/ggml-org/llama.cpp/pull/19597))
  - cont #19375
  - Add `llm_build_delta_net_base` for common delta net builds. Currently used only by `qwen3next`
  - Rename `llm_graph_context_mamba` -> `llm_build_mamba_base`
- **b8071**: Adjust workaround for ROCWMMA_FATTN/GFX9 to only newer ROCm veresions ([#19591](https://github.com/ggml-org/llama.cpp/pull/19591))
  - Avoids issues with ROCm 6.4.4.
  - Closes: https://github.com/ggml-org/llama.cpp/issues/19580
  - Fixes: 6845f7f87 ("Add a workaround for compilation with ROCWMMA_FATTN and gfx9 (#19461)")
- **b8073**: Add support for Tiny Aya Models ([#19611](https://github.com/ggml-org/llama.cpp/pull/19611))
  - This PR adds native support for the CohereLabs/tiny-aya family of models in llama.cpp. These models use a distinct BPE pre-tokenizer (tiny_aya) with a custom digit-grouping regex.
  - Tagging @ngxson for visibility.
- **b8076**: feat: add proper batching to perplexity ([#19661](https://github.com/ggml-org/llama.cpp/pull/19661))
  - This PR updates `llama-perplexity` to allow for batching similarly to how `llama-imatrix` works. The idea being that you can increase `--batch-size` / `--ubatch-size` to process multiple contexts chunks in a batch. This has limited application in VRAM-rich environments (eg, if you're running the entire model in VRAM) but it makes a huge difference when using models in a mixed CPU/GPU setup as it saves `n_seq` trips from the CPU RAM to GPU VRAM per batch.
  - I've double-checked the before and after to make sure the resulting PPL and KLD look correct still.
  - <details>
- **b8077**: convert_hf_to_gguf: add JoyAI-LLM-Flash tokenizer hash mapping to deepseek-v3 ([#19651](https://github.com/ggml-org/llama.cpp/pull/19651))
  - adding hash for `jdopensource/JoyAI-LLM-Flash` mapping to existing `deepseek-v3`
  - `DeepseekV3ForCausalLM` architecture already supported
  - moved `GLM-4.7-Flash` entry together with the other `glm` entries

#### 🚀 Performance Improvements
- **b8053**: models : optimizing qwen3next graph ([#19375](https://github.com/ggml-org/llama.cpp/pull/19375))
  - Rewording the ggml compute graph to avoid too many unnecessary copies.
  - M2 Ultra:
  - | Model                    | Test   |   t/s b7946 |   t/s gg/qwen3-next-opt |   Speedup |
- **b8058**: ggml-cpu: optimize ggml_vec_dot_bf16 for s390x ([#19399](https://github.com/ggml-org/llama.cpp/pull/19399))
  - Similar to #18837, this pull request integrates the SIMD instruction set for BF16 on the s390x platform. We notice a 154.86% performance improvement for Prompt Processing. No performance difference was noticed for Token Generation.
  - | model                          |       size |     params | backend    | threads | mmap |            test |                  t/s |
  - | ------------------------------ | ---------: | ---------: | ---------- | ------: | ---: | --------------: | -------------------: |
- **b8064**: cuda: optimize iq2xxs/iq2xs/iq3xxs dequantization ([#19624](https://github.com/ggml-org/llama.cpp/pull/19624))
  - While looking over quantizations I believe I found a few optimizations for iq2xxs/iq2xs/iq3xxs. With these changes, I get a 5-10% increase in flops in `test-backend-ops` for small `n`, and a few extra flops otherwise:
  - load all 8 int8 for a grid position in one load
  - calculate signs via popcnt instead of fetching from ksigns table
- **b8086**: opencl: optimize mean and sum_row kernels ([#19614](https://github.com/ggml-org/llama.cpp/pull/19614))
  - This PR optimizes the mean op and sum_rows op for the OpenCL backend.
- **b8087**: opencl: refactor expm1 and softplus ([#19404](https://github.com/ggml-org/llama.cpp/pull/19404))
  - This PR refactors the EXPM1 and Softplus OpenCL operators to improve code clarity and reduce duplication.

#### 🐛 Bug Fixes
- **b8056**: cmake: fix KleidiAI install target failure with EXCLUDE_FROM_ALL ([#19581](https://github.com/ggml-org/llama.cpp/pull/19581))
  - Fix for the bug #19501 by adding `EXCLUDE_FROM_ALL` to the `FetchContent_Declare` call for KleidiAI. This properly excludes the KleidiAI library from both the `all` and `install` targets, preventing CMake install failures when building with `GGML_CPU_KLEIDIAI=ON`. The KleidiAI source files are still compiled directly into `libggml-cpu.so`, so functionality is preserved.
- **b8060**: context : fix output reorder with backend sampling ([#19638](https://github.com/ggml-org/llama.cpp/pull/19638))
  - fix #19629
  - Some of the sampling arrays could remain in invalid state after a sequence of enabling/disabling samplers.
- **b8069**: graph : fix KQ mask, lora, cvec reuse checks ([#19644](https://github.com/ggml-org/llama.cpp/pull/19644))
  - cont #14482
  - Graph reuse was never triggered for parallel decoding with non-unified KV cache due to incorrect check of the KQ mask shape.
  - Also fix the checks for reusing lora and control vectors.
- **b8071**: Add a workaround for compilation with ROCWMMA_FATTN and gfx9 ([#19461](https://github.com/ggml-org/llama.cpp/pull/19461))
  - There is an upstream problem [1] with AMD's LLVM 22 fork and rocWMMA 2.2.0 causing compilation issues on devices without native fp16 support (CDNA devices).
  - The specialized types aren't resolved properly:
  - ```c
- **b8083**: ggml: ggml-cpu: force-no-lto-for-cpu-feats ([#19609](https://github.com/ggml-org/llama.cpp/pull/19609))
  - When LTO enabled in build environments it forces all builds to have LTO in place. But feature detection logic is fragile, and causing Illegal instruction errors with lto. This disables LTO for the feature detection code to prevent cross-module optimization from inlining architecture-specific instructions into the score function. Without this, LTO can cause SIGILL when loading backends on older CPUs (e.g., loading power10 backend on power9 crashes before feature check runs).
  - Please also see https://salsa.debian.org/deeplearning-team/ggml/-/merge_requests/6 for more information about the issue we saw on ppc64el builds with LTO enabled in ubuntu.
  - *Make sure to read the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) before submitting a PR*


### Additional Changes
8 minor improvements: 1 documentation, 3 examples, 4 maintenance.

### Full Commit Range
- b8053 to b8087 (28 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b8053...b8087

---

## 2026-02-14: Update to llama.cpp b8040

### Summary
Updated llama.cpp from b8027 to b8040, incorporating 11 upstream commits with breaking changes, new features, and performance improvements.

### Notable Changes

#### ⚠️ Breaking Changes
- **b8027**: llama : remove deprecated codecvt ([#19565](https://github.com/ggml-org/llama.cpp/pull/19565))
  - Using the same conversion function ensures a consistent matching between the regex pattern and the text
- **b8037**: common : update download code ([#19573](https://github.com/ggml-org/llama.cpp/pull/19573))
  - This PR removes the legacy migration code for etag and forces a download if no etag file is found.

#### 🆕 New Features
- **b8028**: Kimi Linear fix conv state update ([#19531](https://github.com/ggml-org/llama.cpp/pull/19531))
  - *Make sure to read the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) before submitting a PR*
  - The current implementation has incorrect conv state update such that it has state corruption when running parallel in llama-server. This is fixed in this PR.
  - ```
- **b8030**: CUDA: Do not mutate cgraph for fused ADDs ([#19566](https://github.com/ggml-org/llama.cpp/pull/19566))
  - 1. We should try to minimize in-place changes to the incoming ggml_cgraph where possible (those should happen in a backends' `graph_optimize` function)
  - 2. Modifying in-place leads to an additional, unnecessary graph capture step as we store the properties before modifying the graph in-place in the cuda-backend: We hit `ggml_cuda_graph_node_set_properties` via `ggml_cuda_graph_update_required` before entering `ggml_cuda_graph_evaluate_and_capture`.
  - Isolated from #19521
- **b8036**: model: support GLM MoE DSA arch (NOTE: indexer is not yet supported) ([#19460](https://github.com/ggml-org/llama.cpp/pull/19460))
  - Ref upstream vllm PR: https://github.com/vllm-project/vllm/pull/34124
  - > [!IMPORTANT]
  - > This PR allows converting safetensors to GGUF while keeping the indexer tensors (for deepseek sparse attention), but they are left unused by the cpp code. **The quality will be suboptimal**

#### 🚀 Performance Improvements
- **b8038**: vulkan: restore -inf check in FA shaders ([#19582](https://github.com/ggml-org/llama.cpp/pull/19582))
  - For #19523.
  - I verified the performance is restored with llama-batched-bench.
- **b8040**: hexagon: further optimizations and refactoring for flash attention ([#19583](https://github.com/ggml-org/llama.cpp/pull/19583))
  - The PR includes some more refactoring and optimizations for flash attention op/kernel:
  - Local fa_context that stores all precomputed values
  - More HVX usage (hvx_vec_expf, ...)

#### 🐛 Bug Fixes
- **b8034**: fix vulkan ggml_acc only works in 3d but not 4d ([#19426](https://github.com/ggml-org/llama.cpp/pull/19426))
  - *Make sure to read the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) before submitting a PR*
  - Discovered ggml_acc for vulkan only works in 3d not 4d while working on
  - https://github.com/ggml-org/llama.cpp/pull/18792
- **b8035**: ggml-cpu: arm64: Fix wrong memcpy length for q4_K block_interleave == 4 ([#19575](https://github.com/ggml-org/llama.cpp/pull/19575))
  - https://github.com/ggml-org/llama.cpp/issues/19561 reports issues with the stack for Q4_K.
  - I can't reproduce the issue locally, but the `make_block_q4_Kx8` function would write past the buffer size 4 extra bytes,  which could be the issue.
  - @taronaeo, since you found the problem, are you able to check if this patch fixes it?


### Additional Changes
2 minor improvements: 1 examples, 1 maintenance.

- **b8033**: cli : support --verbose-prompt ([#19576](https://github.com/ggml-org/llama.cpp/pull/19576))
  - Useful when debugging templates.
- **b8032**: CUDA: loop over ne2*ne3 in case it overflows ([#19538](https://github.com/ggml-org/llama.cpp/pull/19538))

### Full Commit Range
- b8027 to b8040 (11 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b8027...b8040

---

## 2026-02-13: Update to llama.cpp b8018

### Summary
Updated llama.cpp from b7958 to b8018, incorporating 44 upstream commits with breaking changes and new features.

### Notable Changes

#### ⚠️ Breaking Changes
- **b8004**: common : remove unused token util functions ([#19506](https://github.com/ggml-org/llama.cpp/pull/19506))
  - This commit removes two unused functions `common_lcp` and `common_lcs`. The last usage of these functions was removed in Commit 33eff4024084d1f0c8441b79f7208a52fad79858 ("server : vision support via libmtmd") and are no longer used anywhere in the codebase.
- **b8007**: common : replace deprecated codecvt using parse_utf8_codepoint ([#19517](https://github.com/ggml-org/llama.cpp/pull/19517))

#### 🆕 New Features
- **b7964**: Support Step3.5-Flash ([#19283](https://github.com/ggml-org/llama.cpp/pull/19283))
  - This PR adds support for the Step3.5-Flash model architecture.
  - github:
  - https://github.com/stepfun-ai/Step-3.5-Flash/tree/main
- **b7966**: metal : consolidate bin kernels ([#19390](https://github.com/ggml-org/llama.cpp/pull/19390))
  - Refactor and consolidate the implementation of the binary Metal kernels.
  - | Model                    | Test   |   t/s master |   t/s gg/metal-bin-opt |   Speedup |
  - |:-------------------------|:-------|-------------:|-----------------------:|----------:|
- **b7972**: CUDA: Fix non-contig rope ([#19338](https://github.com/ggml-org/llama.cpp/pull/19338))
  - This is a port of https://github.com/ggml-org/llama.cpp/pull/19299 to the CUDA backend, which should fix the broken logic revealed by tests added in https://github.com/ggml-org/llama.cpp/pull/19296
  - Thanks @jeffbolznv for the work in #19299
- **b7973**: [Model] Qwen3.5 dense and MoE support (no vision) ([#19435](https://github.com/ggml-org/llama.cpp/pull/19435))
  - I've gotten a bit tired of Llama.cpp missing all the zero-day releases, so this time I decided to make (or, more precisely, instructed Opus 4.6 to make, based on reference implementations and my guidelines for model adaptation) a conversion based on the Transformers PR ( https://github.com/huggingface/transformers/pull/43830/changes ). It's mostly based on Qwen3Next, but it's rebased on the common-delta-net PR ( #19125 ).
  - Here are the mock models I generated to test it: https://huggingface.co/ilintar/qwen35_testing/tree/main
  - Here are the conversion results from `causal-verify-logits`:
- **b7974**: cmake: add variable to skip installing tests ([#19370](https://github.com/ggml-org/llama.cpp/pull/19370))
  - When packaging downstream, there's usually little point in installing test. The default behaviour remains the same.
- **b7976**: [Model] Qwen3.5 dense and MoE support (no vision) ([#19435](https://github.com/ggml-org/llama.cpp/pull/19435))
  - I've gotten a bit tired of Llama.cpp missing all the zero-day releases, so this time I decided to make (or, more precisely, instructed Opus 4.6 to make, based on reference implementations and my guidelines for model adaptation) a conversion based on the Transformers PR ( https://github.com/huggingface/transformers/pull/43830/changes ). It's mostly based on Qwen3Next, but it's rebased on the common-delta-net PR ( #19125 ).
  - Here are the mock models I generated to test it: https://huggingface.co/ilintar/qwen35_testing/tree/main
  - Here are the conversion results from `causal-verify-logits`:
- **b7976**: revert : "[Model] Qwen3.5 dense and MoE support (no vision) (#19435)" ([#19453](https://github.com/ggml-org/llama.cpp/pull/19453))
  - cont #19435
  - Taking a step back to implement support for Qwen3.5 properly.
- **b7981**: chat: fix case where template accepts type content only ([#19419](https://github.com/ggml-org/llama.cpp/pull/19419))
  - Fix chat template of PaddleOCR-VL, which requires content to be an array (see https://github.com/ggml-org/llama.cpp/pull/18825)
  - This should be able to handle these case:
  - Template supports ONLY string content
- **b7982**: cuda : extend GGML_OP_PAD to work with non-cont src0 ([#19429](https://github.com/ggml-org/llama.cpp/pull/19429))
  - Extend CUDA support
  - Remove redundant assert in CPU implementation
  - Add permuted PAD tests
- **b7983**: CANN: Support MUL_MAT_ID in ACL graph ([#19228](https://github.com/ggml-org/llama.cpp/pull/19228))
  - Implement ggml_cann_mul_mat_id_quant function to support quantized matrix
  - multiplication for Mixture of Experts (MoE) architectures on CANN backend.
  - Key features:
- **b7988**: ggml-cpu: arm64: q6_K repack gemm and gemv (and generic) implementations (dotprod) ([#19360](https://github.com/ggml-org/llama.cpp/pull/19360))
  - https://github.com/ggml-org/llama.cpp/pull/19356 but Q6_K.
  - PR contents:
  - New generics for q6_K_8x4
- **b7991**: [WebGPU] Plug memory leaks and free resources on shutdown ([#19315](https://github.com/ggml-org/llama.cpp/pull/19315))
  - This diff destroys `wgpu::Buffer`s and buffer pools on shutdown. It also fixes memory leaks on the heap, where we allocate `backend`, `backend_ctx`, `buffer_ctx`, and `decisions` on the heap but never delete them. These are either explicitly deleted or changed to be smart pointers.
  - We implement destructors for our buffer pool structs, `webgpu_context` struct and `webgpu_global_context` struct. Since `webgpu_global_context` is a refcounted smart pointer, it will destruct automatically when all thread contexts have been destroyed.
  - <img width="1191" height="220" alt="Screenshot 2026-02-03 at 3 56 11 PM" src="https://github.com/user-attachments/assets/3810b613-4920-4388-bdff-94ef306e8a06" />
- **b7992**: CUDA: Update CCCL-tag for 3.2 to final release from RC ([#19486](https://github.com/ggml-org/llama.cpp/pull/19486))
  - [CCCL 3.2 has been released](https://github.com/NVIDIA/cccl/releases/tag/v3.2.0
  - ) since it was added to llama.cpp as part of the backend-sampling PR (#17004), and it makes sense to update from RC to final released version.
- **b7994**: metal : consolidate unary ops ([#19490](https://github.com/ggml-org/llama.cpp/pull/19490))
  - cont #19390
  - Common implementation of the unary kernels
  - Extend support for non-cont src0
- **b7995**: ggml : extend bin bcast for permuted src1 ([#19484](https://github.com/ggml-org/llama.cpp/pull/19484))
  - Remove CPU asserts preventing `src1` from being permuted
  - Update CUDA kernels to support permuted `src1`
  - Add tests to exercise `src1` permutation
- **b7998**: hexagon: Add ARGSORT, DIV, SQR, SQRT, SUM_ROWS, GEGLU ([#19406](https://github.com/ggml-org/llama.cpp/pull/19406))
  - Catching up on the Op coverage for the Hexagon backend.
  - This PR improves Op coverage for Gemma-3N, LFM2 and other models.
  - All new Ops pass `test-backend-ops` (mostly in f32).
- **b8001**: metal : extend l2_norm support for non-cont src0 ([#19502](https://github.com/ggml-org/llama.cpp/pull/19502))
  - Support non-cont `src0`
  - Support `ne00` non-multiple of 4
- **b8005**: ggml : unary ops support non-cont src0 + metal F16 unary ops ([#19511](https://github.com/ggml-org/llama.cpp/pull/19511))
  - cont #19490
- **b8006**: opencl: add general Q6_K mm and Q4_K mv ([#19347](https://github.com/ggml-org/llama.cpp/pull/19347))
  - Although still slow, this should make Q4_K_M a bit more usable. Q4_K mv is not flattened yet. More specialized Q6_K and Q4_K mm and mv using transposed layouts will be added in follow up PRs.
- **b8008**: hexagon: further optimization and tuning of matmul and dot kernels ([#19407](https://github.com/ggml-org/llama.cpp/pull/19407))
  - This PR adds support for computing 2x2 (2 rows x 2 cols) dot products in parallel.
  - Mostly helps with the Prompt processing that shows 10+ T/S gains for most models.
  - Here are some numbers with Qwen3.
- **b8012**: metal : update sum_rows kernel to support float4 ([#19524](https://github.com/ggml-org/llama.cpp/pull/19524))

#### 🐛 Bug Fixes
- **b7958**: MSVC regex fix ([#19340](https://github.com/ggml-org/llama.cpp/pull/19340))
  - Fix MSVC regex error:
  - ```
  - Regex error: regex_error(error_stack): There was insufficient memory to determine whether the regular expression could match the specified character sequence.
- **b7965**: metal : fix event synchronization in cpy_tensor_async ([#19402](https://github.com/ggml-org/llama.cpp/pull/19402))
  - cont #18966
  - Was incorrectly recording the event in a separate command buffer. Fixes the synchronization issue reported in https://github.com/ggml-org/llama.cpp/pull/19378#issuecomment-3862086179
- **b7987**: ggml: use noexcept overload for is_regular_file in backend registration ([#19452](https://github.com/ggml-org/llama.cpp/pull/19452))
  - using noexcept std::filesystem::directory_entry::is_regular_file overload prevents abnormal termination upon throwing an error (as caused by symlinks to non-existant folders on linux)
  - fixes issue #18560
  - Searched for existing PRs for this issue
- **b7989**: test: fix IMROPE perf test case ([#19465](https://github.com/ggml-org/llama.cpp/pull/19465))
  - Ref: https://github.com/ggml-org/llama.cpp/issues/19464
- **b7997**: fix: correct typos 'occured' and 'occurences' ([#19414](https://github.com/ggml-org/llama.cpp/pull/19414))
  - Fixes minor spelling typos in comments:
  - occurred (1 instance in llama.h)
  - occurrences (3 instances in ngram-map.h and ngram-map.cpp)
- **b7999**: common : improve download error reporting ([#19491](https://github.com/ggml-org/llama.cpp/pull/19491))
  - While debugging the new `cpp-httplib`, the current errors were unusable...
  - Here is a small patch to make life easier for the next person dealing with HTTP issues :)
- **b8011**: Add a workaround for compilation with ROCWMMA_FATTN and gfx9 ([#19461](https://github.com/ggml-org/llama.cpp/pull/19461))
  - There is an upstream problem [1] with AMD's LLVM 22 fork and rocWMMA 2.2.0 causing compilation issues on devices without native fp16 support (CDNA devices).
  - The specialized types aren't resolved properly:
  - ```c
- **b8018**: vendor : update cpp-httplib ([#19537](https://github.com/ggml-org/llama.cpp/pull/19537))
  - The 0.32 version had important bug fixes, but it wasn’t working for us. We need the latest patches.


### Additional Changes
13 minor improvements: 3 documentation, 7 examples, 3 maintenance.

### Full Commit Range
- b7958 to b8018 (44 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7958...b8018

---

## 2026-02-06: Update to llama.cpp b7955

### Summary
Updated llama.cpp from b7926 to b7955, incorporating 24 upstream commits with breaking changes, new features, and performance improvements.

### Notable Changes

#### ⚠️ Breaking Changes
- **b7931**: ggml-virtgpu: make the code thread safe ([#19204](https://github.com/ggml-org/llama.cpp/pull/19204))
  - This PR improves the code of the ggml-virtgpu backend to make it thread safe, by using mutex for accessing the host<>guest shared memory buffers, and by pre-caching, during the initialization, the constant values queried from the backend.
  - The unused `buffer_type_is_host` method is also deprecated.
- **b7933**: spec : fix the check-rate logic of ngram-simple ([#19261](https://github.com/ggml-org/llama.cpp/pull/19261))
  - fix #19231
  - For the `spec-simple` method, we don't need to keep track of the last length to rate-limit the generations. We can simply use an incremental counter. This makes the speculator work with "Regenerate" of last message or branching the conversation from previous messages.
  - Also, removed `struct common_ngram_simple_state` - seemed a bit redundant.

#### 🆕 New Features
- **b7928**: ci : add sanitizer runs for server ([#19291](https://github.com/ggml-org/llama.cpp/pull/19291))
  - Reenable the server sanitizer builds + runs. The thread sanitizer is quite slow, so remains disabled for now.
  - https://github.com/ggerganov/tmp2/actions/runs/21629674042
- **b7929**: metal : add solve_tri ([#19302](https://github.com/ggml-org/llama.cpp/pull/19302))
  - Add `GGML_OP_SOLVE_TRI` implementation for Metal.
  - | Model                  | Test   |   t/s master |   t/s gg/metal-solve-tri |   Speedup |
  - |:-----------------------|:-------|-------------:|-------------------------:|----------:|
- **b7935**: tests : add non-cont, inplace rope tests ([#19296](https://github.com/ggml-org/llama.cpp/pull/19296))
  - ref https://github.com/ggml-org/llama.cpp/pull/18986#issuecomment-3841942982
  - ref https://github.com/ggml-org/llama.cpp/issues/19128#issuecomment-3807441909
  - ref https://github.com/ggml-org/llama.cpp/issues/19292
- **b7941**: vendor : add missing llama_add_compile_flags ([#19322](https://github.com/ggml-org/llama.cpp/pull/19322))
  - ~Hopefully fixes CI~Ensure `httplib` and `boringssl`/`libressl` are built with sanitizer options, see https://github.com/ggml-org/llama.cpp/pull/19291#discussion_r2761613566
- **b7946**: metal : add diag ([#19330](https://github.com/ggml-org/llama.cpp/pull/19330))
  - Add implementation for GGML_OP_DIAG for the Metal backend

#### 🚀 Performance Improvements
- **b7930**: ggml-cpu: use LUT for converting e8->f32 scales on x86 ([#19288](https://github.com/ggml-org/llama.cpp/pull/19288))
  - `perf` showed the e8m0->f32 function as a bottleneck. Use a LUT instead. Tested only on x86
  - | Model                 | Test   |   t/s topk-cuda-refactor |   t/s mxfp4-cpu-scale |   Speedup |
  - |:----------------------|:-------|-------------------------:|----------------------:|----------:|
- **b7951**: metal : adaptive CPU/GPU interleave based on number of nodes ([#19369](https://github.com/ggml-org/llama.cpp/pull/19369))
  - Put a bit more work on the main thread when encoding the graph. This helps to interleave better the CPU/GPU work, especially for larger graphs.
  - | Model                    | Test   |   t/s master |   t/s gg/metal-adaptive-cpu-interleave |   Speedup |
  - |:-------------------------|:-------|-------------:|---------------------------------------:|----------:|
- **b7954**: metal : skip loading all-zero mask ([#19337](https://github.com/ggml-org/llama.cpp/pull/19337))
  - Similar optimization as in #19281 to skip loading the all-zero mask blocks.
  - | Model                 | Test    |   t/s master |   t/s gg/metal-fa-mask-zero-opt |   Speedup |
  - |:----------------------|:--------|-------------:|--------------------------------:|----------:|

#### 🐛 Bug Fixes
- **b7926**: vulkan: disable coopmat1 flash attention on Nvidia Turing ([#19290](https://github.com/ggml-org/llama.cpp/pull/19290))
  - See https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
- **b7927**: sampling : delegate input allocation to the scheduler ([#19266](https://github.com/ggml-org/llama.cpp/pull/19266))
  - fix #18622
  - alt #18636
  - Merge the sampler inputs into the main graph. This way the backend scheduler is responsible for allocating the memory which makes backend sampling compatible with pipeline parallelism
- **b7936**: model: (qwen3next) correct vectorized key_gdiff calculation ([#19324](https://github.com/ggml-org/llama.cpp/pull/19324))
  - Testing with the provided prompt from https://github.com/ggml-org/llama.cpp/issues/19305
  - <img width="837" height="437" alt="image" src="https://github.com/user-attachments/assets/54f19beb-a9d0-4f10-bc33-747057f36fe7" />
- **b7938**: debug: make common_debug_print_tensor readable ([#19331](https://github.com/ggml-org/llama.cpp/pull/19331))
  - Now using 4-space indentation
  - The log is output to stdout, so that I can do `llama-eval-callback ... > debug.log`
  - ```
- **b7940**: vendor: update cpp-httplib version ([#19313](https://github.com/ggml-org/llama.cpp/pull/19313))
  - ref: #19017
  - Sync the `cpp-httplib` library to fix #19017.
- **b7942**: Fix missing includes in metal build ([#19348](https://github.com/ggml-org/llama.cpp/pull/19348))
  - Since commit https://github.com/ggml-org/llama.cpp/commit/6fdddb498780dbda2a14f8b49b92d25601e14764, I get errors when building on Mac.
  - This PR adds the missing includes for `mutex` and `string` to fix the build.
  - ```
- **b7943**: vulkan: fix non-contig rope ([#19299](https://github.com/ggml-org/llama.cpp/pull/19299))
  - For #19296.
- **b7945**: vulkan: fix GPU deduplication logic. ([#19222](https://github.com/ggml-org/llama.cpp/pull/19222))
  - As reported in https://github.com/ggml-org/llama.cpp/issues/19221, the (same uuid, same driver) logic is problematic for windows+intel igpu.
  - Let's just avoid filtering for MoltenVK which is apple-specific, and keep the logic the  same as before 88d23ad5 - just dedup based on UUID.
  - Verified that MacOS + 4xVega still reports 4 GPUs with this version.
- **b7952**: cuda : cuda graphs now compare all node params ([#19383](https://github.com/ggml-org/llama.cpp/pull/19383))
  - ref https://github.com/ggml-org/llama.cpp/pull/19338#issuecomment-3852298933
  - This should fix the CUDA graph usage logic when the ops have variable op params. This issue is most pronounced during `test-backend-ops`.


### Additional Changes
5 minor improvements: 1 examples, 4 maintenance.

- **b7932**: completion : simplify batch (embd) processing ([#19286](https://github.com/ggml-org/llama.cpp/pull/19286))
  - This commit simplifies the processing of embd by removing the for loop that currently exists which uses params.n_batch as its increment. This commit also removes the clamping of n_eval as the size of embd is always at most the size of params.n_batch.
  - The motivation is to clarify the code as it is currently a little confusing when looking at this for loop in isolation and thinking that it can process multiple batches.
- **b7944**: vulkan: Set k_load_shmem to false when K is too large ([#19301](https://github.com/ggml-org/llama.cpp/pull/19301))
  - See https://github.com/ggml-org/llama.cpp/pull/19075/changes#r2726146004.
  - ```
  - Z:\github\jeffbolznv\llama.cpp\build\bin\RelWithDebInfo>llama-bench.exe -fa 1 -m c:\models\GLM-4.7-Flash-Q4_K_M.gguf -p 512 -n 128 -d 0,4096,16384
- **b7947**: vendor : update BoringSSL to 0.20260204.0 ([#19333](https://github.com/ggml-org/llama.cpp/pull/19333))
- **b7950**: vulkan: Preprocess FA mask to detect all-neg-inf and all-zero. ([#19281](https://github.com/ggml-org/llama.cpp/pull/19281))
  - Write out a 2-bit code per block and avoid loading the mask when it matches these two common cases.
  - Apply this optimization when the mask is relatively large (i.e. prompt processing).
  - ```
- **b7955**: vulkan: make FA mask/softcap enables spec constants ([#19309](https://github.com/ggml-org/llama.cpp/pull/19309))
  - ~This is stacked on #19281.~ (merged)
  - This allows the compiler to do a bit better at overlapping loads and math (e.g. loading V can start while computing Q*K^t is still happening). Worth a couple percent for coopmat2, less for coopmat1/scalar.
  - ```

### Full Commit Range
- b7926 to b7955 (24 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7926...b7955

---

## 2026-02-03: Update to llama.cpp b7921

### Summary
Updated llama.cpp from b7907 to b7921, incorporating 11 upstream commits with new features.

### Notable Changes

#### 🆕 New Features
- **b7907**: ggml-backend: fix async set/get fallback sync ([#19179](https://github.com/ggml-org/llama.cpp/pull/19179))
  - While working on an implementation for backend-agnostic tensor parallelism I found what I believe to be a bug in the ggml backend code. For a minimal implementation I did at first not implement `set_tensor_async` and `get_tensor_async` assuming that I could just rely on the synchronous fallback and implement those later. However, `set_tensor_async` and `get_tensor_async` do not call `ggml_backend_synchronize` for their fallback so I got incorrect results. This PR adds the corresponding calls.
- **b7909**: metal : support virtual devices ([#18919](https://github.com/ggml-org/llama.cpp/pull/18919))
  - Support virtual Metal devices. Allows simulating multi-GPU environments on Mac using the new `GGML_METAL_DEVICES` environment variable.
  - ```bash
  - GGML_METAL_DEVICES=4 ./bin/llama-completion -m [model.gguf]
- **b7919**: support infill for Falcon-H1-Tiny-Coder ([#19249](https://github.com/ggml-org/llama.cpp/pull/19249))
  - Added FIM tokens used in Falcon-H1-Tiny-Coder (see https://tiiuae-tiny-h1-blogpost.hf.space/#fim-format, https://huggingface.co/tiiuae/Falcon-H1-Tiny-Coder-90M/blob/main/tokenizer_config.json#L1843) to make the llama-server `POST /infill` handle work.
- **b7921**: ggml: added cleanups in ggml_quantize_free ([#19278](https://github.com/ggml-org/llama.cpp/pull/19278))
  - Add missing cleanup calls for IQ2_S, IQ1_M quantization types and IQ3XS with 512 blocks during quantization cleanup.

#### 🐛 Bug Fixes
- **b7917**: opencl: refactor some ops, concat, repeat, tanh and scale ([#19226](https://github.com/ggml-org/llama.cpp/pull/19226))
  - Gemma-3n-E2B and Gemma-3n-E4B have been producing weird (not really gibberish but apparently not correct) output. Ended up refactoring these ops and the issue is now fixed. In addition, this refactor also improves perf a bit.
  - On X Elite,
  - `gemma-3n-E2B-it-Q8_0`,


### Additional Changes
6 minor improvements: 4 documentation, 1 examples, 1 maintenance.

### Full Commit Range
- b7907 to b7921 (11 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7907...b7921

---

## 2026-02-02: Update to llama.cpp b7907

### Summary
Updated llama.cpp from b7885 to b7907, incorporating 14 upstream commits with breaking changes and new features.

### Notable Changes

#### ⚠️ Breaking Changes
- **b7903**: Remove pipeline cache mutexes ([#19195](https://github.com/ggml-org/llama.cpp/pull/19195))
  - Now that `webgpu_context` is per-thread, we can remove mutexes from pipeline caches. We cannot remove mutexes from `webgpu_buf_pool` since they are allocated and freed in callback threads, and we cannot remove the mutex from the memset buffer pool since it is shared by all ggml buffers.

#### 🆕 New Features
- **b7885**: tests : add GQA=20 FA test ([#19095](https://github.com/ggml-org/llama.cpp/pull/19095))
  - Might be a good idea to have a test that exercises GQA=20 in order to catch any potential regressions.
- **b7895**: lookahead : add example for lookahead decoding ([#4207](https://github.com/ggml-org/llama.cpp/pull/4207))
  - ref #4157
  - Think this should implement the approach from: https://lmsys.org/blog/2023-11-21-lookahead-decoding/
  - The approach requires large batches to be decoded, which in turn requires a lot of FLOPS even for single stream
- **b7895**: Prompt lookup decoding ([#4484](https://github.com/ggml-org/llama.cpp/pull/4484))
  - ref #4226
  - This example implements the "Prompt Lookup Decoding" technique:
  - https://github.com/apoorvumang/prompt-lookup-decoding
- **b7898**: ggml-hexagon: flash-attention and reduce-sum optimizations ([#19141](https://github.com/ggml-org/llama.cpp/pull/19141))
  - Further to the discussion in [PR #19025](vscode-file://vscode-app/f:/Download/OneDrive/sync/tools/editor/VSCode/resources/app/out/vs/code/electron-browser/workbench/workbench.html), this implements the dual row dot product for flash attention.
  - Added `hvx_vec_reduce_sum_qf32x2`, a helper function for efficiently reducing and accumulating two HVX vectors of qf32 values, and refactored several places in the codebase to use this function for dual-accumulation scenarios. [[1]](diffhunk://#diff-a61b8b4ec9b687ceb6adecb4f2de734f398493514475aa35a2ed1697d58e8a78R47-R57) [[2]](diffhunk://#diff-9469cc7ef405748e1379a215fd377726746ae6087c02d975042955268ea40870L468-R469) [[3]](diffhunk://#diff-9469cc7ef405748e1379a215fd377726746ae6087c02d975042955268ea40870L641-R639) [[4]](diffhunk://#diff-9469cc7ef405748e1379a215fd377726746ae6087c02d975042955268ea40870L883-R878) [[5]](diffhunk://#diff-9469cc7ef405748e1379a215fd377726746ae6087c02d975042955268ea40870L960-R952)
  - Introduced new "rx2" (dual accumulation) versions of dot product functions for both f32-f16 and f16-f16 cases (`hvx_dot_f32_f16_aa_rx2`, `hvx_dot_f16_f16_aa_rx2`), improving performance by processing two accumulations in parallel. [[1]](diffhunk://#diff-703a5dfdf5d9711789e72c854d70db2559000749823e0cb8fa9defc4b276e7b8R76-R139) [[2]](diffhunk://#diff-703a5dfdf5d9711789e72c854d70db2559000749823e0cb8fa9defc4b276e7b8R180-R233)
- **b7907**: ggml-backend: fix async set/get fallback sync ([#19179](https://github.com/ggml-org/llama.cpp/pull/19179))
  - While working on an implementation for backend-agnostic tensor parallelism I found what I believe to be a bug in the ggml backend code. For a minimal implementation I did at first not implement `set_tensor_async` and `get_tensor_async` assuming that I could just rely on the synchronous fallback and implement those later. However, `set_tensor_async` and `get_tensor_async` do not call `ggml_backend_synchronize` for their fallback so I got incorrect results. This PR adds the corresponding calls.

#### 🐛 Bug Fixes
- **b7895**: llama : adjust default context size + print warnings ([#10136](https://github.com/ggml-org/llama.cpp/pull/10136))
  - fix #8817, https://github.com/ggerganov/llama.cpp/issues/9563#issuecomment-2452727620
  - By default, the examples will use a context size of 4096, instead of the training context of the model. In a lot of cases, the default training context can be very big - 32k to 128k tokens, which causes enormous KV cache allocation and failures for regular hardware.
  - Also, add warning logs when the specified context size per sequence does not match the training context.


### Additional Changes
7 minor improvements: 3 documentation, 4 examples.

### Full Commit Range
- b7885 to b7907 (14 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7885...b7907

---

## 2026-01-30: Update to llama.cpp b7885

### Summary
Updated llama.cpp from b7871 to b7885, incorporating 9 upstream commits with breaking changes and new features.

### Notable Changes

#### ⚠️ Breaking Changes
- **b7872**: jinja : do not pass empty tools and add some none filters ([#19176](https://github.com/ggml-org/llama.cpp/pull/19176))
  - Passing empty or null `tools` breaks many templates so avoid that.
  - Added several filters to `none` that are accepted by `jinja2`, fixes some templates that will try to use them (like `Functionary`).
  - Fixes #19155
- **b7883**: memory : remove unused tmp_buf ([#19199](https://github.com/ggml-org/llama.cpp/pull/19199))
  - This commit removes the unused tmp_buf variable from llama-kv-cache.cpp and llama-memory-recurrent.cpp.
  - The tmp_buf variable was declared but never used but since it has a non-trivial constructor/desctuctor we don't get an unused variable warning about it.

#### 🆕 New Features
- **b7871**: HIP: add mmf for CDNA ([#18896](https://github.com/ggml-org/llama.cpp/pull/18896))
  - Add mmf for CDNA, CDNA3 is passed, it will be very helpful if anyone can test it on CDNA2 and CDNA1, thank you.
  - [x] Refactor mmf to make rows_per_block as input parameter.
  - [x] Pass MUL_MAT and MUL_MAT_ID.
- **b7881**: add tensor type checking as part of cuda graph properties ([#19186](https://github.com/ggml-org/llama.cpp/pull/19186))
  - Motivated by https://github.com/ggml-org/llama.cpp/pull/15805#issuecomment-3818986820
- **b7885**: tests : add GQA=20 FA test ([#19095](https://github.com/ggml-org/llama.cpp/pull/19095))
  - Might be a good idea to have a test that exercises GQA=20 in order to catch any potential regressions.

#### 🐛 Bug Fixes
- **b7875**: cuda : fix nkvo, offload and cuda graph node properties matching ([#19165](https://github.com/ggml-org/llama.cpp/pull/19165))
  - fix #19158
  - fix #19169
  - cont #19105


### Additional Changes
3 minor improvements: 3 documentation.

- **b7876**: hexagon: enable offloading to Hexagon on Windows on Snapdragon ([#19150](https://github.com/ggml-org/llama.cpp/pull/19150))
  - GGML Hexagon backend updates to support Windows on Snapdragon.
  - Features:
  - Support for building and offloading to NPU on WoS.
- **b7879**: sycl: implement GGML_OP_TRI ([#19089](https://github.com/ggml-org/llama.cpp/pull/19089))
  - Implements GGML_OP_TRI for the SYCL backend (F32).
  - The implementation matches CPU semantics for all ggml_tri_type values
  - (lower/upper, with and without diagonal).
- **b7880**: sycl: implement GGML_UNARY_OP_SOFTPLUS ([#19114](https://github.com/ggml-org/llama.cpp/pull/19114))
  - Implements GGML_UNARY_OP_SOFTPLUS for the SYCL backend.
  - Adds an element-wise softplus kernel integrated through the generic SYCL unary dispatch path.
  - Numerical behavior matches the CPU backend implementation.

### Full Commit Range
- b7871 to b7885 (9 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7871...b7885

---

## 2026-01-29: Update to llama.cpp b7871

### Summary
Updated llama.cpp from b7847 to b7871, incorporating 22 upstream commits with breaking changes, new features, and performance improvements.

### Notable Changes

#### ⚠️ Breaking Changes
- **b7850**: ggml-zendnn : update ZenDNN git tag to main branch ([#19133](https://github.com/ggml-org/llama.cpp/pull/19133))
  - This PR is related to ZenDNN removed their zendnnl branch and moved all the code to main
  - Right now our code is still looking for the old zendnnl branch which no longer exists, so builds break.
  - This fixes it by pointing to the new main branch instead
- **b7852**: sampling : remove sampling branching in output_reserve ([#18811](https://github.com/ggml-org/llama.cpp/pull/18811))
  - This commit updates output_reserve in llama-context.cpp to always allocate sampling buffers regardless of whether sampling is needed for the current batch.
  - The motivation for this is to avoid reallocations and branching based on the sampling requirements of the batch.
- **b7862**: ggml-sycl: remove unused syclcompat header ([#19140](https://github.com/ggml-org/llama.cpp/pull/19140))
  - The `syclcompat/math.hpp` is not used anymore. The change that introduced it was successfully reverted (https://github.com/ggml-org/llama.cpp/pull/17826). This include path will become obsolete and dropped in oneAPI 2026.0 effectively breaking `ggml-sycl` builds.
  - *Make sure to read the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) before submitting a PR*
- **b7868**: CUDA: refactor topk-moe to enable more models (GLM 4.7, Nemotron etc.) ([#19126](https://github.com/ggml-org/llama.cpp/pull/19126))
  - Refactor the topk-moe to enabling various combination of topk-moe. Hopefully this will cover most models. I removed some templates from the code and only kept the bias because it has a extra warp shuffle, the rest of the template code does not provide any significant speedup.
  - 3090
  - | Model                 | Test   |   t/s master |   t/s topk-cuda-refactor |   Speedup |

#### 🆕 New Features
- **b7849**: jinja : implement mixed type object keys ([#18955](https://github.com/ggml-org/llama.cpp/pull/18955))
  - Allow all hashable types as object keys, taking care to replicate special python/jinja behavior between `int`/`float`/`bool`.
  - Fixed array/object output with `string` filter.
  - Fixed object `tojson` output (did not properly escape key string).
- **b7860**: CUDA: use mul_mat_q kernels by default ([#2683](https://github.com/ggml-org/llama.cpp/pull/2683))
  - There seem to have been no further reports of problems with the mul_mat_q kernels so I think it's fine to use them by default. This PR does just that and replaces the `-mmq`/`--mul-mat-q` CLI argument with `-nommq`/`--no-mul-mat-q`. Unless I'm mistaken the long-term plan is to also add equivalent CPU kernels for matrix matrix multiplications. Ideally I think the same CLI argument should then be used for switching the algorithm. So if you think that "mul_mat_q" is a bad name for matrix multiplications using quantized data now would be a good time to tell me.
- **b7870**: arg : add -kvu to llama-batched-bench ([#19172](https://github.com/ggml-org/llama.cpp/pull/19172))
- **b7871**: HIP: add mmf for CDNA ([#18896](https://github.com/ggml-org/llama.cpp/pull/18896))
  - Add mmf for CDNA, CDNA3 is passed, it will be very helpful if anyone can test it on CDNA2 and CDNA1, thank you.
  - [x] Refactor mmf to make rows_per_block as input parameter.
  - [x] Pass MUL_MAT and MUL_MAT_ID.

#### 🚀 Performance Improvements
- **b7847**: CUDA: tune GLM 4.7 Flash FA kernel selection logic ([#19097](https://github.com/ggml-org/llama.cpp/pull/19097))
  - Follow-up to https://github.com/ggml-org/llama.cpp/pull/19092 .
  - Adjusts the kernel selection logic as a function of context depth to squeeze out a few more % on Ampere/Blackwell.
  - | GPU      | Model               |   Microbatch size | Test          |   t/s master |   t/s 8a8b9a8bd |   Speedup |
- **b7858**: ggml: new backend for Virglrenderer API Remoting acceleration (v2) ([#18718](https://github.com/ggml-org/llama.cpp/pull/18718))
  - This is a follow up of https://github.com/ggml-org/llama.cpp/pull/17072
  - The API Remoting backend/frontend allow escaping the VM isolation, with the help of the `virt-gpu` paravirtualization (and the `virglrenderer` library on the host side).
  - `ggml-remotingfrontend` is a GGML API implementation, which intercepts the GGML API calls and forwards them to the `virt-gpu` virtual device
- **b7865**: Vulkan Flash Attention Coopmat1 Refactor ([#19075](https://github.com/ggml-org/llama.cpp/pull/19075))
  - I finally had the time to go through Jeff's Flash Attention shaders in detail and used the chance to refactor the Coopmat1 for AMD. It started out as an attempt to use Coopmats for the Softmax * V matrix multiplication as well and then escalated into a refactor of the whole shader structure.
  - It now uses coopmats for the Softmax result * V matrix multiplication, and I vectorized some variables, changed how shared memory is used, load K and V directly from global memory if possible, otherwise streamed through a shared memory cache.
  - Tests are passing. Performance is up significantly on AMD RX 8060S (Strix Halo). Draft because there is a regression on Nvidia. Let me know if you see anything obvious @jeffbolznv. More tuning is likely required.

#### 🐛 Bug Fixes
- **b7851**: Split shared state (webgpu_context) into global state and per-thread state ([#18976](https://github.com/ggml-org/llama.cpp/pull/18976))
  - Right now, the WebGPU backend has a global `webgpu_context` struct with all the information required to instantiate and run a WebGPU graph.
  - We want to split up the `webgpu_context` struct as follows:
  - Move `get_tensor_sharing_buf` to global state, along with the `mutex`
- **b7853**: llama : disable Direct IO by default ([#19109](https://github.com/ggml-org/llama.cpp/pull/19109))
  - ref https://github.com/ggml-org/llama.cpp/issues/19035#issuecomment-3798971944
  - cont #18012
  - Update `llama_model_params::use_direct_io == false` by default
- **b7856**: cuda : fix "V is K view" check for non-unified KV cache ([#19145](https://github.com/ggml-org/llama.cpp/pull/19145))
  - We weren't handling the case where both V and K are views of the same data with the same offset different from 0. This happens with split KV cache (e.g. `--parallel 4 --no-kv-unified`) and causes the flash attention to fall back to the CPU in such cases.
- **b7860**: vulkan: handle device dedup on MacOS + Vega II Duo cards ([#19058](https://github.com/ggml-org/llama.cpp/pull/19058))
  - Deduplication here relied on the fact that vulkan would return unique UUID for different physical GPUs. It is at the moment not always the case. On Mac Pro 2019 running Mac OS, with 2 Vega II Duo cards (so, 4 GPU total), MotlenVK would assign same UUID to pairs of GPUs, unless they are connected with Infinity Fabric.
  - See more details here: KhronosGroup/MoltenVK#2683.
  - The right way is to fix that in MoltenVK, but until it is fixed, llama.cpp would only recognize 2 of 4 GPUs in such configuration.
- **b7861**: jinja : undefined should be treated as sequence/iterable (return string/array) by filters/tests ([#19147](https://github.com/ggml-org/llama.cpp/pull/19147))
  - Fixes #19130
- **b7869**: ggml-zendnn : resolve ZenDNN backend cross-module symbol dependency ([#19159](https://github.com/ggml-org/llama.cpp/pull/19159))
  - This PR fixes the ZenDNN backend failing to load when `GGML_BACKEND_DL=ON`
  - The issue occurs because MODULE libs cannot access symbols from other MODULE libs, ZenDNN backend was attempting to call `ggml_get_type_traits_cpu()` from ggml-cpu, resulting in an undfined symbol error for `GGML_BACKEND_DL=ON`
  - This fix uses `ggml_get_type_traits()` from ggml-base instead, eliminating the dependency on ggml-cpu


### Additional Changes
5 minor improvements: 3 documentation, 2 maintenance.

- **b7864**: Add self‑speculative decoding (no draft model required) ([#18471](https://github.com/ggml-org/llama.cpp/pull/18471))
  - This PR introduces self-speculative decoding: instead of using a dedicated draft model (which is good, if available, see #18039), the current token history is used to predict future tokens. This can provide a speedup in cases where the output contains repeated parts of the prompt. A typical example is making many small changes in a large source file.
  - **Example 1** (`gpt-oss-120b` in VRAM): Translation of a few comments in a Python script (chosen as a favorable case).
  - ```
- **b7864**: Add self‑speculative decoding (no draft model required) ([#18471](https://github.com/ggml-org/llama.cpp/pull/18471))
  - This PR introduces self-speculative decoding: instead of using a dedicated draft model (which is good, if available, see #18039), the current token history is used to predict future tokens. This can provide a speedup in cases where the output contains repeated parts of the prompt. A typical example is making many small changes in a large source file.
  - **Example 1** (`gpt-oss-120b` in VRAM): Translation of a few comments in a Python script (chosen as a favorable case).
  - ```
- **b7867**: [SYCL] fix norm kernels: l2_norm, group_norm, rms_norm by remove assert ([#19154](https://github.com/ggml-org/llama.cpp/pull/19154))
  - fix norm kernels: l2_norm, group_norm, rms_norm by remove assert.
  - all ut cases of norm are 100% passed.
  - no crash of UT cases.
- **b7855**: CUDA: tune GLM 4.7 Flash FA kernel selection logic (DGX Spark) ([#19142](https://github.com/ggml-org/llama.cpp/pull/19142))
  - cont #19097
  - This is similar to #19097, but for DGX Spark. I used only the `Q8_0` model for the measurements.
  - ```bash
- **b7857**: ggml-cpu: arm64: Q4_K repack (i8mm) scale unroll and vectorization ([#19108](https://github.com/ggml-org/llama.cpp/pull/19108))
  - While working on https://github.com/ggml-org/llama.cpp/pull/18860 I found out a small perf optimization when loading the subblock scales.
  - Behavior unchanged, it's a manual unroll + vectorization.
  - Llama-bench:

### Full Commit Range
- b7847 to b7871 (22 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7847...b7871

---

## 2026-01-27: Update to llama.cpp b7845

### Summary
Updated llama.cpp from b7837 to b7845, incorporating 8 upstream commits with breaking changes, new features, and performance improvements.

### Notable Changes

#### ⚠️ Breaking Changes
- **b7839**: graph : fix nkvo offload with FA ([#19105](https://github.com/ggml-org/llama.cpp/pull/19105))
  - fix #19096
  - The `ggml_flash_attn_ext` was not being offloaded to the CPU when `-nkvo` is specified.
  - Also remove obsolete `strcmp(name, "kqv_merged_cont")` check in the graph callback.

#### 🆕 New Features
- **b7837**: model : add correct type for GLM 4.7 Flash ([#19106](https://github.com/ggml-org/llama.cpp/pull/19106))
  - Fix the displayed model type in the logs:
  - ```bash
  - deepseek2 ?B Q8_0
- **b7843**: common : clarify HTTPS build options in error message ([#19103](https://github.com/ggml-org/llama.cpp/pull/19103))
  - This commit updates the https error message to provide clearer instructions for users who encounter the "HTTPS is not supported" error.
  - The motivation for this is that it might not be clear to users that only one of these options are needed to enable HTTPS support. The LLAMA_OPENSSL option is also added to the message to cover all possible build configurations.
- **b7845**: ggml-cpu: aarm64: q5_K repack gemm and gemv (and generic) implementations (i8mm) ([#18860](https://github.com/ggml-org/llama.cpp/pull/18860))
  - This PR implements the REPACK version of q5_K, following most of the existing design used for q4_K, since Q5_K only differs from q4_K in having the `qh` field with the additional bit.
  - Most of the code is shared, but I didn't know how to abstract the common patterns without creating a convoluted mess of functions. Since only Q4_K and Q5_K share the same 6bit scales and mins decode, I opted to duplicate the code.
  - I also moved around some declarations for Q2_K because the structure seemed weird (it's inverted with what I've seen in quants.c). The Q2_K function declarations were left where they were to avoid polluting the diff and messing the blame. If you want me to revert it, just say so.
- **b7845**: ggml-cpu: aarm64: q6_K repack gemm and gemv (and generic) implementations (i8mm) #18860 ([#18888](https://github.com/ggml-org/llama.cpp/pull/18888))
  - Continuation of repack work  for ARM, since `q4_K_M` and `q5_K_M` quantizations spend ~%20 of compute time on q6_K layers.
  - [x] Still pending rebasing on top of #18860 if that gets merged.
  - Same testing practices from the other repack implementations.

#### 🚀 Performance Improvements
- **b7841**: opencl: add flattened q6_K mv ([#19054](https://github.com/ggml-org/llama.cpp/pull/19054))
  - This PR adds flattened q6_K mv and renames the existing q6_K mv kernel file to better reflect what the kernel does. There should be no performance improvement, but will enable further optimizations.
- **b7842**: ggml-cpu: Enable FP16 MMA kernels on PPC ([#19060](https://github.com/ggml-org/llama.cpp/pull/19060))
  - This change introduces a unified FP16/BF16 MMA kernel selection via mma_instr,
  - allowing FP16 models to leverage Power MMA instructions instead of falling back to scalar/vector paths.
  - Performance impact (Power10, 10 threads, Mistral-7B FP16, llama-batched-bench):


### Additional Changes
1 minor improvements: 1 documentation.

- **b7844**: [CUDA] Reduce CPU-side stalls due to the CUDA command buffer being full ([#19042](https://github.com/ggml-org/llama.cpp/pull/19042))
  - With pipeline parallelism, during prompt processing, the CPU-side CUDA command buffer gets full, stalling the CPU. Due to this, enough work doesn't get submitted to the GPU, resulting in bubbles in the GPU timeline. This PR fixes this by setting the CUDA environment variable CUDA_SCALE_LAUNCH_QUEUES to 4x to increase the command buffer size.
  - The NSight profile below shows the issue in more detail:
  - <img width="1958" height="983" alt="image" src="https://github.com/user-attachments/assets/3efdaaf3-dd58-464b-a9d1-3cd31d3f0030" />

### Full Commit Range
- b7837 to b7845 (8 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7837...b7845

---

## 2026-01-26: Update to llama.cpp b7837

### Summary
Updated llama.cpp from b7837 to b7837, incorporating 1 upstream commits with new features.

### Notable Changes

#### 🆕 New Features
- **b7837**: model : add correct type for GLM 4.7 Flash ([#19106](https://github.com/ggml-org/llama.cpp/pull/19106))
  - Fix the displayed model type in the logs:
  - ```bash
  - deepseek2 ?B Q8_0


### Full Commit Range
- b7837 to b7837 (1 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7837...b7837

---

## 2026-01-26: Update to llama.cpp b7837

### Summary
Updated llama.cpp from b7837 to b7837, incorporating 1 upstream commits with new features.

### Notable Changes

#### 🆕 New Features
- **b7837**: model : add correct type for GLM 4.7 Flash ([#19106](https://github.com/ggml-org/llama.cpp/pull/19106))
  - Fix the displayed model type in the logs:
  - ```bash
  - deepseek2 ?B Q8_0


### Full Commit Range
- b7837 to b7837 (1 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7837...b7837

---

## 2026-01-26: Update to llama.cpp b7837

### Summary
Updated llama.cpp from b7837 to b7837, incorporating 1 upstream commits with new features.

### Notable Changes

#### 🆕 New Features
- **b7837**: model : add correct type for GLM 4.7 Flash ([#19106](https://github.com/ggml-org/llama.cpp/pull/19106))
  - Fix the displayed model type in the logs:
  - ```bash
  - deepseek2 ?B Q8_0


### Full Commit Range
- b7837 to b7837 (1 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7837...b7837

---

## 2026-01-26: Update to llama.cpp b7836

### Summary
Updated llama.cpp from b7836 to b7836, incorporating 1 upstream commits with performance improvements.

### Notable Changes

#### 🚀 Performance Improvements
- **b7836**: CUDA: faster FA for GQA > 1 but not power of 2 ([#19092](https://github.com/ggml-org/llama.cpp/pull/19092))
  - This PR generalizes the CUDA MMA FlashAttention kernel to enable the GQA optimizations for models where the ratio between the number of Q heads and the number of K/V heads is not a power of 2. This is done by simply padding the Q columns per CUDA block to the next higher power of 2. This wastes a bit of compute but particularly for small batch sizes the kernel is I/O-bound anyways.
  - On Ampere or newer this improves performance of GLM 4.7 Flash as well as some random models like Granite 3.0 with a GQA ratio of 3. On Volta the new code path is slower than master so it's disabled. On RDNA4 it seems to be faster but as of right now the performance of the MMA kernel is bad on RDNA for head sizes > 128 so there is no benefit for GLM 4.7 Flash.
  - <details>


### Full Commit Range
- b7836 to b7836 (1 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7836...b7836

---

## 2026-01-26: Update to llama.cpp b7836

### Summary
Updated llama.cpp from b7836 to b7836, incorporating 1 upstream commits with performance improvements.

### Notable Changes

#### 🚀 Performance Improvements
- **b7836**: CUDA: faster FA for GQA > 1 but not power of 2 ([#19092](https://github.com/ggml-org/llama.cpp/pull/19092))
  - This PR generalizes the CUDA MMA FlashAttention kernel to enable the GQA optimizations for models where the ratio between the number of Q heads and the number of K/V heads is not a power of 2. This is done by simply padding the Q columns per CUDA block to the next higher power of 2. This wastes a bit of compute but particularly for small batch sizes the kernel is I/O-bound anyways.
  - On Ampere or newer this improves performance of GLM 4.7 Flash as well as some random models like Granite 3.0 with a GQA ratio of 3. On Volta the new code path is slower than master so it's disabled. On RDNA4 it seems to be faster but as of right now the performance of the MMA kernel is bad on RDNA for head sizes > 128 so there is no benefit for GLM 4.7 Flash.
  - <details>


### Full Commit Range
- b7836 to b7836 (1 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7836...b7836

---

## 2026-01-26: Update to llama.cpp b7836

### Summary
Updated llama.cpp from b7836 to b7836, incorporating 1 upstream commits with performance improvements.

### Notable Changes

#### 🚀 Performance Improvements
- **b7836**: CUDA: faster FA for GQA > 1 but not power of 2 ([#19092](https://github.com/ggml-org/llama.cpp/pull/19092))
  - This PR generalizes the CUDA MMA FlashAttention kernel to enable the GQA optimizations for models where the ratio between the number of Q heads and the number of K/V heads is not a power of 2. This is done by simply padding the Q columns per CUDA block to the next higher power of 2. This wastes a bit of compute but particularly for small batch sizes the kernel is I/O-bound anyways.
  - On Ampere or newer this improves performance of GLM 4.7 Flash as well as some random models like Granite 3.0 with a GQA ratio of 3. On Volta the new code path is slower than master so it's disabled. On RDNA4 it seems to be faster but as of right now the performance of the MMA kernel is bad on RDNA for head sizes > 128 so there is no benefit for GLM 4.7 Flash.
  - <details>


### Full Commit Range
- b7836 to b7836 (1 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7836...b7836

---

## 2026-01-21: Update to llama.cpp b7788

### Summary
Updated llama.cpp from b7772 to b7788, incorporating 13 upstream commits with breaking changes, new features, and performance improvements.

### Notable Changes

#### ⚠️ Breaking Changes
- **b7782**: ggml : cleanup path_str() ([#18928](https://github.com/ggml-org/llama.cpp/pull/18928))
  - Remove pragmas as `std::codecvt_utf8` is not used.
  - Avoid implicit `strlen()`.

#### 🆕 New Features
- **b7774**: ggml : add ggml_build_forward_select ([#18550](https://github.com/ggml-org/llama.cpp/pull/18550))
  - target #18547
  - alt #18549
  - Add `GGML_TENSOR_FLAG_COMPUTE` flag indicating that a tensor in the graph must be computed
- **b7777**: jinja : fix undefined keys and attributes and int/float as bool ([#18924](https://github.com/ggml-org/llama.cpp/pull/18924))
  - Return `undefined` on undefined keys and attributes.
  - Integers and floats can be represented as bools.
  - Added `falsy` tests.

#### 🚀 Performance Improvements
- **b7781**: metal : enable FA for MLA heads ([#18950](https://github.com/ggml-org/llama.cpp/pull/18950))
  - ref #18936
  - Re-enable FA for K head size of 576 (MQA mode of MLA) and adjust simdgroups and loop unrolling for performance.
- **b7783**: CUDA: Replace init_offsets kernel with iterators in cub-based argsort ([#18930](https://github.com/ggml-org/llama.cpp/pull/18930))
  - This is mostly a QOL improvement, saving us the cost of materializing the iterator.
  - --- before
  - ```

#### 🐛 Bug Fixes
- **b7772**: DirectIO Model Loading: Extend and fix Fallback ([#18887](https://github.com/ggml-org/llama.cpp/pull/18887))
  - Due to issues with the DirectIO model loading path on Android this PR adds `EINVAL` errors to the fallback condition. Also there was a bug in the fallback to `mmap` in case `open` with the DirectIO flag fails.
- **b7787**: gguf: display strerrno when cant load a model ([#18884](https://github.com/ggml-org/llama.cpp/pull/18884))
  - I've had issues loading models with llama-server:
  - [44039] E gguf_init_from_file: failed to open GGUF file 'mistral-7b-v0.1.Q8_0.gguf'
  - and I was sure it could access the file. Seems like --models-dir and --models-presets dont interact like I thought they would but I salvaged this snippet that helps troubleshooting
- **b7788**: Fix GLM 4.7 Lite MoE gating func ([#18980](https://github.com/ggml-org/llama.cpp/pull/18980))
  - GLM 4.7 Lite uses SIGMOID, not SOFTMAX like Deepseek.


### Additional Changes
5 minor improvements: 1 documentation, 4 examples.

- **b7786**: CUDA: Fix builds for older CCCL versions by ifdefing strided_iterator ([#18964](https://github.com/ggml-org/llama.cpp/pull/18964))
  - Strided iterator was added in [CCCL 3.1](https://github.com/NVIDIA/cccl/releases/tag/v3.1.0), which is packaged into [CTK
  - 13.1](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5)
  - Should fix #18960
- **b7775**: server: fix memory reservations in populate_token_probs ([#18787](https://github.com/ggml-org/llama.cpp/pull/18787))
  - Fixes the two Vector::reserve calls in the populate_token_probs function.
  - In case post_sampling is true the code now only reserves as much space in the Vector as is needed for the requested number of logprobs. This prevents reserving large amounts of memory that are not used.
  - In case post_sampling is false the code now clamps the reserved size to the maximum number of tokens the model supports. This prevents reserving large amounts of unused memory when the client requests more token logprobs than the model supports and, in extreme cases, crashes from invalid memory allocations.
- **b7779**: server : refactor oai_parser_opt, move it to server_chat_params ([#18937](https://github.com/ggml-org/llama.cpp/pull/18937))
  - In this PR:
  - Rename `oaicompat_parser_options` --> `server_chat_params`
  - Store `common_chat_templates_ptr` inside it
- **b7784**: cli : fix reasoning responses in CLI ([#18961](https://github.com/ggml-org/llama.cpp/pull/18961))
  - The chat format was not populate to task state in CLI, so reasoning content was not parsed correctly
  - With this PR, GLM-4.7 now works correctly on CLI:
  - <img width="996" height="304" alt="image" src="https://github.com/user-attachments/assets/a03545a5-1f32-4c53-acf5-81e58580057d" />
- **b7785**: common, server : use the same User-Agent by default ([#18957](https://github.com/ggml-org/llama.cpp/pull/18957))
  - This commit also ensures that if a custom User-Agent is used, it will be the only one sent.

### Full Commit Range
- b7772 to b7788 (13 commits)
- Upstream releases: https://github.com/ggml-org/llama.cpp/compare/b7772...b7788

---

## 2026-01-05: Update to llama.cpp b7631

- b7622 (b7622) – 2026-01-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7622
- b7624 (b7624) – 2026-01-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7624
- b7625 (b7625) – 2026-01-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7625
  - CUDA: disable cuda graph when using n-cpu-moe
  - call ggml_cuda_set_device
- b7626 (b7626) – 2026-01-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7626
- b7628 (b7628) – 2026-01-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b7628
- b7630 (b7630) – 2026-01-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b7630
  - Implement ggml_cann_op_add_rms_norm_fused() using ACLNN AddRmsNorm
  - Add ggml_cann_can_fuse() to check fusion eligibility
  - Integrate fusion logic into computation graph evaluation
  - Add test cases for ADD + RMS_NORM fusion
  - Update documentation with new environment variable
- b7631 (b7631) – 2026-01-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b7631
  - refactor rope_freq_base/scale_swa conversion and init
  - safe defaults for unknowns
  - update relevant models
  - grammar
  - add get_rope_freq_scale to modern-bert
  - const
  - const
  - log swa info


## 2026-01-03: Update to llama.cpp b7621

- b7489 (b7489) – 2025-12-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7489
- b7490 (b7490) – 2025-12-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7490
- b7491 (b7491) – 2025-12-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7491
  - tests: Avoid floating point precision false positives in SUM
  - also apply to test_mean
- b7492 (b7492) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7492
  - implement sleeping at queue level
  - implement server-context suspend
  - add test
  - add docs
  - optimization: add fast path
  - make sure to free llama_init
  - nits
  - fix use-after-free
  - allow /models to be accessed during sleeping, fix use-after-free
  - don't allow accessing /models during sleep, it is not thread-safe
  - fix data race on accessing props and model_meta
  - small clean up
  - trailing whitespace
  - rm outdated comments
- b7493 (b7493) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7493
- b7495 (b7495) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7495
  - Some improvement on mul_mat_iq2_xs
  - Fix trailing whitespace
- b7496 (b7496) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7496
- b7497 (b7497) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7497
- b7498 (b7498) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7498
- b7499 (b7499) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7499
- b7501 (b7501) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7501
- b7502 (b7502) – 2025-12-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7502
- b7503 (b7503) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7503
- b7506 (b7506) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7506
  - Update release workflow to store XCFramework as Zip file
  - Add comments to document Zip file requirement for XCFramework
  - Apply suggestions from code review
- b7507 (b7507) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7507
- b7508 (b7508) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7508
  - server: prevent data race from HTTP threads
  - fix params
  - fix default_generation_settings
  - nits: make handle_completions_impl looks less strange
  - stricter const
  - fix GGML_ASSERT(idx < states.size())
  - move index to be managed by server_response_reader
  - http: make sure req & res lifecycle are tied together
  - fix compile
  - fix index handling buggy
  - fix data race for lora endpoint
  - nits: fix shadow variable
  - nits: revert redundant changes
  - nits: correct naming for json_webui_settings
- b7509 (b7509) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7509
- b7510 (b7510) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7510
- b7511 (b7511) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7511
- b7512 (b7512) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7512
  - gen-docs: automatically update markdown file
  - also strip whitespace
  - do not add extra newline
  - update TOC
- b7513 (b7513) – 2025-12-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7513
  - feat: working gelu with src0 put on vtcm
  - feat: gelu ping-pong for both in and out
  - fix: fixu compile error
  - break: distinguish dma ddr->vtcm and vtcm->ddr operation
  - fix: fix dma queue size
  - break: update dma api to either pop src or dst ptr
  - fix: fix activation vtcm allocation issue for src1 when swapperd
  - refactor: ping-pong gelu logic to avoid unnecessary if else
  - dma: improved queue interface and prefetch handling
  - gelu: fix N+2 block prefetch
- b7515 (b7515) – 2025-12-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7515
  - constants and tensor mappings for modern bert support, model not supported yet but working on getting conversion to work for encoder only
  - conversion now working, hf -> gguf
  - working on support, now working on building graph
  - some cleanup
  - cleanup
  - continuing
  - correct tensor shape for qkv
  - fixed tensor mappings and working on buildin graph
  - tensor debugging now works -> (llama-eval-callback), instead of simulated gate split with views, GEGLU is now used which does exactly this
  - cleanup
  - cleanup
  - cleanup
  - more cleanup
  - ubatch issues, the assert for checking equal seqs in llama-graph.cpp when building attention  keeps failing, setting ubatch size to 1 when running llama-embedding with --ubatch-size 1 makes it work, but needs to be looked into more
  - added cls token per previous modern bert attempt, still working on checking out the rest
  - fixed pre tokenizer and still working through previous pr
  - working through previous attemp, implimented more accurate conversion per previous attempt, added local sliding window attention that alternates every third layer
  - fixed pre tokenizer
  - working on swa with local and global alternating attention
  - some cleanup and now fails on build attn
  - starting to work, and some cleanup, currently failing on last layer construction in graph build
  - alternating rope implemented and modern bert graph build succeeds
  - fixed asser for equal ubatch seq
  - cleanup
  - added mask check in vocab
  - fixed alternating rope, the hparams.rope_freq_base_train and hparams.rope_freq_base_train_swa were the same and i set them to correct values
  - reuse variable
  - removed repeat
  - standard swa method can be used instead of a new enum being LLAMA_SWA_TYPE_LOCAL
  - correct swa layer indexing, is supposed to be 0, 3, 6 ... instead of 1, 4, 7 ...
  - more modular hparam setting
  - replaced attn out norm with ffn_norm and cosine similarity between hf embds and llama.cpp embds went way up, from 0.05 to 0.24, replaced the cacheless kv with swa todo per the previous conversion
  - Update gguf-py/gguf/tensor_mapping.py
  - Update convert_hf_to_gguf_update.py
  - Update src/llama-model.cpp
  - Update src/llama-vocab.cpp
  - Update src/llama-model.cpp
  - Update gguf-py/gguf/tensor_mapping.py
  - Update convert_hf_to_gguf.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update convert_hf_to_gguf.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update gguf-py/gguf/tensor_mapping.py
  - Update src/llama-graph.cpp
  - Update src/llama-arch.cpp
- b7516 (b7516) – 2025-12-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7516
  - llama-model : fix Nemotron V2 crash by moving MoE parameters calculation
  - remove whitespace
- b7519 (b7519) – 2025-12-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7519
  - refactor: replace ggml_hexagon_mul_mat with template-based binary operation for improved flexibility
  - refactor: replace ggml_hexagon_mul_mat_id with template-based binary operation for improved flexibility
  - refactor: initialize buffer types and streamline dspqueue_buffers_init calls for clarity
  - add comment
  - refactor: remove redundant buffer checks in hexagon supported operations
  - wip
  - add missing include to fix weak symbol warning
  - add ggml_hexagon_op_generic
  - refactor: simplify tensor operation initialization and buffer management in hexagon implementation
  - refactor: streamline hexagon operation initialization and buffer management
  - refactor: update function signatures and streamline request handling in hexagon operations
  - wip
  - ggml-hexagon: clean up code formatting and improve unary operation handling
  - wip
  - rename
  - fix: add support for permuted F16 tensors and enhance quantization checks in matrix operations
  - refactor: replace ggml_hexagon_mul_mat with template-based binary operation for improved flexibility
  - hexagon: fix merge conflicts
  - hexagon: minor cleanup for buffer support checks
  - hexagon: factor out op_desc and the overal op logging
  - hexagon: further simplify and cleanup op dispatch logic
  - snapdragon: update adb scripts to use llama-cli and llama-completion
  - fix pipeline failure
- b7520 (b7520) – 2025-12-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7520
- b7522 (b7522) – 2025-12-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7522
- b7524 (b7524) – 2025-12-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7524
- b7525 (b7525) – 2025-12-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7525
- b7526 (b7526) – 2025-12-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7526
- b7527 (b7527) – 2025-12-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7527
- b7529 (b7529) – 2025-12-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7529
- b7530 (b7530) – 2025-12-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7530
- b7531 (b7531) – 2025-12-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7531
  - model: llama-embed-nemotron
  - minor: python lint
  - changed arch-name
  - templated llm_build_llama to be used for both llama and llama-embed arch
- b7538 (b7538) – 2025-12-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b7538
  - ggml-cuda: fix blackwell native builds
  - replace for GGML_NATIVE=OFF too
  - only replace for native
  - remove 120f-virtual for default compilation
- b7539 (b7539) – 2025-12-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b7539
  - cuda: optimize cumsum cub path
  - remove heavy perf test
- b7540 (b7540) – 2025-12-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b7540
  - ggml-cuda: fix regex for arch list
  - make regex exact
- b7541 (b7541) – 2025-12-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7541
  - CANN: implement SSM_CONV operator
  - CANN: remove custom error limit for SSM_CONV
  - CANN: merge SSM_CONV tensor shape/strides into one line
- b7543 (b7543) – 2025-12-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7543
  - server : fix crash when seq_rm fails for hybrid/recurrent models
  - server : add allow_processing param to clear_slot
- b7544 (b7544) – 2025-12-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7544
- b7545 (b7545) – 2025-12-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7545
- b7547 (b7547) – 2025-12-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7547
- b7548 (b7548) – 2025-12-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7548
  - vulkan: Use BK=32 for coopmat2 mul_mat_id
  - vulkan: optimize decodeFuncB in coopmat2 mul_mat_id shader
- b7549 (b7549) – 2025-12-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7549
- b7550 (b7550) – 2025-12-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7550
- b7551 (b7551) – 2025-12-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7551
- b7552 (b7552) – 2025-12-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7552
- b7553 (b7553) – 2025-12-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7553
  - llama: fix magic number of 999 for GPU layers
  - use strings for -ngl, -ngld
  - enacapsulate n_gpu_layers, split_mode
- b7554 (b7554) – 2025-12-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7554
- b7555 (b7555) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7555
  - opencl: allow resizing transpose buffers instead of using fixed sizes
  - opencl: remove commented code
- b7556 (b7556) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7556
- b7557 (b7557) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7557
  - minor: Consolidated `#include <immintrin.h>` under `ggml-cpu-impl.h`
  - cmake: Added more x86-64 CPU backends when building with `GGML_CPU_ALL_VARIANTS=On`
  - `ivybridge`
  - `piledriver`
  - `cannonlake`
  - `cascadelake`
  - `cooperlake`
  - `zen4`
- b7558 (b7558) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7558
- b7560 (b7560) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7560
- b7561 (b7561) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7561
  - rpc: fix segfault on invalid endpoint format
  - rpc: add error log for failed endpoint connection
- b7562 (b7562) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7562
- b7563 (b7563) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7563
  - plamo3
  - fix plamo3
  - clean code
  - clean up the code
  - fix diff
  - clean up the code
  - clean up the code
  - clean up the code
  - clean up the code
  - clean up the code
  - clean up the code
  - add chat_template if exist
  - clean up the code
  - fix cpu-backend
  - chore: whitespace trim fix + typo fix
  - Fix: address review feedback
  - restore `FREQ_BASE_SWA` constant
  - Fix: address review feedback2
  - Fix:typecheck
  - Fix: address review feedback3
  - final cleanup
- b7564 (b7564) – 2025-12-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7564
- b7566 (b7566) – 2025-12-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7566
  - ggml-cuda: fix race condition in cumsum
  - remove unneccesary sync_threads
- b7567 (b7567) – 2025-12-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7567
- b7568 (b7568) – 2025-12-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7568
  - common: fix return value check for setpriority
  - tools: add logging for process priority setting
- b7569 (b7569) – 2025-12-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7569
- b7571 (b7571) – 2025-12-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7571
- b7572 (b7572) – 2025-12-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7572
  - Fix `msg` typo
  - Fix thread safety in destroy() to support generation abortion in lifecycle callbacks.
  - UI polish: stack new message change from below; fix GGUF margin not in view port
  - Bug fixes: rare racing condition when main thread updating view and and default thread updating messages at the same time; user input not disabled during generation.
  - Bump dependencies' versions; Deprecated outdated dsl usage.
- b7574 (b7574) – 2025-12-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7574
  - Prevent crash if TTFT >300sec, boosted to 90 days
  - server : allow configurable HTTP timeouts for child models
  - server : pass needed timeouts from params only
- b7579 (b7579) – 2025-12-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7579
  - CUDA: add log line when mxfp4 acceleration is used
  - add in backend_get_features
- b7580 (b7580) – 2025-12-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7580
  - kleidiai: add and integrate SVE 256-bit vector-length kernel
  - updated for review comments
- b7581 (b7581) – 2025-12-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7581
- b7582 (b7582) – 2025-12-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7582
  - sampling: reuse token data buffer in llama_sampler_sample
  - move cur buffer before timing section, after samplers
  - minor : fix build
- b7583 (b7583) – 2025-12-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7583
  - lora: count lora nodes in graph_max_nodes
  - 3 nodes per weight
  - 4 nodes
  - keep track n_lora_nodes from llama_model
  - fix assert
  - rm redundant header
  - common: load adapters before context creation
  - use 6 nodes
- b7585 (b7585) – 2025-12-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7585
  - common : default content to an empty string
  - common : fix tests that break when content != null
- b7588 (b7588) – 2025-12-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b7588
  - cmake: work around broken IntelSYCLConfig.cmake in oneAPI 2025.x
  - [AI] sycl: auto-detect and skip incompatible IntelSYCL package
  - refactor: improve SYCL provider handling and error messages in CMake configuration
  - refactor: enhance SYCL provider validation and error handling in CMake configuration
  - ggml-sycl: wrap find_package(IntelSYCL) to prevent build crashes
- b7589 (b7589) – 2025-12-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b7589
- b7590 (b7590) – 2025-12-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b7590
- b7591 (b7591) – 2025-12-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b7591
- b7592 (b7592) – 2025-12-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b7592
  - add count equal for metal
  - remove trailing whitespace
  - updated doc ops table
  - changed shmem to i32
  - added multi tg and templating
  - removed BLAS support from Metal docs
  - Apply suggestions from code review
  - add memset to set dst to 0
  - metal : cleanup
- b7593 (b7593) – 2025-12-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b7593
  - Inital commit, debugging q5_k_s quant
  - Made hf_to_gguf extend whisper to reduce code duplication
  - addressed convert_hf_to_gguf pull request issue
- b7595 (b7595) – 2025-12-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b7595
- b7598 (b7598) – 2026-01-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7598
  - chat: make tool description and parameters optional per OpenAI spec
  - refactor: use value() for cleaner optional field access
- b7599 (b7599) – 2026-01-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7599
- b7600 (b7600) – 2026-01-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7600
  - vulkan: extend topk_moe to handle sigmoid w/exp_probs_b for nemotron
  - change test_topk_moe to allow results in arbitrary order
  - disable sigmoid fusion for moltenvk
- b7601 (b7601) – 2026-01-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7601
- b7603 (b7603) – 2026-01-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7603
  - model: add Solar-Open model
  - vocab: add solar-open to end eog blacklist
  - model: add proper llm type
  - chat: basic template for solar open
  - typo: fix comment about vocab
  - convert: sugested changes
  - convert: suggested changes
  - chat: change reasoning end tag for solar-open
  - llama-chat: add solar-open template
- b7605 (b7605) – 2026-01-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7605
  - WIP: Initial commit for fixing JinaBert original FF type support
  - convert: add jina-v2-de tokenizer variant for German_Semantic_V3
  - convert: fix token collision in BERT phantom vocab conversion
  - convert: add feed_forward_type metadata
  - model: add feed_forward_type metadata for jina-bert-v2
  - model: jina-bert-v2 support standard GELU FFN variant
  - model: remove ffn_type, detect FFN variant from tensor dimensions
  - Update src/llama-model.cpp
  - Update src/llama-model.cpp
  - Update src/models/bert.cpp
  - Update src/models/bert.cpp
  - revert collision fix to be handled in separate PR
- b7607 (b7607) – 2026-01-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7607
  - Support Youtu-VL Model
  - merge code
  - fix bug
  - revert qwen2 code & support rsplit in minja.hpp
  - update warm info
  - fix annotation
  - u
  - revert minja.hpp
  - fix
  - Do not write routed_scaling_factor to gguf when routed_scaling_factor is None
  - fix expert_weights_scale
  - LGTM after whitespace fixes
  - fix
  - fix
  - fix
  - layers to layer_index
  - enum fix
- b7608 (b7608) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7608
  - remove modern-bert iswa template
  - forgotten
- b7609 (b7609) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7609
  - ggml-cuda: fixed assertion in ggml_cuda_cpy (#18140)
  - ggml-cuda: changes in data types to int64_t
  - ggml-cuda: added asserts for CUDA block numbers
  - ggml-cuda: changed the condition for y and z dimension
- b7610 (b7610) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7610
- b7611 (b7611) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7611
  - vocab : reduce debug logs about non-EOG control tokens
  - cont : add comment
- b7612 (b7612) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7612
- b7613 (b7613) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7613
- b7614 (b7614) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7614
  - Add Maincoder model support
  - Removed SPM model vocabulary setting and MOE related GGUF parameters
  - removed set_vocab
  - added new line
  - Fix formatting
  - Add a new line for PEP8
- b7615 (b7615) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7615
- b7616 (b7616) – 2026-01-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7616
  - vulkan: Optimize GGML_OP_CUMSUM
  - use 2 ELEM_PER_THREAD for AMD/Intel
  - address feedback
- b7617 (b7617) – 2026-01-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7617
  - refactor: refactor silu
  - refactor: optimize swiglu
  - refactor: remove unncessary if in swiglu
  - refactor: refactor swiglu_oai
  - chore: fix formatting issue
- b7618 (b7618) – 2026-01-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7618
  - CUDA: Fixed obj byte size instead of obj count being passed to pool alloc (fattn-common, dst_tmp_meta)
  - CUDA: Explicitly casted some of the int alloc counts before multiplication in argsort
- b7619 (b7619) – 2026-01-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7619
- b7620 (b7620) – 2026-01-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7620
- b7621 (b7621) – 2026-01-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7621


## 2025-12-20: Update to llama.cpp b7488

- b7378 (b7378) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7378
- b7379 (b7379) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7379
- b7380 (b7380) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7380
- b7381 (b7381) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7381
- b7382 (b7382) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7382
- b7383 (b7383) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7383
- b7384 (b7384) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7384
- b7385 (b7385) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7385
  - fix - w64devkit build
  - fix - w64devkit build private scope
- b7386 (b7386) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7386
- b7387 (b7387) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7387
- b7388 (b7388) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7388
- b7393 (b7393) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7393
- b7394 (b7394) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7394
  - models : fix YaRN regression + consolidate logic
  - cont : fix the fix
  - cont : remove header
  - cont : add header
- b7397 (b7397) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7397
- b7398 (b7398) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7398
- b7399 (b7399) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7399
  - common : refactor common_sampler + grammar logic changes
  - tests : increase max_tokens to get needed response
  - batched : fix uninitialized samplers
- b7400 (b7400) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7400
- b7401 (b7401) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7401
- b7402 (b7402) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7402
- b7404 (b7404) – 2025-12-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7404
- b7405 (b7405) – 2025-12-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b7405
  - [model] add glm-asr support
  - fix format for ci
  - fix convert format for ci
  - update glm_asr convert script & use build_ffn for glm_asr clip & use build_stack for padding and review
  - check root architecture for convert hf script
  - fix conficlt with upstream
  - fix convert script for glm asr & format clip-impl
  - format
  - restore hparams text
  - improved conversion
- b7406 (b7406) – 2025-12-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b7406
  - support gpt-oss GPU by OP add-id, mul_mat for mxfp4, swiglu_oai, fix warning
  - fix fault ut case, update ops.md
  - rebase, fix format issue
- b7410 (b7410) – 2025-12-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b7410
  - mtmd: refactor audio preprocessing
  - refactor
  - wip
  - wip (2)
  - improve constructor
  - fix use_natural_log
  - fix padding for short input
  - clean up
  - remove need_chunking
- b7411 (b7411) – 2025-12-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b7411
  - metal: use shared buffers on eGPU
  - metal: use shared buffers on eGPU
  - metal: use shared buffers on eGPU
- b7413 (b7413) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7413
  - kv-cache : fix state restore with fragmented cache (#17527)
  - tests : update logic
  - cleanup: tightened state_read_meta sig, added is_contiguous case
  - fix: state_read_meta arg reorder loose ends
- b7414 (b7414) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7414
  - vocab: add KORMo Tokenizer
  - model: add KORMoForCausalLM
  - vocab: change pretokenizer to qwen2
  - lint: fix unintended line removal
  - model: make qwen2 bias tensor optional
  - model: use qwen2 architecture for KORMo
- b7415 (b7415) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7415
  - feat: add run_mtmd script for hexagon
  - fix: fix issue in fp16xfp32 mm
  - fix: remove opt_experiment for fp16xfp32 mm
  - fix: ggml-hexagon: matmul fp16xfp32 support non-contigious src0
  - fix: fix syntax check for run-mtmd.sh for cli
- b7418 (b7418) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7418
  - llama : add support for NVIDIA Nemotron Nano 3
- b7422 (b7422) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7422
  - graph : reuse hybrid graphs
  - graph : reuse recurrent graphs
  - graph : fix reuse check for recurrent inputs
  - memory : move the recurrent state into the memory context
  - Revert "memory : move the recurrent state into the memory context"
  - cont : fix build
- b7423 (b7423) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7423
- b7426 (b7426) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7426
  - common : expose json-schema functionality to extract type info
  - common : fix peg parser negation during needs_more_input
  - common : add some defensive measures in constructed peg parser
  - common : add nemotron nano 3 support
  - common : add nemotron nano 3 tests
  - remove debug line
- b7429 (b7429) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7429
  - convert ok
  - no deepstack
  - less new tensors
  - cgraph ok
  - add mrope for text model
  - faster patch merger
  - add GGML_ROPE_TYPE_MRNORM
  - add support for metal
  - move glm4v do dedicated graph
  - convert: add norm_embd
  - clip: add debugging fn
  - working correctly
  - fix style
  - use bicubic
  - fix mrope metal
  - improve cpu
  - convert to neox ordering on conversion
  - revert backend changes
  - force stop if using old weight
  - support moe variant
  - fix conversion
  - fix convert (2)
  - Update tools/mtmd/clip-graph.h
  - process mrope_section on TextModel base class
  - resolve conflict merge
- b7432 (b7432) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7432
  - It's Qwen3 Next, the lean mean token generation machine!
  - Apply patches from thread
  - Remove recurrent version, only keep chunked and autoregressive
  - Remove unnecessary conts and asserts
  - Remove more extra conts and asserts
  - Cleanup masking
- b7433 (b7433) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7433
  - arg: clarify auto kvu/np being set on server
  - improve docs
  - use invalid_argument
- b7434 (b7434) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7434
  - arch: refactor LLM_TENSOR_NAMES
  - update docs
  - typo
  - fix LLM_ARCH_NEMOTRON_H_MOE
  - show more meaningful error message on missing tensor
  - fix and tested LLM_ARCH_NEMOTRON_H_MOE
- b7436 (b7436) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7436
  - server: fix crash when batch > ubatch with embeddings (#12836)
  - Add parameter validation in main() after common_params_parse()
  - When embeddings enabled and n_batch > n_ubatch:
  - Log warnings explaining the issue
  - Automatically set n_batch = n_ubatch
  - Prevent server crash
  - Build: Compiles successfully
  - Validation triggers: Warns when -b > -ub with --embedding
  - Auto-correction works: Adjusts n_batch = n_ubatch
  - No false positives: Valid params don't trigger warnings
  - Verified on macOS M3 Pro with embedding model
  - Update tools/server/server.cpp
- b7437 (b7437) – 2025-12-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b7437
- b7438 (b7438) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7438
- b7439 (b7439) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7439
- b7440 (b7440) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7440
- b7441 (b7441) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7441
- b7442 (b7442) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7442
- b7444 (b7444) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7444
- b7445 (b7445) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7445
- b7446 (b7446) – 2025-12-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b7446
  - UI: implement basic UI components
  - util: implement performance monitor; wrap it with a viewmodel
  - util: implement user preferences utility
  - UI: implement core flow's screens
  - UI: add a new MainActivity; update manifest
  - [WIP] DI: implement simple local vm factory provider
  - UI: disable triggering drawer via gesture; enable alert dialog on back navigation inside conversation and benchmark
  - UI: allow drawer's gesture control only on Home and Settings screens; enable alert dialog on back navigation inside conversation and benchmark
  - UI: split a nested parent settings screen into separate child settings screens
  - UI: polish system prompt setup UI
  - Deps: bump Kotlin plugin; introduce KSP; apply in :app subproject
  - DB: setup Room database
  - data: introduce repo for System Prompt; flow data from Room to VM
  - bugfix: properly handle user's quitting conversation screen while tokens in generation
  - UI: rename `ModeSelection` to `ModelLoading` for better clarity
  - UI: update app name to be more Arm
  - UI: polish conversation screen
  - data: code polish
  - UI: code polish
  - bugfix: handle user quitting on model loading
  - UI: locks user in alert dialog when model is unloading
  - vm: replace token metrics stubs with actual implementation
  - UI: refactor top app bars
  - nit: combine temperatureMetrics and useFahrenheit
  - DI: introduce Hilt plugin + processor + lib dependencies
  - DI: make app Hilt injectable
  - DI: make viewmodels Hilt injectable
  - DI: replace manual DI with Hilt DI
  - UI: optimize AppContent's composing
  - bugfix: wait for model to load before navigating to benchmark screen; use NavigationActions instead of raw navController
  - UI: navigation with more natural animated transitions
  - DI: Optimize AppModule
  - Feature: Introduce ModelRepository and ModelsManagementViewModel; update AppModule
  - UI: polish UI for ModelsManagementScreen; inject ModelsManagementVieModel
  - DI: abstract the protocol of SystemPromptRepository; update AppModule
  - data: [WIP] prepare for ModelRepository refactor & impl
  - data: introduce Model entity and DAO; update DI module
  - UI: replace Models Management screen's stubbing with instrumentation
  - UI: polish sort order menu
  - data: import local model with file picker
  - bugfix: use List instead of Collection for ModelDao's deletion
  - data: add a util file for extracting file name & size and model metadata
  - UI: enrich ModelManagementState; extract filename to show correct importing UI
  - UI: implement multiple models deletion; update Models Management screen
  - UI: handle back navigation when user is in multi-selection mode
  - util: extract file size formatting into ModelUtils
  - UI: add a confirmation step when user picks a file; refactor model import overlay into AlertDialog
  - UI: extract a shared ModelCard component
  - UI: replace model selection screen's data stubbing; add empty view
  - nit: tidy SystemPromptViewModel
- b7470 (b7470) – 2025-12-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b7470
- b7472 (b7472) – 2025-12-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b7472
- b7475 (b7475) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7475
  - ASR with LFM2-Audio-1.5B
  - Set rope_theta
  - Fix comment
  - Remove rope_theta setting
  - Address PR feedback
  - rename functions to conformer
  - remove some redundant ggml_cont
  - fix missing tensor
  - add prefix "a." for conv tensors
  - remove redundant reshape
  - clean up
  - add test model
- b7476 (b7476) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7476
- b7480 (b7480) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7480
  - presets: refactor, allow cascade presets from different sources
  - update docs
  - fix neg arg handling
  - fix empty mmproj
  - also filter out server-controlled args before to_ini()
  - skip loading custom_models if not specified
  - fix unset_reserved_args
  - fix crash on windows
- b7481 (b7481) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7481
  - llama-server: friendlier error msg when ctx < input
  - llama-server: use string_format inline
  - fix test
- b7482 (b7482) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7482
- b7483 (b7483) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7483
  - arg: fix order to use short form before long form
  - arg: update doc
  - arg: update test-arg-parser
  - arg: address review feedback from ngxson
  - arg: update doc
- b7484 (b7484) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7484
  - feat: implement real Q8_0
  - feat: adding cmake option for configuring FP32 quantize group size
  - typo: set() shall be used
- b7486 (b7486) – 2025-12-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7486
  - remove non-windows zip artifacts
  - add cuda dll links
- b7487 (b7487) – 2025-12-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7487
  - server: support autoload model, support preset-only options
  - add docs
  - load-on-startup
  - fix
  - Update common/arg.cpp
- b7488 (b7488) – 2025-12-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7488


## 2025-12-13: Update to llama.cpp b7376

- b7285 (b7285) – 2025-12-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b7285
- b7296 (b7296) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7296
  - metal : fix build
  - tests : fix context destruction
- b7298 (b7298) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7298
- b7300 (b7300) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7300
- b7301 (b7301) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7301
  - llama : remove quantization sanity check
  - llama : remove unused pruned_attention_w and is_clip_model vars
- b7302 (b7302) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7302
  - Improve error handling for search path existence checks
  - Improve cache file existence check with error code
  - Simplify existence check for search paths
  - Fix logging path in error message for posix_stat
  - Update ggml/src/ggml-backend-reg.cpp
  - Adapt to the coding standard
- b7306 (b7306) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7306
- b7307 (b7307) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7307
  - Feat: Added vulkan circular tiling support
  - Feat: Added cpu circular
  - Feat: Added cuda kernels
  - Added tests
  - Added tests
  - Removed non-pad operations
  - Removed unneded changes
  - removed backend non pad tests
  - Update test-backend-ops.cpp
  - Fixed comment on pad test
  - removed trailing whitespace
  - Removed unneded test in test-backend-ops
  - Removed removed test from calls
  - Update ggml/src/ggml-vulkan/vulkan-shaders/pad.comp
  - Fixed alignment
  - Formatting
  - Format pad
  - Format
  - Clang format
  - format
  - format
  - don't change so much stuff
  - clang format and update to bool
  - fix duplicates
  - don't need to fix the padding
  - make circular bool
  - duplicate again
  - rename vulkan to wrap around
  - Don't need indent
  - moved to const expr
  - removed unneded extra line break
  - More readable method calls
  - Minor wording changes
  - Added final newline
  - Update ggml/include/ggml.h
  - Update ggml/include/ggml.h
  - Added circular pad ext tests
  - Gate non circular pad devices
  - Cleaned gating of non-circular pad devices
- b7310 (b7310) – 2025-12-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b7310
  - vulkan: perf_logger improvements
  - Move perf_logger from device to ctx.
  - Add an env var to control the frequency we dump the stats. If you set a very
  - Add a fusion info string to the tracking, only log one item per fused op.
  - Fix MUL_MAT_ID flops calculation.
  - fix vector sizes
- b7311 (b7311) – 2025-12-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b7311
  - sycl: add missing BF16 conversion support for Intel oneAPI
  - Fix Line 645: Trailing whitespace
- b7312 (b7312) – 2025-12-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b7312
- b7313 (b7313) – 2025-12-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b7313
- b7314 (b7314) – 2025-12-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b7314
  - Optimize Vulkan shader for matrix-vector multiplication
  - Revert changes on compute_outputs and main
  - Fix trailing whitespace
- b7315 (b7315) – 2025-12-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b7315
- b7316 (b7316) – 2025-12-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b7316
  - ggml-cpu: add ggml_thread_cpu_relax with Zihintpause support
  - cmake: enable RISC-V zihintpause extension for Spacemit builds
  - readme : add ZIHINTPAUSE support for RISC-V
- b7317 (b7317) – 2025-12-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b7317
  - ggml-cuda: optimize solve_tri_f32_fast and fix stride handling
  - Switch from using shared memory for the RHS/solution matrix to a register-based approach (x_low, x_high), reducing shared memory pressure and bank conflicts.
  - Implement explicit `fmaf` instructions for the reduction loop.
  - Update kernel arguments to pass strides in bytes rather than elements to align with standard ggml tensor arithmetic (casting to `char *` before addition).
  - Remove unused `MAX_K_FAST` definition.
  - Small cleanup
  - Remove comments in solve_tri.cu
  - Update ggml/src/ggml-cuda/solve_tri.cu
  - Update ggml/src/ggml-cuda/solve_tri.cu
  - Update ggml/src/ggml-cuda/solve_tri.cu
  - Use const for variables in solve_tri.cu
  - Replace fmaf with more readable code
  - remove last fmaf
- b7318 (b7318) – 2025-12-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b7318
- b7324 (b7324) – 2025-12-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b7324
  - support bfloat16 release package
  - add fallback file
- b7325 (b7325) – 2025-12-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b7325
  - server: delegate result_state creation to server_task
  - remove unued states
  - add more docs
- b7327 (b7327) – 2025-12-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b7327
  - use fill instead of scale_bias in grouped expert selection
  - do not explicitly use _inplace
- b7328 (b7328) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7328
  - add support for rnj1
  - refactor gemma3 to support rnj-1
  - address review comments
- b7329 (b7329) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7329
  - llama : add token support to llama-grammar
  - fix inverse token comment
  - refactor trigger_patterns to replay tokens instead of the entire string
  - add token documentation
  - fix test-llama-grammar
  - improve test cases for tokens
- b7330 (b7330) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7330
- b7331 (b7331) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7331
  - cann: add support for partial RoPE and Vision mode
  - Support for partial RoPE (rope_dims < ne0):
  - Split tensor into head (first rope_dims dimensions) and tail portions
  - Apply rotation only to head portion using RotaryPositionEmbedding operator
  - Copy unrotated tail portion directly from source to destination
  - Handle both contiguous and non-contiguous tensor layouts
  - Support for Vision mode (GGML_ROPE_TYPE_VISION):
  - Set rope_dims = ne0 for Vision mode to rotate entire tensor
  - Vision mode pairs dimension i with dimension i+n_dims (where n_dims = ne0/2)
  - No tail handling needed since entire tensor is rotated
  - Use has_tail flag to determine execution path: head/tail splitting when
  - Support both F32 and F16 data types with intermediate F32 conversion
  - Copy non-contiguous tensors to contiguous buffers before calling
  - Improve cache invalidation logic to include rope_dims and indep_sects
  - cann: fix review comment
- b7332 (b7332) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7332
  - console: allow using arrow left/right to edit the line (with UTF-8 support)
  - console: fix arrow keys on Windows using private-use Unicode
  - console: add Home/End key support for Windows and Linux
  - console: add basic Up/Down history navigation
  - fix build
  - console: allow using arrow left/right to edit the line (with UTF-8 support)
  - console: fix arrow keys on Windows using private-use Unicode
  - console: add Home/End key support for Windows and Linux
  - console: add basic Up/Down history navigation
  - console: remove unreachable wc == 0 check after VK switch
  - console: add Ctrl+Left/Right word navigation
  - Add KEY_CTRL_ARROW_LEFT and KEY_CTRL_ARROW_RIGHT codes
  - Windows: detect CTRL modifier via dwControlKeyState
  - Linux: parse ANSI sequences with modifier (1;5D/C)
  - Implement move_word_left/right with space-skipping logic
  - Refactor escape sequence parsing to accumulate params
  - console: add Delete key support
  - Windows: VK_DELETE detection
  - Linux: ESC[3~ sequence parsing
  - Forward character deletion with UTF-8 support
  - console: implement bash-style history editing
  - Edit any history line during UP/DOWN navigation, edits persist
  - Pressing Enter appends edited version as new history entry
  - Original line stay untouched in their positions
  - clean up
  - better history impl
  - fix decode_utf8
- b7333 (b7333) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7333
  - nit, DeepSeek V1 MoE is 16B
  - base type on n_ff_exp instead
- b7334 (b7334) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7334
  - This just sets the Mach-O current version to 0 to get it building
- b7335 (b7335) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7335
- b7336 (b7336) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7336
- b7337 (b7337) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7337
  - fix: Provide macos-specific backtrace printing to avoid terminal death
  - fix: Add GGML_BACKTRACE_LLDB env var to enable using lldb for backtrace
- b7339 (b7339) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7339
  - Add DIAG for CUDA
  - Refactor parameters
- b7340 (b7340) – 2025-12-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7340
  - feat: Add a batched version of ssm_conv
  - feat: Optimized SSM_SCAN kernel for metal
  - test: Add test-backend-ops perf tests for SSM_CONV
  - test: Real representitive tests for SSM_CONV
  - refactor: Use function constant for ssm_conv batch size
  - test: backend op tests for ssm_scan from granite4 1b-h
  - style: remove commented out templates
  - feat: float4 version of ssm_conv_batched
  - fix: Add missing ggml_metal_cv_free
- b7342 (b7342) – 2025-12-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7342
- b7343 (b7343) – 2025-12-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7343
- b7345 (b7345) – 2025-12-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7345
- b7347 (b7347) – 2025-12-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7347
  - model : Qwen3-Next-80B-A3B has 48 layers
  - model : Add 80B-A3B type name
- b7348 (b7348) – 2025-12-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7348
  - wip
  - wip
  - fix logging, add display info
  - handle commands
  - add args
  - wip
  - move old cli to llama-completion
  - rm deprecation notice
  - move server to a shared library
  - move ci to llama-completion
  - add loading animation
  - add --show-timings arg
  - add /read command, improve LOG_ERR
  - add args for speculative decoding, enable show timings by default
  - add arg --image and --audio
  - fix windows build
  - support reasoning_content
  - fix llama2c workflow
  - color default is auto
  - fix merge conflicts
  - properly fix color problem
  - better loading spinner
  - make sure to clean color on force-exit
  - also clear input files on "/clear"
  - simplify common_log_flush
  - add warning in mtmd-cli
  - implement console writter
  - fix data race
  - add attribute
  - fix llama-completion and mtmd-cli
  - add some notes about console::log
  - fix compilation
- b7349 (b7349) – 2025-12-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7349
- b7350 (b7350) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7350
  - ggml : remove GGML_KQ_MASK_PAD constant
  - cont : remove comment
- b7351 (b7351) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7351
  - tests: update barrier test to check for race condition in active threads
  - cpu: combine n_graph and n_threads into a single atomic update
  - tests: add multi-graph test for test_barrier
- b7352 (b7352) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7352
  - llama-server: recursive GGUF loading
  - server : router config POC (INI-based per-model settings)
  - server: address review feedback from @aldehir and @ngxson
  - Simplify parser instantiation (remove arena indirection)
  - Optimize grammar usage (ws instead of zero_or_more, remove optional wrapping)
  - Fix last line without newline bug (+ operator instead of <<)
  - Remove redundant end position check
  - Remove auto-reload feature (will be separate PR per @ngxson)
  - Keep config.ini auto-creation and template generation
  - Preserve per-model customization logic
  - server: adopt aldehir's line-oriented PEG parser
  - Use p.chars(), p.negate(), p.any() instead of p.until()
  - Support end-of-line comments (key=value # comment)
  - Handle EOF without trailing newline correctly
  - Strict identifier validation ([a-zA-Z_][a-zA-Z0-9_.-]*)
  - Simplified visitor (no pending state, no trim needed)
  - Grammar handles whitespace natively via eol rule
  - Reject section names starting with LLAMA_ARG_*
  - Accept only keys starting with LLAMA_ARG_*
  - Require explicit section before key-value pairs
  - server: fix CLI/env duplication in child processes
  - add common/preset.cpp
  - fix compile
  - cont
  - allow custom-path models
  - add falsey check
  - server: fix router model discovery and child process spawning
  - Sanitize model names: replace / and \ with _ for display
  - Recursive directory scan with relative path storage
  - Convert relative paths to absolute when spawning children
  - Filter router control args from child processes
  - Refresh args after port assignment for correct port value
  - Fallback preset lookup for compatibility
  - Fix missing argv[0]: store server binary path before base_args parsing
  - Revert "server: fix router model discovery and child process spawning"
  - clarify about "no-" prefix
  - correct render_args() to include binary path
  - also remove arg LLAMA_ARG_MODELS_PRESET for child
  - add co-author for ini parser code
  - also set LLAMA_ARG_HOST
  - add CHILD_ADDR
  - Remove dead code
- b7353 (b7353) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7353
  - cli: enable jinja by default
  - Update common/arg.cpp
- b7354 (b7354) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7354
  - clip: add support for fused qkv in build_vit
  - use bulid_ffn whenever possible
  - fix internvl
  - mtmd-cli: move image to beginning
  - test script: support custom args
- b7356 (b7356) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7356
  - fix test failure
  - fix: correct scaling calculations in rope_cache_init
  - fix: optimize element copying in rope_hex_f32 using memcpy
  - fix: optimize loop boundaries in rope_hex_f32 for better performance
  - feat: add profiling macros for performance measurement in operations
- b7358 (b7358) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7358
  - batch : fix sequence id ownage
  - cont : reduce allocations
- b7360 (b7360) – 2025-12-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7360
  - Extended TRI
  - Fix whitespace
  - chore: update webui build output
  - Just use cuBLAS for everything...
  - Merge both versions
  - Remove incorrect imports causing failures for CI
  - Still failing... remove all direct cublas imports and rely on common imports from "common.cuh"
  - Defines for hipBlas
  - Aaaand MUSA defines...
  - I hate this job...
  - Stupid typo...
  - Update ggml/src/ggml-cuda/solve_tri.cu
- b7362 (b7362) – 2025-12-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7362
  - enable mmf for RDNA3
  - disable mmf for some shape
  - move some mmvf to mmf
  - more mmfv to mmf
  - 3 is good in mmvf
- b7363 (b7363) – 2025-12-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7363
- b7364 (b7364) – 2025-12-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7364
- b7366 (b7366) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7366
  - arg: add -mm and -mmu as short form of --mmproj and --mmproj-url
  - correct order
  - update docs
- b7368 (b7368) – 2025-12-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7368
- b7369 (b7369) – 2025-12-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7369
  - ggml-cpu:fix RISC-V Q4_0 repack select and RVV feature reporting
  - using the name VLEN instead of CNT
  - Update ggml/include/ggml-cpu.h
- b7370 (b7370) – 2025-12-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7370
- b7371 (b7371) – 2025-12-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7371
  - models : fix the attn_factor for mistral3 graphs
  - cont : rework attn_factor correction logic
  - cont : make deepseek2 consistent
  - cont : add TODO
  - cont : special-case DSv2
  - cont : revert Mistral 3 Large changes
  - cont : fix DS2 to use the original attn_factor
  - cont : minor comments
- b7372 (b7372) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7372
- b7374 (b7374) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7374
- b7375 (b7375) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7375
  - clip: move model cgraphs into their own files
  - more explicit enums
  - fix linux build
  - fix naming
  - missing headers
  - nits: add comments for contributors
- b7376 (b7376) – 2025-12-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7376
  - args: support negated args
  - update docs
  - fix typo
  - add more neg options
  - Apply suggestions from code review
  - rm duplicated arg
  - fix LLAMA_ARG_NO_HOST
  - add test


## 2025-12-05: Update to llama.cpp b7278

- b7218 (b7218) – 2025-12-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7218
- b7219 (b7219) – 2025-12-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7219
- b7220 (b7220) – 2025-12-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7220
- b7222 (b7222) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7222
- b7223 (b7223) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7223
- b7224 (b7224) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7224
- b7225 (b7225) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7225
- b7227 (b7227) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7227
- b7229 (b7229) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7229
  - Revert "rm unused fn"
  - server: explicitly set exec path when create new instance
  - put back TODO
  - only call get_server_exec_path() once
  - add fallback logic
- b7230 (b7230) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7230
- b7231 (b7231) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7231
  - server: remove default "gpt-3.5-turbo" model name
  - do not reflect back model name from request
  - fix test
- b7233 (b7233) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7233
- b7235 (b7235) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7235
- b7236 (b7236) – 2025-12-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b7236
- b7237 (b7237) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7237
- b7239 (b7239) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7239
- b7240 (b7240) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7240
  - Compute row size for the temp buffer based on the output of the first pass.
  - Update shader addressing math to use the output row size
  - Pass the output row size as "ncols_output", what used to be "ncols_output" is now "k"
- b7243 (b7243) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7243
  - server: add --media-path for local media files
  - remove unused fn
- b7245 (b7245) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7245
- b7247 (b7247) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7247
  - Faster tensors (#8)
  - Use map for shader replacements instead of pair of strings
  - Wasm (#9)
  - webgpu : fix build on emscripten
  - more debugging stuff
  - test-backend-ops: force single thread on wasm
  - fix single-thread case for init_tensor_uniform
  - use jspi
  - add pthread
  - test: remember to set n_thread for cpu backend
  - Add buffer label and enable dawn-specific toggles to turn off some checks
  - Intermediate state
  - Fast working f16/f32 vec4
  - Working float fast mul mat
  - Clean up naming of mul_mat to match logical model, start work on q mul_mat
  - Setup for subgroup matrix mat mul
  - Basic working subgroup matrix
  - Working subgroup matrix tiling
  - Handle weirder sg matrix sizes (but still % sg matrix size)
  - Working start to gemv
  - working f16 accumulation with shared memory staging
  - Print out available subgroup matrix configurations
  - Vectorize dst stores for sg matrix shader
  - Gemv working scalar
  - Minor set_rows optimization (#4)
  - updated optimization, fixed errors
  - non vectorized version now dispatches one thread per element
  - Simplify
  - Change logic for set_rows pipelines
  - Comment on dawn toggles
  - Working subgroup matrix code for (semi)generic sizes
  - Remove some comments
  - Cleanup code
  - Update dawn version and move to portable subgroup size
  - Try to fix new dawn release
  - Update subgroup size comment
  - Only check for subgroup matrix configs if they are supported
  - Add toggles for subgroup matrix/f16 support on nvidia+vulkan
  - Make row/col naming consistent
  - Refactor shared memory loading
  - Move sg matrix stores to correct file
  - Working q4_0
  - Formatting
  - Work with emscripten builds
  - Fix test-backend-ops emscripten for f16/quantized types
  - Use emscripten memory64 to support get_memory
  - Add build flags and try ci
  - Remove extra whitespace
  - Move wasm single-thread logic out of test-backend-ops for cpu backend
  - Disable multiple threads for emscripten single-thread builds in ggml_graph_plan
- b7248 (b7248) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7248
  - llama-server: fix duplicate HTTP headers in multiple models mode (#17693)
  - llama-server: address review feedback from ngxson
  - restrict scope of header after std::move
  - simplify header check (remove unordered_set)
- b7250 (b7250) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7250
  - Remove the build of openeuler-cann in release
  - Remove the relevant release files
- b7251 (b7251) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7251
- b7252 (b7252) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7252
- b7253 (b7253) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7253
- b7255 (b7255) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7255
- b7256 (b7256) – 2025-12-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b7256
  - CUDA: generalized (mma) FA, add Volta support
  - use struct for MMA FA kernel config
- b7261 (b7261) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7261
- b7262 (b7262) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7262
  - build: enable parallel builds in msbuild using MTT
  - check LLAMA_STANDALONE
- b7263 (b7263) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7263
- b7264 (b7264) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7264
- b7265 (b7265) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7265
- b7266 (b7266) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7266
- b7268 (b7268) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7268
- b7270 (b7270) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7270
- b7271 (b7271) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7271
- b7273 (b7273) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7273
  - server: move msg diffs tracking to HTTP thread
  - wip
  - tool call tests ok
  - minor : style
  - cont : fix
  - move states to server_response_reader
  - add safe-guard
  - fix
  - fix 2
- b7274 (b7274) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7274
- b7275 (b7275) – 2025-12-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b7275
  - feat(wip): Port initial TRI impl from pervious work
  - fix: Remove argument for constant val override
  - feat: Move the ttype conditional to templating to avoid conditional in kernel
  - fix: Type fixes
  - feat: Add softplus for metal
  - feat: Add EXPM1 for metal
  - feat: Add FILL for metal
  - refactor: Branchless version of tri using _ggml_vec_tri_cmp as a mask
  - fix: Remove unused arguments
  - refactor: Use select instead of branch for softplus non-vec
- b7276 (b7276) – 2025-12-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b7276
  - Add support for CUMSUM and TRI for CUDA.
  - Minor optimizations.
  - Correct warp_prefix_inclusive_sum in float2 variant to return float2
  - Optimize TRI
  - Whitespace
  - Fix strides.
  - Implement double loop
  - Whitespace
  - Fix HIP compilation bugs
  - Optimizations + big case performance tests
  - Implement using CUB with fallback to custom kernel
  - Remove error message.
  - Fixes from code review
  - Comment out CPU-unsupported F16/BF16 cases to fix CI
  - Fine, you win :P
  - Fix last cast, use NO_DEVICE_CODE and GGML_UNUSED_VARS
  - Vary warp-size based on physical warp size
  - Add GGML_UNUSED_VARS in tri as well
  - Use constexpr and call prefix_inclusive with warp_size template param
  - Update ggml/src/ggml-cuda/cumsum.cu
  - Apply suggestions from code review
  - Change to tid % warp_size
  - Fix strides; hardcode mask; add ggml_lane_mask_t
  - Missing renames, remove unused get_warp_mask(), explicit calls to ggml_cuda_info()
  - Too hasty...
- b7278 (b7278) – 2025-12-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b7278
  - transform release binary root dir in tar to llama-bXXXX
  - bsdtar supports -s instead of --transform


## 2025-12-01: Update to llama.cpp b7213

- b7090 (b7090) – 2025-11-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b7090
- b7091 (b7091) – 2025-11-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b7091
- b7096 (b7096) – 2025-11-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b7096
- b7097 (b7097) – 2025-11-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b7097
- b7100 (b7100) – 2025-11-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7100
- b7101 (b7101) – 2025-11-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7101
- b7102 (b7102) – 2025-11-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7102
- b7103 (b7103) – 2025-11-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7103
- b7106 (b7106) – 2025-11-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7106
- b7107 (b7107) – 2025-11-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7107
- b7108 (b7108) – 2025-11-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b7108
- b7109 (b7109) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7109
- b7110 (b7110) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7110
- b7111 (b7111) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7111
- b7112 (b7112) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7112
- b7113 (b7113) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7113
- b7117 (b7117) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7117
- b7118 (b7118) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7118
- b7120 (b7120) – 2025-11-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b7120
- b7122 (b7122) – 2025-11-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7122
- b7123 (b7123) – 2025-11-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7123
- b7124 (b7124) – 2025-11-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7124
- b7126 (b7126) – 2025-11-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7126
- b7127 (b7127) – 2025-11-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7127
- b7128 (b7128) – 2025-11-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b7128
- b7129 (b7129) – 2025-11-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7129
- b7130 (b7130) – 2025-11-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b7130
- b7132 (b7132) – 2025-11-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7132
- b7134 (b7134) – 2025-11-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b7134
- b7136 (b7136) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7136
- b7137 (b7137) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7137
- b7138 (b7138) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7138
- b7139 (b7139) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7139
- b7140 (b7140) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7140
- b7141 (b7141) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7141
- b7142 (b7142) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7142
- b7144 (b7144) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7144
- b7146 (b7146) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7146
- b7148 (b7148) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7148
- b7149 (b7149) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7149
- b7150 (b7150) – 2025-11-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b7150
- b7151 (b7151) – 2025-11-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b7151
- b7152 (b7152) – 2025-11-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b7152
- b7154 (b7154) – 2025-11-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b7154
- b7157 (b7157) – 2025-11-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b7157
- b7158 (b7158) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7158
- b7159 (b7159) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7159
- b7160 (b7160) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7160
- b7161 (b7161) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7161
- b7162 (b7162) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7162
- b7163 (b7163) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7163
- b7164 (b7164) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7164
- b7165 (b7165) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7165
- b7166 (b7166) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7166
- b7167 (b7167) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7167
- b7168 (b7168) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7168
- b7169 (b7169) – 2025-11-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b7169
- b7170 (b7170) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7170
- b7171 (b7171) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7171
- b7172 (b7172) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7172
- b7175 (b7175) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7175
- b7176 (b7176) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7176
- b7177 (b7177) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7177
- b7178 (b7178) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7178
- b7179 (b7179) – 2025-11-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b7179
- b7180 (b7180) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7180
- b7181 (b7181) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7181
- b7182 (b7182) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7182
- b7183 (b7183) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7183
- b7184 (b7184) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7184
- b7185 (b7185) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7185
- b7186 (b7186) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7186
- b7187 (b7187) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7187
- b7188 (b7188) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7188
- b7189 (b7189) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7189
- b7190 (b7190) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7190
- b7191 (b7191) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7191
- b7192 (b7192) – 2025-11-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b7192
- b7194 (b7194) – 2025-11-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7194
- b7195 (b7195) – 2025-11-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7195
- b7196 (b7196) – 2025-11-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7196
- b7197 (b7197) – 2025-11-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7197
- b7198 (b7198) – 2025-11-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7198
- b7199 (b7199) – 2025-11-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7199
- b7200 (b7200) – 2025-11-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b7200
- b7201 (b7201) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7201
- b7202 (b7202) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7202
- b7203 (b7203) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7203
- b7204 (b7204) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7204
- b7205 (b7205) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7205
- b7206 (b7206) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7206
- b7207 (b7207) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7207
- b7208 (b7208) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7208
- b7209 (b7209) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7209
- b7210 (b7210) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7210
- b7211 (b7211) – 2025-11-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b7211
- b7213 (b7213) – 2025-12-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b7213


## 2025-11-14: Update to llama.cpp b7058

- b6959 (b6959) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6959
- b6960 (b6960) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6960
- b6961 (b6961) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6961
- b6962 (b6962) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6962
- b6963 (b6963) – 2025-11-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6963
- b6965 (b6965) – 2025-11-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6965
- b6966 (b6966) – 2025-11-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6966
- b6967 (b6967) – 2025-11-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6967
- b6968 (b6968) – 2025-11-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6968
- b6969 (b6969) – 2025-11-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6969
- b6970 (b6970) – 2025-11-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6970
- b6971 (b6971) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6971
- b6972 (b6972) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6972
- b6973 (b6973) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6973
- b6974 (b6974) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6974
- b6975 (b6975) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6975
- b6976 (b6976) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6976
- b6977 (b6977) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6977
- b6978 (b6978) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6978
- b6979 (b6979) – 2025-11-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6979
- b6980 (b6980) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6980
- b6981 (b6981) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6981
- b6982 (b6982) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6982
- b6983 (b6983) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6983
- b6984 (b6984) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6984
- b6985 (b6985) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6985
- b6986 (b6986) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6986
- b6987 (b6987) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6987
- b6988 (b6988) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6988
- b6989 (b6989) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6989
- b6990 (b6990) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6990
- b6992 (b6992) – 2025-11-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6992
- b6993 (b6993) – 2025-11-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6993
- b6994 (b6994) – 2025-11-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6994
- b6995 (b6995) – 2025-11-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6995
- b6996 (b6996) – 2025-11-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6996
- b6999 (b6999) – 2025-11-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6999
- b7002 (b7002) – 2025-11-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7002
- b7003 (b7003) – 2025-11-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b7003
- b7005 (b7005) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7005
- b7007 (b7007) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7007
- b7008 (b7008) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7008
- b7009 (b7009) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7009
- b7010 (b7010) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7010
- b7011 (b7011) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7011
- b7012 (b7012) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7012
- b7013 (b7013) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7013
- b7014 (b7014) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7014
- b7015 (b7015) – 2025-11-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b7015
- b7016 (b7016) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7016
- b7017 (b7017) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7017
- b7018 (b7018) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7018
- b7020 (b7020) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7020
- b7021 (b7021) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7021
- b7022 (b7022) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7022
- b7023 (b7023) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7023
- b7024 (b7024) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7024
- b7025 (b7025) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7025
- b7027 (b7027) – 2025-11-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b7027
- b7028 (b7028) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7028
- b7030 (b7030) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7030
- b7031 (b7031) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7031
- b7032 (b7032) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7032
- b7033 (b7033) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7033
- b7034 (b7034) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7034
- b7035 (b7035) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7035
- b7037 (b7037) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7037
- b7039 (b7039) – 2025-11-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b7039
- b7041 (b7041) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7041
- b7042 (b7042) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7042
- b7044 (b7044) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7044
- b7045 (b7045) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7045
- b7046 (b7046) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7046
- b7047 (b7047) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7047
- b7048 (b7048) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7048
- b7049 (b7049) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7049
- b7050 (b7050) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7050
- b7051 (b7051) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7051
- b7052 (b7052) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7052
- b7053 (b7053) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7053
- b7054 (b7054) – 2025-11-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b7054
- b7057 (b7057) – 2025-11-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7057
- b7058 (b7058) – 2025-11-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b7058


## 2025-11-05: Update to llama.cpp b6957

- b6919 (b6919) – 2025-11-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6919
- b6920 (b6920) – 2025-11-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6920
- b6922 (b6922) – 2025-11-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6922
- b6923 (b6923) – 2025-11-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6923
- b6924 (b6924) – 2025-11-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6924
- b6927 (b6927) – 2025-11-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6927
- b6929 (b6929) – 2025-11-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6929
- b6931 (b6931) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6931
- b6932 (b6932) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6932
- b6933 (b6933) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6933
- b6934 (b6934) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6934
- b6935 (b6935) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6935
- b6936 (b6936) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6936
- b6937 (b6937) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6937
- b6940 (b6940) – 2025-11-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6940
- b6941 (b6941) – 2025-11-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6941
- b6942 (b6942) – 2025-11-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6942
- b6943 (b6943) – 2025-11-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6943
- b6945 (b6945) – 2025-11-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6945
- b6947 (b6947) – 2025-11-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6947
- b6948 (b6948) – 2025-11-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6948
- b6949 (b6949) – 2025-11-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6949
- b6953 (b6953) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6953
- b6954 (b6954) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6954
- b6955 (b6955) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6955
- b6957 (b6957) – 2025-11-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6957


## 2025-11-01: Update to llama.cpp b6916

- b6904 (b6904) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6904
- b6905 (b6905) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6905
- b6906 (b6906) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6906
- b6907 (b6907) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6907
- b6908 (b6908) – 2025-11-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6908
- b6909 (b6909) – 2025-11-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6909
- b6910 (b6910) – 2025-11-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6910
- b6912 (b6912) – 2025-11-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6912
- b6915 (b6915) – 2025-11-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6915
- b6916 (b6916) – 2025-11-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6916


## 2025-10-31: Update to llama.cpp b6900

- b6793 (b6793) – 2025-10-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6793
- b6794 (b6794) – 2025-10-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6794
- b6795 (b6795) – 2025-10-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6795
- b6799 (b6799) – 2025-10-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b6799
- b6800 (b6800) – 2025-10-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b6800
- b6801 (b6801) – 2025-10-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6801
- b6802 (b6802) – 2025-10-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6802
- b6804 (b6804) – 2025-10-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6804
- b6808 (b6808) – 2025-10-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6808
- b6810 (b6810) – 2025-10-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6810
- b6811 (b6811) – 2025-10-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6811
- b6812 (b6812) – 2025-10-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6812
- b6813 (b6813) – 2025-10-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6813
- b6814 (b6814) – 2025-10-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6814
- b6815 (b6815) – 2025-10-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6815
- b6816 (b6816) – 2025-10-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6816
- b6817 (b6817) – 2025-10-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6817
- b6818 (b6818) – 2025-10-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6818
- b6821 (b6821) – 2025-10-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6821
- b6822 (b6822) – 2025-10-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6822
- b6823 (b6823) – 2025-10-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6823
- b6824 (b6824) – 2025-10-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6824
- b6825 (b6825) – 2025-10-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6825
- b6826 (b6826) – 2025-10-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6826
- b6827 (b6827) – 2025-10-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6827
- b6829 (b6829) – 2025-10-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6829
- b6833 (b6833) – 2025-10-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b6833
- b6834 (b6834) – 2025-10-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b6834
- b6836 (b6836) – 2025-10-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6836
- b6837 (b6837) – 2025-10-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6837
- b6838 (b6838) – 2025-10-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6838
- b6840 (b6840) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6840
- b6841 (b6841) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6841
- b6843 (b6843) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6843
- b6844 (b6844) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6844
- b6845 (b6845) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6845
- b6846 (b6846) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6846
- b6847 (b6847) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6847
- b6848 (b6848) – 2025-10-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6848
- b6849 (b6849) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6849
- b6850 (b6850) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6850
- b6851 (b6851) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6851
- b6852 (b6852) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6852
- b6853 (b6853) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6853
- b6854 (b6854) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6854
- b6855 (b6855) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6855
- b6856 (b6856) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6856
- b6857 (b6857) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6857
- b6858 (b6858) – 2025-10-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6858
- b6859 (b6859) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6859
- b6860 (b6860) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6860
- b6861 (b6861) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6861
- b6862 (b6862) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6862
- b6863 (b6863) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6863
- b6864 (b6864) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6864
- b6865 (b6865) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6865
- b6866 (b6866) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6866
- b6868 (b6868) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6868
- b6869 (b6869) – 2025-10-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6869
- b6870 (b6870) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6870
- b6871 (b6871) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6871
- b6872 (b6872) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6872
- b6873 (b6873) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6873
- b6874 (b6874) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6874
- b6875 (b6875) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6875
- b6876 (b6876) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6876
- b6877 (b6877) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6877
- b6878 (b6878) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6878
- b6879 (b6879) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6879
- b6880 (b6880) – 2025-10-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6880
- b6881 (b6881) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6881
- b6882 (b6882) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6882
- b6883 (b6883) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6883
- b6884 (b6884) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6884
- b6885 (b6885) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6885
- b6886 (b6886) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6886
- b6887 (b6887) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6887
- b6888 (b6888) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6888
- b6889 (b6889) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6889
- b6890 (b6890) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6890
- b6891 (b6891) – 2025-10-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6891
- b6895 (b6895) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6895
- b6896 (b6896) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6896
- b6897 (b6897) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6897
- b6898 (b6898) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6898
- b6900 (b6900) – 2025-10-31 – https://github.com/ggml-org/llama.cpp/releases/tag/b6900


## 2025-10-18: Update to llama.cpp b6792

- b6670 (b6670) – 2025-10-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6670
- b6671 (b6671) – 2025-10-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6671
- b6672 (b6672) – 2025-10-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6672
- b6673 (b6673) – 2025-10-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6673
- b6676 (b6676) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6676
- b6678 (b6678) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6678
- b6679 (b6679) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6679
- b6680 (b6680) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6680
- b6682 (b6682) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6682
- b6683 (b6683) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6683
- b6684 (b6684) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6684
- b6685 (b6685) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6685
- b6686 (b6686) – 2025-10-03 – https://github.com/ggml-org/llama.cpp/releases/tag/b6686
- b6687 (b6687) – 2025-10-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6687
- b6688 (b6688) – 2025-10-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6688
- b6689 (b6689) – 2025-10-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6689
- b6690 (b6690) – 2025-10-04 – https://github.com/ggml-org/llama.cpp/releases/tag/b6690
- b6691 (b6691) – 2025-10-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6691
- b6692 (b6692) – 2025-10-05 – https://github.com/ggml-org/llama.cpp/releases/tag/b6692
- b6695 (b6695) – 2025-10-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6695
- b6697 (b6697) – 2025-10-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6697
- b6699 (b6699) – 2025-10-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6699
- b6700 (b6700) – 2025-10-06 – https://github.com/ggml-org/llama.cpp/releases/tag/b6700
- b6701 (b6701) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6701
- b6702 (b6702) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6702
- b6703 (b6703) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6703
- b6704 (b6704) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6704
- b6706 (b6706) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6706
- b6708 (b6708) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6708
- b6709 (b6709) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6709
- b6710 (b6710) – 2025-10-07 – https://github.com/ggml-org/llama.cpp/releases/tag/b6710
- b6711 (b6711) – 2025-10-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6711
- b6713 (b6713) – 2025-10-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6713
- b6714 (b6714) – 2025-10-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6714
- b6715 (b6715) – 2025-10-08 – https://github.com/ggml-org/llama.cpp/releases/tag/b6715
- b6717 (b6717) – 2025-10-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6717
- b6718 (b6718) – 2025-10-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6718
- b6719 (b6719) – 2025-10-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6719
- b6721 (b6721) – 2025-10-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6721
- b6724 (b6724) – 2025-10-09 – https://github.com/ggml-org/llama.cpp/releases/tag/b6724
- b6726 (b6726) – 2025-10-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b6726
- b6727 (b6727) – 2025-10-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b6727
- b6728 (b6728) – 2025-10-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b6728
- b6729 (b6729) – 2025-10-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b6729
- b6730 (b6730) – 2025-10-10 – https://github.com/ggml-org/llama.cpp/releases/tag/b6730
- b6732 (b6732) – 2025-10-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b6732
- b6733 (b6733) – 2025-10-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b6733
- b6735 (b6735) – 2025-10-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b6735
- b6736 (b6736) – 2025-10-11 – https://github.com/ggml-org/llama.cpp/releases/tag/b6736
- b6737 (b6737) – 2025-10-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b6737
- b6738 (b6738) – 2025-10-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b6738
- b6739 (b6739) – 2025-10-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b6739
- b6741 (b6741) – 2025-10-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b6741
- b6743 (b6743) – 2025-10-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b6743
- b6745 (b6745) – 2025-10-12 – https://github.com/ggml-org/llama.cpp/releases/tag/b6745
- b6746 (b6746) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6746
- b6747 (b6747) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6747
- b6748 (b6748) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6748
- b6750 (b6750) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6750
- b6751 (b6751) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6751
- b6752 (b6752) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6752
- b6753 (b6753) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6753
- b6754 (b6754) – 2025-10-13 – https://github.com/ggml-org/llama.cpp/releases/tag/b6754
- b6756 (b6756) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6756
- b6757 (b6757) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6757
- b6758 (b6758) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6758
- b6759 (b6759) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6759
- b6760 (b6760) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6760
- b6761 (b6761) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6761
- b6762 (b6762) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6762
- b6763 (b6763) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6763
- b6764 (b6764) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6764
- b6765 (b6765) – 2025-10-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6765
- b6766 (b6766) – 2025-10-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6766
- b6767 (b6767) – 2025-10-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6767
- b6768 (b6768) – 2025-10-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6768
- b6769 (b6769) – 2025-10-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6769
- b6770 (b6770) – 2025-10-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6770
- b6773 (b6773) – 2025-10-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6773
- b6774 (b6774) – 2025-10-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6774
- b6776 (b6776) – 2025-10-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6776
- b6777 (b6777) – 2025-10-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6777
- b6778 (b6778) – 2025-10-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6778
- b6779 (b6779) – 2025-10-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6779
- b6780 (b6780) – 2025-10-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6780
- b6782 (b6782) – 2025-10-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6782
- b6783 (b6783) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6783
- b6784 (b6784) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6784
- b6785 (b6785) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6785
- b6786 (b6786) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6786
- b6788 (b6788) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6788
- b6789 (b6789) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6789
- b6790 (b6790) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6790
- b6791 (b6791) – 2025-10-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6791
- b6792 (b6792) – 2025-10-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6792


## 2025-10-02: Update to llama.cpp b6666

- b6499 (b6499) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6499
- b6500 (b6500) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6500
- b6501 (b6501) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6501
- b6502 (b6502) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6502
- b6503 (b6503) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6503
- b6504 (b6504) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6504
- b6505 (b6505) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6505
- b6506 (b6506) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6506
- b6507 (b6507) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6507
- b6508 (b6508) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6508
- b6509 (b6509) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6509
- b6510 (b6510) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6510
- b6511 (b6511) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6511
- b6512 (b6512) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6512
- b6513 (b6513) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6513
- b6514 (b6514) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6514
- b6515 (b6515) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6515
- b6516 (b6516) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6516
- b6517 (b6517) – 2025-09-18 – https://github.com/ggml-org/llama.cpp/releases/tag/b6517
- b6518 (b6518) – 2025-09-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b6518
- b6519 (b6519) – 2025-09-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b6519
- b6521 (b6521) – 2025-09-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b6521
- b6522 (b6522) – 2025-09-19 – https://github.com/ggml-org/llama.cpp/releases/tag/b6522
- b6523 (b6523) – 2025-09-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6523
- b6524 (b6524) – 2025-09-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6524
- b6527 (b6527) – 2025-09-20 – https://github.com/ggml-org/llama.cpp/releases/tag/b6527
- b6528 (b6528) – 2025-09-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6528
- b6529 (b6529) – 2025-09-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6529
- b6532 (b6532) – 2025-09-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6532
- b6533 (b6533) – 2025-09-21 – https://github.com/ggml-org/llama.cpp/releases/tag/b6533
- b6534 (b6534) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6534
- b6535 (b6535) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6535
- b6536 (b6536) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6536
- b6541 (b6541) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6541
- b6543 (b6543) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6543
- b6544 (b6544) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6544
- b6545 (b6545) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6545
- b6548 (b6548) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6548
- b6549 (b6549) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6549
- b6550 (b6550) – 2025-09-22 – https://github.com/ggml-org/llama.cpp/releases/tag/b6550
- b6556 (b6556) – 2025-09-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6556
- b6557 (b6557) – 2025-09-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6557
- b6558 (b6558) – 2025-09-23 – https://github.com/ggml-org/llama.cpp/releases/tag/b6558
- b6565 (b6565) – 2025-09-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b6565
- b6567 (b6567) – 2025-09-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b6567
- b6568 (b6568) – 2025-09-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b6568
- b6569 (b6569) – 2025-09-24 – https://github.com/ggml-org/llama.cpp/releases/tag/b6569
- b6572 (b6572) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6572
- b6574 (b6574) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6574
- b6575 (b6575) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6575
- b6576 (b6576) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6576
- b6578 (b6578) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6578
- b6580 (b6580) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6580
- b6582 (b6582) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6582
- b6583 (b6583) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6583
- b6585 (b6585) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6585
- b6586 (b6586) – 2025-09-25 – https://github.com/ggml-org/llama.cpp/releases/tag/b6586
- b6587 (b6587) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6587
- b6591 (b6591) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6591
- b6593 (b6593) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6593
- b6594 (b6594) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6594
- b6595 (b6595) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6595
- b6598 (b6598) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6598
- b6601 (b6601) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6601
- b6602 (b6602) – 2025-09-26 – https://github.com/ggml-org/llama.cpp/releases/tag/b6602
- b6603 (b6603) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6603
- b6604 (b6604) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6604
- b6605 (b6605) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6605
- b6606 (b6606) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6606
- b6607 (b6607) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6607
- b6608 (b6608) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6608
- b6610 (b6610) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6610
- b6611 (b6611) – 2025-09-27 – https://github.com/ggml-org/llama.cpp/releases/tag/b6611
- b6612 (b6612) – 2025-09-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6612
- b6613 (b6613) – 2025-09-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6613
- b6615 (b6615) – 2025-09-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6615
- b6619 (b6619) – 2025-09-28 – https://github.com/ggml-org/llama.cpp/releases/tag/b6619
- b6621 (b6621) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6621
- b6622 (b6622) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6622
- b6623 (b6623) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6623
- b6624 (b6624) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6624
- b6627 (b6627) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6627
- b6628 (b6628) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6628
- b6634 (b6634) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6634
- b6635 (b6635) – 2025-09-29 – https://github.com/ggml-org/llama.cpp/releases/tag/b6635
- b6638 (b6638) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6638
- b6640 (b6640) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6640
- b6641 (b6641) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6641
- b6642 (b6642) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6642
- b6643 (b6643) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6643
- b6644 (b6644) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6644
- b6646 (b6646) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6646
- b6647 (b6647) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6647
- b6648 (b6648) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6648
- b6650 (b6650) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6650
- b6651 (b6651) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6651
- b6653 (b6653) – 2025-09-30 – https://github.com/ggml-org/llama.cpp/releases/tag/b6653
- b6660 (b6660) – 2025-10-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6660
- b6661 (b6661) – 2025-10-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6661
- b6662 (b6662) – 2025-10-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6662
- b6663 (b6663) – 2025-10-01 – https://github.com/ggml-org/llama.cpp/releases/tag/b6663
- b6666 (b6666) – 2025-10-02 – https://github.com/ggml-org/llama.cpp/releases/tag/b6666


This file lists notable changes synchronized from upstream llama.cpp releases.
Each entry corresponds to the vendor submodule update in this package.

## 2025-09-17: Update to llama.cpp b6497

- b6469 (b6469) – 2025-09-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6469
- b6470 (b6470) – 2025-09-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6470
- b6471 (b6471) – 2025-09-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6471
- b6473 (b6473) – 2025-09-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6473
- b6474 (b6474) – 2025-09-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6474
- b6475 (b6475) – 2025-09-14 – https://github.com/ggml-org/llama.cpp/releases/tag/b6475
- b6476 (b6476) – 2025-09-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6476
- b6477 (b6477) – 2025-09-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6477
- b6478 (b6478) – 2025-09-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6478
- b6479 (b6479) – 2025-09-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6479
- b6480 (b6480) – 2025-09-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6480
- b6482 (b6482) – 2025-09-15 – https://github.com/ggml-org/llama.cpp/releases/tag/b6482
- b6483 (b6483) – 2025-09-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6483
- b6484 (b6484) – 2025-09-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6484
- b6488 (b6488) – 2025-09-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6488
- b6490 (b6490) – 2025-09-16 – https://github.com/ggml-org/llama.cpp/releases/tag/b6490
- b6491 (b6491) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6491
- b6492 (b6492) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6492
- b6493 (b6493) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6493
- b6494 (b6494) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6494
- b6496 (b6496) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6496
- b6497 (b6497) – 2025-09-17 – https://github.com/ggml-org/llama.cpp/releases/tag/b6497

