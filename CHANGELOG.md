# Changelog

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

