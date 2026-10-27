# llama.cpp fork - dc740

| branch | functionality |
|---|---|
| `dev-master` | everything below merged onto upstream master |
| `numa-mirror-v2` | NUMA weight mirroring: per-node replicas of host weight buffers, expert-upload remap into upstream sched_copy_experts |
| `k3-iq1-types` | unsloth IQ1_XS/IQ1_XXS/IQ1_XXXS tensor types (ids 64-66), scalar vec_dot - for Kimi-K3 UD quants |
| `k3-iq1-avx2` | stacks on k3-iq1-types: AVX2/BMI2 vec_dot for the three types (verified bit-exact) |
| `fa-mla-576` | fattn gate fix: MLA 576/512 routes to tile kernel when KV len is unaligned |
| `fa2-gfx906` | FA2-style flash attention kernel for gfx906: LDS-tiled online softmax, v_dot2/v_pk_fma fp16, never materializes SxS (plan: docs/gfx906-fa2-plan.md) |
| `fix-llama-bench-markers` | drops a merge conflict marker committed into llama-bench usage output |
| `gfx906-vec-d256` | stock VEC attention nthreads_KQ_q 4->8 for D==256 on gfx906 (+0.6/+1.5% tg) |
| `fa2-gfx906-d256` | gated D256 attention experiments (pair + lane-per-key kernels, q8 tile loaders); correct but slower than stock, OFF by default behind env gates |
| `gfx906-mmvq` | GCN matvec/MoE launch tuning for gfx906 (+3.7% tg on Qwen27B) |
| `gfx906-opprof` | env-gated per-op event profiler `GGML_GFX906_OP_PROF` + round-4 profile map |
| `gfx906-gdn-state` | fused gated-delta-net state slot read (kills per-layer GET_ROWS; +4.3/+8.0% tg) |
| `gfx906-moe-n1` | packed batch-1 MoE expert matvec `mul_mat_vec_q_moe_n1` (+14.5% tg on Qwen35B-A3B) |
| `gfx906-vecdot-attempt` | docs only: vec_dot instruction-reduction attempt (negative result, reverted) |
| `fix-iq4nl-vdr` | vec_dot_iq4_nl_q8_1 uses VDR_IQ4_NL_Q8_1_MMVQ instead of the borrowed Q4_0 constant |
| `fix-mmq-tiling-revert` | reverts upstream #29948 fused gate/up MMQ: restored the pre-fusion plain tiling that gfx906 needs (GLM pp +43% on revert A/B) |
| `fix-mirror-bind` | NUMA mirror: when the move_pages probe fails, home node gets a bound anonymous copy too - file-cache pages can't be relocated or steered, only an anon mbind copy guarantees home-local reads |
| `gfx906-mirror-replica-pin` | page-lock NUMA-mirror replica pages through the GPU backend's host-register hook (GGML_CUDA_REGISTER_HOST now covers replicas, the exps=CPU upload sources) |
| `fix-mirror-bind-cap` | NUMA mirror: cap the probe-fail home anon copies at 40% of the home node's free+reclaimable memory - unbounded they double per-node weight RAM and evict the model's own pages |
| `fix-mirror-primary-home` | NUMA mirror: file-mapped weight buffers always get a bound anonymous home replica - the resident-pages probe lies for sparse file buffers and their remaining pages fault ~50/50 across nodes, unrelocatable once cached |

## measured (`dev-master`)

Aggregate throughput serving **8 simultaneous streams** (`llama-batched-bench -npl 8`, `-t 28`, `--numa mirror`, experts on CPU):

| model | config | pp t/s | tg t/s |
|---|---|---|---|
| GLM-5.3 UD-Q4_K_XL (744B, 435GiB) | `-ngl 99 -dev ROCm0 -ot exps=CPU` | **14.63** | **10.70** |
| Kimi-K3 UD-TQ1_0 (2.8T, ~474GB) | `-ngl 99 -dev ROCm0,ROCm1,ROCm2 -ot exps=CPU` | **5.47** | **4.17** |

Re-measured 2026-10-27 after `fix-numa-mirror-tiled`, `fix-mmq-tiling-revert`, `fix-mirror-bind{,-cap}` (PP512 row, `-c 16384` GLM / `-c 8192` K3). Prior rows: GLM 13.74/9.90, K3 5.59/4.19.

Single-GPU `llama-bench` (-ngl 999 -fa on -ctk/-ctv q8_0 -t8 -p512 -n64 -d8192):

| model | pp t/s | tg t/s |
|---|---|---|
| Qwen3.8-27B UD-Q6_K | 154.79 | 23.33 |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 831.27 | 83.24 |

Per-stream = agg/8: GLM ~1.4 t/s, K3 ~0.5 t/s per user. GLM scales further: npl16 agg 11.73, real chat API @16 streams 9.29 t/s. Do not run the batched -t28 mirror benches under `numactl --cpunodebind` - 28 threads oversubscribe one 18-core socket and tg drops ~30%.

## branch details

- `numa-mirror-v2` - mirrors every host weight buffer into one replica per NUMA node; scheduler expert-upload sources are remapped to the replica local to the destination GPU's node (not the calling thread's), so weight reads never cross the inter-node link. Mirror buffers are allocated lazily during graph scan and freed with their owning backend buffer. `GGML_NUMA_MIRROR_MAX_BUFS` raised to 1024: Kimi-K3's hundreds of ~1.5GB expert shards overflowed the original 16-entry table and silently fell back to unmirrored reads.
- `k3-iq1-types` - adds the three unsloth IQ1 quant formats (block layout, grid tables, dequantize + scalar vec_dot + traits) needed to load unsloth UD quants of Kimi-K3. Generic-scalar only - correctness floor, not speed.
- `k3-iq1-avx2` - x86 AVX2/BMI2 vec_dot for all three types: pdep-packed index decode + grid lookup + SIMD multiply/add, adapted from the existing iq1_s kernel. Verified bit-exact vs the generic path. ~3x CPU speedup on K3.
- `fa-mla-576` - relaxes the flash-attn tile-kernel gate so MLA head sizes 576/512 are used when the KV length is not a multiple of the padding constraint; fixes a ~13% throughput regression on absorbed-MLA shapes (GLM-4.6/K3) where -fa was slower than off.
- `dev-master` - integration of all of the above on current upstream master. This is the branch that produced the numbers in the table.

## build parameters (build-mirror, all measured numbers)

```
cmake -B build-mirror \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_VULKAN=ON -DGGML_NATIVE=ON \
  -DGGML_HIP_RCCL=OFF
```

Source-level knobs that matter (not env vars - compile-time `#define`):

- `GGML_NUMA_MIRROR_MAX_BUFS` = **1024** in `ggml/src/ggml-cpu/ggml-cpu.c` - original 16 silently overflowed on Kimi-K3's hundreds of ~1.5GB expert-shard buffers, falling back to unmirrored reads. Set at build time; no env override exists.
- `GGML_NUMA_MIRROR_HOME` env var exists for mirror home-node override (`--membind` via numactl also works; `move_pages` probe failure warns but mirror still builds via fallback path).

## gfx906 attention controls

Flash attention must be enabled with `--flash-attn on`. Eligible gfx906 operations use the custom kernel automatically; unsupported shapes keep the stock path. Native q8_0 K/V is supported for head dimensions 64 and 128, plus experimental 256 when explicitly enabled. D512 still converts quantized KV to fp16. The fitting Qwen3.8-27B and Qwen3.6-35B-A3B GGUFs have head dimension 256 and keep stock attention by default.

| environment variable | purpose and behavior |
|---|---|
| `GGML_GFX906_NO_Q8` | A/B or troubleshooting switch: when set, disables native q8_0 K/V and restores conversion to fp16 before custom attention. Does not disable flash attention itself. Unset by default, allowing native q8_0 on eligible inputs. Conversion needs extra scratch memory and can be slower. |
| `GGML_GFX906_D256` | Experimental opt-in: set to `1` before startup to route eligible D256 attention through the custom kernel. Leave unset for stock dispatch. Correctness tests pass, but Qwen27B/35B benchmarks at 8K KV depth regress, so this is for development, not a recommended speed setting. |
| `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` | GPU visibility filters that can change device numbering. With `CUDA_VISIBLE_DEVICES=1`, physical GPU1 appears as process-local `ROCm0`. Leave these unset when selecting physical GPU1 by its unremapped `ROCm1` name. |

`GGML_GFX906_NO_Q8` is read once per process. Any set value, including `0` or an empty string, disables native q8_0; use `unset GGML_GFX906_NO_Q8` to restore it and restart the process.

`GGML_GFX906_D256` also uses presence semantics and is cached per process; unset it and restart to restore stock D256 dispatch. To exercise D256 in the tests below, add `GGML_GFX906_D256=1` to each `env` command. These switches have no effect on other GPU architectures.

On the measurement host, physical GPU1 is on NUMA node 1; GPU0 is on node 0 and may be busy with ComfyUI. These correctness-test commands select only the ROCm backend on GPU1, not Vulkan, and bind CPU work and host allocations to node 1:

```bash
# Native q8_0 enabled where eligible:
env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
    -u GGML_GFX906_NO_Q8 \
    numactl --cpunodebind=1 --membind=1 \
    ./build_devmaster/bin/test-backend-ops test -b ROCm1 -o FLASH_ATTN_EXT

# Same tests with q8_0 converted to fp16:
env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
    GGML_GFX906_NO_Q8=1 \
    numactl --cpunodebind=1 --membind=1 \
    ./build_devmaster/bin/test-backend-ops test -b ROCm1 -o FLASH_ATTN_EXT
```

For `llama-cli` or `llama-bench`, use `--device ROCm1` with the same visibility settings and NUMA binding. Check `--list-devices` before running on a different host; device numbering and NUMA topology are host-specific.

## hardware (measurement host)

- Dell PowerEdge R740, 2x Intel Xeon Gold 6254 (Cascade Lake) @ 3.10GHz - 18 cores/socket, HT off, 36 cores total, 2 NUMA nodes
- ~1.5TB DDR4-2666 ECC across both sockets (dup channels, one set per node) - this is why mirroring weights per-node pays off: both sockets' memory bandwidth get used instead of crossing the UPI link
- 3x AMD Instinct MI50-class, gfx906:sramecc+xnack-, 32GB HBM2 each (~98GB total VRAM), ROCm; also exposed as RADV VEGA20 to Vulkan
- Model storage on NFS (`<NFS>`) - 400+GB model loads take ~30min; not part of steady-state serving cost
