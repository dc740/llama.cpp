# llama.cpp fork - dc740

| branch | functionality | pp t/s | tg t/s |
|---|---|---|---|
| `dev-master` | everything below merged onto upstream master | **5.59** | **4.19** |
| `numa-mirror-v2` | NUMA weight mirroring: per-node replicas of host weight buffers, expert-upload remap into upstream sched_copy_experts | - | - |
| `k3-iq1-types` | unsloth IQ1_XS/IQ1_XXS/IQ1_XXXS tensor types (ids 64-66), scalar vec_dot - for Kimi-K3 UD quants | - | - |
| `k3-iq1-avx2` | stacks on k3-iq1-types: AVX2/BMI2 vec_dot for the three types (verified bit-exact) | - | - |
| `fa-mla-576` | fattn gate fix: MLA 576/512 routes to tile kernel when KV len is unaligned | - | - |

Numbers = aggregate throughput serving **8 simultaneous streams** (`llama-batched-bench -npl 8`), Kimi-K3 UD-TQ1_0 (2.8T, ~474GB) on 3x gfx906 MI50 + 2-socket NUMA host, `-ngl 99 -dev ROCm0,ROCm1,ROCm2 -ot exps=CPU --numa mirror -t 28`. Per-stream ~= tg/8 (~0.5 t/s).

## branch details

- `numa-mirror-v2` - mirrors every host weight buffer into one replica per NUMA node; scheduler expert-upload sources are remapped to the replica local to the destination GPU's node (not the calling thread's), so weight reads never cross the inter-node link. Mirror buffers are allocated lazily during graph scan and freed with their owning backend buffer. `GGML_NUMA_MIRROR_MAX_BUFS` raised to 1024: Kimi-K3's hundreds of ~1.5GB expert shards overflowed the original 16-entry table and silently fell back to unmirrored reads.
- `k3-iq1-types` - adds the three unsloth IQ1 quant formats (block layout, grid tables, dequantize + scalar vec_dot + traits) needed to load unsloth UD quants of Kimi-K3. Generic-scalar only - correctness floor, not speed.
- `k3-iq1-avx2` - x86 AVX2/BMI2 vec_dot for all three types: pdep-packed index decode + grid lookup + SIMD multiply/add, adapted from the existing iq1_s kernel. Verified bit-exact vs the generic path. ~3x CPU speedup on K3.
- `fa-mla-576` - relaxes the flash-attn tile-kernel gate so MLA head sizes 576/512 are used when the KV length is not a multiple of the padding constraint; fixes a ~13% throughput regression on absorbed-MLA shapes (GLM-4.6/K3) where -fa was slower than off.
- `dev-master` - integration of all of the above on current upstream master. This is the branch that produced the numbers in the table.
