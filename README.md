# llama.cpp fork - dc740

| branch | functionality |
|---|---|
| `dev-master` | everything below merged onto upstream master |
| `numa-mirror-v2` | NUMA weight mirroring: per-node replicas of host weight buffers, expert-upload remap into upstream sched_copy_experts |
| `k3-iq1-types` | unsloth IQ1_XS/IQ1_XXS/IQ1_XXXS tensor types (ids 64-66), scalar vec_dot - for Kimi-K3 UD quants |
| `k3-iq1-avx2` | stacks on k3-iq1-types: AVX2/BMI2 vec_dot for the three types (verified bit-exact) |
| `fa-mla-576` | fattn gate fix: MLA 576/512 routes to tile kernel when KV len is unaligned |
