#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// fused-kernel recurrent-state input: the state slot of each sequence is gathered
// in-kernel straight from the cache instead of a separate get_rows
struct ggml_cuda_gated_delta_net_fused_state {
    const float *   rows;       // cache rows (get_rows src0)
    const int32_t * ids;        // state slot per sequence (get_rows src1)
    int64_t         row_stride; // elements between cache rows
    int64_t         n_rows;     // rows in the cache tensor (bounds for ids)
};

// detects that the state input of `gdn` is a zero-offset view of a one-row
// get_rows(states, ids) gather and fills `out` so the kernel can read the row directly
bool ggml_cuda_gdn_fused_state_src(const ggml_tensor * gdn, ggml_backend_buffer_type_t buft,
                                   ggml_cuda_gated_delta_net_fused_state & out);

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
