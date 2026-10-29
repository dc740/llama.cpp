#include "common.cuh"

// conv-state fusion: the ssm_conv input's first d_conv-1 columns are the recurrent conv
// state of the sequence (normally materialized by GET_ROWS + CONCAT) and the updated
// tail is copied back into the state slot (a CPY node). When fused, the kernel reads the
// state row via state_ids and writes the tail itself, so gather/concat/cpy are skipped.
struct ggml_cuda_ssm_conv_state_fusion {
    const float *   state_rows;       // conv state cache base
    const int32_t * state_ids;        // per-seq state slot ids (graph input)
    int64_t         state_row_stride; // floats per slot row
    int64_t         state_n_rows;     // rows in the cache tensor (bounds for ids)
    const float *   tok;              // new token columns (the concat's second source)
    size_t          tok_nb0;
    size_t          tok_nb1;
    size_t          tok_nb2;
    float *         wb;               // write-back dst (the cpy's cache view), may be null
    int64_t         wb_seq_stride;    // floats between per-seq slots in wb
};

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr, const ggml_cuda_ssm_conv_state_fusion * state_fusion = nullptr);
