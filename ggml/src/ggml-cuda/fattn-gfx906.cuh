#include "common.cuh"
#include "fattn-common.cuh"

// FA2-style flash attention kernels for gfx906 (MI50/MI60, GCN5).
// gfx906 has no usable matrix instructions, so all compute runs on VALU:
// v_dot2_f32_f16 for the KQ dots and packed fp16 FMA for the PV accumulation.
// Design is ported from the verified standalone kernel (see docs/gfx906-fa2-plan.md):
//   - ncols1*ncols2 (Q column, Q head) rows per block, SPLIT=4 lanes share one row,
//     each lane owns DE/SPLIT head dims (pair kernel, DE <= 128)
//   - one lane per key for the KQ dot, dim-slice lanes for PV (D == 512 kernel)
//   - K/V tiles of 16 rows staged in LDS as half2 with +8 half2 row padding
//   - online softmax in fp32, probs in fp16, PV partial sums folded into fp32
//     accumulators at the end of each tile
// Q arrives as fp32, K/V as fp16 (converted by launch_fattn), dst as fp32.
// Dispatch (fattn.cu) guarantees: cc == gfx906, DE == DV in {64, 128, 512},
// Q/K/V nb[1..3] % 16 == 0 (16-byte aligned vectorized loads), ncols2 | gqa_ratio.

#if defined(GGML_USE_HIP) && defined(FLASH_ATTN_AVAILABLE)

// K/V columns per launch_fattn parallel block iteration (4 K/V tiles of 16).
#define FATTN_GFX906_NBATCH_FA 64

// 2 fp16 muls + fp32 acc in one issue on gfx906; scalar fallback so the code
// still compiles for the other architectures in a multi-arch HIP build.
static __device__ __forceinline__ float fattn_gfx906_dot2(float acc, half2 a, half2 b) {
#if defined(__gfx906__)
    return __builtin_amdgcn_fdot2(a, b, acc, false);
#else
    const float2 fa = __half22float2(a);
    const float2 fb = __half22float2(b);
    return acc + fa.x*fb.x + fa.y*fb.y;
#endif // __gfx906__
}

// Pair kernel for DE in {64, 128}: SPLIT lanes share one Q row, each owns
// DE/SPLIT head dims; the partial dots are reduced via __shfl_xor.
template<int DE, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ncols1*ncols2*4)
static __global__ void flash_attn_gfx906(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {

    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    constexpr int TILE_M = ncols1*ncols2;   // (Q column, Q head) rows per block
    constexpr int TILE_N = 16;              // K/V rows per tile
    constexpr int SPLIT  = 4;               // lanes per Q row
    constexpr int DH     = DE/SPLIT;        // head dims per lane
    constexpr int DQ     = DH/2;            // half2 per lane
    constexpr int D2     = DE/2;            // half2 per K/V row
    constexpr int CPR    = DE/8;            // uint4 chunks per K/V row

    static_assert(DE == 64 || DE == 128, "bad DE");
    static_assert(TILE_M*SPLIT % 32 == 0, "bad TILE_M");

    const int tid      = threadIdx.y*blockDim.x + threadIdx.x;
    const int sub      = tid % SPLIT;       // head-dim slice of this lane
    const int row      = tid / SPLIT;
    const int col_Q_0  = blockIdx.x*ncols1;
    const int q_local  = row / ncols2;
    const int hc       = row % ncols2;
    const int qg       = col_Q_0 + q_local; // global Q column index

    const int sequence  = blockIdx.z / (ne02/ncols2);
    const int head0     = blockIdx.z*ncols2 - sequence*ne02;
    const int head      = head0 + hc;
    const int gqa_ratio = ne02/ne12;
    const int kvh       = head0 / gqa_ratio; // same K/V head for the whole block

    const char * K_h = K + nb13*sequence + nb12*kvh;
    const char * V_h = V + nb23*sequence + nb22*kvh;
    const bool live  = qg < int(ne01.z);

    __shared__ __align__(16) half2 sK[TILE_N][D2 + 8];
    __shared__ __align__(16) half2 sV[TILE_N][D2 + 8];

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*qg) : nullptr;
    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head, n_head_log2, m0, m1) : 1.0f;

    // Q row slice -> fp16 registers, scale folded in:
    half2 qreg[DQ] = {};
    if (live) {
        const float4 * qp = (const float4 *) (Q + nb01*qg + nb02*head + nb03*sequence + sub*DH*int32_t(sizeof(float)));
#pragma unroll
        for (int d4 = 0; d4 < DH/4; ++d4) {
            const float4 f = qp[d4];
            qreg[2*d4 + 0] = __floats2half2_rn(f.x*scale, f.y*scale);
            qreg[2*d4 + 1] = __floats2half2_rn(f.z*scale, f.w*scale);
        }
    }

    float m = -FLT_MAX/2.0f, l = 0.0f;
    float acc[DH] = {};

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    for (int k0 = blockIdx.y*FATTN_GFX906_NBATCH_FA; k0 < k_VKQ_max; k0 += gridDim.y*FATTN_GFX906_NBATCH_FA) {
        const int k_lim = min(k0 + FATTN_GFX906_NBATCH_FA, k_VKQ_max);
        for (int n0 = k0; n0 < k_lim; n0 += TILE_N) {
            // cooperative K,V tile load (uint4 = 8 fp16 values each):
            for (int c = tid; c < TILE_N*CPR; c += TILE_M*SPLIT) {
                const int i = c / CPR;
                const int kv = n0 + i;
                if (kv < k_lim) {
                    const int off = (c % CPR);
                    *reinterpret_cast<uint4 *>(&sK[i][off*4]) =
                        reinterpret_cast<const uint4 *>(K_h + (int64_t) kv*nb11)[off];
                    *reinterpret_cast<uint4 *>(&sV[i][off*4]) =
                        reinterpret_cast<const uint4 *>(V_h + (int64_t) kv*nb21)[off];
                }
            }
            __syncthreads();

            const int nmax = min(TILE_N, k_lim - n0);
            if (live) {
                // scores: dot(q, sK[i]) via dot2 + split-lane reduction:
                half p[TILE_N];
                float m_new = m;
#pragma unroll 8
                for (int i = 0; i < nmax; ++i) {
                    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                    for (int d2 = 0; d2 < DQ; d2 += 4) {
                        const int d2h = sub*DQ + d2;
                        a0 = fattn_gfx906_dot2(a0, qreg[d2],     sK[i][d2h]);
                        a0 = fattn_gfx906_dot2(a0, qreg[d2 + 2], sK[i][d2h + 2]);
                        a1 = fattn_gfx906_dot2(a1, qreg[d2 + 1], sK[i][d2h + 1]);
                        a1 = fattn_gfx906_dot2(a1, qreg[d2 + 3], sK[i][d2h + 3]);
                    }
                    float pd = a0 + a1;
#pragma unroll
                    for (int s = SPLIT/2; s > 0; s >>= 1) {
                        pd += __shfl_xor(pd, s);
                    }
                    float sc = pd;
                    if (use_logit_softcap) {
                        sc = logit_softcap*tanhf(sc);
                    }
                    if (maskh) {
                        sc += slope*__half2float(maskh[n0 + i]);
                    }
                    p[i] = __float2half(sc);
                    m_new = fmaxf(m_new, sc + FATTN_KQ_MAX_OFFSET);
                }

                float psum = 0.0f;
#pragma unroll 8
                for (int i = 0; i < nmax; ++i) {
                    const float e = expf(__half2float(p[i]) - m_new);
                    p[i] = __float2half(e);
                    psum += e;
                }
                const float rs = expf(m - m_new);
#pragma unroll
                for (int d = 0; d < DH; ++d) {
                    acc[d] *= rs;
                }

                // PV: fp16x2 partial sums over this tile, folded into fp32:
                half2 hacc[DQ] = {};
#pragma unroll 4
                for (int i = 0; i < nmax; ++i) {
                    const half2 pp = __half2half2(p[i]);
#pragma unroll
                    for (int d2 = 0; d2 < DQ; ++d2) {
                        hacc[d2] = __hfma2(pp, sV[i][sub*DQ + d2], hacc[d2]);
                    }
                }
#pragma unroll
                for (int d2 = 0; d2 < DQ; ++d2) {
                    const float2 f = __half22float2(hacc[d2]);
                    acc[2*d2]     += f.x;
                    acc[2*d2 + 1] += f.y;
                }
                l = l*rs + psum;
                m = m_new;
            }
            __syncthreads();
        }
    }

    // Attention sink: extra logit, first of all parallel blocks only:
    if (sinks && blockIdx.y == 0 && live) {
        const float sink  = ((const float *) sinks)[head];
        const float m_new = fmaxf(m, sink);
        const float rs    = expf(m - m_new);
        l = l*rs + expf(sink - m_new);
        m = m_new;
#pragma unroll
        for (int d = 0; d < DH; ++d) {
            acc[d] *= rs;
        }
    }

    if (live) {
        const int64_t j_dst = ((int64_t) (sequence*int(ne01.z) + qg)*ne02 + head)*gridDim.y + blockIdx.y;
        const float inv = gridDim.y == 1 ? 1.0f/l : 1.0f;
        float4 * drow = (float4 *) (dst + j_dst*DE + sub*DH);
#pragma unroll
        for (int d4 = 0; d4 < DH/4; ++d4) {
            drow[d4] = make_float4(acc[4*d4 + 0]*inv, acc[4*d4 + 1]*inv, acc[4*d4 + 2]*inv, acc[4*d4 + 3]*inv);
        }
        if (gridDim.y != 1 && sub == 0) {
            dst_meta[j_dst] = make_float2(m, l);
        }
    }

    GGML_UNUSED_VARS(ne00, ne03, ne10, ne13, ne31, ne32, nb32);
}

// D == 512 kernel: lane owns one key of the KQ phase (full 512-dim dot from sQ
// x sKV, no cross-lane reduction), then remaps to a DE/TILE_N-dim slice for PV.
// sKV is aliased: K tile for the dot phase, V tile for the accumulate phase.
template<int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ncols1*ncols2*16)
static __global__ void flash_attn_gfx906_512(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {

    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    constexpr int DE     = 512;
    constexpr int TM     = ncols1*ncols2;   // rows per block, fixed by sQ
    constexpr int TN     = 16;              // K/V rows per tile
    constexpr int D2     = DE/2;            // half2 per row
    constexpr int DH     = DE/TN;           // out dims per lane in the V phase
    constexpr int CPR    = DE/8;            // uint4 chunks per K/V row

    static_assert(TM == 32, "bad TM");

    const int tid      = threadIdx.y*blockDim.x + threadIdx.x;
    const int sub      = tid % TN;          // key slot (KQ) / dim slice (PV)
    const int row      = tid / TN;
    const int col_Q_0  = blockIdx.x*ncols1;
    const int q_local  = row / ncols2;
    const int hc       = row % ncols2;
    const int qg       = col_Q_0 + q_local;

    const int sequence  = blockIdx.z / (ne02/ncols2);
    const int head0     = blockIdx.z*ncols2 - sequence*ne02;
    const int head      = head0 + hc;
    const int gqa_ratio = ne02/ne12;
    const int kvh       = head0 / gqa_ratio;

    const char * K_h = K + nb13*sequence + nb12*kvh;
    const char * V_h = V + nb23*sequence + nb22*kvh;
    const bool live  = qg < int(ne01.z);

    __shared__ __align__(16) half2 sQ [TM][D2 + 8];
    __shared__ __align__(16) half2 sKV[TN][D2 + 8];
    __shared__ float sE[TM][TN];

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*qg) : nullptr;
    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head, n_head_log2, m0, m1) : 1.0f;

    // stage Q rows into LDS once (fp32 -> half2, scale folded in):
    for (int c = tid; c < TM*CPR; c += TM*TN) {
        const int r    = c / CPR;
        const int hoff = (c % CPR)*8; // head-dim offset in halves
        const int q    = col_Q_0 + r/ncols2;
        const int h    = head0 + r%ncols2;
        if (q < int(ne01.z)) {
            const float4 * qp = (const float4 *) (Q + nb01*q + nb02*h + nb03*sequence + hoff*int32_t(sizeof(float)));
            __align__(16) half2 t[4];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float4 f = qp[e];
                t[2*e + 0] = __floats2half2_rn(f.x*scale, f.y*scale);
                t[2*e + 1] = __floats2half2_rn(f.z*scale, f.w*scale);
            }
            *reinterpret_cast<uint4 *>(&sQ[r][hoff/2]) = *reinterpret_cast<uint4 *>(t);
        }
    }
    __syncthreads();

    float m = -FLT_MAX/2.0f, l = 0.0f;
    float acc[DH] = {};

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    for (int k0 = blockIdx.y*FATTN_GFX906_NBATCH_FA; k0 < k_VKQ_max; k0 += gridDim.y*FATTN_GFX906_NBATCH_FA) {
        const int k_lim = min(k0 + FATTN_GFX906_NBATCH_FA, k_VKQ_max);
        for (int n0 = k0; n0 < k_lim; n0 += TN) {
            // cooperative K tile load:
            for (int c = tid; c < TN*CPR; c += TM*TN) {
                const int i = c / CPR;
                const int kv = n0 + i;
                if (kv < k_lim) {
                    const int off = (c % CPR);
                    *reinterpret_cast<uint4 *>(&sKV[i][off*4]) =
                        reinterpret_cast<const uint4 *>(K_h + (int64_t) kv*nb11)[off];
                }
            }
            __syncthreads();

            const int nmax = min(TN, k_lim - n0);
            // KQ phase: lane sub owns key n0+sub, full 512-dim dot:
            if (live) {
                float sc;
                if (sub < nmax) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (int d2 = 0; d2 < D2; d2 += 4) {
                        a0 = fattn_gfx906_dot2(a0, sQ[row][d2],     sKV[sub][d2]);
                        a1 = fattn_gfx906_dot2(a1, sQ[row][d2 + 1], sKV[sub][d2 + 1]);
                        a2 = fattn_gfx906_dot2(a2, sQ[row][d2 + 2], sKV[sub][d2 + 2]);
                        a3 = fattn_gfx906_dot2(a3, sQ[row][d2 + 3], sKV[sub][d2 + 3]);
                    }
                    sc = a0 + a1 + a2 + a3;
                    if (use_logit_softcap) {
                        sc = logit_softcap*tanhf(sc);
                    }
                    if (maskh) {
                        sc += slope*__half2float(maskh[n0 + sub]);
                    }
                } else {
                    sc = -INFINITY;
                }
                sE[row][sub] = sc;
            }
            __syncthreads();

            // K is dead: reload sKV with the V tile:
            for (int c = tid; c < TN*CPR; c += TM*TN) {
                const int i = c / CPR;
                const int kv = n0 + i;
                if (kv < k_lim) {
                    const int off = (c % CPR);
                    *reinterpret_cast<uint4 *>(&sKV[i][off*4]) =
                        reinterpret_cast<const uint4 *>(V_h + (int64_t) kv*nb21)[off];
                }
            }
            __syncthreads();

            // softmax + PV phase: lane owns dims {TN*4*j + sub*4 .. +3} half2s:
            if (live) {
                float m_new = m;
#pragma unroll
                for (int i = 0; i < TN; ++i) {
                    if (i < nmax) {
                        m_new = fmaxf(m_new, sE[row][i] + FATTN_KQ_MAX_OFFSET);
                    }
                }
                const float rs = expf(m - m_new);
                float psum = 0.0f;
#pragma unroll
                for (int d = 0; d < DH; ++d) {
                    acc[d] *= rs;
                }
                half2 hacc[DH/2] = {};
#pragma unroll 4
                for (int i = 0; i < nmax; ++i) {
                    const float e = expf(sE[row][i] - m_new);
                    psum += e;
                    const half2 ee = __floats2half2_rn(e, e);
#pragma unroll
                    for (int d2 = 0; d2 < DH/2; d2 += 4) {
                        const int d2h = d2*TN + sub*4;
                        hacc[d2]     = __hfma2(ee, sKV[i][d2h],     hacc[d2]);
                        hacc[d2 + 1] = __hfma2(ee, sKV[i][d2h + 1], hacc[d2 + 1]);
                        hacc[d2 + 2] = __hfma2(ee, sKV[i][d2h + 2], hacc[d2 + 2]);
                        hacc[d2 + 3] = __hfma2(ee, sKV[i][d2h + 3], hacc[d2 + 3]);
                    }
                }
#pragma unroll
                for (int d2 = 0; d2 < DH/2; ++d2) {
                    const float2 f = __half22float2(hacc[d2]);
                    acc[2*d2]     += f.x;
                    acc[2*d2 + 1] += f.y;
                }
                l = l*rs + psum;
                m = m_new;
            }
            __syncthreads();
        }
    }

    // Attention sink: extra logit, first of all parallel blocks only:
    if (sinks && blockIdx.y == 0 && live) {
        const float sink  = ((const float *) sinks)[head];
        const float m_new = fmaxf(m, sink);
        const float rs    = expf(m - m_new);
        l = l*rs + expf(sink - m_new);
        m = m_new;
#pragma unroll
        for (int d = 0; d < DH; ++d) {
            acc[d] *= rs;
        }
    }

    if (live) {
        // acc[8j..8j+7] holds dims (TN*4*j + sub*4 .. +3)*2 -> float4s at 128*j + 8*sub:
        const int64_t j_dst = ((int64_t) (sequence*int(ne01.z) + qg)*ne02 + head)*gridDim.y + blockIdx.y;
        float * drow = dst + j_dst*DE;
        const float inv = gridDim.y == 1 ? 1.0f/l : 1.0f;
#pragma unroll
        for (int j = 0; j < DH/8; ++j) {
            float4 * t = (float4 *) (drow + 128*j + 8*sub);
            t[0] = make_float4(acc[8*j + 0]*inv, acc[8*j + 1]*inv, acc[8*j + 2]*inv, acc[8*j + 3]*inv);
            t[1] = make_float4(acc[8*j + 4]*inv, acc[8*j + 5]*inv, acc[8*j + 6]*inv, acc[8*j + 7]*inv);
        }
        if (gridDim.y != 1 && sub == 0) {
            dst_meta[j_dst] = make_float2(m, l);
        }
    }

    GGML_UNUSED_VARS(ne00, ne03, ne10, ne13, ne31, ne32, nb32);
}

template<int DKQ, int ncols1, int ncols2, bool use_logit_softcap>
static void ggml_cuda_flash_attn_ext_gfx906_sc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nthreads = DKQ == 512 ? ncols1*ncols2*16 : ncols1*ncols2*4;
    fattn_kernel_t fattn_kernel;
    if constexpr (DKQ == 512) {
        fattn_kernel = flash_attn_gfx906_512<ncols1, ncols2, use_logit_softcap>;
    } else {
        fattn_kernel = flash_attn_gfx906<DKQ, ncols1, ncols2, use_logit_softcap>;
    }
    launch_fattn<DKQ, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nthreads/32, 0, FATTN_GFX906_NBATCH_FA, true, true, false, false, 32);
}

template<int DKQ, int ncols1, int ncols2>
static void ggml_cuda_flash_attn_ext_gfx906_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        ggml_cuda_flash_attn_ext_gfx906_sc<DKQ, ncols1, ncols2, false>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_gfx906_sc<DKQ, ncols1, ncols2, true>(ctx, dst);
    }
}

// Kernel supports D in {64, 128, 512} with fp16 K/V (converted) and fp32 Q.
// Needs 16-byte aligned rows for the vectorized global loads; anything else
// falls through to the stock kernels.
static bool ggml_cuda_fattn_gfx906_eligible(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (K->ne[0] != 64 && K->ne[0] != 128 && K->ne[0] != 512) {
        return false;
    }
    if (Q->ne[0] != K->ne[0] || V->ne[0] != K->ne[0]) {
        return false;
    }
    if ((uintptr_t) dst->data % 16 != 0) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V}) {
        // K/V tensors not in fp16 are converted to contiguous fp16 by launch_fattn:
        if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32) {
            continue;
        }
        if ((uintptr_t) t->data % 16 != 0) {
            return false;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    return true;
}

static void ggml_cuda_flash_attn_ext_gfx906(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    int ncols2 = 1;
    if (mask && max_bias == 0.0f) {
        ncols2 = gqa_ratio % 8 == 0 ? 8 : gqa_ratio % 4 == 0 ? 4 : gqa_ratio % 2 == 0 ? 2 : 1;
    }

    if (K->ne[0] == 512) {
        // D == 512 kernel has 32 rows/block and 16 lanes/row:
        switch (ncols2) {
            case 8:  ggml_cuda_flash_attn_ext_gfx906_case<512,  4, 8>(ctx, dst); return;
            case 4:  ggml_cuda_flash_attn_ext_gfx906_case<512,  8, 4>(ctx, dst); return;
            case 2:  ggml_cuda_flash_attn_ext_gfx906_case<512, 16, 2>(ctx, dst); return;
            default: ggml_cuda_flash_attn_ext_gfx906_case<512, 32, 1>(ctx, dst); return;
        }
    }

    if (K->ne[0] == 128) {
        if (Q->ne[1] <= 8) {
            switch (ncols2) {
                case 8:  ggml_cuda_flash_attn_ext_gfx906_case<128,  8, 8>(ctx, dst); return;
                case 4:  ggml_cuda_flash_attn_ext_gfx906_case<128,  8, 4>(ctx, dst); return;
                case 2:  ggml_cuda_flash_attn_ext_gfx906_case<128,  8, 2>(ctx, dst); return;
                default: ggml_cuda_flash_attn_ext_gfx906_case<128,  8, 1>(ctx, dst); return;
            }
        }
        switch (ncols2) {
            case 8:  ggml_cuda_flash_attn_ext_gfx906_case<128,  8, 8>(ctx, dst); return;
            case 4:  ggml_cuda_flash_attn_ext_gfx906_case<128, 16, 4>(ctx, dst); return;
            case 2:  ggml_cuda_flash_attn_ext_gfx906_case<128, 32, 2>(ctx, dst); return;
            default: ggml_cuda_flash_attn_ext_gfx906_case<128, 64, 1>(ctx, dst); return;
        }
    }

    GGML_ASSERT(K->ne[0] == 64);
    if (Q->ne[1] <= 8) {
        switch (ncols2) {
            case 8:  ggml_cuda_flash_attn_ext_gfx906_case<64,  8, 8>(ctx, dst); return;
            case 4:  ggml_cuda_flash_attn_ext_gfx906_case<64,  8, 4>(ctx, dst); return;
            case 2:  ggml_cuda_flash_attn_ext_gfx906_case<64,  8, 2>(ctx, dst); return;
            default: ggml_cuda_flash_attn_ext_gfx906_case<64,  8, 1>(ctx, dst); return;
        }
    }
    switch (ncols2) {
        case 8:  ggml_cuda_flash_attn_ext_gfx906_case<64,  8, 8>(ctx, dst); return;
        case 4:  ggml_cuda_flash_attn_ext_gfx906_case<64, 16, 4>(ctx, dst); return;
        case 2:  ggml_cuda_flash_attn_ext_gfx906_case<64, 32, 2>(ctx, dst); return;
        default: ggml_cuda_flash_attn_ext_gfx906_case<64, 64, 1>(ctx, dst); return;
    }
}

#endif // defined(GGML_USE_HIP) && defined(FLASH_ATTN_AVAILABLE)
