#include "common.cuh"
#include "fattn-common.cuh"

// FA2-style flash attention kernels for gfx906 (MI50/MI60, GCN5).
// gfx906 has no usable matrix instructions, so all compute runs on VALU:
// v_dot2_f32_f16 for the KQ dots and packed fp16 FMA for the PV accumulation.
// Design is ported from the verified standalone kernel (see docs/gfx906-fa2-plan.md):
//   - ncols1*ncols2 (Q column, Q head) rows per block, 4 or 8 lanes share one row,
//     each lane owns at most 32 head dims (pair kernel, DE <= 256)
//   - one lane per key for the KQ dot, dim-slice lanes for PV (D == 512 kernel)
//   - K/V tiles of 16 rows staged in LDS as half2 with +8 half2 row padding
//   - online softmax in fp32, probs in fp16, PV partial sums folded into fp32
//     accumulators at the end of each tile
//   - optional native q8_0 K/V path: K stays in q8_0 blocks, the KQ dot runs on
//     v_dot4_i32_i8 against Q quantized to int8 per lane slice, q8_0 V is
//     dequantized into the fp16 LDS tile during the load
// Q arrives as fp32, K/V as fp16 or raw q8_0, dst as fp32.
// Dispatch (fattn.cu) guarantees: cc == gfx906, DE == DV in {64, 128, 256, 512},
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

// Pair kernel for DE in {64, 128, 256}: SPLIT lanes share one Q row, each owns
// DE/SPLIT head dims; the partial dots are reduced via __shfl_xor.
// KQ8/VQ8: K/V arrive as raw q8_0 block rows instead of converted fp16.
template<int DE, int ncols1, int ncols2, bool use_logit_softcap, bool KQ8 = false, bool VQ8 = false>
__launch_bounds__(ncols1*ncols2*(DE == 256 ? 8 : 4))
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
    constexpr int SPLIT  = DE == 256 ? 8 : 4; // keep each lane within one q8_0 block
    constexpr int DH     = DE/SPLIT;        // head dims per lane
    constexpr int DQ     = DH/2;            // half2 per lane
    constexpr int D2     = DE/2;            // half2 per K/V row
    constexpr int CPR    = DE/8;            // uint4 chunks per K/V row
    constexpr int NB     = DE/32;           // q8_0 blocks per K/V row
    constexpr int QW     = DH/4;            // packed i8 words of Q per lane

    // For DE == 256 the fp16 path gives each lane 4-dim groups spread across
    // the row (dim = 32*g + 4*sub): K/V LDS reads land on 8 distinct banks
    // instead of colliding on 2. The q8_0 K path keeps one whole 32-dim block
    // per lane (single block scale) with sKq staged word-major instead.
    static_assert(DE == 64 || DE == 128 || DE == 256, "bad DE");
    static_assert(TILE_M*SPLIT % 16 == 0, "bad TILE_M");

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

    __shared__ __align__(16) half2  sK [KQ8 ? 1 : TILE_N][D2 + 8];
    __shared__ __align__(16) int8_t sKq[KQ8 ? TILE_N : 1][DE];   // q8_0 quants only
    __shared__ __align__(16) half   sKd[KQ8 ? TILE_N : 1][NB];   // q8_0 block scales
    __shared__ __align__(16) half2  sV [TILE_N][D2 + 8];

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*qg) : nullptr;
    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head, n_head_log2, m0, m1) : 1.0f;

    // Q row slice -> fp16 registers, scale folded in. For the q8_0 K path the
    // lane's DH dims are quantized to int8 instead; they always lie inside the
    // single K block sub*DH/32, so one lane-local scale is enough.
    half2 qreg[KQ8 ? 1 : DQ] = {};
    int   q8i [KQ8 ? QW : 1] = {};
    float q8d = 0.0f;
    if (live) {
        const float4 * qp = (const float4 *) (Q + nb01*qg + nb02*head + nb03*sequence);
        if constexpr (KQ8) {
            float amax = 0.0f;
#pragma unroll
            for (int d4 = 0; d4 < DH/4; ++d4) {
                const float4 f = qp[sub*(DH/4) + d4];
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
            }
            const float a127 = amax > 0.0f ? 127.0f/amax : 0.0f;
            q8d = amax*scale/127.0f;
#pragma unroll
            for (int w = 0; w < QW; ++w) {
                const float4 f = qp[sub*(DH/4) + w];
                int v = 0;
                v |= int(uint8_t(int8_t(roundf(f.x*a127))));
                v |= int(uint8_t(int8_t(roundf(f.y*a127)))) <<  8;
                v |= int(uint8_t(int8_t(roundf(f.z*a127)))) << 16;
                v |= int(uint8_t(int8_t(roundf(f.w*a127)))) << 24;
                q8i[w] = v;
            }
        } else {
#pragma unroll
            for (int d4 = 0; d4 < DH/4; ++d4) {
                const float4 f = qp[DE == 256 ? 8*d4 + sub : sub*(DH/4) + d4];
                qreg[2*d4 + 0] = __floats2half2_rn(f.x*scale, f.y*scale);
                qreg[2*d4 + 1] = __floats2half2_rn(f.z*scale, f.w*scale);
            }
        }
    }

    float m = -FLT_MAX/2.0f, l = 0.0f;
    float acc[DH] = {};

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    for (int k0 = blockIdx.y*FATTN_GFX906_NBATCH_FA; k0 < k_VKQ_max; k0 += gridDim.y*FATTN_GFX906_NBATCH_FA) {
        const int k_lim = min(k0 + FATTN_GFX906_NBATCH_FA, k_VKQ_max);
        for (int n0 = k0; n0 < k_lim; n0 += TILE_N) {
            // cooperative K,V tile load. fp16 rows copy as uint4 (8 fp16 each).
            // q8_0 rows are 34B blocks (half scale + 32 i8); the qs sit at odd
            // byte offsets so they move as ushort pairs into split LDS arrays.
            if constexpr (KQ8) {
                for (int c = tid; c < TILE_N*9*NB; c += TILE_M*SPLIT) {
                    const int i  = c / (9*NB);
                    const int kv = n0 + i;
                    if (kv < k_lim) {
                        const int u = c % (9*NB);
                        const int b = u/9, o = u%9;
                        const unsigned short * row16 = reinterpret_cast<const unsigned short *>(K_h + (int64_t) kv*nb11);
                        if (o == 0) {
                            sKd[i][b] = reinterpret_cast<const half *>(row16)[b*17];
                        } else if constexpr (DE == 256) {
                            // word-major staging: block b word w lands at word
                            // w*NB + b so the 8 lanes read 8 distinct banks
                            reinterpret_cast<uint *>(sKq[i])[(o - 1)*NB + b] =
                                row16[b*17 + 2*o - 1] | (unsigned(row16[b*17 + 2*o]) << 16);
                        } else {
                            reinterpret_cast<uint *>(sKq[i])[b*8 + o - 1] =
                                row16[b*17 + 2*o - 1] | (unsigned(row16[b*17 + 2*o]) << 16);
                        }
                    }
                }
            }
            if constexpr (VQ8) {
                for (int c = tid; c < TILE_N*8*NB; c += TILE_M*SPLIT) {
                    const int i  = c / (8*NB);
                    const int kv = n0 + i;
                    if (kv < k_lim) {
                        const int u = c % (8*NB);
                        const int b = u/8, w = u%8;
                        const unsigned short * row16 = reinterpret_cast<const unsigned short *>(V_h + (int64_t) kv*nb21);
                        const float d = __half2float(reinterpret_cast<const half *>(row16)[b*17]);
                        const unsigned short u0 = row16[b*17 + 2*w + 1];
                        const unsigned short u1 = row16[b*17 + 2*w + 2];
                        __align__(8) half2 hv[2];
                        hv[0] = __floats2half2_rn(d*(int8_t) (u0 & 0xFF), d*(int8_t) (u0 >> 8));
                        hv[1] = __floats2half2_rn(d*(int8_t) (u1 & 0xFF), d*(int8_t) (u1 >> 8));
                        reinterpret_cast<uint2 *>(sV[i])[b*8 + w] = *reinterpret_cast<uint2 *>(hv);
                    }
                }
            }
            if constexpr (!KQ8 || !VQ8) {
                for (int c = tid; c < TILE_N*CPR; c += TILE_M*SPLIT) {
                    const int i = c / CPR;
                    const int kv = n0 + i;
                    if (kv < k_lim) {
                        const int off = (c % CPR);
                        if constexpr (!KQ8) {
                            *reinterpret_cast<uint4 *>(&sK[i][off*4]) =
                                reinterpret_cast<const uint4 *>(K_h + (int64_t) kv*nb11)[off];
                        }
                        if constexpr (!VQ8) {
                            *reinterpret_cast<uint4 *>(&sV[i][off*4]) =
                                reinterpret_cast<const uint4 *>(V_h + (int64_t) kv*nb21)[off];
                        }
                    }
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
                    float pd;
                    if constexpr (KQ8) {
                        // i8 x i8 -> s32 dots; lane dims sit inside K block
                        // sub*DH/32, so a lane-local Q scale is exact. DE == 256
                        // reads the word-major (transposed) sKq staging.
                        const int * kq = reinterpret_cast<const int *>(sKq[i]);
                        int s0 = 0, s1 = 0;
#pragma unroll
                        for (int w = 0; w < QW; w += 2) {
                            if constexpr (DE == 256) {
                                s0 = ggml_cuda_dp4a(q8i[w],     kq[w*NB + sub],       s0);
                                s1 = ggml_cuda_dp4a(q8i[w + 1], kq[(w + 1)*NB + sub], s1);
                            } else {
                                s0 = ggml_cuda_dp4a(q8i[w],     kq[sub*QW + w],     s0);
                                s1 = ggml_cuda_dp4a(q8i[w + 1], kq[sub*QW + w + 1], s1);
                            }
                        }
                        pd = (s0 + s1)*(q8d*__half2float(sKd[i][sub*DH/32]));
                    } else {
                        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                        for (int g = 0; g < DH/4; ++g) {
                            const int col = DE == 256 ? 16*g + 2*sub : sub*DQ + 2*g;
                            const uint2 kv2 = *reinterpret_cast<const uint2 *>(&sK[i][col]);
                            a0 = fattn_gfx906_dot2(a0, qreg[2*g],     reinterpret_cast<const half2 *>(&kv2)[0]);
                            a1 = fattn_gfx906_dot2(a1, qreg[2*g + 1], reinterpret_cast<const half2 *>(&kv2)[1]);
                        }
                        pd = a0 + a1;
                    }
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

                // PV: fp16x2 partial sums over this tile, folded into fp32.
                // Lane group g covers dims {4*sub + 32*g .. +3} for DE == 256
                // (col = 16*g + 2*sub) or the contiguous slice for DE < 256:
                half2 hacc[DQ] = {};
#pragma unroll 4
                for (int i = 0; i < nmax; ++i) {
                    const half2 pp = __half2half2(p[i]);
#pragma unroll
                    for (int g = 0; g < DH/4; ++g) {
                        const int col = DE == 256 ? 16*g + 2*sub : sub*DQ + 2*g;
                        const uint2 vv2 = *reinterpret_cast<const uint2 *>(&sV[i][col]);
                        hacc[2*g]     = __hfma2(pp, reinterpret_cast<const half2 *>(&vv2)[0], hacc[2*g]);
                        hacc[2*g + 1] = __hfma2(pp, reinterpret_cast<const half2 *>(&vv2)[1], hacc[2*g + 1]);
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
        float4 * drow = (float4 *) (dst + j_dst*DE);
#pragma unroll
        for (int d4 = 0; d4 < DH/4; ++d4) {
            drow[DE == 256 ? 8*d4 + sub : sub*(DH/4) + d4] =
                make_float4(acc[4*d4 + 0]*inv, acc[4*d4 + 1]*inv, acc[4*d4 + 2]*inv, acc[4*d4 + 3]*inv);
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

// D == 256 kernel: lane owns one key of the KQ phase (full 256-dim dot against
// Q staged in LDS, no cross-lane reduction), scores are exchanged through sE,
// then lanes remap to a DE/TN-dim slice for PV. K is staged once per tile,
// either as fp16 rows or as raw q8_0 words in a [word][key] transposed layout
// so the 16 key-lanes read conflict-free; V is always staged as fp16.
// LDS row stride is D2 + 4 half2 so 16B reads land on distinct bank quads.
template<int TN_, int ncols1, int ncols2, bool use_logit_softcap, bool KQ8 = false, bool VQ8 = false>
__launch_bounds__(ncols1*ncols2*TN_)
static __global__ void flash_attn_gfx906_256(
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

    constexpr int DE  = 256;
    constexpr int TM  = ncols1*ncols2;   // rows per block
    constexpr int TN  = TN_;             // K/V rows per tile
    constexpr int D2  = DE/2;            // half2 per row
    constexpr int DH  = DE/TN;           // out dims per lane in the V phase
    constexpr int CPR = DE/8;            // uint4 chunks per fp16 row
    constexpr int NB  = DE/32;           // q8_0 blocks per row
    constexpr int W8  = NB*8;            // packed i8 words per q8_0 row
    constexpr int SP  = D2 + 4;          // LDS row stride in half2

    // Small fp16-V blocks read V straight from global in the PV phase
    // (L2-friendly, no staging cost); q8_0 V and large tiles keep a staged
    // fp16 sV where the redundant global reads would dominate.
    constexpr bool VGLOB = !VQ8 && TM <= 12;

    static_assert(TM*TN <= 768, "bad TM");
    static_assert(DH % 8 == 0, "bad TN"); // PV reads 16B per lane

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

    // VGLOB blocks pipeline the K tiles (stage t+1 during KQ t) and scores
    // through double buffers, so one __syncthreads per tile suffices; the
    // staged variants keep single buffers and three syncs per tile.
    __shared__ __align__(16) half2 sQ [KQ8 ? 1     : TM][KQ8 ? 1   : SP];
    __shared__ __align__(16) int   sQq[KQ8 ? TM    : 1 ][KQ8 ? W8  : 1 ];
    __shared__ __align__(16) float sQd[KQ8 ? TM    : 1 ][KQ8 ? NB  : 1 ];
    __shared__ __align__(16) half2 sK [KQ8 ? 1 : (VGLOB ? 2 : 1)*TN][KQ8 ? 1 : SP];
    __shared__ __align__(16) int4  sKq[KQ8 ? (VGLOB ? 2 : 1)*W8/4 : 1 ][KQ8 ? TN : 1 ]; // [quad word][key]
    __shared__ __align__(16) float sKd[KQ8 ? (VGLOB ? 2 : 1)*NB   : 1 ][KQ8 ? TN : 1 ];
    __shared__ __align__(16) half2 sV [VGLOB ? 1 : TN][SP];
    __shared__ float sE[VGLOB ? 2 : 1][TM][TN];

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*qg) : nullptr;
    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head, n_head_log2, m0, m1) : 1.0f;

    // stage Q rows into LDS once. fp16 path: half2 with the softmax scale
    // folded in. q8_0 K path: int8 quants + fp32 scale per 32-dim block so the
    // v_dot4 result pairs exactly with the q8_0 K block scales.
    if constexpr (KQ8) {
        for (int c = tid; c < TM*NB; c += TM*TN) {
            const int r = c / NB;
            const int b = c % NB;
            const int q = col_Q_0 + r/ncols2;
            const int h = head0 + r%ncols2;
            if (q < int(ne01.z)) {
                const float4 * qp = (const float4 *) (Q + nb01*q + nb02*h + nb03*sequence) + 8*b;
                __align__(16) float4 f[8];
                float amax = 0.0f;
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    f[w] = qp[w];
                    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f[w].x), fabsf(f[w].y)),
                                             fmaxf(fabsf(f[w].z), fabsf(f[w].w))));
                }
                const float a127 = amax > 0.0f ? 127.0f/amax : 0.0f;
                sQd[r][b] = amax*scale/127.0f;
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    int v = 0;
                    v |= int(uint8_t(int8_t(roundf(f[w].x*a127))));
                    v |= int(uint8_t(int8_t(roundf(f[w].y*a127)))) <<  8;
                    v |= int(uint8_t(int8_t(roundf(f[w].z*a127)))) << 16;
                    v |= int(uint8_t(int8_t(roundf(f[w].w*a127)))) << 24;
                    sQq[r][b*8 + w] = v;
                }
            }
        }
    } else {
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
    }
    __syncthreads();

    float m = -FLT_MAX/2.0f, l = 0.0f;
    float acc[DH] = {};

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    // cooperative K tile load into buffer buf. q8_0 rows are 34B blocks (half
    // scale + 32 i8); the qs sit at odd byte offsets so they move as ushort
    // pairs into the transposed [word][key] layout.
    auto stage_k = [&](const int n0s, const int buf) {
        if constexpr (KQ8) {
            for (int c = tid; c < TN*9*NB; c += TM*TN) {
                const int i  = c / (9*NB);
                const int kv = n0s + i;
                if (kv < k_VKQ_max) {
                    const int u = c % (9*NB);
                    const int b = u/9, o = u%9;
                    const unsigned short * row16 = reinterpret_cast<const unsigned short *>(K_h + (int64_t) kv*nb11);
                    if (o == 0) {
                        sKd[buf*NB + b][i] = __half2float(reinterpret_cast<const half *>(row16)[b*17]);
                    } else {
                        reinterpret_cast<int *>(&sKq[buf*(W8/4) + (b*8 + o - 1)/4][i])[(o - 1) % 4] =
                            row16[b*17 + 2*o - 1] | (int(unsigned(row16[b*17 + 2*o])) << 16);
                    }
                }
            }
        } else {
            for (int c = tid; c < TN*CPR; c += TM*TN) {
                const int i = c / CPR;
                const int kv = n0s + i;
                if (kv < k_VKQ_max) {
                    const int off = (c % CPR);
                    *reinterpret_cast<uint4 *>(&sK[buf*TN + i][off*4]) =
                        reinterpret_cast<const uint4 *>(K_h + (int64_t) kv*nb11)[off];
                }
            }
        }
    };

    // V tile load into buffer buf, fp16 rows as uint4 or q8_0 dequantized:
    auto stage_v = [&](const int n0s, const int buf) {
        if constexpr (VQ8) {
            for (int c = tid; c < TN*8*NB; c += TM*TN) {
                const int i  = c / (8*NB);
                const int kv = n0s + i;
                if (kv < k_VKQ_max) {
                    const int u = c % (8*NB);
                    const int b = u/8, w = u%8;
                    const unsigned short * row16 = reinterpret_cast<const unsigned short *>(V_h + (int64_t) kv*nb21);
                    const float d = __half2float(reinterpret_cast<const half *>(row16)[b*17]);
                    const unsigned short u0 = row16[b*17 + 2*w + 1];
                    const unsigned short u1 = row16[b*17 + 2*w + 2];
                    __align__(8) half2 hv[2];
                    hv[0] = __floats2half2_rn(d*(int8_t) (u0 & 0xFF), d*(int8_t) (u0 >> 8));
                    hv[1] = __floats2half2_rn(d*(int8_t) (u1 & 0xFF), d*(int8_t) (u1 >> 8));
                    reinterpret_cast<uint2 *>(sV[buf*TN + i])[b*8 + w] = *reinterpret_cast<uint2 *>(hv);
                }
            }
        } else {
            for (int c = tid; c < TN*CPR; c += TM*TN) {
                const int i = c / CPR;
                const int kv = n0s + i;
                if (kv < k_VKQ_max) {
                    const int off = (c % CPR);
                    *reinterpret_cast<uint4 *>(&sV[buf*TN + i][off*4]) =
                        reinterpret_cast<const uint4 *>(V_h + (int64_t) kv*nb21)[off];
                }
            }
        }
    };

    // KQ phase: lane sub owns key n0+sub, full 256-dim dot per row:
    auto kq_phase = [&](const int n0, const int nmax, const int buf) {
        if (live) {
            float sc;
            if (sub < nmax) {
                if constexpr (KQ8) {
                    float sacc = 0.0f;
#pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
                        for (int w = 0; w < 2; ++w) {
                            const int4 qw = reinterpret_cast<const int4 *>(sQq[row])[b*2 + w];
                            const int4 kw = sKq[buf*(W8/4) + b*2 + w][sub];
                            s0 = ggml_cuda_dp4a(qw.x, kw.x, s0);
                            s1 = ggml_cuda_dp4a(qw.y, kw.y, s1);
                            s2 = ggml_cuda_dp4a(qw.z, kw.z, s2);
                            s3 = ggml_cuda_dp4a(qw.w, kw.w, s3);
                        }
                        sacc += ((s0 + s1) + (s2 + s3))*(sQd[row][b]*sKd[buf*NB + b][sub]);
                    }
                    sc = sacc;
                } else {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (int d2 = 0; d2 < D2; d2 += 4) {
                        const uint4 kv4 = *reinterpret_cast<const uint4 *>(&sK[buf*TN + sub][d2]);
                        const uint4 qv4 = *reinterpret_cast<const uint4 *>(&sQ[row][d2]);
                        const half2 * kh = reinterpret_cast<const half2 *>(&kv4);
                        const half2 * qh = reinterpret_cast<const half2 *>(&qv4);
                        a0 = fattn_gfx906_dot2(a0, qh[0], kh[0]);
                        a1 = fattn_gfx906_dot2(a1, qh[1], kh[1]);
                        a2 = fattn_gfx906_dot2(a2, qh[2], kh[2]);
                        a3 = fattn_gfx906_dot2(a3, qh[3], kh[3]);
                    }
                    sc = a0 + a1 + a2 + a3;
                }
                if (use_logit_softcap) {
                    sc = logit_softcap*tanhf(sc);
                }
                if (maskh) {
                    sc += slope*__half2float(maskh[n0 + sub]);
                }
            } else {
                sc = -INFINITY;
            }
            sE[buf][row][sub] = sc;
        }
    };

    // softmax + PV phase: lane owns dims {8*TN*j + 8*sub .. +7}:
    auto pv_phase = [&](const int n0, const int nmax, const int buf) {
        if (live) {
            float m_new = m;
#pragma unroll
            for (int i = 0; i < TN; ++i) {
                if (i < nmax) {
                    m_new = fmaxf(m_new, sE[buf][row][i] + FATTN_KQ_MAX_OFFSET);
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
                const float e = expf(sE[buf][row][i] - m_new);
                psum += e;
                const half2 ee = __floats2half2_rn(e, e);
#pragma unroll
                for (int j = 0; j < DH/8; ++j) {
                    uint4 vv4;
                    if constexpr (VGLOB) {
                        const char * vrow = V_h + (int64_t)(n0 + i)*nb21;
                        vv4 = *reinterpret_cast<const uint4 *>(vrow + 16*(TN*j + sub));
                    } else {
                        vv4 = *reinterpret_cast<const uint4 *>(&sV[i][4*TN*j + 4*sub]);
                    }
                    const half2 * vh = reinterpret_cast<const half2 *>(&vv4);
                    hacc[4*j + 0] = __hfma2(ee, vh[0], hacc[4*j + 0]);
                    hacc[4*j + 1] = __hfma2(ee, vh[1], hacc[4*j + 1]);
                    hacc[4*j + 2] = __hfma2(ee, vh[2], hacc[4*j + 2]);
                    hacc[4*j + 3] = __hfma2(ee, vh[3], hacc[4*j + 3]);
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
    };

    if constexpr (VGLOB) {
        int buf = 0;
        for (int k0 = blockIdx.y*FATTN_GFX906_NBATCH_FA; k0 < k_VKQ_max; k0 += gridDim.y*FATTN_GFX906_NBATCH_FA) {
            const int k_lim = min(k0 + FATTN_GFX906_NBATCH_FA, k_VKQ_max);
            stage_k(k0, buf);
            __syncthreads();
            for (int n0 = k0; n0 < k_lim; n0 += TN) {
                const int nmax = min(TN, k_lim - n0);
                kq_phase(n0, nmax, buf);
                // prefetch the next K tile into the other buffer while PV runs:
                if (n0 + TN < k_lim) {
                    stage_k(n0 + TN, buf ^ 1);
                }
                __syncthreads();
                pv_phase(n0, nmax, buf);
                buf ^= 1;
            }
        }
    } else {
        for (int k0 = blockIdx.y*FATTN_GFX906_NBATCH_FA; k0 < k_VKQ_max; k0 += gridDim.y*FATTN_GFX906_NBATCH_FA) {
            const int k_lim = min(k0 + FATTN_GFX906_NBATCH_FA, k_VKQ_max);
            for (int n0 = k0; n0 < k_lim; n0 += TN) {
                stage_k(n0, 0);
                __syncthreads();
                const int nmax = min(TN, k_lim - n0);
                kq_phase(n0, nmax, 0);
                stage_v(n0, 0);
                __syncthreads();
                pv_phase(n0, nmax, 0);
                __syncthreads();
            }
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
        // acc[8j..8j+7] holds dims 8*TN*j + 8*sub .. +7:
        const int64_t j_dst = ((int64_t) (sequence*int(ne01.z) + qg)*ne02 + head)*gridDim.y + blockIdx.y;
        float * drow = dst + j_dst*DE;
        const float inv = gridDim.y == 1 ? 1.0f/l : 1.0f;
#pragma unroll
        for (int j = 0; j < DH/8; ++j) {
            float4 * t = (float4 *) (drow + 8*TN*j + 8*sub);
            t[0] = make_float4(acc[8*j + 0]*inv, acc[8*j + 1]*inv, acc[8*j + 2]*inv, acc[8*j + 3]*inv);
            t[1] = make_float4(acc[8*j + 4]*inv, acc[8*j + 5]*inv, acc[8*j + 6]*inv, acc[8*j + 7]*inv);
        }
        if (gridDim.y != 1 && sub == 0) {
            dst_meta[j_dst] = make_float2(m, l);
        }
    }

    GGML_UNUSED_VARS(ne00, ne03, ne10, ne13, ne31, ne32, nb32);
}

// The pair kernel consumes q8_0 K/V natively; the D == 512 kernel and rows
// whose strides break the 2B-aligned ushort loads keep the fp16 conversion.
// Set GGML_GFX906_NO_Q8 before startup to force the fp16 conversion path for A/B.
static bool fattn_gfx906_can_use_q8(const ggml_tensor * t, const int64_t head_size) {
    static const bool disable_q8 = (getenv("GGML_GFX906_NO_Q8") != nullptr);
    if (disable_q8 || head_size == 512 || t->type != GGML_TYPE_Q8_0 || (uintptr_t) t->data % 2 != 0) {
        return false;
    }
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (t->nb[i] % 2 != 0) {
            return false;
        }
    }
    return true;
}

template<int DKQ, int ncols1, int ncols2, bool use_logit_softcap>
static void ggml_cuda_flash_attn_ext_gfx906_sc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool k_q8 = fattn_gfx906_can_use_q8(K, DKQ);
    const bool v_q8 = fattn_gfx906_can_use_q8(V, DKQ);

    // D256 lane-per-key: q8_0 K blocks small enough for it get 32 keys per
    // tile (fewer barriers per key); fp16 K and large blocks keep 16.
    const bool tn32 = k_q8 && DKQ == 256 && ncols1*ncols2 <= 24;
    const int nthreads = ncols1*ncols2*(DKQ == 512 ? 16 : DKQ == 256 ? (tn32 ? 32 : 16) : 4);
    const int ws = nthreads % 32 == 0 ? 32 : 16;
    fattn_kernel_t fattn_kernel;
    if constexpr (DKQ == 512) {
        fattn_kernel = flash_attn_gfx906_512<ncols1, ncols2, use_logit_softcap>;
    } else if constexpr (DKQ == 256) {
        if (k_q8 && ncols1*ncols2 <= 24) {
            if constexpr (ncols1*ncols2 <= 24) {
                fattn_kernel = v_q8 ? (fattn_kernel_t) flash_attn_gfx906_256<32, ncols1, ncols2, use_logit_softcap, true,  true>
                                    : (fattn_kernel_t) flash_attn_gfx906_256<32, ncols1, ncols2, use_logit_softcap, true,  false>;
            }
        } else if (k_q8) {
            fattn_kernel = v_q8 ? (fattn_kernel_t) flash_attn_gfx906_256<16, ncols1, ncols2, use_logit_softcap, true,  true>
                                : (fattn_kernel_t) flash_attn_gfx906_256<16, ncols1, ncols2, use_logit_softcap, true,  false>;
        } else {
            fattn_kernel = v_q8 ? (fattn_kernel_t) flash_attn_gfx906_256<16, ncols1, ncols2, use_logit_softcap, false, true>
                                : (fattn_kernel_t) flash_attn_gfx906_256<16, ncols1, ncols2, use_logit_softcap, false, false>;
        }
        launch_fattn<DKQ, ncols1, ncols2>
            (ctx, dst, fattn_kernel, nthreads/ws, 0, FATTN_GFX906_NBATCH_FA, !k_q8, !v_q8, false, false, ws);
        return;
    } else if (k_q8) {
        fattn_kernel = v_q8 ? (fattn_kernel_t) flash_attn_gfx906<DKQ, ncols1, ncols2, use_logit_softcap, true,  true>
                            : (fattn_kernel_t) flash_attn_gfx906<DKQ, ncols1, ncols2, use_logit_softcap, true,  false>;
    } else {
        fattn_kernel = v_q8 ? (fattn_kernel_t) flash_attn_gfx906<DKQ, ncols1, ncols2, use_logit_softcap, false, true>
                            : (fattn_kernel_t) flash_attn_gfx906<DKQ, ncols1, ncols2, use_logit_softcap, false, false>;
    }
    launch_fattn<DKQ, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nthreads/ws, 0, FATTN_GFX906_NBATCH_FA, !k_q8, !v_q8, false, false, ws);
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

// Kernel supports D in {64, 128, 256, 512} with fp16 or native q8_0 K/V and fp32 Q.
// Needs 16-byte aligned rows for the vectorized global loads; anything else
// falls through to the stock kernels.
static bool ggml_cuda_fattn_gfx906_eligible(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (K->ne[0] != 64 && K->ne[0] != 128 && K->ne[0] != 256 && K->ne[0] != 512) {
        return false;
    }
    if (K->ne[0] == 256) {
        // Correctness passes, but Qwen model benchmarks are slower than stock.
        static const bool enable_d256 = (getenv("GGML_GFX906_D256") != nullptr);
        if (!enable_d256) {
            return false;
        }
    }
    if (Q->ne[0] != K->ne[0] || V->ne[0] != K->ne[0]) {
        return false;
    }
    if ((uintptr_t) dst->data % 16 != 0) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V}) {
        // quant K/V either stay q8_0 for the native path or are converted to
        // contiguous fp16 by launch_fattn, so their strides are not checked here:
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

    if (K->ne[0] == 256) {
        // Eight lanes per row; cap the tile at 48 rows (384 threads). ncols2 is
        // the largest divisor of gqa_ratio <= 8; non-pow2 6 is allowed so Qwen
        // GQA6 tiles run with no dead rows. Small batches pack ncols1 = nb.
        int nc2 = ncols2;
        if (mask && max_bias == 0.0f && gqa_ratio % 8 != 0 && gqa_ratio % 6 == 0) {
            nc2 = 6;
        }
        if (nc2 == 6) {
            if (Q->ne[1] <= 1) { ggml_cuda_flash_attn_ext_gfx906_case<256, 1, 6>(ctx, dst); return; }
            if (Q->ne[1] <= 2) { ggml_cuda_flash_attn_ext_gfx906_case<256, 2, 6>(ctx, dst); return; }
            if (Q->ne[1] <= 4) { ggml_cuda_flash_attn_ext_gfx906_case<256, 4, 6>(ctx, dst); return; }
            ggml_cuda_flash_attn_ext_gfx906_case<256, 8, 6>(ctx, dst); return;
        }
        if (Q->ne[1] <= 2 && nc2 == 8) {
            if (Q->ne[1] <= 1) { ggml_cuda_flash_attn_ext_gfx906_case<256, 1, 8>(ctx, dst); return; }
            ggml_cuda_flash_attn_ext_gfx906_case<256, 2, 8>(ctx, dst); return;
        }
        if (Q->ne[1] <= 8) {
            switch (nc2) {
                case 8:  ggml_cuda_flash_attn_ext_gfx906_case<256, 4, 8>(ctx, dst); return;
                case 4:  ggml_cuda_flash_attn_ext_gfx906_case<256, 8, 4>(ctx, dst); return;
                case 2:  ggml_cuda_flash_attn_ext_gfx906_case<256, 8, 2>(ctx, dst); return;
                default: ggml_cuda_flash_attn_ext_gfx906_case<256, 8, 1>(ctx, dst); return;
            }
        }
        switch (nc2) {
            case 8:  ggml_cuda_flash_attn_ext_gfx906_case<256,  4, 8>(ctx, dst); return;
            case 4:  ggml_cuda_flash_attn_ext_gfx906_case<256,  8, 4>(ctx, dst); return;
            case 2:  ggml_cuda_flash_attn_ext_gfx906_case<256, 16, 2>(ctx, dst); return;
            default: ggml_cuda_flash_attn_ext_gfx906_case<256, 32, 1>(ctx, dst); return;
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
