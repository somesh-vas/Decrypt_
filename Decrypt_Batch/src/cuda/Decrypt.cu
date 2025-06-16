#include "decrypt.h"               // KATNUM, SYS_T, SYS_N, sb, etc.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using C4 = uchar4;                  // four packed ciphertext bytes

#define CHECK_CUDA(x)                                                     \
    do {                                                                  \
        cudaError_t err__ = (x);                                          \
        if (err__ != cudaSuccess) {                                       \
            fprintf(stderr,"CUDA error %s @ %s:%d – %s\n",                \
                    #x,__FILE__,__LINE__,cudaGetErrorString(err__));      \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
//  Kernel 1 – compute 2·t syndromes
////////////////////////////////////////////////////////////////////////////////
__global__ void computeSyndromesKernel(const C4 * __restrict__ d_ct4,
                                       const gf * __restrict__ d_inv,
                                       gf       * __restrict__ d_syn)
{
    extern __shared__ gf shared[];
    gf *c = shared;

    const int tid      = threadIdx.x;
    const int ct       = blockIdx.y;
    const int wordsPer = (SYND_BYTES + 3) / 4;
    const size_t base  = static_cast<size_t>(ct) * wordsPer;

    for (int i = tid; i < sb; i += blockDim.x) c[i] = 0;
    __syncthreads();

    if (tid < wordsPer) {
        C4 w = d_ct4[base + tid];
        const int bit0 = tid * 32;
        #pragma unroll
        for (int by = 0; by < 4; ++by) {
            unsigned char B = *(&w.x + by);
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                int idx = bit0 + by * 8 + b;
                if (idx < sb) c[idx] = (B >> b) & 1u;
            }
        }
    }
    __syncthreads();

    if (tid < 2 * SYS_T) {
        const gf *col = d_inv + tid;
        const int stride = 2 * SYS_T;
        gf acc = 0;
        for (int i = 0; i < sb; ++i) {
            gf mask = gf(-int(c[i] & 1));
            acc ^= (col[0] & mask);
            col += stride;
        }
        d_syn[ct * (2 * SYS_T) + tid] = acc;
    }
}

////////////////////////////////////////////////////////////////////////////////
//  Kernel 2 – branch‑free Berlekamp–Massey (no constant‑mem masks)
////////////////////////////////////////////////////////////////////////////////
__global__ void berlekampMasseyKernel(const gf * __restrict__ d_syn,
                                      gf       * __restrict__ d_loc)
{
    __shared__ gf S[2 * SYS_T];
    __shared__ gf C[SYS_T + 1];
    __shared__ gf Btmp[SYS_T + 1];
    __shared__ gf B[SYS_T + 1];
    __shared__ gf warpSum[(SYS_T + 32) / 32];

    const int tid = threadIdx.x;
    const int ct  = blockIdx.y;

    for (int i = tid; i < 2 * SYS_T; i += blockDim.x)
        S[i] = __ldg(&d_syn[ct * (2 * SYS_T) + i]);
    __syncthreads();

    if (tid <= SYS_T) {
        C[tid] = (tid == 0);
        B[tid] = (tid == 1);
    }
    __shared__ gf b;  __shared__ int L;
    if (tid == 0) { b = 1;  L = 0; }
    __syncthreads();

    #pragma unroll 1
    for (int N = 0; N < 2 * SYS_T; ++N) {
        const int max_j = min(N, SYS_T);

        gf part = 0;
        for (int j = tid; j <= max_j; j += blockDim.x)
            part ^= mul(C[j], S[N - j]);
        for (int off = 16; off; off >>= 1)
            part ^= __shfl_down_sync(0xFFFFFFFFu, part, off);
        if ((tid & 31) == 0) warpSum[tid >> 5] = part;
        __syncthreads();

        __shared__ gf d, f, mask_ne, mask_le;
        if (tid == 0) {
            d = 0; for (int w = 0; w <= max_j / 32; ++w) d ^= warpSum[w];
            const bool nz  = (d != 0);
            const bool big = nz && (2 * L <= N);
            mask_ne = nz  ? (gf)0xFFFF : 0;
            mask_le = big ? (gf)0xFFFF : 0;
            f       = nz ? p_gf_frac(b, d) : 0;
            if (big) L = N + 1 - L;
            b = (b & ~mask_le) | (d & mask_le);
        }
        __syncthreads();

        if (tid <= SYS_T) {
            const gf oldC = C[tid];
            const gf oldB = B[tid];
            C[tid]   = oldC ^ (mul(f, oldB) & mask_ne);
            Btmp[tid] = (oldB & ~mask_le) | (oldC & mask_le);
        }
        __syncthreads();

        if (tid <= SYS_T) {
            B[tid] = (tid ? Btmp[tid - 1] : 0);
        }
        __syncthreads();
    }

    if (tid <= SYS_T) d_loc[ct * (SYS_T + 1) + tid] = C[SYS_T - tid];
}

////////////////////////////////////////////////////////////////////////////////
//  Kernel 3 – Chien search
////////////////////////////////////////////////////////////////////////////////
// __global__ void chien_search_kernel(const gf * __restrict__ d_loc,
//                                     unsigned char * __restrict__ d_err)
// {
//     const int ct  = blockIdx.y;
//     const int pos = blockIdx.x * blockDim.x + threadIdx.x;
//     if (pos >= SYS_N) return;

//     const gf *sigma = d_loc + ct * (SYS_T + 1);
//     gf a   = d_L[pos];
//     gf val = sigma[SYS_T];
//     for (int j = SYS_T - 1; j >= 0; --j) val = mul(val, a) ^ sigma[j];
//     d_err[ct * SYS_N + pos] = (val == 0);
// }
__global__ void chien_search_kernel(
    const gf* __restrict__ d_sigma_all,    // Input:  KATNUM x (SYS_T+1) coefficients
    unsigned char* __restrict__ d_error_all // Output: KATNUM x SYS_N error flags
) {
    // --- Shared Memory Declaration & Loading ---

    // Declare shared memory for the error-locator polynomial coefficients.
    // Size is SYS_T + 1 coefficients.
    extern __shared__ gf s_sigma[];

    const int tid = threadIdx.x;
    const int cipherIdx = blockIdx.y;

    // The stride for accessing the next polynomial must be (SYS_T + 1).
    const gf* sigma_global = d_sigma_all + cipherIdx * (SYS_T + 1);

    // Cooperatively load all SYS_T + 1 coefficients into shared memory.
    if (tid <= SYS_T) {
        s_sigma[tid] = sigma_global[tid];
    }
    __syncthreads(); // Barrier: Ensure all threads have finished loading before proceeding.


    // --- Chien Search Evaluation ---

    int posIdx = blockIdx.x * blockDim.x + tid;
    
    // Boundary check is crucial.
    if (posIdx < SYS_N) {
        // Get the field element alpha^posIdx for this position from constant memory
        gf a = d_L[posIdx];

        // CRITICAL FIX: Reverted to the correct evaluation logic for the
        // RECIPROCAL polynomial: x^t + C_1*x^(t-1) + ... + C_t.
        // The coefficients are stored as [C_t, ..., C_1, C_0=1].
        // Initialize with C_0=1, which is the last coefficient in the array.
        gf val = s_sigma[SYS_T];

        #pragma unroll
        for (int j = SYS_T - 1; j >= 0; j--) {
            // --- Inlined, Constant-Time Galois Field Multiplication: val = val * a ---
            uint32_t t0 = val;
            uint32_t t1 = a;
            uint32_t tmp = 0;

            // Bitsliced "Russian Peasant" multiplication loop
            #pragma unroll
            for (int b = 0; b < GFBITS; b++) {
                // Equivalent to: if (t1 & (1 << b)) tmp ^= (t0 << b);
                // This branchless version is constant-time.
                uint32_t mask = -((t1 >> b) & 1);
                tmp ^= (t0 << b) & mask;
            }

            // --- Fast, Branchless Reduction (Modulo GF_POLY for GF(2^12)) ---
            // This logic is specific to the primitive polynomial x^12 + x^3 + 1
            uint32_t t = tmp & 0x7FC000;
            tmp ^= t >> 9;
            tmp ^= t >> 12;
            t    = tmp & 0x3000;
            tmp ^= t >> 9;
            tmp ^= t >> 12;
            val = (gf)(tmp & GFMASK);
            // --- End of Multiplication ---

            // Complete the Horner's method step, reading the next coefficient from shared memory.
            val ^= s_sigma[j];
        }

        // If val is 0, alpha^posIdx is a root, meaning an error exists at this position.
        // The write to global memory is inside the bounds check.
        d_error_all[cipherIdx * SYS_N + posIdx] = (val == 0);
    }
}
////////////////////////////////////////////////////////////////////////////////
//  Host pipeline
////////////////////////////////////////////////////////////////////////////////
static void decrypt_mass_separate(void)
{
    const int SYNS  = 2 * SYS_T;
    const int LOCS  = SYS_T + 1;
    const int words = (SYND_BYTES + 3) / 4;

    const size_t packedCtBytes = size_t(KATNUM) * words * sizeof(C4);
    const size_t synBytes      = size_t(KATNUM) * SYNS * sizeof(gf);
    const size_t locBytes      = size_t(KATNUM) * LOCS * sizeof(gf);
    const size_t errBytes      = size_t(KATNUM) * SYS_N * sizeof(unsigned char);

    C4 *h_ct4 = (C4 *)malloc(packedCtBytes);
    memset(h_ct4, 0, packedCtBytes);

    for (int ct = 0; ct < KATNUM; ++ct) {
        const unsigned char *src = &ciphertexts[ct][0];
        memcpy(reinterpret_cast<unsigned char *>(h_ct4) + ct * words * 4,
               src, crypto_kem_CIPHERTEXTBYTES);
    }

    gf *h_syn = (gf *)malloc(synBytes);
    gf *h_loc = (gf *)malloc(locBytes);
    unsigned char *h_err = (unsigned char *)malloc(errBytes);

    C4 *d_ct4   = nullptr; CHECK_CUDA(cudaMalloc(&d_ct4, packedCtBytes));
    gf *d_syn   = nullptr; CHECK_CUDA(cudaMalloc(&d_syn, synBytes));
    gf *d_loc   = nullptr; CHECK_CUDA(cudaMalloc(&d_loc, locBytes));
    unsigned char *d_err = nullptr; CHECK_CUDA(cudaMalloc(&d_err, errBytes));

    CHECK_CUDA(cudaMemcpy(d_ct4, h_ct4, packedCtBytes, cudaMemcpyHostToDevice));

    dim3 gridSyn(1, KATNUM);
    const int threadsSyn = 256;
    const int shSyn      = (sb + 2 * SYS_T) * sizeof(gf);
    computeSyndromesKernel<<<gridSyn, threadsSyn, shSyn>>>(d_ct4, d_inverse_elements, d_syn);
    CHECK_CUDA(cudaGetLastError());

    const int threadsBM = ((SYS_T + 1 + 31) & ~31);
    berlekampMasseyKernel<<< dim3(1, KATNUM), threadsBM >>>(d_syn, d_loc);
    CHECK_CUDA(cudaGetLastError());

    const int threadsCS = 256;
    const int blocksCS  = (SYS_N + threadsCS - 1) / threadsCS;
    dim3 gridCS(blocksCS, KATNUM);
    chien_search_kernel<<<gridCS, threadsCS>>>(d_loc, d_err);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_err, d_err, errBytes, cudaMemcpyDeviceToHost));

    // for (int ct = 0; ct < KATNUM; ++ct) {
    //     for (int i = 0; i < SYS_N; ++i) if (h_err[ct * SYS_N + i]) printf("%d ", i);
    //     printf("\n");
    // }

    free(h_ct4); free(h_syn); free(h_loc); free(h_err);
    CHECK_CUDA(cudaFree(d_ct4));
    CHECK_CUDA(cudaFree(d_syn));
    CHECK_CUDA(cudaFree(d_loc));
    CHECK_CUDA(cudaFree(d_err));
    CHECK_CUDA(cudaDeviceReset());
}

////////////////////////////////////////////////////////////////////////////////
//  main – build field tables, then call pipeline
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
    initialisation(secretkeys, ciphertexts, sk, L, g);
    compute_inverses();
    InitializeC();
    decrypt_mass_separate();
    return 0;
}
