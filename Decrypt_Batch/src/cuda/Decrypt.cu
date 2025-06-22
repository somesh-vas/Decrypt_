#include <sys/stat.h>   // mkdir
#include "decrypt.h"               // KATNUM, SYS_T, SYS_N, sb, etc.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using C4 = uchar4;                  // four packed ciphertext bytes
typedef uchar4 C4;
// cuda error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

////////////////////////////////////////////////////////////////////////////////
//  Kernel 1 – compute 2·t syndromes
////////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// Tensor‐core‐style kernel: 4 warps / block, each warp → syndrome
//-----------------------------------------------------------------------------

__global__ void computeSyndromesKernel(
    const C4  * __restrict__ d_ct4,   // [ batchSize × ((SYND_BYTES+3)/4) ]
    const gf  * __restrict__ d_inv,   // [ sb × (2*SYS_T) ]
          gf  * __restrict__ d_syn)   // [ batchSize × (2*SYS_T) ]
{
    extern __shared__ gf shared[];
    gf *c = shared;                   // length sb = SYND_BYTES*8

    const int tid      = threadIdx.x;
    const int ct       = blockIdx.y;  // which codeword in this batch
    const int wordsPer= (SYND_BYTES + 3) / 4;
    const size_t base = size_t(ct) * wordsPer;

    // 1) zero-out the bit-buffer
    for (int i = tid; i < sb; i += blockDim.x) {
        c[i] = 0;
    }
    __syncthreads();

    // 2) unpack each 32-bit word into 32 bits
    if (tid < wordsPer) {
        C4 w = d_ct4[base + tid];
        const int bit0 = tid * 32;
        #pragma unroll
        for (int by = 0; by < 4; ++by) {
            unsigned char B = *(&w.x + by);
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                int idx = bit0 + by * 8 + b;
                if (idx < sb) {
                    c[idx] = (B >> b) & 1u;
                }
            }
        }
    }
    __syncthreads();

    // 3) tensor-warp dot-product: one column per tid<2*SYS_T
    if (tid < 2 * SYS_T) {
        const gf *col    = d_inv + tid; 
        const int stride = 2 * SYS_T;
        gf acc = 0;
        #pragma unroll 4
        for (int i = 0; i < sb; ++i) {
            gf mask = (gf)(-int(c[i] & 1));
            acc ^= (col[0] & mask);
            col += stride;
        }
        d_syn[ ct * (2 * SYS_T) + tid ] = acc;
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

__global__
void chien_search_kernel(
    const gf* __restrict__ d_sigma_all,
    unsigned char* __restrict__ d_error_all
) {
    // Shared memory to cache the sigma polynomial for the current block.
    extern __shared__ gf s_sigma[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int cipherIdx = blockIdx.y;
    const int posIdx = blockIdx.x * block_size + tid;

    // --- Step 1: Cache Sigma Polynomial in Shared Memory ---
    // Each thread loads one coefficient. This is a coalesced read.
    for(int i = tid; i <= SYS_T; i += block_size) {
        s_sigma[i] = d_sigma_all[cipherIdx * (SYS_T + 1) + i];
    }
    __syncthreads();

    if (posIdx >= SYS_N) return;

    // --- Step 2: Polynomial Evaluation with Horner's Method ---
    // Get the field element alpha^posIdx for this position from constant memory
    gf a = d_L[posIdx];
    gf val = s_sigma[SYS_T];

    // Evaluate the RECIPROCAL polynomial. All reads of sigma are
    // now fast reads from shared memory.
    #pragma unroll
    for (int j = SYS_T - 1; j >= 0; j--) {
        val = mul(val, a) ^ s_sigma[j];
    }
    
    // --- Step 3: Write Results ---
    // If val is 0, alpha^posIdx is a root, meaning an error exists at this position.
    d_error_all[cipherIdx * SYS_N + posIdx] = (val == 0);
}



// void decrypt_mass_streamed_to_disk(unsigned char (*ciphertexts)[crypto_kem_CIPHERTEXTBYTES]) {
//     const int total      = KATNUM;
//     const int batchSize  = BATCH_SIZE;
//     const int numStreams = 4;

//     // Shared kernel inputs (reused per stream)
//     cudaStream_t streams[numStreams];
//     unsigned char *d_ct[numStreams];
//     gf           *d_syn[numStreams], *d_loc[numStreams];
//     unsigned char *d_err[numStreams], *h_err_batch[numStreams];

//     for (int i = 0; i < numStreams; ++i) {
//         CUDA_CHECK(cudaStreamCreate(&streams[i]));

//         CUDA_CHECK(cudaMalloc(&d_ct[i],   batchSize * crypto_kem_CIPHERTEXTBYTES));
//         CUDA_CHECK(cudaMalloc(&d_syn[i],  batchSize * 2 * SYS_T * sizeof(gf)));
//         CUDA_CHECK(cudaMalloc(&d_loc[i],  batchSize * (SYS_T + 1) * sizeof(gf)));
//         CUDA_CHECK(cudaMalloc(&d_err[i],  batchSize * SYS_N * sizeof(unsigned char)));
//         CUDA_CHECK(cudaMallocHost(&h_err_batch[i], batchSize * SYS_N * sizeof(unsigned char)));
//     }

//     float totalH2Dms = 0.0f;
//     float totalD2Hms = 0.0f;
//     float totalKernelMs = 0.0f;
//     float totalBatchMs = 0.0f;

//     int batchCount = (total + batchSize - 1) / batchSize;

//     for (int b = 0; b < batchCount; ++b) {
//         int streamId    = b % numStreams;
//         cudaStream_t s  = streams[streamId];
//         int offset      = b * batchSize;
//         int actualBatch = (offset + batchSize > total) ? (total - offset) : batchSize;

//         // Events
//         cudaEvent_t evH2DStart, evH2DStop;
//         cudaEvent_t evKernelStart, evKernelStop;
//         cudaEvent_t evD2HStart, evD2HStop;
//         cudaEvent_t evBatchStart, evBatchStop;

//         cudaEventCreate(&evH2DStart);     cudaEventCreate(&evH2DStop);
//         cudaEventCreate(&evKernelStart);  cudaEventCreate(&evKernelStop);
//         cudaEventCreate(&evD2HStart);     cudaEventCreate(&evD2HStop);
//         cudaEventCreate(&evBatchStart);   cudaEventCreate(&evBatchStop);

//         // (1) Start total batch timer
//         cudaEventRecord(evBatchStart, s);

//         // (2) H2D transfer
//         cudaEventRecord(evH2DStart, s);
//         CUDA_CHECK(cudaMemcpyAsync(
//             d_ct[streamId],
//             &ciphertexts[offset],
//             actualBatch * crypto_kem_CIPHERTEXTBYTES,
//             cudaMemcpyHostToDevice,
//             s
//         ));
//         cudaEventRecord(evH2DStop, s);

//         // (3) Kernel launch
//         cudaEventRecord(evKernelStart, s);

//         computeSyndromesKernel<<<
//             dim3(1, actualBatch), 256, sb * sizeof(gf), s
//         >>>((C4*)d_ct[streamId], d_inverse_elements, d_syn[streamId]);
//         CUDA_CHECK(cudaGetLastError());

//         berlekampMasseyKernel<<<
//             dim3(1, actualBatch),
//             ((SYS_T + 1 + 31) & ~31),
//             0, s
//         >>>(d_syn[streamId], d_loc[streamId]);
//         CUDA_CHECK(cudaGetLastError());

//         chien_search_kernel<<<
//             dim3((SYS_N + 255) / 256, actualBatch),
//             256,
//             (SYS_T + 1) * sizeof(gf),
//             s
//         >>>(d_loc[streamId], d_err[streamId]);
//         CUDA_CHECK(cudaGetLastError());

//         cudaEventRecord(evKernelStop, s);

//         // (4) D2H copy
//         cudaEventRecord(evD2HStart, s);
//         CUDA_CHECK(cudaMemcpyAsync(
//             h_err_batch[streamId],
//             d_err[streamId],
//             actualBatch * SYS_N * sizeof(unsigned char),
//             cudaMemcpyDeviceToHost,
//             s
//         ));
//         cudaEventRecord(evD2HStop, s);

//         // (5) End batch
//         cudaEventRecord(evBatchStop, s);

//         // (6) Launch async write-to-disk on host after sync
//         cudaStreamSynchronize(s);  // required before accessing h_err_batch

//         // Measure times
//         float h2dMs = 0, d2hMs = 0, kernelMs = 0, batchMs = 0;
//         cudaEventElapsedTime(&h2dMs,     evH2DStart,    evH2DStop);
//         cudaEventElapsedTime(&kernelMs,  evKernelStart, evKernelStop);
//         cudaEventElapsedTime(&d2hMs,     evD2HStart,    evD2HStop);
//         cudaEventElapsedTime(&batchMs,   evBatchStart,  evBatchStop);

//         totalH2Dms     += h2dMs;
//         totalD2Hms     += d2hMs;
//         totalKernelMs  += kernelMs;
//         totalBatchMs   += batchMs;

//         float throughput = actualBatch * 1000.f / batchMs;
//         printf("[Batch %2d] Total: %.2f ms | H2D: %.2f ms | Kernel: %.2f ms | D2H: %.2f ms → %.2f cw/s\n",
//                b, batchMs, h2dMs, kernelMs, d2hMs, throughput);

//         // Write output to file
//         char filename[128];
//         snprintf(filename, sizeof(filename), "Output/errorstream%d.bin", b);
//         FILE *fout = fopen(filename, "wb");
//         if (!fout) {
//             perror("Error writing batch result");
//             exit(EXIT_FAILURE);
//         }
//         fwrite(h_err_batch[streamId], sizeof(unsigned char), actualBatch * SYS_N, fout);
//         fclose(fout);

//         // Cleanup events for this batch
//         cudaEventDestroy(evH2DStart);     cudaEventDestroy(evH2DStop);
//         cudaEventDestroy(evKernelStart);  cudaEventDestroy(evKernelStop);
//         cudaEventDestroy(evD2HStart);     cudaEventDestroy(evD2HStop);
//         cudaEventDestroy(evBatchStart);   cudaEventDestroy(evBatchStop);
//     }

//     // Final cleanup
//     for (int i = 0; i < numStreams; ++i) {
//         CUDA_CHECK(cudaFree(d_ct[i]));
//         CUDA_CHECK(cudaFree(d_syn[i]));
//         CUDA_CHECK(cudaFree(d_loc[i]));
//         CUDA_CHECK(cudaFree(d_err[i]));
//         CUDA_CHECK(cudaFreeHost(h_err_batch[i]));
//         CUDA_CHECK(cudaStreamDestroy(streams[i]));
//     }

//     cudaDeviceReset();

//     // Print final summary
//     printf("\n===== Summary =====\n");
//     printf("Total Host→Device (H2D) transfer time : %.2f ms\n", totalH2Dms);
//     printf("Total Device→Host (D2H) transfer time : %.2f ms\n", totalD2Hms);
//     printf("Total Kernel execution time           : %.2f ms\n", totalKernelMs);
//     printf("Total End-to-End batch time           : %.2f ms\n", totalBatchMs);
// }
void decrypt(unsigned char (*ciphertexts)[crypto_kem_CIPHERTEXTBYTES]) {
    const int total     = KATNUM;
    const int batchSize = BATCH_SIZE;

    // Single kernel input/output buffers
    unsigned char *d_ct;
    gf           *d_syn, *d_loc;
    unsigned char *d_err, *h_err_batch;

    CUDA_CHECK(cudaMalloc(&d_ct,   batchSize * crypto_kem_CIPHERTEXTBYTES));
    CUDA_CHECK(cudaMalloc(&d_syn,  batchSize * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_loc,  batchSize * (SYS_T + 1) * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_err,  batchSize * SYS_N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMallocHost(&h_err_batch, batchSize * SYS_N * sizeof(unsigned char)));

    float totalH2Dms = 0.0f;
    float totalD2Hms = 0.0f;
    float totalKernelMs = 0.0f;
    float totalBatchMs = 0.0f;

    int batchCount = (total + batchSize - 1) / batchSize;

    for (int b = 0; b < batchCount; ++b) {
        int offset      = b * batchSize;
        int actualBatch = (offset + batchSize > total) ? (total - offset) : batchSize;

        // Events
        cudaEvent_t evH2DStart, evH2DStop;
        cudaEvent_t evKernelStart, evKernelStop;
        cudaEvent_t evD2HStart, evD2HStop;
        cudaEvent_t evBatchStart, evBatchStop;

        cudaEventCreate(&evH2DStart);     cudaEventCreate(&evH2DStop);
        cudaEventCreate(&evKernelStart);  cudaEventCreate(&evKernelStop);
        cudaEventCreate(&evD2HStart);     cudaEventCreate(&evD2HStop);
        cudaEventCreate(&evBatchStart);   cudaEventCreate(&evBatchStop);

        // (1) Start total batch timer
        cudaEventRecord(evBatchStart);

        // (2) H2D transfer
        cudaEventRecord(evH2DStart);
        CUDA_CHECK(cudaMemcpy(
            d_ct,
            &ciphertexts[offset],
            actualBatch * crypto_kem_CIPHERTEXTBYTES,
            cudaMemcpyHostToDevice
        ));
        cudaEventRecord(evH2DStop);

        // (3) Kernel launch
        cudaEventRecord(evKernelStart);

        computeSyndromesKernel<<<
            dim3(1, actualBatch), 256, sb * sizeof(gf)
        >>>((C4*)d_ct, d_inverse_elements, d_syn);
        CUDA_CHECK(cudaGetLastError());

        berlekampMasseyKernel<<<
            dim3(1, actualBatch),
            ((SYS_T + 1 + 31) & ~31)
        >>>(d_syn, d_loc);
        CUDA_CHECK(cudaGetLastError());

        chien_search_kernel<<<
            dim3((SYS_N + 255) / 256, actualBatch),
            512,
            (SYS_T + 1) * sizeof(gf)
        >>>(d_loc, d_err);
        CUDA_CHECK(cudaGetLastError());

        cudaEventRecord(evKernelStop);

        // (4) D2H copy
        cudaEventRecord(evD2HStart);
        CUDA_CHECK(cudaMemcpy(
            h_err_batch,
            d_err,
            actualBatch * SYS_N * sizeof(unsigned char),
            cudaMemcpyDeviceToHost
        ));
        cudaEventRecord(evD2HStop);

        // (5) End batch
        cudaEventRecord(evBatchStop);
        cudaEventSynchronize(evBatchStop); // Required for elapsed time measurement

        // Measure times
        float h2dMs = 0, d2hMs = 0, kernelMs = 0, batchMs = 0;
        cudaEventElapsedTime(&h2dMs,     evH2DStart,    evH2DStop);
        cudaEventElapsedTime(&kernelMs,  evKernelStart, evKernelStop);
        cudaEventElapsedTime(&d2hMs,     evD2HStart,    evD2HStop);
        cudaEventElapsedTime(&batchMs,   evBatchStart,  evBatchStop);

        totalH2Dms     += h2dMs;
        totalD2Hms     += d2hMs;
        totalKernelMs  += kernelMs;
        totalBatchMs   += batchMs;

        float throughput = actualBatch * 1000.f / batchMs;
        printf("[Batch %2d] Total: %.2f ms | H2D: %.2f ms | Kernel: %.2f ms | D2H: %.2f ms → %.2f cw/s\n",
               b, batchMs, h2dMs, kernelMs, d2hMs, throughput);

        // Write output to file
        char filename[128];
        snprintf(filename, sizeof(filename), "Output/errorstream%d.bin", b);
        FILE *fout = fopen(filename, "wb");
        if (!fout) {
            perror("Error writing batch result");
            exit(EXIT_FAILURE);
        }
        fwrite(h_err_batch, sizeof(unsigned char), actualBatch * SYS_N, fout);
        fclose(fout);

        // Cleanup events
        cudaEventDestroy(evH2DStart);     cudaEventDestroy(evH2DStop);
        cudaEventDestroy(evKernelStart);  cudaEventDestroy(evKernelStop);
        cudaEventDestroy(evD2HStart);     cudaEventDestroy(evD2HStop);
        cudaEventDestroy(evBatchStart);   cudaEventDestroy(evBatchStop);
    }

    // Final cleanup
    CUDA_CHECK(cudaFree(d_ct));
    CUDA_CHECK(cudaFree(d_syn));
    CUDA_CHECK(cudaFree(d_loc));
    CUDA_CHECK(cudaFree(d_err));
    CUDA_CHECK(cudaFreeHost(h_err_batch));

    cudaDeviceReset();

    // Final summary
    printf("\n===== Summary =====\n");
    printf("Total Host→Device (H2D) transfer time : %.2f ms\n", totalH2Dms);
    printf("Total Device→Host (D2H) transfer time : %.2f ms\n", totalD2Hms);
    printf("Total Kernel execution time           : %.2f ms\n", totalKernelMs);
    printf("Total End-to-End batch time           : %.2f ms\n", totalBatchMs);
}

int main(void)
{ 
    //   unsigned char (*ciphertexts)[crypto_kem_CIPHERTEXTBYTES] =  malloc(KATNUM * sizeof(*ciphertexts));
    cudaDeviceReset();
    initialisation(secretkeys, ciphertexts, sk, L, g);
    compute_inverses();
    InitializeC();
    decrypt(ciphertexts);
    return 0;
}
// -----------------------------------------------------------------------------
