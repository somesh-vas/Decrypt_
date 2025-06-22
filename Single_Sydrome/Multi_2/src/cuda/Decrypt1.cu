//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// Multi-2 syndrome kernel: 2 ciphertexts per block, 256 threads/block
// – unpack both ciphers into shared memory
// – each block handles ctIdx0 = 2*blockIdx.y, ctIdx1 = ctIdx0+1
// – threads 0..127 compute 128 coeffs for ct0, 128..255 compute for ct1
// ------------------------------------------------------------------------
__global__ void computeOut1(
    const gf*  __restrict__ d_inverse_elements,  // [sb][2*SYS_T] per CT
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES per CT
          gf*  __restrict__ out1                  // KATNUM × (2*SYS_T)
) {
    extern __shared__ gf shared[];
    gf* c0    = shared;                        // sb elements
    gf* c1    = c0 + sb;                       // sb elements
    gf* s0    = c1 + sb;                       // 2*SYS_T elements
    gf* s1    = s0 + 2*SYS_T;                  // 2*SYS_T elements

    int tid    = threadIdx.x;                  // 0..255
    int ctPair = blockIdx.y;                   // one block per pair
    int ct0    = ctPair * 2;
    int ct1    = ct0 + 1;
    const int stride = 2 * SYS_T;
    const int bytes  = SYND_BYTES;             // (sb+7)/8

    // 1) unpack both ciphertexts’ bits into shared memory
    for (int bIdx = tid; bIdx < bytes; bIdx += blockDim.x) {
        unsigned char r0 = d_ciphertexts[ct0 * bytes + bIdx];
        unsigned char r1 = d_ciphertexts[ct1 * bytes + bIdx];
    #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            int idx = bIdx * 8 + bit;
            if (idx < sb) {
                c0[idx] = (r0 >> bit) & 1U;
                c1[idx] = (r1 >> bit) & 1U;
            }
        }
    }
    __syncthreads();

    // 2) compute syndrome coefficients
    //    threads 0..stride-1 → s0[0..stride), threads stride..2*stride-1 → s1[0..stride)
    if (tid < 2 * stride) {
        bool isFirst = tid < stride;
        int  coeff   = isFirst ? tid : (tid - stride);
        const gf *colBase = d_inverse_elements + (isFirst ? ct0 : ct1) * stride;
        gf sum = 0;

    #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            gf mask = (gf)(- (int)( isFirst ? c0[bit] : c1[bit] ));
            sum ^= (colBase[ coeff + bit*stride ] & mask);
        }
        if (isFirst) s0[coeff] = sum;
        else          s1[coeff] = sum;
    }
    __syncthreads();

    // 3) write back both sets of 2*SYS_T syndromes
    if (tid < stride) {
        out1[ct0 * stride + tid] = s0[tid];
        out1[ct1 * stride + tid] = s1[tid];
    }
}

// ------------------------------------------------------------------------
// Host function: synd_f() launcher for multi-2 variant
// ------------------------------------------------------------------------
int synd_f() {
    InitializeC();  // set up constant memory, d_L, gf_inverse_table, etc.

    const int threadsPerBlock = 256;
    int dev, smCount;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));

    // one block per 2 ciphertexts
    int pairs = (KATNUM + 1) / 2;
    dim3 grid(1, pairs);

    // shared memory: 2*sb + 2*(2*SYS_T) elements
    size_t sharedMem = (size_t)(2*sb + 4*SYS_T) * sizeof(gf);

    // allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Multi-2 syndrome kernel (2 cts/block): %f ms\n", ms);

    // copy back & print first CT’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; i++) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}






int main() {

    
    initialisation(secretkeys,ciphertexts,sk,L,g);	
	
	compute_inverses();

	InitializeC(); // only for test purpose
   


	synd_f();
	// Dec();

    return KAT_SUCCESS;
}


