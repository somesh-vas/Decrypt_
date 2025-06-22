//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

#include <mma.h>
using namespace nvcuda::wmma;


// ------------------------------------------------------------------------
// Warp‐shuffle syndrome kernel (32 threads = 1 warp per coefficient)
//  • grid = {2*SYS_T, KATNUM}, blockDim = 32
//  • each warp computes one syndrome coefficient cf ∈ [0, 2*SYS_T)
//  • bit‐vector of length sb is reduced via lane‐strided loads + __shfl_xor_sync
// ------------------------------------------------------------------------
__global__ void computeOut1(
    const gf* __restrict__ d_inverse_elements, 
    const unsigned char* __restrict__ d_ciphertexts,
          gf* __restrict__ out1
) {
    const int cf    = blockIdx.x;      // which syndrome coefficient
    const int ct   = blockIdx.y;      // which ciphertext index
    const int lane = threadIdx.x;      // 0..31
    const int stride = 2 * SYS_T;

    // accumulate partial XOR of those bits where c[bit]==1
    gf sum = 0;
    for (int bit = lane; bit < sb; bit += 32) {
        unsigned char r = d_ciphertexts[ ct * SYND_BYTES + (bit >> 3) ];
        if ((r >> (bit & 7)) & 1U) {
            // original column‐major: inv[bit][cf] at d_inverse_elements[ bit*stride + cf ]
            sum ^= d_inverse_elements[ bit * stride + cf ];
        }
    }

    // warp‐wide XOR reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum ^= __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // lane 0 writes the final result
    if (lane == 0) {
        out1[ ct * stride + cf ] = sum;
    }
}


// ------------------------------------------------------------------------
// Host launcher: synd_f_warpshuffle()
// ------------------------------------------------------------------------
int synd_f() {
    // 1) initialize constants & global memory
    InitializeC();

    // 2) launch configuration
    const int threadsPerBlock = 32;  // one warp
    dim3 grid(2 * SYS_T, KATNUM);

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1<<<grid, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Warp‐shuffle syndrome kernel (32 tpb): %f ms\n", ms);

    // 6) copy back & print first ciphertext’s 2*SYS_T syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // 7) cleanup
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


