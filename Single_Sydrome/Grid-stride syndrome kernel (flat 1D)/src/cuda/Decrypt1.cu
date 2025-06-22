//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

#include <mma.h>
using namespace nvcuda::wmma;


// ------------------------------------------------------------------------
// Grid-stride syndrome kernel (flat 1D): no shared mem or atomics
//  • gridDim.x = ceil(KATNUM*2*SYS_T / threadsPerBlock)
//  • each thread handles multiple (ct,cf) pairs in a 1D loop
//  • uses __ldg for read-only cache on inverse elements
// ------------------------------------------------------------------------
__global__ void computeOut1_flat(
    const gf*  __restrict__ d_inverse_elements,  // layout: bit-major [bit*stride + cf]
    const unsigned char* __restrict__ d_ciphertexts,
          gf*  __restrict__ out1
) {
    const int threadsPerBlock = blockDim.x;
    const int stride         = 2 * SYS_T;
    const int SYNC_BYTES     = SYND_BYTES;       // = (sb+7)/8
    const int totalItems     = KATNUM * stride;

    // Global 1D index across all (ciphertext, coefficient) pairs
    int idx = blockIdx.x * threadsPerBlock + threadIdx.x;
    int gridSize = gridDim.x * threadsPerBlock;

    while (idx < totalItems) {
        int ctIdx = idx / stride;    // which ciphertext
        int cf    = idx % stride;    // which syndrome coefficient

        // Compute the XOR-sum over all sb bits
        gf sum = 0;
        int baseCipher = ctIdx * SYNC_BYTES;
        const gf* invBase = d_inverse_elements + cf;  // we'll step by stride

        for (int bit = 0; bit < sb; ++bit) {
            unsigned char byte = d_ciphertexts[ baseCipher + (bit >> 3) ];
            if ((byte >> (bit & 7)) & 1U) {
                // use __ldg to hit the read-only cache
                sum ^= __ldg(&invBase[ bit * stride ]);
            }
        }

        // Write result
        out1[idx] = sum;

        idx += gridSize;
    }
}


// ------------------------------------------------------------------------
// Host function: synd_f_flat()
// ------------------------------------------------------------------------
int synd_f() {
    InitializeC();  // constants & global memory set up

    // 1D launch parameters
    const int threadsPerBlock = 256;
    int totalItems = KATNUM * 2 * SYS_T;
    int blocks = (totalItems + threadsPerBlock - 1) / threadsPerBlock;

    // allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1_flat<<<blocks, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms=0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Grid-stride syndrome kernel (1D, %d blocks): %f ms\n", blocks, ms);

    // copy back & print first ciphertext’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
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


