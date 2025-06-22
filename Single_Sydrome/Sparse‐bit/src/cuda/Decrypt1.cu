//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

#include <mma.h>
using namespace nvcuda::wmma;


// ------------------------------------------------------------------------
// “Sparse‐bit” grid‐stride kernel:
//   • one thread per (ct,cf) pair in a 1D grid
//   • each thread walks only the set bits of its ciphertext, using __ffs
//   • uses __ldg() for read‐only caching of inverse elements
// ------------------------------------------------------------------------
__global__ void computeOut1_sparse(
    const gf*  __restrict__ d_inverse_elements,  // layout: bit-major [bit*stride + cf]
    const unsigned char* __restrict__ d_ciphertexts,
          gf*  __restrict__ out1                  // length = KATNUM * (2*SYS_T)
) {
    const int stride     = 2 * SYS_T;
    const int SYNC_BYTES = SYND_BYTES;          // (sb+7)/8
    const int totalItems = KATNUM * stride;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    while (idx < totalItems) {
        int ctIdx = idx / stride;    // ciphertext index
        int cf    = idx % stride;    // syndrome coefficient

        // walk set bits only
        gf sum = 0;
        const unsigned char* ctBase = d_ciphertexts + ctIdx * SYNC_BYTES;
        for (int word = 0; word < (sb + 31)/32; ++word) {
            // load 32 bits of the ciphertext
            uint32_t bits = *(const uint32_t*)(ctBase + word*4);
            // for each set bit in 'bits', XOR the corresponding inverse element
            while (bits) {
                int b = __ffs(bits) - 1; 
                bits &= bits - 1;
                int bitIdx = word*32 + b;
                if (bitIdx < sb) {
                    sum ^= __ldg(&d_inverse_elements[ bitIdx*stride + cf ]);
                }
            }
        }

        out1[idx] = sum;
        idx += step;
    }
}


// ------------------------------------------------------------------------
// Host launcher: synd_f_sparse()
// ------------------------------------------------------------------------
int synd_f() {
    InitializeC();  // load constants, allocate & copy globals

    // 1D grid‐stride launch
    const int threadsPerBlock = 256;
    int totalItems = KATNUM * 2 * SYS_T;
    int blocks     = (totalItems + threadsPerBlock - 1) / threadsPerBlock;

    // device buffers
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
        SYND_BYTES * KATNUM,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf),
        cudaMemcpyHostToDevice));

    // time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1_sparse<<<blocks, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Sparse‐bit grid‐stride kernel: %f ms\n", ms);

    // copy back & print first ciphertext’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf),
        cudaMemcpyDeviceToHost));
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


