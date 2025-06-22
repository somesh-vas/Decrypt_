//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

#include <mma.h>
using namespace nvcuda::wmma;


// ------------------------------------------------------------------------
// Thread-per-ciphertext syndrome kernel:
//   • each CUDA thread handles one ciphertext end-to-end
//   • blockDim = 64, gridDim = ceil(KATNUM/64)
//   • no shared memory, no atomics—each thread writes its own 2*SYS_T outputs
// ------------------------------------------------------------------------
__global__ void computeOut1(
    const gf*  __restrict__ d_inverse_elements,  // [bit][coeff]
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES per CT
          gf*  __restrict__ out1                  // KATNUM × (2*SYS_T)
) {
    int ctIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ctIdx >= KATNUM) return;

    // accumulator registers
    gf reg[2 * SYS_T];
    #pragma unroll
    for (int j = 0; j < 2 * SYS_T; ++j) {
        reg[j] = 0;
    }

    // scan each bit of ciphertext
    for (int byte = 0; byte < SYND_BYTES; ++byte) {
        unsigned char r = d_ciphertexts[ctIdx * SYND_BYTES + byte];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if ((r >> b) & 1U) {
                int bitIdx = byte * 8 + b;
                if (bitIdx < sb) {
                    const gf* col = d_inverse_elements + bitIdx * (2 * SYS_T);
                    #pragma unroll
                    for (int j = 0; j < 2 * SYS_T; ++j) {
                        reg[j] ^= col[j];
                    }
                }
            }
        }
    }

    // write back full syndrome vector for this ciphertext
    int base = ctIdx * (2 * SYS_T);
    #pragma unroll
    for (int j = 0; j < 2 * SYS_T; ++j) {
        out1[base + j] = reg[j];
    }
}


// ------------------------------------------------------------------------
// synd_f(): host launcher for Thread-per-ciphertext variant
// ------------------------------------------------------------------------
int synd_f() {
    InitializeC();  // upload constants, allocate d_ciphertexts & d_inverse_elements

    const int threadsPerBlock = 64;
    int blocks = (KATNUM + threadsPerBlock - 1) / threadsPerBlock;

    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1<<<blocks, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Thread-per-ciphertext syndrome kernel (64 tpb): %f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

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


