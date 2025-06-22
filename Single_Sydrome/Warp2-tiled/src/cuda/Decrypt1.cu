//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// tiled (shared-padding) syndrome kernel
//   • 1 block per ciphertext (gridDim = {1, KATNUM})
//   • 256 threads per block
//   • shared memory: sb + 2*SYS_T elements of gf
// ------------------------------------------------------------------------
__global__ void computeOut1(
    const gf*  __restrict__ d_inverse_elements,  // [sb][2*SYS_T]
    const unsigned char* __restrict__ d_ciphertexts, // compact bytes, length = SYND_BYTES
          gf*  __restrict__ out1                  // output syndromes: KATNUM × (2*SYS_T)
) {
    extern __shared__ gf shared[];
    gf* c     = shared;                 // [0 .. sb-1]
    gf* s_out = shared + sb;            // [sb .. sb + 2*SYS_T - 1]

    int tid    = threadIdx.x;
    int ctIdx  = blockIdx.y;            // which ciphertext

    // 1) unpack bits into c[]
    //    SYND_BYTES = (sb + 7)/8
    for (int byte = tid; byte < SYND_BYTES; byte += blockDim.x) {
        unsigned char r = d_ciphertexts[ ctIdx * SYND_BYTES + byte ];
    #pragma unroll
        for (int b = 0; b < 8; ++b) {
            int idx = byte * 8 + b;
            if (idx < sb) c[idx] = (r >> b) & 1;
        }
    }
    __syncthreads();

    // 2) each of the first 2*SYS_T threads computes one syndrome coefficient
    if (tid < 2 * SYS_T) {
        const int stride = 2 * SYS_T;
        const gf* col0   = d_inverse_elements + tid; 
        gf sum = 0;

    #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            // mask = 0xFFFF if c[bit]==1, else 0
            gf mask = (gf)(- (int)c[bit]);
            sum ^= (col0[bit * stride] & mask);
        }
        s_out[tid] = sum;
    }
    __syncthreads();

    // 3) write back
    if (tid < 2 * SYS_T) {
        out1[ ctIdx * (2*SYS_T) + tid ] = s_out[tid];
    }
}


// ------------------------------------------------------------------------
// synd_f(): host launcher for the tiled kernel above
// ------------------------------------------------------------------------
int synd_f() {
    // 1) set up constant/device globals
    InitializeC();

    // 2) threads/blocks
    const int threadsPerBlock = 256;
    dim3    grid(1, KATNUM);   // one block per ciphertext

    // 3) shared‐mem size = (sb + 2*SYS_T) × sizeof(gf)
    size_t sharedMem = (size_t)(sb + 2*SYS_T) * sizeof(gf);

    // 4) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;

    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 5) copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 6) time & launch
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
    printf("Tiled (shared-padding) syndrome kernel: %f ms\n", ms);

    // 7) copy back & print first ciphertext’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; i++) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // 8) cleanup
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


