//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

#include <mma.h>
using namespace nvcuda::wmma;


// ------------------------------------------------------------------------
// Block‐reduction syndrome kernel:
//   • one block per (coeff, ciphertext) pair: grid = {2*SYS_T, KATNUM}
//   • 256 threads per block
//   • each thread processes a strided subset of bits, XORs partial sums
//   • warp‐wide __shfl_down_sync reduction, then block‐wide shared‐mem reduction
// ------------------------------------------------------------------------
__global__ void computeOut1(
    const gf*           __restrict__ d_inverse_elements, // [bit][coeff]
    const unsigned char* __restrict__ d_ciphertexts,     // SYND_BYTES per CT
          gf*           __restrict__ out1                // KATNUM×(2*SYS_T)
) {
    extern __shared__ gf warp_sums[];  // one entry per warp (256/32 = 8 warps)

    const int cf      = blockIdx.x;    // coefficient index [0..2*SYS_T)
    const int ct      = blockIdx.y;    // ciphertext index [0..KATNUM)
    const int lane    = threadIdx.x;   // 0..255
    const int warpId  = lane >> 5;     // 0..7
    const int laneInW = lane & 31;     // 0..31
    const int stride  = 2 * SYS_T;

    // 1) each thread XOR-accumulates its share of sb bits
    gf sum = 0;
    for (int bit = lane; bit < sb; bit += blockDim.x) {
        unsigned char r = d_ciphertexts[ ct * SYND_BYTES + (bit >> 3) ];
        if ((r >> (bit & 7)) & 1U) {
            sum ^= d_inverse_elements[ bit * stride + cf ];
        }
    }

    // 2) warp‐local reduction via shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum ^= __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    // lane 0 of each warp writes its partial into shared memory
    if (laneInW == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // 3) block‐wide reduction of the 8 warp sums (only warp 0 participates)
    if (warpId == 0) {
        if (laneInW < (blockDim.x / 32)) {
            sum = warp_sums[laneInW];
            // accumulate the other warp sums
            for (int w = laneInW + 1; w < (blockDim.x/32); ++w) {
                sum ^= warp_sums[w];
            }
            // lane 0 writes the final syndrome
            if (laneInW == 0) {
                out1[ ct * stride + cf ] = sum;
            }
        }
    }
}


// ------------------------------------------------------------------------
// Host function: synd_f_blockred()
// ------------------------------------------------------------------------
int synd_f() {
    // 1) initialize constants & global memory
    InitializeC();

    // 2) launch config
    const int threadsPerBlock = 256;
    dim3 grid(2 * SYS_T, KATNUM);  // one block per (coeff, CT)
    // shared mem = one gf per warp
    size_t sharedMem = (threadsPerBlock / 32) * sizeof(gf);

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
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
    printf("Block‐reduction syndrome kernel (256 tpb): %f ms\n", ms);

    // 6) copy back & print first ciphertext’s syndromes
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


