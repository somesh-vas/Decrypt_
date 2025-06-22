//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

#include <mma.h>
using namespace nvcuda::wmma;


// ------------------------------------------------------------------------
// “u64-padded” syndrome kernel
//  • 1 block per ciphertext (grid = {1, KATNUM})
//  • 256 threads/block
//  • uses coalesced uint64_t loads to unpack 64 bits at a time
// ------------------------------------------------------------------------
__global__ void computeOut1_u64(
    const gf*  __restrict__ d_inverse_elements,  // [sb][2*SYS_T] per CT
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES per CT
          gf*  __restrict__ out1                  // KATNUM × (2*SYS_T)
) {
    __shared__ gf c[sb];
    __shared__ gf s_out[2 * SYS_T];

    const int tid   = threadIdx.x;      // 0..255
    const int ctIdx = blockIdx.y;       // which ciphertext
    const int stride = 2 * SYS_T;
    const int bytes  = SYND_BYTES;
    const int u64Count = (bytes + 7) / 8;
    // pointer to this CT’s ciphertext as 64-bit words
    const uint64_t* ct64 = reinterpret_cast<const uint64_t*>(
        d_ciphertexts + ctIdx * bytes
    );

    // 1) unpack bits into shared c[] via 64-bit loads
    for (int v = tid; v < u64Count; v += blockDim.x) {
        uint64_t chunk = ct64[v];
        #pragma unroll
        for (int b = 0; b < 64; ++b) {
            int idx = v * 64 + b;
            if (idx < sb) {
                c[idx] = (chunk >> b) & 1U;
            }
        }
    }
    __syncthreads();

    // 2) each of the first 2*SYS_T threads computes one syndrome
    if (tid < stride) {
        const gf* col0 = d_inverse_elements + tid;
        gf sum = 0;
    #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            // mask=0xFFFF if c[bit]==1 else 0
            gf mask = (gf)(- (int)c[bit]);
            sum ^= (col0[bit * stride] & mask);
        }
        s_out[tid] = sum;
    }
    __syncthreads();

    // 3) write back
    if (tid < stride) {
        out1[ ctIdx * stride + tid ] = s_out[tid];
    }
}


// ------------------------------------------------------------------------
// Host function: synd_f_u64()
// ------------------------------------------------------------------------
int synd_f() {
    // 1) initialize constant memory & globals
    InitializeC();

    // 2) launch parameters
    const int threadsPerBlock = 256;
    dim3 grid(1, KATNUM);    // one block per ciphertext

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // zero output buffer
    CUDA_CHECK(cudaMemset(d_images, 0,
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
    computeOut1_u64<<<grid, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("u64-padded syndrome kernel: %f ms\n", ms);

    // 6) copy back & print first CT’s syndromes
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


