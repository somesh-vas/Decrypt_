//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// Vec4-padded + log/antilog-mul syndrome kernel
//   • 256 threads/block, grid = (1, KATNUM)
//   • packs d_ciphertexts into uchar4, unpacks into bits[],
//     then uses mul() for GF-multiply instead of bit-mask
// ------------------------------------------------------------------------
__global__ void computeOut1(
    const gf*  __restrict__ d_inverse_elements,    // [sb][2*SYS_T] per CT
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES × KATNUM
          gf*  __restrict__ out1                    // KATNUM × (2*SYS_T)
) {
    __shared__ gf bits[sb];
    __shared__ gf s_out[2 * SYS_T];

    const int tid   = threadIdx.x;     // 0..255
    const int ctIdx = blockIdx.y;      // which ciphertext [0..KATNUM)
    const int stride = 2 * SYS_T;

    // 1) coalesced vec4 loads + unpack to bits[]
    const int vecCount = SYND_BYTES / 4;
    const uchar4* ct4 = reinterpret_cast<const uchar4*>(
        d_ciphertexts + ctIdx * SYND_BYTES
    );

    for (int v = tid; v < vecCount; v += blockDim.x) {
        uchar4 chunk = ct4[v];
        unsigned char b0 = chunk.x,
                      b1 = chunk.y,
                      b2 = chunk.z,
                      b3 = chunk.w;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx0 = v*4*8 + i;
            int idx1 = v*4*8 + 8 + i;
            int idx2 = v*4*8 + 16 + i;
            int idx3 = v*4*8 + 24 + i;
            if (idx0 < sb) bits[idx0] = (b0 >> i) & 1U;
            if (idx1 < sb) bits[idx1] = (b1 >> i) & 1U;
            if (idx2 < sb) bits[idx2] = (b2 >> i) & 1U;
            if (idx3 < sb) bits[idx3] = (b3 >> i) & 1U;
        }
    }
    __syncthreads();

    // 2) each of the first 2*SYS_T threads computes one syndrome via mul()
    if (tid < stride) {
        const gf* col0 = d_inverse_elements + tid;
        gf sum = 0;
        #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            // multiply col0[ bit*stride ] × bits[bit]
            sum = add(sum, mul(col0[bit * stride], bits[bit]));
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
// synd_f(): host launcher for the vec4+log/antilog-mul variant
// ------------------------------------------------------------------------
int synd_f() {
    // 1) init constants & global memory
    InitializeC();

    // 2) launch config
    const int threadsPerBlock = 256;
    dim3 grid(1, KATNUM);

    // 3) allocate + copy
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
    CUDA_CHECK(cudaMemset(d_images, 0,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) time & launch
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
    printf("Vec4+padded+log/antilog-mul kernel: %f ms\n", ms);

    // 5) copy back & print
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // 6) cleanup
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


