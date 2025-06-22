//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1
// tile shared version of the syndrome kernel 
// Kernel name and parameter list unchanged
__global__ void computeOut1(gf *d_inverse_elements,
                            unsigned char *d_ciphertexts,
                            gf *out1)
{
    // Shared buffers for one ciphertext
    __shared__ gf c[sb];
    __shared__ gf s_out[2 * SYS_T];

    int tid     = threadIdx.x;
    // one block per ciphertext
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    // 1) Unpack bits into c[] (coalesced by byte)
    for (int byte = tid; byte < SYND_BYTES; byte += blockDim.x) {
        unsigned char r = d_ciphertexts[byte];
    #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            int idx = byte * 8 + bit;
            if (idx < sb) c[idx] = (r >> bit) & 1U;
        }
    }
    __syncthreads();

    // 2) Compute syndromes using 4 warps per CT (each warp = 32 threads)
    int warpId = tid >> 5; // which warp within the block
    int laneId = tid & 31; // lane within the warp

    if (warpId < 4) {
        int coeff = warpId * 32 + laneId;  // coefficient index [0..127)
        if (coeff < 2 * SYS_T) {
            const int stride = 2 * SYS_T;
            // point at the column for this (ciphertext, coeff)
            const gf *col = d_inverse_elements + blockId * stride + coeff;
            gf sum = 0;
            // each thread accumulates over all sb bits
            for (int b = 0; b < sb; ++b) {
                gf mask = (gf)(- (int)(c[b] & 1));  // 0xFFFF if bit==1, else 0
                sum ^= (col[0] & mask);
                col  += stride;
            }
            s_out[coeff] = sum;
        }
    }
    __syncthreads();

    // 3) Write back all 2*SYS_T results
    int outIdx = blockId * (2 * SYS_T) + tid;
    if (tid < 2 * SYS_T) {
        out1[outIdx] = s_out[tid];
    }
    __syncthreads();
}


int synd_f() {
    int threadsPerBlock = 256;
    int dev, smCount;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);
    int numBlocks = max((SYS_N + threadsPerBlock - 1) / threadsPerBlock, smCount);

    gf *d_images;
    gf *d_out2;
    gf *d_error;
    int *d_e;
    unsigned char *d_error_all;

    cudaMalloc(&d_error_all, KATNUM * SYS_N);
    const int TPB = threadsPerBlock;
    dim3 grid((SYS_N + TPB - 1) / TPB, KATNUM);

    cudaMalloc(&d_error, SYS_T * sizeof(gf));
    cudaMalloc(&d_images, 3488 * sizeof(gf));
    cudaMalloc(&d_out2, 128 * sizeof(gf));
    cudaMalloc(&d_e, (SYS_N + 7) / 8);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(d_ciphertexts, ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("H ----> D time: %f microseconds\n", milliseconds * 1000);

    cudaEventRecord(start);
    computeOut1<<<numBlocks, threadsPerBlock>>>(d_inverse_elements, d_ciphertexts, d_images);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernels execution time: %f microseconds\n", milliseconds * 1000);

    cudaEventRecord(start);
    cudaMemcpy(h_out1, d_images, 2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 2 * SYS_T; i++) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("D ---> H time: %f microseconds\n", milliseconds * 1000);

    cudaFree(d_images);
    cudaFree(d_ciphertexts);
    cudaFree(d_inverse_elements);
    cudaFree(d_L);
    cudaFree(d_error_all);
    cudaFree(d_error);
    cudaFree(d_e);
    cudaFree(d_out2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();

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


