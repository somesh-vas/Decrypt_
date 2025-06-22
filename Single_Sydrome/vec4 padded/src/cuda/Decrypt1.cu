//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1

// Vec4-padded baseline version of the syndrome kernel
// Kernel name and parameter list unchanged
__global__ void computeOut1(gf *d_inverse_elements,
                            unsigned char *d_ciphertexts,
                            gf *out1)
{
    // Shared buffers
    __shared__ gf c[sb];
    __shared__ gf s_out[2 * SYS_T];

    int tid       = threadIdx.x;
    int blockId   = blockIdx.x + blockIdx.y * gridDim.x;
    int globalIdx = blockId * blockDim.x + tid;

    // 1. Unpack ciphertext bits into c[] using coalesced uchar4 (vec4) loads
    //    We assume SYND_BYTES is padded to a multiple of 4 bytes.
    const int vecCount = SYND_BYTES / 4;
    for (int v = tid; v < vecCount; v += blockDim.x) {
        // Read 4 bytes at once
        uchar4 chunk = reinterpret_cast<uchar4*>(d_ciphertexts)[v];

        // Unpack each of the 4 bytes into bits
        unsigned char r0 = chunk.x;
        unsigned char r1 = chunk.y;
        unsigned char r2 = chunk.z;
        unsigned char r3 = chunk.w;

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            int idx0 = (v * 4 + 0) * 8 + b;
            if (idx0 < sb) c[idx0] = (r0 >> b) & 1U;

            int idx1 = (v * 4 + 1) * 8 + b;
            if (idx1 < sb) c[idx1] = (r1 >> b) & 1U;

            int idx2 = (v * 4 + 2) * 8 + b;
            if (idx2 < sb) c[idx2] = (r2 >> b) & 1U;

            int idx3 = (v * 4 + 3) * 8 + b;
            if (idx3 < sb) c[idx3] = (r3 >> b) & 1U;
        }
    }
    __syncthreads();

    // 2. Compute 2T syndromes (exactly as in the baseline)
    if (globalIdx < 2 * SYS_T) {
        const int stride = 2 * SYS_T;
        const gf *col    = d_inverse_elements + globalIdx;
        gf sum = 0;

        #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            gf mask = (gf)(- (int)(c[bit] & 1));  // all-ones if bit is 1, else 0
            sum ^= (col[0] & mask);               // conditional XOR
            col += stride;
        }
        s_out[globalIdx] = sum;
    }
    __syncthreads();

    // 3. Write back to global memory
    if (globalIdx < 2 * SYS_T) {
        out1[globalIdx] = s_out[globalIdx];
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


