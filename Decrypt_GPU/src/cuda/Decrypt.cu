//************** header files


#include <decrypt.h>




__global__ void computeOut1(gf *d_inverse_elements, unsigned char *d_ciphertexts, gf *out1)
{
    // Shared buffers
    __shared__ gf c[sb];
    __shared__ gf s_out[2 * SYS_T];

    __shared__ uint16_t T[SYS_T + 1];
    __shared__ uint16_t C[SYS_T + 1];
    __shared__ uint16_t B[SYS_T + 1];

    __shared__ gf b, d, f;

    int tid       = threadIdx.x;
    int blockId   = blockIdx.x + blockIdx.y * gridDim.x;
    int globalIdx = blockId * blockDim.x + tid;

    // 1. Unpack ciphertext bits into c[] (coalesced by byte)
    for (int byte = tid; byte < SYND_BYTES; byte += blockDim.x) {
        unsigned char r = d_ciphertexts[byte];
    #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            int idx = byte * 8 + bit;
            if (idx < sb) c[idx] = (r >> bit) & 1U;
        }
    }
    __syncthreads();

    // 2. Compute 2T syndromes with manual ×8 unroll
	if (globalIdx < 2 * SYS_T) {
        const int stride = 2 * SYS_T;          // distance between rows in d_inverse_elements
        const gf *col    = d_inverse_elements + globalIdx; // start of this column
        gf sum = 0;
    #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            // sum ^= mul(col[0], c[bit]);       // col points at element (bit, globalIdx)
            // col += stride;                    // advance to next row (same column)
			gf mask = (gf)(-(int)(c[bit] & 1)); // 0xFFFF.. when 1, 0 when 0
            sum ^= (col[0] & mask);             // add col value iff bit==1
            col += stride;
        }
        s_out[globalIdx] = sum;
    }

    // 3. Initialise Berlekamp–Massey variables
    if (tid <= SYS_T) {
        T[tid] = C[tid] = B[tid] = 0;
        if (tid == 0) {
            C[0] = 1;   // C(x) starts with 1
            B[1] = 1;   // B(x) starts with x
            b    = 1;   // discrepancy baseline
        }
    }
    __syncthreads();

	  for (int N = 0; N < 2 * SYS_T; ++N) {
        int max_j = min(N, SYS_T);

        // 3.a compute discrepancy d via warp+shared reduction ----------------
        gf part = 0;
        // each thread handles strided contributions
        for (int j = tid; j <= max_j; j += blockDim.x) {
            part ^= mul(C[j], s_out[N - j]);
        }
        // warp-level XOR reduction
        for (int offs = 16; offs > 0; offs >>= 1) {
            part ^= __shfl_down_sync(0xFFFFFFFF, part, offs);
        }
        // warp leader writes its partial sum
        if ((tid & 31) == 0) {
            T[tid >> 5] = part;
        }
        // __syncthreads();
        // thread 0 combines warp results
        if (tid == 0) {
            d = 0;
            int numWarps = (max_j / 32) + 1;
            for (int w = 0; w < numWarps; ++w) {
                d ^= (gf)T[w];
            }
        }
        __syncthreads();

        if (tid == 0) {
            f = p_gf_frac(b, d);          // f = d / b
            b = (b & ~mle[N]) | (d & mle[N]); // update discrepancy baseline
        }
        __syncthreads();

        // 3.b merge C and B update into one parallel pass ------------------------
        if (tid <= SYS_T) {
            // backup old values
            gf oldC = C[tid];
            gf oldB = B[tid];
            // update C
            C[tid] = oldC ^ (mul(f, oldB) & mne);
            // update B polynomial coefficient before shift
            B[tid] = (oldB & ~mle[N]) | (oldC & mle[N]);
        }
        __syncthreads();
        
        // 3.c parallel shift of B polynomial ----------------------------------
        __shared__ gf shiftBuf[SYS_T + 1];
        if (tid <= SYS_T) {
            shiftBuf[tid] = (tid == 0) ? 0 : B[tid - 1];
        }
        __syncthreads();
        if (tid <= SYS_T) {
            B[tid] = shiftBuf[tid];
        }
        __syncthreads();
    }
    // 5. Write locator coefficients in reverse order
    if (tid <= SYS_T) {
        out1[tid] = C[SYS_T - tid];
    }
    // Replace your existing BM loop with this optimized version:
   

}

__global__ void chien_search_kernel(
    const gf* __restrict__ d_sigma_all,
    unsigned char* __restrict__ d_error_all
) {
    int cipherIdx = blockIdx.y;                               // which ciphertext
    int posIdx    = blockIdx.x * blockDim.x + threadIdx.x;    // which code position

    if (posIdx >= SYS_N) return;

    // pointer to this block’s σ
    const gf *sigma = d_sigma_all + cipherIdx * (SYS_T + 1);

    // Horner’s method: val = σ(S) where S = L[posIdx]
    gf val = sigma[SYS_T];     // start with highest coeff
    gf a   = d_L[posIdx];      // your support array in const memory
    for (int j = SYS_T - 1; j >= 0; j--) {
        val = mul(val, a) ^ sigma[j];
    }
    // if σ(a)==0 → error at posIdx
    d_error_all[cipherIdx * SYS_N + posIdx] = (val == 0);
}




int synd_f() {
	    // --- sanity-check our lookup tables on the device ---


	int threadsPerBlock = 256;  // Max threads per block based on your device capabilities
    // int totalElements = SYS_N;  // Total number of elements to be processed
    // int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;  // Calculate the necessary number of blocks
    int dev, smCount;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smCount,
                       cudaDevAttrMultiProcessorCount,
                       dev);
    int numBlocks = max((SYS_N + threadsPerBlock - 1)/threadsPerBlock, smCount);  // ensure ≥ SM count
	

    gf *d_images;
	gf *d_out2;
	gf *d_error;
	int *d_e;

    unsigned char *d_error_all;
    cudaMalloc(&d_error_all, KATNUM * SYS_N);
    const int TPB = threadsPerBlock;
    dim3 grid( (SYS_N + TPB - 1)/TPB, KATNUM );

	cudaMalloc(&d_error, SYS_T * sizeof(gf));
    cudaMalloc(&d_images, 3488 * sizeof(gf));
	cudaMalloc(&d_out2, 128 * sizeof(gf));

    cudaMalloc(&d_e, (SYS_N + 7) / 8); // size of e in bytes are   

	

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(d_ciphertexts, ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM, cudaMemcpyHostToDevice);

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("H ----> D time: %f microseconds\n", milliseconds*1000);


    cudaEventRecord(start);
    computeOut1<<<numBlocks, threadsPerBlock>>>(d_inverse_elements,d_ciphertexts, d_images);

    chien_search_kernel<<< grid, TPB >>>(
        d_images,
        d_error_all
    );
    cudaDeviceSynchronize();
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernels execution time: %f microseconds\n", milliseconds*1000);

	

    cudaEventRecord(start);
    cudaMemcpy(h_error, d_error_all,
		KATNUM * SYS_N,
		cudaMemcpyDeviceToHost);


	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("D ---> H time: %f microseconds\n", milliseconds*1000);

	

	
	// cudaMemcpy(images, d_images, SYS_N * sizeof(gf), cudaMemcpyDeviceToHost);
    




    // print h_error like this if (images[i] != 0) printf("%d ", i);
    for (int i = 0; i < KATNUM; i++) {
        printf("Ciphertext %d: ", i);
        for (int j = 0; j < SYS_N; j++) {
            if (h_error[i][j] != 0) {
                printf("%d ", j);
            }
        }
        printf("\n");
    }


    


	cudaFree(d_images);
    cudaFree(d_ciphertexts);
    cudaFree(d_inverse_elements);

    cudaFree(d_L);
    // cudaDestroyTextureObject(tex_inverse_elements);
	cudaFree(d_error_all);
	cudaFree(d_error);
	cudaFree(d_e);
	cudaFree(d_out2);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset(); // Reset the device to clean up resources
	

    
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


