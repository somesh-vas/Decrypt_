//************** header files


#include <decrypt.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

typedef uchar4 C4;

__global__ void computeSyndromesKernel(
    const C4*  __restrict__ d_ct4,
    const gf*  __restrict__ d_inverse_elements,
          gf*  __restrict__ d_syndromes
) {
    extern __shared__ gf shared[];      // sb + 2*SYS_T elements
    gf* c     = shared;                 // [0 .. sb-1]
    int   tid = threadIdx.x;
    int   ct  = blockIdx.y;
    int   wordsPerCt = (SYND_BYTES + 3)/4;
    size_t base     = size_t(ct)*wordsPerCt;

    // 1) unpack into c[]
    for (int i = tid; i < sb; i += blockDim.x) c[i] = 0;
    __syncthreads();
    if (tid < wordsPerCt) {
        C4 v = d_ct4[base + tid];
        int bitBase = tid*32;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            unsigned char byte = *(&v.x + b);
            #pragma unroll
            for (int bit = 0; bit < 8; ++bit) {
                int idx = bitBase + b*8 + bit;
                if (idx < sb) c[idx] = (byte >> bit)&1U;
            }
        }
    }
    __syncthreads();

    // 2) compute each syndrome
    if (tid < 2*SYS_T) {
        const int stride = 2*SYS_T;
        const gf* col    = d_inverse_elements + tid;
        gf sum = 0;
        for (int i = 0; i < sb; i++) {
            gf mask = gf(-int(c[i]&1));
            sum ^= (col[0] & mask);
            col += stride;
        }
        d_syndromes[ct*(2*SYS_T) + tid] = sum;
    }
}

// ─── Kernel #2: run Berlekamp–Massey on each syndrome vector ─────────
__global__ void berlekampMasseyKernel(
    const gf* __restrict__ d_syndromes,
          gf* __restrict__ d_locator    // size = KATNUM*(SYS_T+1)
) {
    __shared__ gf C[SYS_T+1], B[SYS_T+1], Tsw[(SYS_T+31)/32];
    __shared__ gf shiftBuf[SYS_T+1];
    __shared__ gf b, d, f;

    int tid    = threadIdx.x;
    int ct     = blockIdx.y;
    // int lane   = tid;
    int nWarp  = (SYS_T+31)/32;

    // 1) init
    if (tid <= SYS_T) {
        C[tid] = (tid==0);
        B[tid] = (tid==1);
    }
    if (tid == 0) {
        b = 1;
        for (int w = 0; w < nWarp; w++) Tsw[w] = 0;
    }
    __syncthreads();

    // 2) BM main loop
    for (int N = 0; N < 2*SYS_T; N++) {
        int max_j = min(N, SYS_T);

        // 2.a) discrepancy d
        gf part = 0;
        for (int j = tid; j <= max_j; j += blockDim.x) {
            part ^= mul(C[j],
              d_syndromes[ct*(2*SYS_T) + (N - j)]);
        }
        for (int off = 16; off > 0; off >>= 1)
            part ^= __shfl_down_sync(0xFFFFFFFFu, part, off);
        if ((tid & 31) == 0) Tsw[tid>>5] = part;
        __syncthreads();

        if (tid == 0) {
            d = 0;
            int nW = (max_j/32) + 1;
            for (int w = 0; w < nW; w++)
                d ^= Tsw[w];
            // compute f = d/b and update b
            f = p_gf_frac(b, d);
            b = (b & ~mle[N]) | (d & mle[N]);
        }
        __syncthreads();

        // 2.b) update C and B
        if (tid <= SYS_T) {
            gf oldC = C[tid], oldB = B[tid];
            C[tid] = oldC ^ ( mul(f, oldB) & mne );
            B[tid] = (oldB & ~mle[N]) | (oldC & mle[N]);
        }
        __syncthreads();

        // 2.c) shift B right by 1
        if (tid <= SYS_T)
            shiftBuf[tid] = (tid ? B[tid-1] : 0);
        __syncthreads();
        if (tid <= SYS_T)
            B[tid] = shiftBuf[tid];
        __syncthreads();
    }

    // 3) write locator C[SYS_T..0]
    if (tid <= SYS_T) {
        d_locator[ ct*(SYS_T+1) + tid ]
          = C[SYS_T - tid];
    }
}

// ─── Host launcher: split version ────────────────────────────────────
void decrypt_mass_separate() {
    const int SYNS = 2*SYS_T;
    const int LOCS = SYS_T+1;
    size_t ctWords = (SYND_BYTES+3)/4;
    size_t ctBytes = size_t(KATNUM)*ctWords*sizeof(C4);
    size_t synBytes= size_t(KATNUM)*SYNS *sizeof(gf);
    size_t locBytes= size_t(KATNUM)*LOCS *sizeof(gf);

    // 1) pin & upload ciphertexts as C4
    C4 *h_ct4 = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_ct4, ctBytes));
    memcpy(h_ct4, ciphertexts,
           size_t(KATNUM)*crypto_kem_CIPHERTEXTBYTES);
    C4 *d_ct4 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ct4, ctBytes));
    CHECK_CUDA(cudaMemcpy(d_ct4, h_ct4, ctBytes,
                          cudaMemcpyHostToDevice));

    // 2) allocate device buffers
    gf *d_synd = nullptr, *d_loc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_synd, synBytes));
    CHECK_CUDA(cudaMalloc(&d_loc,  locBytes));

    // 3) launch syndrome kernel
    dim3 gridSyn(1, KATNUM);
    int threadsSyn = 256;
    int sharedSyn  = (sb + 2*SYS_T)*sizeof(gf);
    computeSyndromesKernel<<<gridSyn,threadsSyn,sharedSyn>>>(
         d_ct4,
         d_inverse_elements,
         d_synd
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4) launch BM kernel
    dim3 gridBM(1, KATNUM);
    int threadsBM = max(SYS_T+1, 32);
    berlekampMasseyKernel<<<gridBM,threadsBM>>>(
         d_synd,
         d_loc
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5) copy back & print locator only
    gf *h_loc = (gf*)malloc(locBytes);
    CHECK_CUDA(cudaMemcpy(h_loc, d_loc, locBytes,
                          cudaMemcpyDeviceToHost));
    // for (int t = 0; t < KATNUM; t++) {
    //     for (int i = 0; i < LOCS; i++)
    //         printf("%04x ", h_loc[t*LOCS + i]);
    //     printf("\n");
    // }

    // 6) cleanup
    free(h_loc);
    CHECK_CUDA(cudaFreeHost(h_ct4));
    CHECK_CUDA(cudaFree(d_ct4));
    CHECK_CUDA(cudaFree(d_synd));
    CHECK_CUDA(cudaFree(d_loc));
}


///

int main() {

    
    initialisation(secretkeys,ciphertexts,sk,L,g);	
	
	compute_inverses();

	InitializeC(); // only for test purpose
    decrypt_mass_separate(); // run the separate syndrome + BM kernels
    // decrypt_mass_warpTiny(); // run the new warpTiny kernel
    // decrypt_mass_with_BM(); // run the new BM kernel
    // decrypt_mass_warpTiny_and_BM(); // run the new warpTiny+BM kernel
    // decrypt_mass_with_BM(); // run the new BM kernel
    // decrypt_mass_separate(); // run the separate syndrome + BM kernels
    return KAT_SUCCESS;
}


