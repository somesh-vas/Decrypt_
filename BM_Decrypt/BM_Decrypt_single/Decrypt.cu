//************** header files

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "common.h"
#include "gf.h"
#include "bm.h"
#include "root.h"
#include <time.h>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<vector>
#include <assert.h>  // For C code
using namespace std;


__constant__ uint16_t mne;
__constant__ uint16_t mle[2 * SYS_T];
__constant__ gf d_L[ SYS_N ];
__constant__ gf        gf_inverse_table[1<<GFBITS];
// for convenience:


unsigned char *d_ciphertexts;
__restrict__ gf *d_inverse_elements;
gf images[SYS_N];
gf error[SYS_T];
int tv; //test_vector
// __constant__ unsigned char d_secretkey[crypto_kem_SECRETKEYBYTES];
unsigned char secretkeys[crypto_kem_SECRETKEYBYTES];
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES]; // size 
int e[SYS_N / 8];
int i,w = 0,j,k;
gf g[ SYS_T+1 ]; // goppa polynomial
gf L[ SYS_N ]; // support
gf s[ SYS_T*2 ];
gf e_inv_LOOP_1D[sb * 2 * SYS_T];
gf inverse_elements[sb][2*SYS_T];
gf temp;
gf e_inv[SYS_N];

unsigned char r[ SYS_N/8 ]; 
// gf out[ SYS_T*2 ]; // random string s
gf locator[ SYS_T+1 ]; // error locator 
// gf images[ SYS_N ]; // 
gf t,c[SYS_N];
clock_t start, end;
double avg_cpu_time_used;
double cpu_printing;
double synd_time = 0, bm_time = 0, root_time = 0;
unsigned char *sk = NULL;
int count;
unsigned char h_error[KATNUM][SYS_N];

#define GF_POLY_MOD ((1<<GFBITS)-1)      // 4095 for GFBITS=12
#define POLY_MOD ((1<<GFBITS)-1)


__device__ __forceinline__ gf add(gf in0, gf in1)
{
	return in0 ^ in1;
}

__device__ __forceinline__ gf mul(gf in0, gf in1) {
    int i;
    uint32_t tmp = 0;
    uint32_t t0 = in0;
    uint32_t t1 = in1;
    uint32_t t;

    // Perform multiplication bit by bit
    tmp = t0 * (t1 & 1);
    for (i = 1; i < GFBITS; i++) {
        tmp ^= (t0 * (t1 & (1 << i)));
    }

  
    t = tmp & 0x7FC000; 
    tmp ^= t >> 9;
    tmp ^= t >> 12;

    t = tmp & 0x3000; // Another example mask for further reduction
    tmp ^= t >> 9;
    tmp ^= t >> 12;
    // print tem & ((1 << GFBITS) - 1) just
  
    return tmp & ((1 << GFBITS) - 1); // Return result masked to GFBITS
}


//----------------------------------------------------------------------------
// Branchless, log/antilog based GF(2^12) multiply
//----------------------------------------------------------------------------


// // remove your existing __device__ mul(…) and insert:

// __device__ __forceinline__ gf mul(gf a, gf b)
// {
//     // 1) detect zero inputs
//     uint32_t za = (a == 0);
//     uint32_t zb = (b == 0);

//     // 2) look up logs (host_log) — host_log[0] = 0 by convention
//     uint32_t la = gf_log_table_padded[a];
//     uint32_t lb = gf_log_table_padded[b];

//     // 3) add & branch-less modulo (FIELD_ORDER = (1<<GFBITS)-1)
//     uint32_t s = la + lb;
//     uint32_t flag = (uint32_t)(- (int32_t)(s >= ((1<<GFBITS)-1)));
//     s = s - (((1<<GFBITS)-1) & flag);

//     // 4) antilog lookup
//     gf  prod = gf_antilog_table[s];

//     // 5) zero if either input was zero
//     uint32_t keep = (~(za | zb)) & 1;
//     return prod * (gf)keep;
// }

// #define mul(a, b) mul_optimized(a, b)


__device__ __forceinline__ gf p_gf_inv(gf in)
{
    return gf_inverse_table[in];
}



__device__ gf p_gf_frac(gf den, gf num)
{
	return mul(p_gf_inv(den), num);
}


__device__ gf d_gf_iszero(gf a)
{
	uint32_t t = a;

	t -= 1;
	t >>= 19;

	return (gf) t;
}



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

void InitializeC() {
    // Define host-side parameters
	uint16_t h_mne = 0xFFFFu;
    uint16_t h_mle[2*SYS_T];
    for(int i=0;i<2*SYS_T;i++)
        h_mle[i] = (i&1)?0:0xFFFFu;
        const int N = 1<<GFBITS;
        // 1) build host inverse table
        gf host_inv[N];
        host_inv[0] = 0;
        for(int i=1;i<N;i++) host_inv[i] = gf_inv((gf)i);
    
        // 2) upload to constant memory
        cudaMemcpyToSymbol(gf_inverse_table, host_inv, sizeof(host_inv));
    
    // Copy to constant memory
    cudaMemcpyToSymbol(mle, h_mle, sizeof(h_mle));
    cudaMemcpyToSymbol(mne, &h_mne, sizeof(h_mne));
    cudaMemcpyToSymbol(d_L, L, SYS_N * sizeof(gf));

    // Allocate and copy ciphertexts (if necessary)
    cudaMalloc(&d_ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM);

    // Setup global memory for inverse elements
    size_t size = sizeof(gf) * sb * 2 * SYS_T;
    cudaMalloc(&d_inverse_elements, size);
    cudaMemcpy(d_inverse_elements, inverse_elements, size, cudaMemcpyHostToDevice);

   
}



int synd_f() {
	    // --- sanity-check our lookup tables on the device ---


	int threadsPerBlock = 256;  // Max threads per block based on your device capabilities
    int totalElements = SYS_N;  // Total number of elements to be processed
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


