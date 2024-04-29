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



// Device variables
// __constant__ int div8Lookup[SYS_N/8];
// __constant__ int mod8Lookup[SYS_N/8];
// __constant__ unsigned short mne;
__constant__ uint16_t mne;
__constant__ uint16_t mle[2 * SYS_T];
//  gf *d_L;
__constant__ gf d_L[ SYS_N ];

unsigned char *d_ciphertexts;
gf *d_inverse_elements;

gf images[SYS_N];
gf error[SYS_T];

int tv; //test_vector
unsigned char secretkeys[crypto_kem_SECRETKEYBYTES];
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES];
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

__device__ gf add(gf in0, gf in1)
{
	return in0 ^ in1;
}

__device__ gf mul(gf in0, gf in1) {
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

    // Reduce using the irreducible polynomial (example might need adjustment based on actual polynomial)
    t = tmp & 0x7FC000; // Example mask for reduction
    tmp ^= t >> 9;
    tmp ^= t >> 12;

    t = tmp & 0x3000; // Another example mask for further reduction
    tmp ^= t >> 9;
    tmp ^= t >> 12;

    return tmp & ((1 << GFBITS) - 1); // Return result masked to GFBITS
}
static __device__ inline gf p_gf_sq(gf in)
{
	const uint32_t B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};

	uint32_t x = in; 
	uint32_t t;

	x = (x | (x << 8)) & B[3];
	x = (x | (x << 4)) & B[2];
	x = (x | (x << 2)) & B[1];
	x = (x | (x << 1)) & B[0];

	t = x & 0x7FC000;
	x ^= t >> 9;
	x ^= t >> 12;

	t = x & 0x3000;
	x ^= t >> 9;
	x ^= t >> 12;

	return x & ((1 << GFBITS)-1);
}

__device__ gf p_gf_inv(gf in)
{
	gf tmp_11;
	gf tmp_1111;

	gf out = in;

	out = p_gf_sq(out);
	tmp_11 = mul(out, in); // 11

	out = p_gf_sq(tmp_11);
	out = p_gf_sq(out);
	tmp_1111 = mul(out, tmp_11); // 1111

	out = p_gf_sq(tmp_1111);
	out = p_gf_sq(out);
	out = p_gf_sq(out);
	out = p_gf_sq(out);
	out = mul(out, tmp_1111); // 11111111

	out = p_gf_sq(out);
	out = p_gf_sq(out);
	out = mul(out, tmp_11); // 1111111111

	out = p_gf_sq(out);
	out = mul(out, in); // 11111111111

	return p_gf_sq(out); // 111111111110
}

__device__ gf p_gf_frac(gf den, gf num)
{
	return mul(p_gf_inv(den), num);
}

__device__ gf peval(gf *f, gf a)
{
	int i;
	gf r;
	
	r = f[ SYS_T ];

	for (i = SYS_T-1; i >= 0; i--)
	{
		r = mul(r, a);
		// r = gf_add(r, f[i]);
		r ^= f[i];
	}

	return r;
}
__device__ gf d_gf_iszero(gf a)
{
	uint32_t t = a;

	t -= 1;
	t >>= 19;

	return (gf) t;
}



__global__ void computeOut(gf *d_inverse_elements, unsigned char *d_ciphertexts,gf *d_images)
{
    __shared__ gf c[sb];
	__shared__ gf s_out[2 * SYS_T];
	__shared__ uint16_t T[SYS_T + 1];
    __shared__ uint16_t C[SYS_T + 1];
    __shared__ uint16_t B[SYS_T + 1];
	// __shared__ gf shared_coeffs[SYS_T + 1];  // Adjust size as needed based on the degree 
    __shared__ gf b, d, f;

	int tid = threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int globalIdx = blockId * blockDim.x + tid;
	

	if (tid < SYND_BYTES) {
		unsigned char r = d_ciphertexts[tid];
		// Completely unroll the loop
		if (tid * 8 + 0 < sb) c[tid * 8 + 0] = (r >> 0) & 1;
		if (tid * 8 + 1 < sb) c[tid * 8 + 1] = (r >> 1) & 1;
		if (tid * 8 + 2 < sb) c[tid * 8 + 2] = (r >> 2) & 1;
		if (tid * 8 + 3 < sb) c[tid * 8 + 3] = (r >> 3) & 1;
		if (tid * 8 + 4 < sb) c[tid * 8 + 4] = (r >> 4) & 1;
		if (tid * 8 + 5 < sb) c[tid * 8 + 5] = (r >> 5) & 1;
		if (tid * 8 + 6 < sb) c[tid * 8 + 6] = (r >> 6) & 1;
		if (tid * 8 + 7 < sb) c[tid * 8 + 7] = (r >> 7) & 1;
	}
	__syncthreads(); // Ensure 'c' is fully populated before proceeding

    // Compute 'out' elements in parallel
	if (globalIdx < 2 * SYS_T) {
		gf sum = 0;
		int i = 0;
		for (i = 0; i <= sb - 8; i += 8) {
			sum = add(sum, mul(d_inverse_elements[i * 2 * SYS_T + globalIdx], c[i]));
			sum = add(sum, mul(d_inverse_elements[(i + 1) * 2 * SYS_T + globalIdx], c[i + 1]));
			sum = add(sum, mul(d_inverse_elements[(i + 2) * 2 * SYS_T + globalIdx], c[i + 2]));
			sum = add(sum, mul(d_inverse_elements[(i + 3) * 2 * SYS_T + globalIdx], c[i + 3]));
			sum = add(sum, mul(d_inverse_elements[(i + 4) * 2 * SYS_T + globalIdx], c[i + 4]));
			sum = add(sum, mul(d_inverse_elements[(i + 5) * 2 * SYS_T + globalIdx], c[i + 5]));
			sum = add(sum, mul(d_inverse_elements[(i + 6) * 2 * SYS_T + globalIdx], c[i + 6]));
			sum = add(sum, mul(d_inverse_elements[(i + 7) * 2 * SYS_T + globalIdx], c[i + 7]));
		
		// The loop is fully unrolled and sb is a multiple of 8, so no need for extra handling
	
		s_out[globalIdx] = sum;
	}
	   // Initialize Berlekamp-Massey variables
	   if (tid <= SYS_T) {
        T[tid] = C[tid] = B[tid] = 0;
        if (tid == 0) {
            C[0] = 1;  // C(x) starts with 1
            B[1] = 1;  // B(x) starts with x
            b = 1;     // b starts with 1
            
        }
    }   
}
	__syncthreads(); // Ensure 's_out' is fully populated before proceeding
	
	
	// Main loop of the Berlekamp-Massey algorithm executed for 2 * SYS_T iterations
	for (int N = 0; N < 2 * SYS_T; N++) {
		int max_j = min(N, SYS_T);
		if (tid <= max_j) {
			d = 0;
			for (int j = 0; j <= max_j; j++) {
				d ^= mul(C[j], s_out[N - j]);
			}
		}
	
		__syncthreads(); // Necessary for consistent 'd' before proceeding
	
		if (tid == 0) {
			f = p_gf_frac(b, d);  // Compute the fractional update
			b = (b & ~mle[N]) | (d & mle[N]);
		}
	
		__syncthreads(); // Necessary for 'f' and 'b' update
	
		if (tid <= SYS_T) {
			T[tid] = C[tid]; // Backup 'C'
			C[tid] ^= mul(f, B[tid]) & mne; // Update 'C'
		}

	
		if (tid <= SYS_T) {
			B[tid] = (B[tid] & ~mle[N]) | (T[tid] & mle[N]); // Shift 'B'
		}
	
		if (tid == SYS_T) {
			for (int i = SYS_T; i > 0; i--) {
				B[i] = B[i - 1];
			}
			B[0] = 0;
		}

	}
	__syncthreads(); // Ensure 'C' is fully updated before proceeding
   // copy C to out1
   if (tid <= SYS_T) {
		d_images[tid] = C[SYS_T - tid];
		
	}
	
	
}

__global__ void eval_root(gf *d_images) {
    int tid = threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;  // This is useful if you need unique block identification within a 2D grid.
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;  // Standard calculation for a 1D block arrangement within a 1D or 2D grid.
	__shared__ gf shared_coeffs[SYS_T + 1];
	__shared__ int error_positions[SYS_T];  // Buffer to store positions of non-zero errors up to SYS_T
	__shared__ int e[SYS_N/8];

	if (tid <= SYS_T) {
		shared_coeffs[tid] = d_images[tid];
		
	}
	__syncthreads();

    if (globalIdx < SYS_N) {
        gf point = d_L[globalIdx];  // Each thread uniquely accesses an element of d_L based on globalIdx.
        gf result = 0;
        gf power = 1;
		gf term;
		// uint32_t t;
		// int r;
        for (int i = 0; i <= SYS_T; i++) {
            term = mul(shared_coeffs[i], power);
            result ^= term;
            power = mul(power, point);
        }
		d_images[globalIdx] = d_gf_iszero(result) & 1;
    }
    __syncthreads();  // Synchronize to ensure all computations are done before exiting the kernel.
	
	
}


int synd_f() {
	
	int threadsPerBlock = 1024;  // Max threads per block based on your device capabilities
    int totalElements = SYS_N;  // Total number of elements to be processed
    int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;  // Calculate the necessary number of blocks

	

    gf *d_images;
	gf *d_out2;
	gf *d_error;
	int *d_e;
	cudaMalloc(&d_error, SYS_T * sizeof(gf));
    cudaMalloc(&d_images, 3488 * sizeof(gf));
	cudaMalloc(&d_out2, 128 * sizeof(gf));
	cudaMalloc(&d_e, (SYS_N + 7) / 8);
	cudaEvent_t start, stop;
    float milliseconds = 0;
    float microseconds = 0;



    cudaMemcpy(d_ciphertexts, ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM, cudaMemcpyHostToDevice);
	
    computeOut<<<numBlocks, threadsPerBlock>>>(d_inverse_elements, d_ciphertexts, d_images);	
	eval_root<<<numBlocks, threadsPerBlock>>>(d_images);
    cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaMemcpy(images, d_images, SYS_N * sizeof(gf), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
   microseconds = milliseconds * 1000;  // Convert milliseconds to microseconds
    printf("Time for Host to Device transfer: %f us\n", microseconds);

    cudaEventDestroy(stop);

	for (i = 0; i < SYS_N; i++) {
		 if (images[i] != 0) printf("%d ", i);
	}
	printf("\n");
	cudaFree(d_images);
    cudaFree(d_ciphertexts);
    cudaFree(d_inverse_elements);
    // cudaFree(d_out1);
    cudaFree(d_L);
    return 0;
}


void InitializeC() {

	uint16_t h_mne = 65535;
	uint16_t h_mle[2 * SYS_T] = {65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0,65535,0
	};
	


	cudaMemcpyToSymbol(mle, h_mle, sizeof(h_mle));
	cudaMemcpyToSymbol(mne, &h_mne, sizeof(h_mne));
	// cudaMalloc(&d_L, SYS_N * sizeof(gf));
	// cudaMemcpy(d_L, L, SYS_N * sizeof(gf), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_L, L, SYS_N * sizeof(gf));
	cudaMalloc(&d_inverse_elements, sizeof(gf) * sb * 2 * SYS_T);
    cudaMemcpy(d_inverse_elements, inverse_elements, sizeof(gf) * sb * 2 * SYS_T, cudaMemcpyHostToDevice);
	cudaMalloc(&d_ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM);
	


	
}

int main() {

    
    initialisation(secretkeys,ciphertexts,sk,L,g);	
	
	compute_inverses();

	InitializeC(); // only for test purpose



	synd_f();
	// Dec();

    return KAT_SUCCESS;
}



