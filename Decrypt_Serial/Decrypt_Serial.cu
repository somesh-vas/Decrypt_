#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//********************************************************************************
//*************************** CUDA KERNELS **************************************
__constant__ int div8Lookup[SYS_N];
__constant__ int mod8Lookup[SYS_N];
//***************************** device varialbles ********************************
gf *d_L;
gf (*d_e_inv_LOOP_device)[SYS_N][2*SYS_T];
gf (*d_c)[SYS_N];
unsigned char (*d_ciphertexts)[crypto_kem_CIPHERTEXTBYTES];
gf (*d_out)[2*SYS_T];
gf (*d_locator)[SYS_T + 1];
gf *d_images;
//*****************************cpu variables********************************
char secretkeys[1][crypto_kem_SECRETKEYBYTES];   
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES];
gf g[SYS_T + 1], L[SYS_N];
gf e_inv_LOOP[SYS_N][2*SYS_T];
gf  e_inv[SYS_N];
int i,j,k,tv,w = 0;
gf c[KATNUM][SYS_N];
gf out[KATNUM][2*SYS_T];
gf locator[KATNUM][SYS_T + 1];
gf images[KATNUM][SYS_N];
int e[KATNUM][SYS_N / 8];
gf t;
//*******************************************************************************

//*********************** device functions **************************************
__device__ gf d_gf_sq(gf in) 
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
__device__ gf d_gf_inv(gf in)
{
	gf tmp_11;
	gf tmp_1111;

	gf out = in;

	out = d_gf_sq(out);
	tmp_11 = mul(out, in); // 11

	out = d_gf_sq(tmp_11);
	out = d_gf_sq(out);
	tmp_1111 = mul(out, tmp_11); // 1111

	out = d_gf_sq(tmp_1111);
	out = d_gf_sq(out);
	out = d_gf_sq(out);
	out = d_gf_sq(out);
	out = mul(out, tmp_1111); // 11111111

	out = d_gf_sq(out);
	out = d_gf_sq(out);
	out = mul(out, tmp_11); // 1111111111

	out = d_gf_sq(out);
	out = mul(out, in); // 11111111111

	return d_gf_sq(out); // 111111111110
}
__device__ gf d_gf_frac(gf den, gf num) {
    return mul(d_gf_inv(den), num);
}
__device__ gf add(gf in0, gf in1)
{
	return in0 ^ in1;
}
__device__ gf d_eval(gf *f, gf a)
{
	int i;
	gf r;
	
	r = f[ SYS_T ];

	for (i = SYS_T-1; i >= 0; i--)
	{
		r = mul(r, a);
		r = add(r, f[i]);
	}

	return r;
}
__device__ gf iszero(gf a)
{
	uint32_t t = a;

	t -= 1;
	t >>= 19;

	return (gf) t;
}
__global__ void d_root(gf *d_out, gf (*d_f)[SYS_T + 1], gf *d_L, int katNum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SYS_N) {
        for (int k = 0; k < katNum; k++) {
            d_out[k * SYS_N + i] = d_eval(d_f[k], d_L[i]);
        }
    }
}
__global__ void d_bm(gf (*out)[SYS_T + 1], gf (*s)[2 * SYS_T]) {
	int katIdx = blockIdx.x * blockDim.x + threadIdx.x; // Index for KATNUM
	int sysTIdx = blockIdx.y * blockDim.y + threadIdx.y; // Index for 2*SYS_T

	if(katIdx < KATNUM && sysTIdx < 2*SYS_T) {
		int N = 0;
		int L = 0;
		gf mle;
		gf mne;
		gf T[SYS_T + 1];
		gf C[SYS_T + 1];
		gf B[SYS_T + 1];
		gf b = 1, d, f;

		for (int i = 0; i < SYS_T + 1; i++) {
			C[i] = B[i] = 0;
		}
		B[1] = C[0] = 1;

		for (N = 0; N < 2 * SYS_T; N++) {
			d = 0;
			for (int i = 0; i <= min(N, SYS_T); i++) {
				d ^= mul(C[i], s[katIdx][N - i]);
			}

			mne = ((d - 1) >> 15) - 1;
			mle = N; mle -= 2 * L; mle >>= 15; mle -= 1;
			mle &= mne;

			for (int i = 0; i <= SYS_T; i++) {
				T[i] = C[i];
			}

			f = d_gf_frac(b, d);

			for (int i = 0; i <= SYS_T; i++) {
				C[i] ^= mul(f, B[i]) & mne;
			}

			L = (L & ~mle) | ((N + 1 - L) & mle);

			for (int i = 0; i <= SYS_T; i++) {
				B[i] = (B[i] & ~mle) | (T[i] & mle);
			}

			b = (b & ~mle) | (d & mle);

			for (int i = SYS_T; i >= 1; i--) {
				B[i] = B[i - 1];	
			}
			B[0] = 0;
		}

		for (int i = 0; i <= SYS_T; i++) {
			out[katIdx][i] = C[SYS_T - i];
		}

	}
}
__global__ void initRAndComputeC(unsigned char (*d_ciphertexts)[crypto_kem_CIPHERTEXTBYTES], gf (*c)[SYS_N]) {
    int tv = blockIdx.x * blockDim.x + threadIdx.x; // KATNUM index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // SYS_N index

    if(tv < KATNUM && j < SYS_N) {
        unsigned char bitValue;
        if(j < SYND_BYTES * 8) {
            int byteIdx = div8Lookup[j];
            int bitPos = mod8Lookup[j];
            bitValue = (d_ciphertexts[tv][byteIdx] >> bitPos) & 1;
        } else {
            bitValue = 0;
        }
        c[tv][j] = bitValue;
    }
}
__global__ void matrixVectorMulKernel(gf (*d_c)[SYS_N], gf (*d_out)[2*SYS_T], gf (*d_e_inv_LOOP_device)[SYS_N][2*SYS_T]) {
    int katIdx = blockIdx.x * blockDim.x + threadIdx.x; // Index for KATNUM
    int sysTIdx = blockIdx.y * blockDim.y + threadIdx.y; // Index for 2*SYS_T

    if(katIdx < KATNUM && sysTIdx < 2*SYS_T) {
        gf sum = 0;
        for(int i = 0; i < SYS_N; i++) {
            sum ^= mul(d_c[katIdx][i], (*d_e_inv_LOOP_device)[i][sysTIdx]);
        }
        d_out[katIdx][sysTIdx] = sum;
    }
}
//*********************** end of device functions ********************************
//*********************** gf.c methods ******************************************
gf gf_iszero(gf a)
{
	uint32_t t = a;

	t -= 1;
	t >>= 19;

	return (gf) t;
}
gf gf_add(gf in0, gf in1)
{
	return in0 ^ in1;
}
gf gf_mul(gf in0, gf in1)
{
	int i;

	uint32_t tmp;
	uint32_t t0;
	uint32_t t1;
	uint32_t t;
	t0 = in0;
	t1 = in1;
	tmp = t0 * (t1 & 1);
	for (i = 1; i < GFBITS; i++)
		tmp ^= (t0 * (t1 & (1 << i)));
	t = tmp & 0x7FC000;
	tmp ^= t >> 9;
	tmp ^= t >> 12;

	t = tmp & 0x3000;
	tmp ^= t >> 9;
	tmp ^= t >> 12;

	return tmp & ((1 << GFBITS)-1);
}
/* input: field element in */
/* return: in^2 */
static inline gf gf_sq(gf in)
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
gf gf_inv(gf in)
{
	gf tmp_11;
	gf tmp_1111;

	gf out = in;

	out = gf_sq(out);
	tmp_11 = gf_mul(out, in); // 11

	out = gf_sq(tmp_11);
	out = gf_sq(out);
	tmp_1111 = gf_mul(out, tmp_11); // 1111

	out = gf_sq(tmp_1111);
	out = gf_sq(out);
	out = gf_sq(out);
	out = gf_sq(out);
	out = gf_mul(out, tmp_1111); // 11111111

	out = gf_sq(out);
	out = gf_sq(out);
	out = gf_mul(out, tmp_11); // 1111111111

	out = gf_sq(out);
	out = gf_mul(out, in); // 11111111111

	return gf_sq(out); // 111111111110
}
/* input: field element den, num */
/* return: (num/den) */
gf gf_frac(gf den, gf num)
{
	return gf_mul(gf_inv(den), num);
}
/* input: in0, in1 in GF((2^m)^t)*/
/* output: out = in0*in1 */
void GF_mul(gf *out, gf *in0, gf *in1)
{
	int i, j;

	gf prod[ SYS_T*2-1 ];

	for (i = 0; i < SYS_T*2-1; i++)
		prod[i] = 0;

	for (i = 0; i < SYS_T; i++)
		for (j = 0; j < SYS_T; j++)
			prod[i+j] ^= gf_mul(in0[i], in1[j]);

	//
 
	for (i = (SYS_T-1)*2; i >= SYS_T; i--)
	{
		prod[i - SYS_T + 3] ^= prod[i];
		prod[i - SYS_T + 1] ^= prod[i];
		prod[i - SYS_T + 0] ^= gf_mul(prod[i], (gf) 2);
	}

	for (i = 0; i < SYS_T; i++)
		out[i] = prod[i];
}
//******************** endof gf.c methods
gf eval(gf *f, gf a)
{
	int i;
	gf r;
	
	r = f[ SYS_T ];

	for (i = SYS_T-1; i >= 0; i--)
	{
		r = gf_mul(r, a);
		r = gf_add(r, f[i]);
	}

	return r;
}
uint16_t load_gf(const unsigned char *src)
{	
	
	uint16_t a; // 2 byte 

	a = src[1]; 
	a <<= 8; // Left-shift by 8 bits (one byte)
	a |= src[0]; 

	return a & GFMASK; 

}
gf bitrev(gf a)
{
	a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8); // Swap Adjacent Bytes:
	a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4); // Swap Nibbles within Bytes:
	a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2); // Swap Pairs of Bits within Nibbles:
	a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1); // Swap Individual Bits within Pairs:
	
	return a >> 4; // Right Shift by 4 to Discard Lower 4 Bits:
}
uint64_t load8(const unsigned char * in)
{
	int i;
	uint64_t ret = in[7];
	
	for (i = 6; i >= 0; i--)
	{
		ret <<= 8;
		ret |= in[i];
	}
	
	return ret;
}
void transpose_64x64(uint64_t * out, uint64_t * in)
{
	int i, j, s, d;

	uint64_t x, y;
	uint64_t masks[6][2] = {
	                        {0x5555555555555555, 0xAAAAAAAAAAAAAAAA},
	                        {0x3333333333333333, 0xCCCCCCCCCCCCCCCC},
	                        {0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0},
	                        {0x00FF00FF00FF00FF, 0xFF00FF00FF00FF00},
	                        {0x0000FFFF0000FFFF, 0xFFFF0000FFFF0000},
	                        {0x00000000FFFFFFFF, 0xFFFFFFFF00000000}
	                       };

	for (i = 0; i < 64; i++)
		out[i] = in[i];

	for (d = 5; d >= 0; d--)
	{
		s = 1 << d;

		for (i = 0; i < 64; i += s*2)
		for (j = i; j < i+s; j++)
		{
			x = (out[j] & masks[d][0]) | ((out[j+s] & masks[d][0]) << s);
			y = ((out[j] & masks[d][1]) >> s) | (out[j+s] & masks[d][1]);

			out[j+0] = x;
			out[j+s] = y;
		}
	}
}
static void layer(uint64_t * data, uint64_t * bits, int lgs)
{
	int i, j, s;

	uint64_t d;

	s = 1 << lgs;

	for (i = 0; i < 64; i += s*2)
	for (j = i; j < i+s; j++)
	{

		d = (data[j+0] ^ data[j+s]);
		d &= (*bits++);
		data[j+0] ^= d;
		data[j+s] ^= d;
	}
}
uint32_t load4(const unsigned char * in)
{
	int i;
	uint32_t ret = in[3];

	for (i = 2; i >= 0; i--)
	{
		ret <<= 8;
		ret |= in[i];
	}

	return ret;
}
void store8(unsigned char *out, uint64_t in)
{
	out[0] = (in >> 0x00) & 0xFF;
	out[1] = (in >> 0x08) & 0xFF;
	out[2] = (in >> 0x10) & 0xFF;
	out[3] = (in >> 0x18) & 0xFF;
	out[4] = (in >> 0x20) & 0xFF;
	out[5] = (in >> 0x28) & 0xFF;
	out[6] = (in >> 0x30) & 0xFF;
	out[7] = (in >> 0x38) & 0xFF;
}
void apply_benes(unsigned char * r, const unsigned char * bits, int rev)
{
	int i;

	const unsigned char *cond_ptr; 
	int inc, low;

	uint64_t bs[64];
	uint64_t cond[64];

	//
	
		// printf("testing ");
		// for(i = 0; i < 64; i++) {
		// 	printf("i: %d, r[%d]: %u\n", i, i, (unsigned int)r[i]);

		// }

	for (i = 0; i < 64; i++)
	{
		bs[i] = load8(r + i*8);
	}

	if (rev == 0) 
	{
		inc = 256;
		cond_ptr = bits;
	}
	else
	{
		inc = -256;
		cond_ptr = bits + (2*GFBITS-2)*256;
	}

	//

	transpose_64x64(bs, bs);

	for (low = 0; low <= 5; low++) 
	{ 
		for (i = 0; i < 64; i++) cond[i] = load4(cond_ptr + i*4);
		transpose_64x64(cond, cond);
		layer(bs, cond, low); 
		cond_ptr += inc; 
	}
	
	transpose_64x64(bs, bs);
	
	for (low = 0; low <= 5; low++) 
	{ 
		for (i = 0; i < 32; i++) cond[i] = load8(cond_ptr + i*8);
		layer(bs, cond, low); 
		cond_ptr += inc; 
	}
	for (low = 4; low >= 0; low--) 
	{ 
		for (i = 0; i < 32; i++) cond[i] = load8(cond_ptr + i*8);
		layer(bs, cond, low); 
		cond_ptr += inc; 
	}

	transpose_64x64(bs, bs);
	
	for (low = 5; low >= 0; low--) 
	{ 
		for (i = 0; i < 64; i++) cond[i] = load4(cond_ptr + i*4);
		transpose_64x64(cond, cond);
		layer(bs, cond, low); 
		cond_ptr += inc; 
	}

	transpose_64x64(bs, bs);

	//

	for (i = 0; i < 64; i++)
	{
		store8(r + i*8, bs[i]);
	}
}
void support_gen(gf * s, const unsigned char *c) // typedef uint16_t gf 2 bytes
{
	gf a; //  uint16_t a
	int i, j;
	unsigned char L[ GFBITS ][ (1 << GFBITS)/8 ];

	for (i = 0; i < GFBITS; i++)
		for (j = 0; j < (1 << GFBITS)/8; j++)
			L[i][j] = 0;

	
	for (i = 0; i < (1 << GFBITS); i++) // O TO 4096
	{
		a = bitrev((gf) i); 
		for (j = 0; j < GFBITS; j++) // 0 to 12
			L[j][ i/8 ] |= ((a >> j) & 1) << (i%8);
	}
			
	for (j = 0; j < GFBITS; j++)
		apply_benes(L[j], c, 0);

	for (i = 0; i < SYS_N; i++)
	{
		s[i] = 0;
		for (j = GFBITS-1; j >= 0; j--)
		{
			s[i] <<= 1;
			s[i] |= (L[j][i/8] >> (i%8)) & 1;
		}
	}
	
}
//********************************* endofcpu funtions **********************************************

void precomputeLookupTables() {
    int hostDiv8[SYS_N];
    int hostMod8[SYS_N];

    for (int j = 0; j < SYS_N; j++) {
        hostDiv8[j] = j / 8;
        hostMod8[j] = j % 8;
    }

    cudaMemcpyToSymbol(div8Lookup, hostDiv8, SYS_N * sizeof(int));
    cudaMemcpyToSymbol(mod8Lookup, hostMod8, SYS_N * sizeof(int));
}
//******************** endof bm.c method
int keysetup() {
    unsigned char *sk = 0;	
    

    FILE *file1 = fopen("ct.bin", "rb");
    FILE *file2 = fopen("sk.bin", "rb");

    if (file1 == NULL || file2 == NULL) {
        perror("Error opening file");
        return 1;
    }
    if (fread(secretkeys, crypto_kem_SECRETKEYBYTES, 1, file2) != 1) {
        fprintf(stderr, "Error reading from file_sk");
        fclose(file2);
        return 1;
    }	
    if (fread(ciphertexts, crypto_kem_CIPHERTEXTBYTES, KATNUM, file1) != KATNUM) {
        fprintf(stderr, "Error reading from file_ct");
        fclose(file1);
        return 1;
    }
	fclose(file1);
    sk = (unsigned char *)secretkeys + 40; // Adjusted to pointer arithmetic
    fclose(file2);
    
    for (i = 0; i < SYS_T; i++) {
        g[i] = load_gf(sk);
        sk += 2;
    }
    g[SYS_T] = 1;
    support_gen(L, sk);	

    for (i = 0; i < SYS_N; i++) {
        gf e = eval(g, L[i]);
        e_inv[i] = gf_inv(gf_mul(e, e));
    }
    for (i = 0; i < SYS_N; i++) {
        e_inv_LOOP[i][0] = e_inv[i];
        for (j = 1; j < 2*SYS_T; j++) {
            e_inv_LOOP[i][j] = gf_mul(e_inv_LOOP[i][j-1], L[i]);
        }
    }

    precomputeLookupTables();
    cudaMalloc(&d_L, SYS_N * sizeof(gf));
    cudaMemcpy(d_L, L, sizeof(gf) * SYS_N, cudaMemcpyHostToDevice);
    cudaMalloc(&d_e_inv_LOOP_device, sizeof(gf) * SYS_N * 2 * SYS_T);
    cudaMemcpy(d_e_inv_LOOP_device, e_inv_LOOP, sizeof(gf) * SYS_N * 2 * SYS_T, cudaMemcpyHostToDevice);

    return 0; // Assuming successful execution
}
int main()
{	

	keysetup();
	clock_t start, end;
    double cpu_time_used;
    start = clock();
//********************************************************
	// Initialization
	dim3 blocksPerGridc((KATNUM + 15) / 16, (SYS_N + 15) / 16);
	dim3 threadsPerBlockc(16, 16);
	int numBlocksX = (KATNUM + threadsPerBlockc.x - 1) / threadsPerBlockc.x;
	int numBlocksY = (2 * SYS_T + threadsPerBlockc.y - 1) / threadsPerBlockc.y;
	dim3 blocksPerGridm(numBlocksX, numBlocksY);
	int threadsPerBlockb = 256;
	int blocksPerGridb = (KATNUM + threadsPerBlockb - 1) / threadsPerBlockb;
	int blocksPerGridR = (SYS_N + threadsPerBlockb - 1) / threadsPerBlockb;
	cudaMalloc(&d_ciphertexts, sizeof(ciphertexts));
	cudaMalloc(&d_c, sizeof(c)); // Device memory for c
	cudaMalloc(&d_out, sizeof(gf) * KATNUM * 2 * SYS_T); // Device memory for out
	cudaMemcpy(d_ciphertexts, ciphertexts, sizeof(ciphertexts), cudaMemcpyHostToDevice);
	cudaMalloc(&d_locator, sizeof(gf) * KATNUM * 2 * SYS_T);
	cudaMalloc(&d_images, sizeof(gf) * KATNUM * SYS_N);

	//**************** kernels ***************************************************************
	initRAndComputeC<<<blocksPerGridc, threadsPerBlockc>>>(d_ciphertexts, d_c);
	matrixVectorMulKernel<<<blocksPerGridm, threadsPerBlockc>>>(d_c, d_out, d_e_inv_LOOP_device);
	d_bm<<<blocksPerGridb, threadsPerBlockb>>>(d_locator,d_out);
	d_root<<<blocksPerGridR, threadsPerBlockb>>>(d_images, d_locator, d_L, KATNUM);
	//**************** end of kernels ********************************************************
	cudaDeviceSynchronize();
	cudaMemcpy(images, d_images, sizeof(gf) * KATNUM * SYS_N, cudaMemcpyDeviceToHost);



	// for(j = 0; j < KATNUM; j++) {


	// 	for (i = 0; i < SYS_N; i++) {
			
	// 		t = gf_iszero(images[j][i]) & 1;
	// 		e[j][ i/8 ] |= t << (i%8);
	// 		w += t;
	// 	}

	// 	printf("decrypt e: positions : ");
    // 	for (k = 0;k < SYS_N;++k)
    //   		if (e[j][k/8] & (1 << (k&7)))
    //     		printf(" %d",k);
    // 	printf("\n\n");
	// }
	//**************** end of kernels ********************************************************
//**********************************************************
	end = clock(); // Record the end time
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // Calculate the CPU time used
	

    printf("%f \n", cpu_time_used);
	cudaFree(d_e_inv_LOOP_device);
	cudaFree(d_L);	
	cudaFree(d_c);	
	cudaFree(d_out);
	cudaFree(d_locator);
	cudaFree(d_images);
	cudaFree(d_ciphertexts);

    
	return 0;
}
