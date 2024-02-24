//************** header files

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

//*************** endof header files
//******************** parameter define
#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_CRYPTO_FAILURE  -4
#define KATNUM 10
#define crypto_kem_SECRETKEYBYTES 6492 
#define crypto_kem_CIPHERTEXTBYTES 96
#define GFBITS 12 // Size of each element in the Galois field: 12 bits
#define SYS_N 3488 
#define SYS_T 64
#define COND_BYTES ((1 << (GFBITS-4))*(2*GFBITS - 1))
#define IRR_BYTES (SYS_T * 2)  // Size of irreducible polynomial in bytes
#define PK_NROWS (SYS_T*GFBITS) 
#define PK_NCOLS (SYS_N - PK_NROWS)
#define PK_ROW_BYTES ((PK_NCOLS + 7)/8)
#define SYND_BYTES ((PK_NROWS + 7)/8)
#define GFMASK ((1 << GFBITS) - 1) // Bitmask for the Galois field
#define min(a, b) ((a < b) ? a : b)

typedef uint16_t gf;

//******************** endof parameters
//******************** prototypes
// Function prototypes
void support_gen(gf *s, const unsigned char *c);

void root(gf *out, gf *f, gf *L);
void transpose_64x64(uint64_t *out, uint64_t *in);
gf eval(gf *f, gf a);

////////////////////// endof prototypes
//******************** util.c methods
void store_gf(unsigned char *dest, gf a)
{
	dest[0] = a & 0xFF;
	dest[1] = a >> 8;
}
uint16_t load_gf(const unsigned char *src)
{	
	
	uint16_t a; // 2 byte 

	a = src[1]; 
	a <<= 8; // Left-shift by 8 bits (one byte)
	a |= src[0]; 

	return a & GFMASK; 
	/*
	GFMASK is defined as 4095, which has 12 bits set to 1), and it discards any higher-order bits in a beyond the 12th bit
	ex:
	a        : 0000 0110 1111 0111
	GFMASK   : 0000 1111 1111 1111
	-------------------------------
	result   : 0000 0110 1111 0111

		
	*/
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
uint64_t load8(const unsigned char * in)
{
	int i;
	uint64_t ret = in[7];
	
	for (i = 6; i >= 0; i--)
	{
		ret <<= 8;
		ret |= in[i];
	}
	/*	Initially, ret is set to the last byte (in[7]), which is 0x08.
		In the loop, the code shifts ret left by 8 bits in each iteration 
		and then performs a bitwise OR with the current byte (in[i]). 
		This process effectively combines the bytes to form the 64-bit integer.
	*/
	return ret;
}
gf bitrev(gf a)
{
	a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8); // Swap Adjacent Bytes:
	a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4); // Swap Nibbles within Bytes:
	a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2); // Swap Pairs of Bits within Nibbles:
	a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1); // Swap Individual Bits within Pairs:
	
	return a >> 4; // Right Shift by 4 to Discard Lower 4 Bits:
}
//******************** endof util.c methods
//******************** gf.c methods
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
//******************** endof gf.c methods

//******************** bm.c method
/* the Berlekamp-Massey algorithm */
/* input: s, sequence of field elements */
/* output: out, minimal polynomial of s */
void bm(gf *out, gf *s)
{
	int i;
	uint16_t N = 0;
	uint16_t L = 0;
	uint16_t mle;
	uint16_t mne;
	gf T[ SYS_T+1  ];
	gf C[ SYS_T+1 ];
	gf B[ SYS_T+1 ];
	gf b = 1, d, f;
	for (i = 0; i < SYS_T+1; i++)
		C[i] = B[i] = 0;
	B[1] = C[0] = 1;

	for (N = 0; N < 2 * SYS_T; N++)
	{	
		// printf("N: %d\n", N);
		d = 0;
		for (i = 0; i <= min(N, SYS_T); i++)
		{	d ^= gf_mul(C[i], s[ N-i]);
			// printf("d[%d]: %d ",i, d);
		}
		// printf("\nbIntermediate - d[%d]: %d\n", N, d);
	
	
		// mne = d; mne -= 1;   mne >>= 15; mne -= 1;
		mne = ((d-1)>>15)-1;
		mle = N; mle -= 2*L; mle >>= 15; mle -= 1;
		mle &= mne;

		for (i = 0; i <= SYS_T; i++)			
			T[i] = C[i];

		f = gf_frac(b, d);

		for (i = 0; i <= SYS_T; i++)			
			C[i] ^= gf_mul(f, B[i]) & mne;

		L = (L & ~mle) | ((N+1-L) & mle);

		for (i = 0; i <= SYS_T; i++)			
			B[i] = (B[i] & ~mle) | (T[i] & mle);

		b = (b & ~mle) | (d & mle);

		for (i = SYS_T; i >= 1; i--) B[i] = B[i-1];
		B[0] = 0;
	}

	for (i = 0; i <= SYS_T; i++)
		out[i] = C[ SYS_T-i ];
}
//******************** endof bm.c method

//******************** benes.c methods
/* one layer of the benes network */
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

/* input: r, sequence of bits to be permuted */
/*        bits, condition bits of the Benes network */
/*        rev, 0 for normal application; !0 for inverse */
/* output: r, permuted bits */
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

/* input: condition bits c */
/* output: support s */
void support_gen(gf * s, const unsigned char *c) // typedef uint16_t gf 2 bytes
{
	gf a; //  uint16_t a
	int i, j;
	unsigned char L[ GFBITS ][ (1 << GFBITS)/8 ];

	for (i = 0; i < GFBITS; i++)
		for (j = 0; j < (1 << GFBITS)/8; j++)
			L[i][j] = 0;

	// gf a[1<<GFBITS];
	// for(i = 0; i < (1<<GFBITS); i++) a[i] = bitrev((gf) i);

	for (i = 0; i < (1 << GFBITS); i++) // O TO 4096
	{
		a = bitrev((gf) i); 
		for (j = 0; j < GFBITS; j++) // 0 to 12
			L[j][ i/8 ] |= ((a >> j) & 1) << (i%8);
	}
	
	

    // printf("Values of L in hexadecimal:\n");
    // for (i = 0; i < GFBITS; i++) {
    //     printf("\n");
    //     for (j = 0; j < (1 << GFBITS) / 8; j++)
    //         printf("%04X ", L[i][j]);  // %02X prints in hexadecimal format with leading zeros
    // }
    // printf("\n");


			
	for (j = 0; j < GFBITS; j++)
		apply_benes(L[j], c, 0);

	
	// 	printf("=======================\n");
	// 	for (i = 0; i < GFBITS; i++) {
    // for (j = 0; j < (1<<GFBITS)/8 ; j++) {
    //     printf("%04X ", L[i][j]);
    // }
    // printf("\n");
// }


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
//********************endof benes.c methods
//******************** transpose.c method
/* input: in, a 64x64 matrix over GF(2) */
/* output: out, transpose of in */
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


//******************** endof 
//******************** synd.c method
/* input: Goppa polynomial f, support L, received word r */
/* output: out, the syndrome of length 2t */

//******************** endof synd.c method
//******************** root.c methods
/* input: polynomial f and field element a */
/* return f(a) */
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

/* input: polynomial f and list of field elements L */
/* output: out = [ f(a) for a in L ] */
void root(gf *out, gf *f, gf *L)
{
	int i; 

	for (i = 0; i < SYS_N; i++)
		out[i] = eval(f, L[i]);
}
int tv; //test_vector
unsigned char secretkeys[1][crypto_kem_SECRETKEYBYTES];
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES];
int e[SYS_N / 8];
int i,w = 0,j,k;
gf g[ SYS_T+1 ]; // goppa polynomial
gf L[ SYS_N ]; // support
gf e_inv_LOOP[SYS_N][2*SYS_T];
gf  e_inv[SYS_N];
unsigned char r[ SYS_N/8 ]; 
gf out[ SYS_T*2 ]; // random string s
gf locator[ SYS_T+1 ]; // error locator 
gf images[ SYS_N ]; // 
gf t,c[SYS_N],temp;
clock_t start, end;
double cpu_time_used;
unsigned char *sk = NULL;
//******************** endof root.c methods
int main() {
    
    
    
       

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
	fclose(file2);
    
    
	sk = secretkeys[0] + 40;
    start = clock();
	
	for (i = 0; i < SYS_T; i++) { g[i] = load_gf(sk); sk += 2; } g[ SYS_T ] = 1; // load goppa polynomial from sk to g[] // 'load_gf' is utility function from util.c
	
	support_gen(L, sk); 
	
	for (i = 0; i < SYS_N; i++) {
        temp = eval(g, L[i]);
        e_inv[i] = gf_inv(gf_mul(temp, temp));
    }
    for (i = 0; i < SYS_N; i++) {
        e_inv_LOOP[i][0] = e_inv[i];
        for (j = 1; j < 2*SYS_T; j++) {
            e_inv_LOOP[i][j] = gf_mul(e_inv_LOOP[i][j-1], L[i]);
        }
    }
    
		for (tv = 0; tv < KATNUM; tv++) {

			memset(r + SYND_BYTES, 0, (SYS_N/8 - SYND_BYTES));
			memset(c, 0, sizeof(c));
			memset(out, 0, sizeof(out));			
			memset(locator, 0, sizeof(locator));
			memset(images, 0, sizeof(images));
			memset(e, 0, SYS_N/8);

			for (i = 0; i < SYND_BYTES; i++)       
				r[i] = ciphertexts[tv][i];

			for(i = 0; i < SYS_N; i++) 
				c[i] = (r[i/8] >> (i%8)) & 1;		
			
			for (i = 0; i < SYS_N; i++)	
				for (j = 0; j < 2*SYS_T; j++)
					out[j] = gf_add(out[j], gf_mul(e_inv_LOOP[i][j], c[i]));
			

			bm(locator, out);

			root(images, locator, L);
			

			for (i = 0; i < SYS_N; i++) {
				t = gf_iszero(images[i]) & 1;
				e[ i/8 ] |= t << (i%8);
				w += t;
			}

			
			printf("decrypt e: positions : ");
			for (k = 0;k < SYS_N;++k)
				if (e[k/8] & (1 << (k&7)))
					printf(" %d",k);
				printf("\n\n");
			
		}

	end = clock(); 

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // Calculate the CPU time used

    printf("%f \n", cpu_time_used);  

    return KAT_SUCCESS;
}



