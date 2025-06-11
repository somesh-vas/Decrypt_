#include "common.h"
#include "gf.h"

//******************** benes.c methods
/* one layer of the benes network */
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
		// r = gf_add(r, f[i]);
		r ^= f[i];
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
