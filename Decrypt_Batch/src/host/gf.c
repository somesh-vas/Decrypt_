#include "gf.h"

gf bitrev(gf a)
{
	a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8); // Swap Adjacent Bytes:
	a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4); // Swap Nibbles within Bytes:
	a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2); // Swap Pairs of Bits within Nibbles:
	a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1); // Swap Individual Bits within Pairs:
	
	return a >> 4; // Right Shift by 4 to Discard Lower 4 Bits:
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

void store_gf(unsigned char *dest, gf a)
{
	dest[0] = a & 0xFF;
	dest[1] = a >> 8;
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
