#include "common.h"

uint16_t load_gf(const unsigned char *src)
{	
	
	uint16_t a; // 2 byte 

	a = src[1]; 
	a <<= 8; // Left-shift by 8 bits (one byte)
	a |= src[0]; 

	return a & GFMASK; 

}
