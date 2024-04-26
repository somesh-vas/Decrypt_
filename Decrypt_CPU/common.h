#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
// #include <cstring>
#include <time.h>
#include <stdlib.h>
// #include <sys/utime.h>
#include <string.h>
// #include <fstream> 
#include <stdint.h>

//******************** parameter define
#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_CRYPTO_FAILURE  -4
#define KATNUM 1
#define ciphert 20
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
#define sb SYND_BYTES * 8
typedef uint16_t gf;



#endif // !COMMON_H