#ifndef COMMON_H
#define COMMON_H

/* --------------------  standard headers  --------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* --------------------  library / test parameters  ------------------ */
#define KAT_SUCCESS           0
#define KAT_FILE_OPEN_ERROR  -1
#define KAT_CRYPTO_FAILURE   -4

#define KATNUM                         1     /* test-vectors to run      */
#define CIPHERT                        20    /* (left as-is from legacy) */

#define crypto_kem_SECRETKEYBYTES   6492
#define crypto_kem_CIPHERTEXTBYTES    96

/* --------------------  McEliece code parameters  ------------------- */
#define GFBITS   12                    /* field size → GF(2^12)        */
#define SYS_N  3488
#define SYS_T    64

#define COND_BYTES  ((1 << (GFBITS-4)) * (2*GFBITS - 1))
#define IRR_BYTES   (SYS_T * 2)

#define PK_NROWS    (SYS_T * GFBITS)
#define PK_NCOLS    (SYS_N - PK_NROWS)
#define PK_ROW_BYTES  ((PK_NCOLS + 7) / 8)

#define SYND_BYTES  ((PK_NROWS + 7) / 8)

#define GFMASK      ((1 << GFBITS) - 1)
#define sb          (SYND_BYTES * 8)

/* helper */
#define min(a,b)    (( (a) < (b) ) ? (a) : (b))

/* -------------------------------------------------------------------
 * Fundamental field type
 * ------------------------------------------------------------------- */
typedef uint16_t gf;                   /* 12-bit field elements live here */

/* -------------------------------------------------------------------
 * Constant-memory lookup tables (log / antilog) for GF(2^12)
 * ------------------------------------------------------------------- */
#ifndef GF_LUT_H
#define GF_LUT_H

/* multiplicative order (α^4095 = 1), and table sizes                */
#define GFLUT_ORDER   ((1u << GFBITS) - 1u)     /* 4095 */
#define GFLUT_SIZE    (1u << GFBITS)            /* 4096 (includes zero) */

#endif /* GF_LUT_H */

#endif /* COMMON_H */
