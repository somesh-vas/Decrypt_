#ifndef ROOT_H
#define ROOT_H
#include "common.h"
#include "gf.h"

#ifdef __cplusplus
extern "C" {
#endif


void support_gen(gf *s, const unsigned char *c);
void root(gf *out, gf *f, gf *L);
void transpose_64x64(uint64_t *out,uint64_t *in);
gf eval(gf *f, gf a);
void layer(uint64_t * data, uint64_t * bits, int lgs);
void apply_benes(unsigned char * r, const unsigned char * bits, int rev);
void transpose_64x64(uint64_t *out,uint64_t *in);

// Change the signature to accept a pointer to an array of fixed size
int initialisation(unsigned char *secretkeys, unsigned char (*ciphertexts)[crypto_kem_CIPHERTEXTBYTES], unsigned char *sk, gf *L, gf *g);

void compute_inverses();
// void synd(gf *out, gf *f, gf *L, unsigned char *r, unsigned char *ciphertexts);
void synd(gf *out, unsigned char *r);


#ifdef __cplusplus
}
#endif

#endif // !ROOT_H