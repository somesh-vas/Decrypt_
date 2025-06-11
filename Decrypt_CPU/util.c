#include "common.h"
#include "root.h"
#include "gf.h"

extern gf L[SYS_N];
extern gf g[SYS_T + 1];
extern gf e_inv[SYS_N];
extern gf inverse_elements[SYS_N][2 * SYS_T];


// Change the signature to accept a pointer to an array of fixed size
int initialisation(unsigned char *secretkeys, unsigned char (*ciphertexts)[crypto_kem_CIPHERTEXTBYTES], unsigned char *sk, gf *L, gf *g) {

       

    FILE *file1 = fopen("../Cipher_Sk_1/ct.bin", "rb");
    FILE *file2 = fopen("../Cipher_Sk_1/sk.bin", "rb");

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
    sk = secretkeys + 40;	
	for (int i = 0; i < SYS_T; i++) { g[i] = load_gf(sk); sk += 2; } g[ SYS_T ] = 1; 	
	support_gen(L, sk); 

    
    
    return 0;
}

void compute_inverses() {
    int i, j;
    gf temp;

    for (i = 0; i < sb; i++) {
        temp = eval(g, L[i]);
        e_inv[i] = gf_inv(gf_mul(temp, temp));
        inverse_elements[i][0] = e_inv[i];
        for (j = 1; j < 2 * SYS_T; j++) {
            inverse_elements[i][j] = gf_mul(inverse_elements[i][j - 1], L[i]);
        }
    }
}
void synd(gf *out, unsigned char *r) {
    int i, j;
    gf c;

    memset(out, 0, 2 * SYS_T * sizeof(gf)); // Reset the output array

    // Extract the bits from r and compute the syndrome using precomputed inverses
    for (i = 0; i < sb; i++) {
        c = (r[i / 8] >> (i % 8)) & 1; // Extract bit
        for (j = 0; j < 2 * SYS_T; j++) {
            out[j] = gf_add(out[j], gf_mul(inverse_elements[i][j], c));
        }
    }
}

