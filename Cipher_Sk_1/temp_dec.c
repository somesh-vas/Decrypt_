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



int tv; //test_vector
unsigned char secretkeys[crypto_kem_SECRETKEYBYTES];
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES];
int e[SYS_N / 8];
int i,w = 0,j,k;
gf g[ SYS_T+1 ]; // goppa polynomial
gf L[ SYS_N ]; // support

gf e_inv_LOOP_1D[sb * 2 * SYS_T];
gf inverse_elements[sb][2*SYS_T];
gf temp;
gf e_inv[SYS_N];

unsigned char r[ SYS_N/8 ]; 
gf out[ SYS_T*2 ]; // random string s
gf locator[ SYS_T+1 ]; // error locator 
gf images[ SYS_N ]; // 
gf t,c[SYS_N];
clock_t start, end;
double cpu_time_used;
double cpu_printing;
unsigned char *sk = NULL;
int count;
//******************** endof root.c methods
int main() {

    
    initialisation(secretkeys,ciphertexts,sk,L,g);	
	
	compute_inverses();

	for (tv = 0; tv < KATNUM; tv++) {
			start = clock();

			// memset(r + SYND_BYTES, 0, (SYS_N/8 - SYND_BYTES));
			// memset(c, 0, sizeof(c));
			// memset(out, 0, sizeof(out));			
			// memset(locator, 0, sizeof(locator));
			// memset(images, 0, sizeof(images));
			// memset(e, 0, (SYS_N / 8));

			// for (i = 0; i < SYND_BYTES; i++)       
			// 	r[i] = ciphertexts[tv][i];

			for(i = 0; i < SYS_N; i++) 
			{	
				c[i] = (r[(i)/8] >> ((i)%8)) & 1;	
			}

			for (i = 0; i < SYS_N; i++)	
				for (j = 0; j < 2*SYS_T; j++)
					out[j] = gf_add(out[j], gf_mul(inverse_elements[i][j], c[i]));
			
			synd(out, g, L, r, ciphertexts[tv]);  // Call the synd function here

			bm(locator, out);

			root(images, locator, L);

			end = clock(); // Record the end time
			cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC; // Calculate the CPU time used

			start = clock();

			w = 0;
			for (i = 0; i < SYS_N; i++) {
				t = gf_iszero(images[i]) & 1;
				e[ i/8 ] |= t << (i%8);
				w += t;
			}
			
			
			printf("decrypt e: positions : ");
			for (k = 0; k < SYS_N; ++k) {
				if (e[k / 8] & (1 << (k & 7))) {
					printf(" %d", k);
					++count;
				}
			}
			printf("\n\n");
			printf("Number of errors: %d\n", count);
			count = 0;
			end = clock(); // Record the end time
			cpu_printing += ((double) (end - start)) / CLOCKS_PER_SEC; // Calculate the CPU time used
		}
//**************** end ********************************************
	
    printf("Total Kernel Time : %f \n", cpu_time_used);
	// printf("Error positions printing time : %f \n", cpu_printing);

    return KAT_SUCCESS;
}



