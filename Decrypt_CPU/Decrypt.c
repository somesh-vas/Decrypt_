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
#include <time.h>


int tv; //test_vector
unsigned char secretkeys[crypto_kem_SECRETKEYBYTES];
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES];
int e[SYS_N / 8];
int i,w = 0,j,k;
gf g[ SYS_T+1 ]; // goppa polynomial
gf L[ SYS_N ]; // support
gf s[ SYS_T*2 ];

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
double avg_cpu_time_used;
double cpu_printing;
double synd_time = 0, bm_time = 0, root_time = 0;
unsigned char *sk = NULL;
int count;
//******************** endof root.c methods
int main() {

    
    initialisation(secretkeys,ciphertexts,sk,L,g);	
	
	compute_inverses();
	
	
	for (tv = 0; tv < KATNUM; tv++) {
		
		start = clock();
		
		for (i = 0; i < SYND_BYTES; i++)       r[i] = ciphertexts[tv][i];

		for (i = SYND_BYTES; i < SYS_N/8; i++) r[i] = 0;
		// print r

		synd(s, r);
		



		 
		bm(locator, s);
		


		
		root(images, locator, L);	


		

		

		for (i = 0; i < SYS_N/8; i++) 
			e[i] = 0;			
	
		for (i = 0; i < SYS_N; i++)
		{
			t = gf_iszero(images[i]) & 1;			
			e[ i/8 ] |= t << (i%8);
		}

		end = clock();
		avg_cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cpu_printing = avg_cpu_time_used / KATNUM;
	
	double micros = avg_cpu_time_used * 1000000; // Convert to microseconds

    printf("CPU time used: %.2f microseconds\n", micros);
		
    	for (k = 0;k < SYS_N;++k)
    	  if (e[k/8] & (1 << (k&7))){
    	    // printf(" %d",k);
			
		  }
		  printf("\n");
    
		
	}
	
	
			
    return KAT_SUCCESS;
}



