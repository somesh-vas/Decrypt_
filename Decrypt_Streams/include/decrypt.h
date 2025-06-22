#ifndef DECRYPT_H
#define DECRYPT_H
#include <nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "gf.h"
#include "common.h"
#include "root.h"

#define GFLUT_ORDER   ((1u << GFBITS) - 1u)
#define GFLUT_SIZE    (1u << GFBITS)

__constant__ gf d_L[SYS_N];
__constant__ gf gf_inverse_table[1 << GFBITS];

__constant__ uint16_t d_gf_log[1<<GFBITS];
__constant__ uint16_t d_gf_exp[(1<<GFBITS)-1];
__constant__ gf d_tab0[16][SYS_T+1];
__constant__ gf d_tab1[16][SYS_T+1];
__constant__ gf d_tab2[16][SYS_T+1];

static gf       _gf_exp_table[GFLUT_ORDER];
static uint16_t _gf_log_table[GFLUT_SIZE];
static int      _gf_tables_initialized = 0;

// void init_gf_log_exp_tables(void)
// {
//     if (_gf_tables_initialized) return;
//     gf x = 1;
//     for (int i = 0; i < (int)GFLUT_ORDER; i++) {
//         _gf_exp_table[i] = x;
//         _gf_log_table[x] = (uint16_t)i;
//         x = gf_mul(x, (gf)2);
//     }
//     _gf_log_table[0] = 0;
//     _gf_tables_initialized = 1;
// }



unsigned char *d_ciphertexts;
gf *d_inverse_elements;
__device__       gf d_L_global[SYS_N];
gf images[SYS_N];
gf error[SYS_T];
int tv;
unsigned char secretkeys[crypto_kem_SECRETKEYBYTES];
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES];
int e[SYS_N / 8];
int i, w = 0, j, k;
gf g[SYS_T + 1];
gf L[SYS_N];
gf s[SYS_T * 2];
gf e_inv_LOOP_1D[sb * 2 * SYS_T];
gf inverse_elements[sb][2 * SYS_T];
gf temp;
gf e_inv[SYS_N];
unsigned char r[SYS_N / 8];
gf locator[SYS_T + 1];
gf t, c[SYS_N];
clock_t start, end;
double avg_cpu_time_used;
double cpu_printing;
double synd_time = 0, bm_time = 0, root_time = 0;
unsigned char *sk = NULL;
int count;
// unsigned char h_error[KATNUM][SYS_N];

__device__ __forceinline__ gf add(gf in0, gf in1) {
    return in0 ^ in1;
}

__device__ __forceinline__ gf mul(gf in0, gf in1) {
    int i;
    uint32_t tmp = 0;
    uint32_t t0 = in0;
    uint32_t t1 = in1;
    uint32_t t;
    tmp = t0 * (t1 & 1);
    for (i = 1; i < GFBITS; i++) {
        tmp ^= (t0 * (t1 & (1 << i)));
    }
    t = tmp & 0x7FC000;
    tmp ^= t >> 9;
    tmp ^= t >> 12;
    t = tmp & 0x3000;
    tmp ^= t >> 9;
    tmp ^= t >> 12;
    return tmp & ((1 << GFBITS) - 1);
}

__device__ __forceinline__ gf p_gf_inv(gf in) {
    return gf_inverse_table[in];
}

__device__ __forceinline__ gf p_gf_frac(gf den, gf num) {
    return mul(p_gf_inv(den), num);
}

void InitializeC() {
    const int N = 1 << GFBITS;
    gf host_inv[N];
    host_inv[0] = 0;
    for (int i = 1; i < N; i++)
        host_inv[i] = gf_inv((gf)i);
    cudaMemcpyToSymbol(gf_inverse_table, host_inv, sizeof(host_inv), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_L, L, SYS_N * sizeof(gf), 0, cudaMemcpyHostToDevice);
    cudaMemcpy(
     d_L_global,
     L,
     SYS_N * sizeof(gf),
     cudaMemcpyHostToDevice
 ) ;
    cudaMalloc(&d_ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM);
    size_t size = sizeof(gf) * sb * 2 * SYS_T;
    cudaMalloc(&d_inverse_elements, size);
    cudaMemcpy(d_inverse_elements, inverse_elements, size, cudaMemcpyHostToDevice);
    
}

#endif // DECRYPT_H
