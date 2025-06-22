#ifndef DECRYPT_H
#define DECRYPT_H
#include <nvToolsExt.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
                // for std::min, if needed
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "gf.h"
#include "common.h"
#include "root.h"
// in common.h, after typedef uint16_t gf;
#define GFLUT_ORDER   ((1u << GFBITS) - 1u)  // 4095
#define GFLUT_SIZE    (1u << GFBITS)         // 4096

// Device constants
__constant__ gf d_L[SYS_N];
__constant__ gf gf_inverse_table[1 << GFBITS];
// in decrypt.h (or common header)
#include "common.h"

// constant lookup tables for GF(2^12):
//   log:  maps α^i → i   (0 → 0 by convention)
//   exp:  maps i   → α^i (only indices 0..4094 used)
__constant__ uint16_t d_gf_log[1<<GFBITS];
__constant__ uint16_t d_gf_exp[(1<<GFBITS)-1];
__constant__ gf d_tab0[16][SYS_T+1];
__constant__ gf d_tab1[16][SYS_T+1];
__constant__ gf d_tab2[16][SYS_T+1];

__device__ __forceinline__ gf gf_mul_lut(gf x, gf y) {
    // if either operand is zero, product is zero
    if (x == 0 || y == 0) return 0;
    // look up logs, add (mod order), then antilog
    uint16_t lx = d_gf_log[x];
    uint16_t ly = d_gf_log[y];
    uint16_t s  = lx + ly;
    // GFLUT_ORDER == (1<<GFBITS)-1
    if (s >= GFLUT_ORDER) s -= GFLUT_ORDER;
    return d_gf_exp[s];
}



// Device pointers
unsigned char *d_ciphertexts;
gf *d_inverse_elements;

// Host variables
gf images[SYS_N];
gf error[SYS_T];
int tv; // test_vector
unsigned char secretkeys[crypto_kem_SECRETKEYBYTES];
unsigned char ciphertexts[KATNUM][crypto_kem_CIPHERTEXTBYTES];
int e[SYS_N / 8];
int i, w = 0, j, k;
gf g[SYS_T + 1]; // goppa polynomial
gf L[SYS_N];     // support
gf s[SYS_T * 2];
gf e_inv_LOOP_1D[sb * 2 * SYS_T];
gf inverse_elements[sb][2 * SYS_T];
gf temp;
gf e_inv[SYS_N];
unsigned char r[SYS_N / 8];
gf locator[SYS_T + 1]; // error locator
gf t, c[SYS_N];
clock_t start, end;
double avg_cpu_time_used;
double cpu_printing;
double synd_time = 0, bm_time = 0, root_time = 0;
unsigned char *sk = NULL;
int count;
unsigned char h_error[KATNUM][SYS_N];

// Device functions
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


// branch-free multiply via log/exp in GF(2^12)
// __device__ __forceinline__ gf mul(gf a, gf b) {
//     uint16_t la = d_gf_log[a];
//     uint16_t lb = d_gf_log[b];
//     // la==GFLUT_ORDER  or lb==GFLUT_ORDER  ⇒ la+lb ≥ GFLUT_ORDER ⇒ exp[...] was set to 0
//     return d_gf_exp[ la + lb ];
// }
// in decrypt.h, replace your old mul() with:

__device__ __forceinline__ gf tmul(gf a, gf b) {
    // fetch logs
    uint16_t la = d_gf_log[a];
    uint16_t lb = d_gf_log[b];

    // sum the logs
    uint16_t idx = la + lb;  // in [0 .. 2*ORDER-2]

    // get α^(la+lb) from wrapped exp[]
    gf v = d_gf_exp[idx];

    // if either a==0 or b==0, we must return 0
    // mask = 0xFFFF when a!=0 && b!=0, else 0
    uint16_t nz = (uint16_t)((a!=0) & (b!=0));
    gf mask = (gf)(- (int)nz);

    return v & mask;
}


__device__ __forceinline__ gf p_gf_inv(gf in) {
    return gf_inverse_table[in];
}

__device__ __forceinline__ gf p_gf_frac(gf den, gf num) {
    return mul(p_gf_inv(den), num);
}

// -----------------------------------------------------------------------------
// initialization of all constants
// -----------------------------------------------------------------------------
static void InitializeC(int batchSize, const gf *host_locator, gf primitive) {
    // 1) inverse table & L
    {
        const int N = 1<<GFBITS;
        gf host_inv[N];
        host_inv[0]=0;
        for(int i=1;i<N;i++) host_inv[i]=gf_inv((gf)i);
        cudaMemcpyToSymbol(gf_inverse_table, host_inv, sizeof host_inv);
        cudaMemcpyToSymbol(d_L,             L,            SYS_N*sizeof(gf));
    }

    // 2) build & upload log/exp tables
    {
        uint16_t host_log[GFLUT_SIZE];
        uint16_t host_exp[GFLUT_ORDER];
        for(int i=0;i<GFLUT_SIZE;i++){
            host_log[i] = gf_log((gf)i);
            if(i<GFLUT_ORDER) host_exp[i] = gf_exp(i);
        }
        cudaMemcpyToSymbol(d_gf_log, host_log, sizeof host_log);
        cudaMemcpyToSymbol(d_gf_exp, host_exp, sizeof host_exp);
    }

    // 3) build nibble-tables from host_locator[ batchSize ][ SYS_T+1 ]
    {
        static gf host_tab0[16][SYS_T+1],
                  host_tab1[16][SYS_T+1],
                  host_tab2[16][SYS_T+1];
        for(int cw=0; cw<batchSize; cw++){
            const gf *σ = host_locator + cw*(SYS_T+1);
            for(int nib=0; nib<16; nib++){
                gf x0 = gf_exp(nib);
                gf x1 = gf_exp(nib<<4);
                gf x2 = gf_exp(nib<<8);
                host_tab0[nib][0] = σ[SYS_T];
                host_tab1[nib][0] = σ[SYS_T];
                host_tab2[nib][0] = σ[SYS_T];
                for(int j=1;j<=SYS_T;j++){
                    host_tab0[nib][j] = tmul(host_tab0[nib][j-1], x0) ^ σ[SYS_T-j];
                    host_tab1[nib][j] = tmul(host_tab1[nib][j-1], x1) ^ σ[SYS_T-j];
                    host_tab2[nib][j] = tmul(host_tab2[nib][j-1], x2) ^ σ[SYS_T-j];
                }
            }
        }
        cudaMemcpyToSymbol(d_tab0, host_tab0, sizeof host_tab0);
        cudaMemcpyToSymbol(d_tab1, host_tab1, sizeof host_tab1);
        cudaMemcpyToSymbol(d_tab2, host_tab2, sizeof host_tab2);
    }
}
// // Host initialization function
// void InitializeC() {

//     // Prepare inverse table
//     const int N = 1 << GFBITS;
//     gf host_inv[N];
//     host_inv[0] = 0;
//     for (int i = 1; i < N; i++)
//         host_inv[i] = gf_inv((gf)i);

//     // Upload to constant memory
//     cudaMemcpyToSymbol(gf_inverse_table, host_inv, sizeof(host_inv), 0, cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(d_L, L, SYS_N * sizeof(gf), 0, cudaMemcpyHostToDevice);
//     // Allocate and copy ciphertexts (if necessary)
//     cudaMalloc(&d_ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM);
//     // Setup global memory for inverse elements
//     size_t size = sizeof(gf) * sb * 2 * SYS_T;
//     cudaMalloc(&d_inverse_elements, size);
//     cudaMemcpy(d_inverse_elements, inverse_elements, size, cudaMemcpyHostToDevice);
//         // build & upload 12-bit log/exp tables

//         // For each codeword index cw=0..KATNUM-1:
// for(int cw=0; cw<batchSize; cw++){
//   gf *σ = host_locator + cw*(SYS_T+1);
//   // fill every nibble-table
//   for(int nib=0; nib<16; nib++){
//     // tab0[nib][j] = evaluate σ at α^(nib) masked to low 4 bits
//     gf x0 = gf_pow(primitive, nib);       // α^nib
//     gf x1 = gf_pow(primitive, nib<<4);    // α^(nib<<4)
//     gf x2 = gf_pow(primitive, nib<<8);    // α^(nib<<8)
//     // Horner at those three points:
//     d_tab0[nib][0] = σ[SYS_T];
//     d_tab1[nib][0] = σ[SYS_T];
//     d_tab2[nib][0] = σ[SYS_T];
//     for(int j=1;j<=SYS_T;j++){
//       d_tab0[nib][j] = mul(d_tab0[nib][j-1], x0) ^ σ[SYS_T-j];
//       d_tab1[nib][j] = mul(d_tab1[nib][j-1], x1) ^ σ[SYS_T-j];
//       d_tab2[nib][j] = mul(d_tab2[nib][j-1], x2) ^ σ[SYS_T-j];
//     }
//   }
// }
// // then cudaMemcpyToSymbol all three tables once
//     cudaMemcpyToSymbol(d_tab0, host_tab0, sizeof(host_tab0), 0, cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(d_tab1, host_tab1, sizeof(host_tab1), 0, cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(d_tab2, host_tab2, sizeof(host_tab2), 0, cudaMemcpyHostToDevice);

//     // Prepare log/exp tables
//     uint16_t host_log[GFLUT_SIZE];
//     uint16_t host_exp[GFLUT_ORDER];
//     for (int i = 0; i < GFLUT_SIZE; i++) {
//         host_log[i] = gf_log((gf)i);
//         if (i < GFLUT_ORDER) {
//             host_exp[i] = gf_exp(i);
//         }
//     }
//     cudaMemcpyToSymbol(d_gf_log, host_log, sizeof(host_log), 0, cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(d_gf_exp, host_exp, sizeof(host_exp), 0, cudaMemcpyHostToDevice);


// }

#endif // DECRYPT_H
