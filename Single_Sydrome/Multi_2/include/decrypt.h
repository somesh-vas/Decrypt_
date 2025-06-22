#ifndef DECRYPT_H
#define DECRYPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gf.h"
#include "common.h"
#include "root.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Device constants
__constant__ uint16_t mne;
__constant__ uint16_t mle[2 * SYS_T];
__constant__ gf d_L[SYS_N];
__constant__ gf gf_inverse_table[1 << GFBITS];

// Device pointers
unsigned char *d_ciphertexts;
__restrict__ gf *d_inverse_elements;

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

#define GF_POLY_MOD ((1 << GFBITS) - 1)
#define POLY_MOD    ((1 << GFBITS) - 1)

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

__device__ __forceinline__ gf p_gf_inv(gf in) {
    return gf_inverse_table[in];
}

__device__ gf p_gf_frac(gf den, gf num) {
    return mul(p_gf_inv(den), num);
}

__device__ gf d_gf_iszero(gf a) {
    uint32_t t = a;
    t -= 1;
    t >>= 19;
    return (gf)t;
}

// Host initialization function
void InitializeC() {
    // Define host-side parameters
    uint16_t h_mne = 0xFFFFu;
    uint16_t h_mle[2 * SYS_T];
    for (int i = 0; i < 2 * SYS_T; i++)
        h_mle[i] = (i & 1) ? 0 : 0xFFFFu;

    const int N = 1 << GFBITS;
    gf host_inv[N];
    host_inv[0] = 0;
    for (int i = 1; i < N; i++)
        host_inv[i] = gf_inv((gf)i);

    // Upload to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(gf_inverse_table, host_inv, sizeof(host_inv)));
    CUDA_CHECK(cudaMemcpyToSymbol(mle, h_mle, sizeof(h_mle)));
    CUDA_CHECK(cudaMemcpyToSymbol(mne, &h_mne, sizeof(h_mne)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_L, L, SYS_N * sizeof(gf)));

    // Allocate and copy ciphertexts (if necessary)
    CUDA_CHECK(cudaMalloc(&d_ciphertexts, crypto_kem_CIPHERTEXTBYTES * KATNUM));

    // Setup global memory for inverse elements
    size_t size = sizeof(gf) * sb * 2 * SYS_T;
    CUDA_CHECK(cudaMalloc(&d_inverse_elements, size));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements, size, cudaMemcpyHostToDevice));
}

#endif // DECRYPT_H
