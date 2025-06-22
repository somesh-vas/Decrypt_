//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1



// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// Tensor-warp syndrome kernel (4 warps = 128 threads/block)
// ------------------------------------------------------------------------
__global__ void computeOut1(
    const gf*  __restrict__ d_inverse_elements,  // [sb][2*SYS_T]
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES per CT
          gf*  __restrict__ out1                  // KATNUM × (2*SYS_T)
) {
    extern __shared__ gf c_tile[];  // bitsPerTile entries

    const int tid            = threadIdx.x;     // 0..127
    const int tileId         = blockIdx.x;      // which bit-tile
    const int ctIdx          = blockIdx.y;      // which ciphertext
    const int threadsPerBlock= blockDim.x;      // must be 128
    const int stride         = 2 * SYS_T;       // number of syndromes
    const int numTiles       = gridDim.x;
    const int bitsPerTile    = (sb + numTiles - 1) / numTiles;
    const int bitStart       = tileId * bitsPerTile;
    const int bitEnd         = min(bitStart + bitsPerTile, sb);

    // 1) unpack bits for this tile
    for (int b = bitStart + tid; b < bitEnd; b += threadsPerBlock) {
        int byte = b >> 3, bit = b & 7;
        unsigned char r = d_ciphertexts[ ctIdx * SYND_BYTES + byte ];
        c_tile[b - bitStart] = (r >> bit) & 1U;
    }
    __syncthreads();

    // 2) four warps each handle 1/4 of the 2*SYS_T coefficients
    int warpId = tid >> 5;  // 0..3
    int lane   = tid & 31;  // 0..31
    if (warpId < 4) {
        int chunk = stride / 4;                  // typically 32
        int base  = warpId * chunk;              // start coeff idx
        for (int coeff = base + lane; coeff < base + chunk; coeff += 32) {
            const gf *col = d_inverse_elements + ctIdx * stride + coeff;
            gf partial = 0;
            for (int i = 0; i < (bitEnd - bitStart); ++i) {
                gf mask = (gf)(- (int)c_tile[i]);      // 0xFFFF if bit==1
                partial ^= (col[(bitStart + i) * stride] & mask);
            }
            // pack into 32-bit word and atomicXor
            int idx        = ctIdx * stride + coeff;
            size_t byteOff = idx * sizeof(gf);       // gf=2 bytes
            size_t aligned = byteOff & ~0x3UL;       // align down to 4
            int    shift   = (byteOff & 0x2) ? 16 : 0;
            unsigned int *wordPtr = (unsigned int*)((char*)out1 + aligned);
            unsigned int  val     = (unsigned int)partial << shift;
            atomicXor(wordPtr, val);
        }
    }
}


// ------------------------------------------------------------------------
// Host function: synd_f() launcher for tensor-warp variant
// ------------------------------------------------------------------------
int synd_f() {
    InitializeC();  // sets up constant memory & globals

    // 1) configure threads & tiles
    const int threadsPerBlock = 128;  
    int dev, smCount;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));

    int tilesPerCt  = max((sb + threadsPerBlock - 1) / threadsPerBlock, smCount);
    int bitsPerTile = (sb + tilesPerCt - 1) / tilesPerCt;
    size_t sharedMem = bitsPerTile * sizeof(gf);

    // 2) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // zero output for atomic XOR accumulation
    CUDA_CHECK(cudaMemset(d_images, 0,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 3) copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 4) timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 5) launch kernel
    dim3 grid(tilesPerCt, KATNUM);
    CUDA_CHECK(cudaEventRecord(start));
    computeOut1<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Tensor-warp syndrome kernel (128 tpb): %f ms\n", ms);

    // 6) copy back & print
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // 7) cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}






int main() {

    
    initialisation(secretkeys,ciphertexts,sk,L,g);	
	
	compute_inverses();

	InitializeC(); // only for test purpose
   


	synd_f();
	// Dec();

    return KAT_SUCCESS;
}


