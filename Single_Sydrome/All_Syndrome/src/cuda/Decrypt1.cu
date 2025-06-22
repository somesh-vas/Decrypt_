//************** header files


#include <decrypt.h>

gf h_out1[2 * SYS_T]; // output array for out1

// src/cuda/Decrypt1.cu
#include "decrypt.h"    // brings in InitializeC(), common definitions, etc.
#include <cuda_runtime.h>

#include <mma.h>
using namespace nvcuda::wmma;


// ------------------------------------------------------------------------
// Block‐reduction syndrome kernel:
//   • one block per (coeff, ciphertext) pair: grid = {2*SYS_T, KATNUM}
//   • 256 threads per block
//   • each thread processes a strided subset of bits, XORs partial sums
//   • warp‐wide __shfl_down_sync reduction, then block‐wide shared‐mem reduction
// ------------------------------------------------------------------------
__global__ void Block_reduction_syndrome_kernel(
    const gf*           __restrict__ d_inverse_elements, // [bit][coeff]
    const unsigned char* __restrict__ d_ciphertexts,     // SYND_BYTES per CT
          gf*           __restrict__ out1                // KATNUM×(2*SYS_T)
) {
    extern __shared__ gf warp_sums[];  // one entry per warp (256/32 = 8 warps)

    const int cf      = blockIdx.x;    // coefficient index [0..2*SYS_T)
    const int ct      = blockIdx.y;    // ciphertext index [0..KATNUM)
    const int lane    = threadIdx.x;   // 0..255
    const int warpId  = lane >> 5;     // 0..7
    const int laneInW = lane & 31;     // 0..31
    const int stride  = 2 * SYS_T;

    // 1) each thread XOR-accumulates its share of sb bits
    gf sum = 0;
    for (int bit = lane; bit < sb; bit += blockDim.x) {
        unsigned char r = d_ciphertexts[ ct * SYND_BYTES + (bit >> 3) ];
        if ((r >> (bit & 7)) & 1U) {
            sum ^= d_inverse_elements[ bit * stride + cf ];
        }
    }

    // 2) warp‐local reduction via shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum ^= __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    // lane 0 of each warp writes its partial into shared memory
    if (laneInW == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // 3) block‐wide reduction of the 8 warp sums (only warp 0 participates)
    if (warpId == 0) {
        if (laneInW < (blockDim.x / 32)) {
            sum = warp_sums[laneInW];
            // accumulate the other warp sums
            for (int w = laneInW + 1; w < (blockDim.x/32); ++w) {
                sum ^= warp_sums[w];
            }
            // lane 0 writes the final syndrome
            if (laneInW == 0) {
                out1[ ct * stride + cf ] = sum;
            }
        }
    }
}
// ------------------------------------------------------------------------
// Host function: synd_f_blockred()
// ------------------------------------------------------------------------
int synd_f() {
    // 1) initialize constants & global memory
    // InitializeC();

    // 2) launch config
    const int threadsPerBlock = 256;
    dim3 grid(2 * SYS_T, KATNUM);  // one block per (coeff, CT)
    // shared mem = one gf per warp
    size_t sharedMem = (threadsPerBlock / 32) * sizeof(gf);

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Block_reduction_syndrome_kernel<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Block‐reduction syndrome kernel (256 tpb): %f ms\n", ms);

    // 6) copy back & print first ciphertext’s syndromes
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
///
// Vec4-padded baseline version of the syndrome kernel
// Kernel name and parameter list unchanged
__global__ void Vec4_padded(gf *d_inverse_elements,
                            unsigned char *d_ciphertexts,
                            gf *out1)
{
    // Shared buffers
    __shared__ gf c[sb];
    __shared__ gf s_out[2 * SYS_T];

    int tid       = threadIdx.x;
    int blockId   = blockIdx.x + blockIdx.y * gridDim.x;
    int globalIdx = blockId * blockDim.x + tid;

    // 1. Unpack ciphertext bits into c[] using coalesced uchar4 (vec4) loads
    //    We assume SYND_BYTES is padded to a multiple of 4 bytes.
    const int vecCount = SYND_BYTES / 4;
    for (int v = tid; v < vecCount; v += blockDim.x) {
        // Read 4 bytes at once
        uchar4 chunk = reinterpret_cast<uchar4*>(d_ciphertexts)[v];

        // Unpack each of the 4 bytes into bits
        unsigned char r0 = chunk.x;
        unsigned char r1 = chunk.y;
        unsigned char r2 = chunk.z;
        unsigned char r3 = chunk.w;

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            int idx0 = (v * 4 + 0) * 8 + b;
            if (idx0 < sb) c[idx0] = (r0 >> b) & 1U;

            int idx1 = (v * 4 + 1) * 8 + b;
            if (idx1 < sb) c[idx1] = (r1 >> b) & 1U;

            int idx2 = (v * 4 + 2) * 8 + b;
            if (idx2 < sb) c[idx2] = (r2 >> b) & 1U;

            int idx3 = (v * 4 + 3) * 8 + b;
            if (idx3 < sb) c[idx3] = (r3 >> b) & 1U;
        }
    }
    __syncthreads();

    // 2. Compute 2T syndromes (exactly as in the baseline)
    if (globalIdx < 2 * SYS_T) {
        const int stride = 2 * SYS_T;
        const gf *col    = d_inverse_elements + globalIdx;
        gf sum = 0;

        #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            gf mask = (gf)(- (int)(c[bit] & 1));  // all-ones if bit is 1, else 0
            sum ^= (col[0] & mask);               // conditional XOR
            col += stride;
        }
        s_out[globalIdx] = sum;
    }
    __syncthreads();

    // 3. Write back to global memory
    if (globalIdx < 2 * SYS_T) {
        out1[globalIdx] = s_out[globalIdx];
    }
    __syncthreads();
}
int synd_Vec4_padded() {
    // 1) initialize constants & global memory
    // InitializeC();

    // 2) launch config
    const int threadsPerBlock = 256;
    dim3 grid(2 * SYS_T, KATNUM);  // one block per (coeff, CT)
    // shared mem = one gf per warp
    size_t sharedMem = (threadsPerBlock / 32) * sizeof(gf);

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Vec4_padded<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Vec4-padded baseline version of the syndrome kernel %f ms\n", ms);

    // 6) copy back & print first ciphertext’s syndromes
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
///
///
// Warp-4 version of the syndrome kernel (32 × 4 threads per ciphertext)
// Kernel name and parameter list unchanged
__global__ void syndrome_Warp4(gf *d_inverse_elements,
                            unsigned char *d_ciphertexts,
                            gf *out1)
{
    // Shared buffers for one ciphertext
    __shared__ gf c[sb];
    __shared__ gf s_out[2 * SYS_T];

    int tid     = threadIdx.x;
    // one block per ciphertext
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    // 1) Unpack bits into c[] (coalesced by byte)
    for (int byte = tid; byte < SYND_BYTES; byte += blockDim.x) {
        unsigned char r = d_ciphertexts[byte];
    #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            int idx = byte * 8 + bit;
            if (idx < sb) c[idx] = (r >> bit) & 1U;
        }
    }
    __syncthreads();

    // 2) Compute syndromes using 4 warps per CT (each warp = 32 threads)
    int warpId = tid >> 5; // which warp within the block
    int laneId = tid & 31; // lane within the warp

    if (warpId < 4) {
        int coeff = warpId * 32 + laneId;  // coefficient index [0..127)
        if (coeff < 2 * SYS_T) {
            const int stride = 2 * SYS_T;
            // point at the column for this (ciphertext, coeff)
            const gf *col = d_inverse_elements + blockId * stride + coeff;
            gf sum = 0;
            // each thread accumulates over all sb bits
            for (int b = 0; b < sb; ++b) {
                gf mask = (gf)(- (int)(c[b] & 1));  // 0xFFFF if bit==1, else 0
                sum ^= (col[0] & mask);
                col  += stride;
            }
            s_out[coeff] = sum;
        }
    }
    __syncthreads();

    // 3) Write back all 2*SYS_T results
    int outIdx = blockId * (2 * SYS_T) + tid;
    if (tid < 2 * SYS_T) {
        out1[outIdx] = s_out[tid];
    }
    __syncthreads();
}
int synd_Warp4() {
    // 1) initialize constants & global memory
    // InitializeC();

    // 2) launch config
    const int threadsPerBlock = 256;
    dim3 grid(2 * SYS_T, KATNUM);  // one block per (coeff, CT)
    // shared mem = one gf per warp
    size_t sharedMem = (threadsPerBlock / 32) * sizeof(gf);

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    syndrome_Warp4<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Warp-4 version of the syndrome kernel (32 × 4 threads per ciphertext) %f ms\n", ms);

    // 6) copy back & print first ciphertext’s syndromes
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
///
///


// ------------------------------------------------------------------------
// tiled (shared-padding) syndrome kernel
//   • 1 block per ciphertext (gridDim = {1, KATNUM})
//   • 256 threads per block
//   • shared memory: sb + 2*SYS_T elements of gf
// ------------------------------------------------------------------------
__global__ void syndrome_shared_padding(
    const gf*  __restrict__ d_inverse_elements,  // [sb][2*SYS_T]
    const unsigned char* __restrict__ d_ciphertexts, // compact bytes, length = SYND_BYTES
          gf*  __restrict__ out1                  // output syndromes: KATNUM × (2*SYS_T)
) {
    extern __shared__ gf shared[];
    gf* c     = shared;                 // [0 .. sb-1]
    gf* s_out = shared + sb;            // [sb .. sb + 2*SYS_T - 1]

    int tid    = threadIdx.x;
    int ctIdx  = blockIdx.y;            // which ciphertext

    // 1) unpack bits into c[]
    //    SYND_BYTES = (sb + 7)/8
    for (int byte = tid; byte < SYND_BYTES; byte += blockDim.x) {
        unsigned char r = d_ciphertexts[ ctIdx * SYND_BYTES + byte ];
    #pragma unroll
        for (int b = 0; b < 8; ++b) {
            int idx = byte * 8 + b;
            if (idx < sb) c[idx] = (r >> b) & 1;
        }
    }
    __syncthreads();

    // 2) each of the first 2*SYS_T threads computes one syndrome coefficient
    if (tid < 2 * SYS_T) {
        const int stride = 2 * SYS_T;
        const gf* col0   = d_inverse_elements + tid; 
        gf sum = 0;

    #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            // mask = 0xFFFF if c[bit]==1, else 0
            gf mask = (gf)(- (int)c[bit]);
            sum ^= (col0[bit * stride] & mask);
        }
        s_out[tid] = sum;
    }
    __syncthreads();

    // 3) write back
    if (tid < 2 * SYS_T) {
        out1[ ctIdx * (2*SYS_T) + tid ] = s_out[tid];
    }
}


// ------------------------------------------------------------------------
// synd_f(): host launcher for the tiled kernel above
// ------------------------------------------------------------------------
int synd_syndrome_shared_padding() {


    // 2) threads/blocks
    const int threadsPerBlock = 256;
    dim3    grid(1, KATNUM);   // one block per ciphertext

    // 3) shared‐mem size = (sb + 2*SYS_T) × sizeof(gf)
    size_t sharedMem = (size_t)(sb + 2*SYS_T) * sizeof(gf);

    // 4) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;

    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 5) copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 6) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    syndrome_shared_padding<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Tiled (shared-padding) syndrome kernel: %f ms\n", ms);

    // 7) copy back & print first ciphertext’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; i++) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // 8) cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

///
///
// src/cuda/Decrypt1.cu
#include "decrypt.h"
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// Tensor-warp syndrome kernel (4 warps = 128 threads/block)
// ------------------------------------------------------------------------
__global__ void Tensor_warp_syndrome(
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
int synd_Tensor_warp_syndrome() {
   
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
    Tensor_warp_syndrome<<<grid, threadsPerBlock, sharedMem>>>(
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

///
///
// src/cuda/Decrypt1.cu
#include "decrypt.h"
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// Multi-2 syndrome kernel: 2 ciphertexts per block, 256 threads/block
// – unpack both ciphers into shared memory
// – each block handles ctIdx0 = 2*blockIdx.y, ctIdx1 = ctIdx0+1
// – threads 0..127 compute 128 coeffs for ct0, 128..255 compute for ct1
// ------------------------------------------------------------------------
__global__ void Multi_2_syndrome(
    const gf*  __restrict__ d_inverse_elements,  // [sb][2*SYS_T] per CT
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES per CT
          gf*  __restrict__ out1                  // KATNUM × (2*SYS_T)
) {
    extern __shared__ gf shared[];
    gf* c0    = shared;                        // sb elements
    gf* c1    = c0 + sb;                       // sb elements
    gf* s0    = c1 + sb;                       // 2*SYS_T elements
    gf* s1    = s0 + 2*SYS_T;                  // 2*SYS_T elements

    int tid    = threadIdx.x;                  // 0..255
    int ctPair = blockIdx.y;                   // one block per pair
    int ct0    = ctPair * 2;
    int ct1    = ct0 + 1;
    const int stride = 2 * SYS_T;
    const int bytes  = SYND_BYTES;             // (sb+7)/8

    // 1) unpack both ciphertexts’ bits into shared memory
    for (int bIdx = tid; bIdx < bytes; bIdx += blockDim.x) {
        unsigned char r0 = d_ciphertexts[ct0 * bytes + bIdx];
        unsigned char r1 = d_ciphertexts[ct1 * bytes + bIdx];
    #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            int idx = bIdx * 8 + bit;
            if (idx < sb) {
                c0[idx] = (r0 >> bit) & 1U;
                c1[idx] = (r1 >> bit) & 1U;
            }
        }
    }
    __syncthreads();

    // 2) compute syndrome coefficients
    //    threads 0..stride-1 → s0[0..stride), threads stride..2*stride-1 → s1[0..stride)
    if (tid < 2 * stride) {
        bool isFirst = tid < stride;
        int  coeff   = isFirst ? tid : (tid - stride);
        const gf *colBase = d_inverse_elements + (isFirst ? ct0 : ct1) * stride;
        gf sum = 0;

    #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            gf mask = (gf)(- (int)( isFirst ? c0[bit] : c1[bit] ));
            sum ^= (colBase[ coeff + bit*stride ] & mask);
        }
        if (isFirst) s0[coeff] = sum;
        else          s1[coeff] = sum;
    }
    __syncthreads();

    // 3) write back both sets of 2*SYS_T syndromes
    if (tid < stride) {
        out1[ct0 * stride + tid] = s0[tid];
        out1[ct1 * stride + tid] = s1[tid];
    }
}

// ------------------------------------------------------------------------
// Host function: synd_f() launcher for multi-2 variant
// ------------------------------------------------------------------------
int synd_Multi_2_syndrome() {


    const int threadsPerBlock = 256;
    int dev, smCount;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));

    // one block per 2 ciphertexts
    int pairs = (KATNUM + 1) / 2;
    dim3 grid(1, pairs);

    // shared memory: 2*sb + 2*(2*SYS_T) elements
    size_t sharedMem = (size_t)(2*sb + 4*SYS_T) * sizeof(gf);

    // allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Multi_2_syndrome<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Multi-2 syndrome kernel (2 cts/block): %f ms\n", ms);

    // copy back & print first CT’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; i++) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

///
///


// ------------------------------------------------------------------------
// Vec4-padded + log/antilog-mul syndrome kernel
//   • 256 threads/block, grid = (1, KATNUM)
//   • packs d_ciphertexts into uchar4, unpacks into bits[],
//     then uses mul() for GF-multiply instead of bit-mask
// ------------------------------------------------------------------------
__global__ void Vec4_padded_log_antilog_mul_syndrome(
    const gf*  __restrict__ d_inverse_elements,    // [sb][2*SYS_T] per CT
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES × KATNUM
          gf*  __restrict__ out1                    // KATNUM × (2*SYS_T)
) {
    __shared__ gf bits[sb];
    __shared__ gf s_out[2 * SYS_T];

    const int tid   = threadIdx.x;     // 0..255
    const int ctIdx = blockIdx.y;      // which ciphertext [0..KATNUM)
    const int stride = 2 * SYS_T;

    // 1) coalesced vec4 loads + unpack to bits[]
    const int vecCount = SYND_BYTES / 4;
    const uchar4* ct4 = reinterpret_cast<const uchar4*>(
        d_ciphertexts + ctIdx * SYND_BYTES
    );

    for (int v = tid; v < vecCount; v += blockDim.x) {
        uchar4 chunk = ct4[v];
        unsigned char b0 = chunk.x,
                      b1 = chunk.y,
                      b2 = chunk.z,
                      b3 = chunk.w;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx0 = v*4*8 + i;
            int idx1 = v*4*8 + 8 + i;
            int idx2 = v*4*8 + 16 + i;
            int idx3 = v*4*8 + 24 + i;
            if (idx0 < sb) bits[idx0] = (b0 >> i) & 1U;
            if (idx1 < sb) bits[idx1] = (b1 >> i) & 1U;
            if (idx2 < sb) bits[idx2] = (b2 >> i) & 1U;
            if (idx3 < sb) bits[idx3] = (b3 >> i) & 1U;
        }
    }
    __syncthreads();

    // 2) each of the first 2*SYS_T threads computes one syndrome via mul()
    if (tid < stride) {
        const gf* col0 = d_inverse_elements + tid;
        gf sum = 0;
        #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            // multiply col0[ bit*stride ] × bits[bit]
            sum = add(sum, mul(col0[bit * stride], bits[bit]));
        }
        s_out[tid] = sum;
    }
    __syncthreads();

    // 3) write back
    if (tid < stride) {
        out1[ ctIdx * stride + tid ] = s_out[tid];
    }
}

// ------------------------------------------------------------------------
// synd_f(): host launcher for the vec4+log/antilog-mul variant
// ------------------------------------------------------------------------
int synd_Vec4_padded_log_antilog_mul_syndrome() {
    // 1) init constants & global memory
    

    // 2) launch config
    const int threadsPerBlock = 256;
    dim3 grid(1, KATNUM);

    // 3) allocate + copy
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_images, 0,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Vec4_padded_log_antilog_mul_syndrome<<<grid, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Vec4+padded+log/antilog-mul kernel: %f ms\n", ms);

    // 5) copy back & print
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // 6) cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

///
///


// ------------------------------------------------------------------------
// Tensor-Core (WMMA) syndrome kernel
//  • processes one ciphertext per block (grid = {1, KATNUM})
//  • 128 threads/block (4 warps), each warp computes 16 coeffs at a time
//  • tiles the sb‐length “bits” vector into 16‐wide chunks
//  • uses FP16 fragments to accumulate inner‐products via tensor cores
// ------------------------------------------------------------------------
__global__ void Tensor_Core_WMMA_syndrome(
    const gf*           __restrict__ d_inverse_elements, // [KATNUM][2*SYS_T][sb]
    const unsigned char* __restrict__ d_ciphertexts,      // [KATNUM][SYND_BYTES]
          gf*           __restrict__ out1                 // [KATNUM][2*SYS_T]
) {
    extern __shared__ half smem[];       // shared workspace
    half* bits    = smem;                // sb entries of half
    half* invTile = bits + sb;           // 16×16 tile per warp = 256 entries

    int tid    = threadIdx.x;            // 0..127
    int warpId = tid >> 5;               // 0..3
    int lane   = tid & 31;               // 0..31
    int ctIdx  = blockIdx.y;             // which ciphertext
    const int stride   = 2 * SYS_T;      // # coeffs
    const int numTiles = (sb + 15) / 16; // bit-tiles of width 16

    // 1) unpack the bits into shared fp16 “bits[]”
    for (int i = tid; i < sb; i += blockDim.x) {
        unsigned char byte = d_ciphertexts[ctIdx * SYND_BYTES + (i >> 3)];
        bits[i] = __float2half(float((byte >> (i & 7)) & 1));
    }
    __syncthreads();

    // 2) each warp computes 16 consecutive coefficients
    int coeffBase = warpId * 16;
    fragment<matrix_a,16,16,16,half,row_major>   aFrag;
    fragment<matrix_b,16,16,16,half,col_major>   bFrag;
    fragment<accumulator,16,16,16,float>        cFrag;
    fill_fragment(cFrag, 0.0f);

    // loop over each 16-bit tile of the “bits” vector
    for (int tile = 0; tile < numTiles; ++tile) {
        int bitOffset = tile * 16;
        // load 16-wide subvector into bFrag
        load_matrix_sync(bFrag, bits + bitOffset, 16);

        // build a 16×16 tile of d_inverse_elements for this warp
        // row i = coefficient (coeffBase + i), col j = bitOffset + j
        for (int i = 0; i < 16; ++i) {
            int coeff = coeffBase + i;
            for (int j = 0; j < 16; ++j) {
                int bitIdx = bitOffset + j;
                half v = 0;
                if (bitIdx < sb && coeff < stride) {
                    // gather from global: layout [ctIdx][coeff][bitIdx]
                    v = __float2half(float(
                        d_inverse_elements[
                            ctIdx * stride * sb +
                            coeff * sb +
                            bitIdx
                        ]
                    ));
                }
                invTile[i * 16 + j] = v;
            }
        }
        // load that tile into aFrag
        load_matrix_sync(aFrag, invTile, 16);
        // tensor‐core multiply-accumulate: cFrag += aFrag × bFrag
        mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // 3) write back results: each lane writes one of the 16 outputs
    if (lane < 16 && coeffBase + lane < stride) {
        float sum = cFrag.x[lane];
        // convert float back to integer GF element (mod 2^GFBITS)
        uint32_t val = __float2int_rn(sum) & ((1u << GFBITS) - 1);
        out1[ctIdx * stride + (coeffBase + lane)] = (gf)val;
    }
}


// ------------------------------------------------------------------------
// Host launcher: synd_f_tensor()
// ------------------------------------------------------------------------
int synd_f_tensor() {
   

    const int threadsPerBlock = 128;            // 4 warps
    dim3 grid(1, KATNUM);                       // one block per ciphertext
    int dev, smCount;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));

    // shared memory = sb fp16 bits + 16×16 fp16 tile
    size_t sharedMem = (size_t)sb * sizeof(half) + 16 * 16 * sizeof(half);

    // allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        2 * SYS_T * sb * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        2 * SYS_T * sb * sizeof(gf), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_images, 0,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Tensor_Core_WMMA_syndrome<<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("WMMA (tensor-core) syndrome kernel: %f ms\n", ms);

    // copy back & print
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

///
///
// ------------------------------------------------------------------------
// Warp‐shuffle syndrome kernel (32 threads = 1 warp per coefficient)
//  • grid = {2*SYS_T, KATNUM}, blockDim = 32
//  • each warp computes one syndrome coefficient cf ∈ [0, 2*SYS_T)
//  • bit‐vector of length sb is reduced via lane‐strided loads + __shfl_xor_sync
// ------------------------------------------------------------------------
__global__ void Warp_shuffle_syndrome_kernel(
    const gf* __restrict__ d_inverse_elements, 
    const unsigned char* __restrict__ d_ciphertexts,
          gf* __restrict__ out1
) {
    const int cf    = blockIdx.x;      // which syndrome coefficient
    const int ct   = blockIdx.y;      // which ciphertext index
    const int lane = threadIdx.x;      // 0..31
    const int stride = 2 * SYS_T;

    // accumulate partial XOR of those bits where c[bit]==1
    gf sum = 0;
    for (int bit = lane; bit < sb; bit += 32) {
        unsigned char r = d_ciphertexts[ ct * SYND_BYTES + (bit >> 3) ];
        if ((r >> (bit & 7)) & 1U) {
            // original column‐major: inv[bit][cf] at d_inverse_elements[ bit*stride + cf ]
            sum ^= d_inverse_elements[ bit * stride + cf ];
        }
    }

    // warp‐wide XOR reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum ^= __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // lane 0 writes the final result
    if (lane == 0) {
        out1[ ct * stride + cf ] = sum;
    }
}


// ------------------------------------------------------------------------
// Host launcher: synd_f_warpshuffle()
// ------------------------------------------------------------------------
int synd_f_warpshuffle() {
   
    // 2) launch configuration
    const int threadsPerBlock = 32;  // one warp
    dim3 grid(2 * SYS_T, KATNUM);

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Warp_shuffle_syndrome_kernel<<<grid, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Warp‐shuffle syndrome kernel (32 tpb): %f ms\n", ms);

    // 6) copy back & print first ciphertext’s 2*SYS_T syndromes
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
///
///

// src/cuda/Decrypt1.cu
#include "decrypt.h"
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// Block‐reduction syndrome kernel:
//   • one block per (coeff, ciphertext) pair: grid = {2*SYS_T, KATNUM}
//   • 256 threads per block
//   • each thread processes a strided subset of bits, XORs partial sums
//   • warp‐wide __shfl_down_sync reduction, then block‐wide shared‐mem reduction
// ------------------------------------------------------------------------
__global__ void Block_reduction_syndrome (
    const gf*           __restrict__ d_inverse_elements, // [bit][coeff]
    const unsigned char* __restrict__ d_ciphertexts,     // SYND_BYTES per CT
          gf*           __restrict__ out1                // KATNUM×(2*SYS_T)
) {
    extern __shared__ gf warp_sums[];  // one entry per warp (256/32 = 8 warps)

    const int cf      = blockIdx.x;    // coefficient index [0..2*SYS_T)
    const int ct      = blockIdx.y;    // ciphertext index [0..KATNUM)
    const int lane    = threadIdx.x;   // 0..255
    const int warpId  = lane >> 5;     // 0..7
    const int laneInW = lane & 31;     // 0..31
    const int stride  = 2 * SYS_T;

    // 1) each thread XOR-accumulates its share of sb bits
    gf sum = 0;
    for (int bit = lane; bit < sb; bit += blockDim.x) {
        unsigned char r = d_ciphertexts[ ct * SYND_BYTES + (bit >> 3) ];
        if ((r >> (bit & 7)) & 1U) {
            sum ^= d_inverse_elements[ bit * stride + cf ];
        }
    }

    // 2) warp‐local reduction via shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum ^= __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    // lane 0 of each warp writes its partial into shared memory
    if (laneInW == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // 3) block‐wide reduction of the 8 warp sums (only warp 0 participates)
    if (warpId == 0) {
        if (laneInW < (blockDim.x / 32)) {
            sum = warp_sums[laneInW];
            // accumulate the other warp sums
            for (int w = laneInW + 1; w < (blockDim.x/32); ++w) {
                sum ^= warp_sums[w];
            }
            // lane 0 writes the final syndrome
            if (laneInW == 0) {
                out1[ ct * stride + cf ] = sum;
            }
        }
    }
}


// ------------------------------------------------------------------------
// Host function: synd_f_blockred()
// ------------------------------------------------------------------------
int synd_f_blockred() {


    // 2) launch config
    const int threadsPerBlock = 256;
    dim3 grid(2 * SYS_T, KATNUM);  // one block per (coeff, CT)
    // shared mem = one gf per warp
    size_t sharedMem = (threadsPerBlock / 32) * sizeof(gf);

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Block_reduction_syndrome <<<grid, threadsPerBlock, sharedMem>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Block‐reduction syndrome kernel (256 tpb): %f ms\n", ms);

    // 6) copy back & print first ciphertext’s syndromes
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


///
///

// ------------------------------------------------------------------------
// “u64-padded” syndrome kernel
//  • 1 block per ciphertext (grid = {1, KATNUM})
//  • 256 threads/block
//  • uses coalesced uint64_t loads to unpack 64 bits at a time
// ------------------------------------------------------------------------
__global__ void computeOut1_u64(
    const gf*  __restrict__ d_inverse_elements,  // [sb][2*SYS_T] per CT
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES per CT
          gf*  __restrict__ out1                  // KATNUM × (2*SYS_T)
) {
    __shared__ gf c[sb];
    __shared__ gf s_out[2 * SYS_T];

    const int tid   = threadIdx.x;      // 0..255
    const int ctIdx = blockIdx.y;       // which ciphertext
    const int stride = 2 * SYS_T;
    const int bytes  = SYND_BYTES;
    const int u64Count = (bytes + 7) / 8;
    // pointer to this CT’s ciphertext as 64-bit words
    const uint64_t* ct64 = reinterpret_cast<const uint64_t*>(
        d_ciphertexts + ctIdx * bytes
    );

    // 1) unpack bits into shared c[] via 64-bit loads
    for (int v = tid; v < u64Count; v += blockDim.x) {
        uint64_t chunk = ct64[v];
        #pragma unroll
        for (int b = 0; b < 64; ++b) {
            int idx = v * 64 + b;
            if (idx < sb) {
                c[idx] = (chunk >> b) & 1U;
            }
        }
    }
    __syncthreads();

    // 2) each of the first 2*SYS_T threads computes one syndrome
    if (tid < stride) {
        const gf* col0 = d_inverse_elements + tid;
        gf sum = 0;
    #pragma unroll 8
        for (int bit = 0; bit < sb; ++bit) {
            // mask=0xFFFF if c[bit]==1 else 0
            gf mask = (gf)(- (int)c[bit]);
            sum ^= (col0[bit * stride] & mask);
        }
        s_out[tid] = sum;
    }
    __syncthreads();

    // 3) write back
    if (tid < stride) {
        out1[ ctIdx * stride + tid ] = s_out[tid];
    }
}


// ------------------------------------------------------------------------
// Host function: synd_f_u64()
// ------------------------------------------------------------------------
int synd_f_u64() {

    // 2) launch parameters
    const int threadsPerBlock = 256;
    dim3 grid(1, KATNUM);    // one block per ciphertext

    // 3) allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // zero output buffer
    CUDA_CHECK(cudaMemset(d_images, 0,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // 4) copy inputs up
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // 5) time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1_u64<<<grid, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("u64-padded syndrome kernel: %f ms\n", ms);

    // 6) copy back & print first CT’s syndromes
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
///
///
// src/cuda/Decrypt1.cu
#include "decrypt.h"
#include <cuda_runtime.h>

// ------------------------------------------------------------------------
// Grid-stride syndrome kernel (flat 1D): no shared mem or atomics
//  • gridDim.x = ceil(KATNUM*2*SYS_T / threadsPerBlock)
//  • each thread handles multiple (ct,cf) pairs in a 1D loop
//  • uses __ldg for read-only cache on inverse elements
// ------------------------------------------------------------------------
__global__ void computeOut1_flat(
    const gf*  __restrict__ d_inverse_elements,  // layout: bit-major [bit*stride + cf]
    const unsigned char* __restrict__ d_ciphertexts,
          gf*  __restrict__ out1
) {
    const int threadsPerBlock = blockDim.x;
    const int stride         = 2 * SYS_T;
    const int SYNC_BYTES     = SYND_BYTES;       // = (sb+7)/8
    const int totalItems     = KATNUM * stride;

    // Global 1D index across all (ciphertext, coefficient) pairs
    int idx = blockIdx.x * threadsPerBlock + threadIdx.x;
    int gridSize = gridDim.x * threadsPerBlock;

    while (idx < totalItems) {
        int ctIdx = idx / stride;    // which ciphertext
        int cf    = idx % stride;    // which syndrome coefficient

        // Compute the XOR-sum over all sb bits
        gf sum = 0;
        int baseCipher = ctIdx * SYNC_BYTES;
        const gf* invBase = d_inverse_elements + cf;  // we'll step by stride

        for (int bit = 0; bit < sb; ++bit) {
            unsigned char byte = d_ciphertexts[ baseCipher + (bit >> 3) ];
            if ((byte >> (bit & 7)) & 1U) {
                // use __ldg to hit the read-only cache
                sum ^= __ldg(&invBase[ bit * stride ]);
            }
        }

        // Write result
        out1[idx] = sum;

        idx += gridSize;
    }
}


// ------------------------------------------------------------------------
// Host function: synd_f_flat()
// ------------------------------------------------------------------------
int synd_f_flat() {
    // 1D launch parameters
    const int threadsPerBlock = 256;
    int totalItems = KATNUM * 2 * SYS_T;
    int blocks = (totalItems + threadsPerBlock - 1) / threadsPerBlock;

    // allocate device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    // time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1_flat<<<blocks, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms=0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Grid-stride syndrome kernel (1D, %d blocks): %f ms\n", blocks, ms);

    // copy back & print first ciphertext’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

///
///


// ------------------------------------------------------------------------
// Thread-per-ciphertext syndrome kernel:
//   • each CUDA thread handles one ciphertext end-to-end
//   • blockDim = 64, gridDim = ceil(KATNUM/64)
//   • no shared memory, no atomics—each thread writes its own 2*SYS_T outputs
// ------------------------------------------------------------------------
__global__ void Thread_per_ciphertext_syndrome(
    const gf*  __restrict__ d_inverse_elements,  // [bit][coeff]
    const unsigned char* __restrict__ d_ciphertexts, // SYND_BYTES per CT
          gf*  __restrict__ out1                  // KATNUM × (2*SYS_T)
) {
    int ctIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ctIdx >= KATNUM) return;

    // accumulator registers
    gf reg[2 * SYS_T];
    #pragma unroll
    for (int j = 0; j < 2 * SYS_T; ++j) {
        reg[j] = 0;
    }

    // scan each bit of ciphertext
    for (int byte = 0; byte < SYND_BYTES; ++byte) {
        unsigned char r = d_ciphertexts[ctIdx * SYND_BYTES + byte];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if ((r >> b) & 1U) {
                int bitIdx = byte * 8 + b;
                if (bitIdx < sb) {
                    const gf* col = d_inverse_elements + bitIdx * (2 * SYS_T);
                    #pragma unroll
                    for (int j = 0; j < 2 * SYS_T; ++j) {
                        reg[j] ^= col[j];
                    }
                }
            }
        }
    }

    // write back full syndrome vector for this ciphertext
    int base = ctIdx * (2 * SYS_T);
    #pragma unroll
    for (int j = 0; j < 2 * SYS_T; ++j) {
        out1[base + j] = reg[j];
    }
}


// ------------------------------------------------------------------------
// synd_f(): host launcher for Thread-per-ciphertext variant
// ------------------------------------------------------------------------
int synd_Thread_per_ciphertext_syndrome() {
   
    const int threadsPerBlock = 64;
    int blocks = (KATNUM + threadsPerBlock - 1) / threadsPerBlock;

    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Thread_per_ciphertext_syndrome<<<blocks, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Thread-per-ciphertext syndrome kernel (64 tpb): %f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

///
///
// ------------------------------------------------------------------------
// “Sparse‐bit” grid‐stride kernel:
//   • one thread per (ct,cf) pair in a 1D grid
//   • each thread walks only the set bits of its ciphertext, using __ffs
//   • uses __ldg() for read‐only caching of inverse elements
// ------------------------------------------------------------------------

__global__ void computeOut1_sparse(
    const gf*  __restrict__ d_inverse_elements,  // layout: bit-major [bit*stride + cf]
    const unsigned char* __restrict__ d_ciphertexts,
          gf*  __restrict__ out1                  // length = KATNUM * (2*SYS_T)
) {
    const int stride     = 2 * SYS_T;
    const int SYNC_BYTES = SYND_BYTES;          // (sb+7)/8
    const int totalItems = KATNUM * stride;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    while (idx < totalItems) {
        int ctIdx = idx / stride;    // ciphertext index
        int cf    = idx % stride;    // syndrome coefficient

        // walk set bits only
        gf sum = 0;
        const unsigned char* ctBase = d_ciphertexts + ctIdx * SYNC_BYTES;
        for (int word = 0; word < (sb + 31)/32; ++word) {
            // load 32 bits of the ciphertext
            uint32_t bits = *(const uint32_t*)(ctBase + word*4);
            // for each set bit in 'bits', XOR the corresponding inverse element
            while (bits) {
                int b = __ffs(bits) - 1; 
                bits &= bits - 1;
                int bitIdx = word*32 + b;
                if (bitIdx < sb) {
                    sum ^= __ldg(&d_inverse_elements[ bitIdx*stride + cf ]);
                }
            }
        }

        out1[idx] = sum;
        idx += step;
    }
}



// ------------------------------------------------------------------------
// Host launcher: synd_f_sparse()
// ------------------------------------------------------------------------
int synd_f_sparse() {
   

    // 1D grid‐stride launch
    const int threadsPerBlock = 256;
    int totalItems = KATNUM * 2 * SYS_T;
    int blocks     = (totalItems + threadsPerBlock - 1) / threadsPerBlock;

    // device buffers
    unsigned char *d_ciphertexts;
    gf            *d_inverse_elements, *d_images;
    CUDA_CHECK(cudaMalloc(&d_ciphertexts,
        SYND_BYTES * KATNUM));
    CUDA_CHECK(cudaMalloc(&d_inverse_elements,
        sb * 2 * SYS_T * sizeof(gf)));
    CUDA_CHECK(cudaMalloc(&d_images,
        KATNUM * 2 * SYS_T * sizeof(gf)));

    // copy inputs
    CUDA_CHECK(cudaMemcpy(d_ciphertexts, ciphertexts,
        SYND_BYTES * KATNUM,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse_elements, inverse_elements,
        sb * 2 * SYS_T * sizeof(gf),
        cudaMemcpyHostToDevice));

    // time & launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    computeOut1_sparse<<<blocks, threadsPerBlock>>>(
        d_inverse_elements,
        d_ciphertexts,
        d_images
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Sparse‐bit grid‐stride kernel: %f ms\n", ms);

    // copy back & print first ciphertext’s syndromes
    CUDA_CHECK(cudaMemcpy(h_out1, d_images,
        2 * SYS_T * sizeof(gf),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2 * SYS_T; ++i) {
        printf("%04x ", h_out1[i]);
    }
    printf("\n");

    // cleanup
    CUDA_CHECK(cudaFree(d_ciphertexts));
    CUDA_CHECK(cudaFree(d_inverse_elements));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
///

int main() {

    
    initialisation(secretkeys,ciphertexts,sk,L,g);	
	
	compute_inverses();

	InitializeC(); // only for test purpose
   


	synd_f();
    synd_Vec4_padded();
    synd_Warp4(); 
    synd_syndrome_shared_padding();
    synd_Tensor_warp_syndrome();
    synd_Multi_2_syndrome();    
    synd_Vec4_padded_log_antilog_mul_syndrome();
    synd_f_tensor();
    synd_f_warpshuffle();
    synd_f_blockred();
    synd_f_u64();
    synd_f_flat();
    synd_Thread_per_ciphertext_syndrome();
    synd_f_sparse();


    


    return KAT_SUCCESS;
}


