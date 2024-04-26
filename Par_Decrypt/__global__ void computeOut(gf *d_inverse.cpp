__global__ void computeOut(gf *d_inverse_elements, unsigned char *d_ciphertexts,gf *out1)
{
    __shared__ gf c[sb];
	__shared__ gf s_out[2 * SYS_T];
	__shared__ uint16_t results[SYS_N];
	__shared__ uint16_t T[SYS_T + 1];
    __shared__ uint16_t C[SYS_T + 1];
    __shared__ uint16_t B[SYS_T + 1];
	__shared__ gf shared_coeffs[SYS_T + 1];  // Adjust size as needed based on the degree 
    __shared__ gf b, d, f;

	int tid = threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int globalIdx = blockId * blockDim.x + tid;
	
	if (tid < SYND_BYTES) {
		unsigned char r = d_ciphertexts[tid];
		// Completely unroll the loop
		if (tid * 8 + 0 < sb) c[tid * 8 + 0] = (r >> 0) & 1;
		if (tid * 8 + 1 < sb) c[tid * 8 + 1] = (r >> 1) & 1;
		if (tid * 8 + 2 < sb) c[tid * 8 + 2] = (r >> 2) & 1;
		if (tid * 8 + 3 < sb) c[tid * 8 + 3] = (r >> 3) & 1;
		if (tid * 8 + 4 < sb) c[tid * 8 + 4] = (r >> 4) & 1;
		if (tid * 8 + 5 < sb) c[tid * 8 + 5] = (r >> 5) & 1;
		if (tid * 8 + 6 < sb) c[tid * 8 + 6] = (r >> 6) & 1;
		if (tid * 8 + 7 < sb) c[tid * 8 + 7] = (r >> 7) & 1;
	}
	__syncthreads(); // Ensure 'c' is fully populated before proceeding

    // Compute 'out' elements in parallel
	if (globalIdx < 2 * SYS_T) {
		gf sum = 0;
		int i = 0;
		for (i = 0; i <= sb - 8; i += 8) {
			sum = add(sum, mul(d_inverse_elements[i * 2 * SYS_T + globalIdx], c[i]));
			sum = add(sum, mul(d_inverse_elements[(i + 1) * 2 * SYS_T + globalIdx], c[i + 1]));
			sum = add(sum, mul(d_inverse_elements[(i + 2) * 2 * SYS_T + globalIdx], c[i + 2]));
			sum = add(sum, mul(d_inverse_elements[(i + 3) * 2 * SYS_T + globalIdx], c[i + 3]));
			sum = add(sum, mul(d_inverse_elements[(i + 4) * 2 * SYS_T + globalIdx], c[i + 4]));
			sum = add(sum, mul(d_inverse_elements[(i + 5) * 2 * SYS_T + globalIdx], c[i + 5]));
			sum = add(sum, mul(d_inverse_elements[(i + 6) * 2 * SYS_T + globalIdx], c[i + 6]));
			sum = add(sum, mul(d_inverse_elements[(i + 7) * 2 * SYS_T + globalIdx], c[i + 7]));
		
		// The loop is fully unrolled and sb is a multiple of 8, so no need for extra handling
	
		s_out[globalIdx] = sum;
	}
	   // Initialize Berlekamp-Massey variables
	   if (tid <= SYS_T) {
        T[tid] = C[tid] = B[tid] = 0;
        if (tid == 0) {
            C[0] = 1;  // C(x) starts with 1
            B[1] = 1;  // B(x) starts with x
            b = 1;     // b starts with 1
            
        }
    }   
	
	__syncthreads(); // Ensure 's_out' is fully populated before proceeding
	
	
	// Main loop of the Berlekamp-Massey algorithm executed for 2 * SYS_T iterations
	for (int N = 0; N < 2 * SYS_T; N++) {
		int max_j = min(N, SYS_T);
		if (tid <= max_j) {
			d = 0;
			for (int j = 0; j <= max_j; j++) {
				d ^= mul(C[j], s_out[N - j]);
			}
		}
	
		__syncthreads(); // Necessary for consistent 'd' before proceeding
	
		if (tid == 0) {
			f = p_gf_frac(b, d);  // Compute the fractional update
			b = (b & ~mle[N]) | (d & mle[N]);
		}
	
		__syncthreads(); // Necessary for 'f' and 'b' update
	
		if (tid <= SYS_T) {
			T[tid] = C[tid]; // Backup 'C'
			C[tid] ^= mul(f, B[tid]) & mne; // Update 'C'
		}

	
		if (tid <= SYS_T) {
			B[tid] = (B[tid] & ~mle[N]) | (T[tid] & mle[N]); // Shift 'B'
		}
	
		if (tid == SYS_T) {
			for (int i = SYS_T; i > 0; i--) {
				B[i] = B[i - 1];
			}
			B[0] = 0;
		}

	}
   // copy C to out1
   if (tid <= SYS_T) {
		shared_coeffs[tid] = C[SYS_T - tid];
		// out1[tid] = C[SYS_T - tid];
	}
	__syncthreads(); 
 // Calculate global index
    if (globalIdx < SYS_N) {
        gf point = d_L[globalIdx];
        gf result = 0;
        gf power = 1;
        for (int i = 0; i <= SYS_T; i++) {
            gf term = mul(shared_coeffs[i], power);
            result ^= term;
            power = mul(power, point);
        }
        out1[globalIdx] = result;
    }
	__syncthreads();
	


}
    
}