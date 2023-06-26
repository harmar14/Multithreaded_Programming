__kernel void Multiplication(const int N, const int K, const int M, const __global float* A, const __global float* B, __global float* C)
{
	// tile is TSxTS elements
	
	// const int VS = 4;
	float4 inter_sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	
	// getting work-item IDs with dimension index
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int global_row = get_global_id(1);
	const int global_col = get_global_id(0);
	
	__local float Asub[TS * TS];
	__local float Bsub[TS * TS];
	
	const int Astart = K * TS * get_group_id(1);
	const int Aend = Astart + K - 1;
	const int Bstart = TS * get_group_id(0);
	
	for (int a = Astart, b = Bstart; a <= Aend; a += TS, b += TS * N) {
		Asub[row * TS + col] = A[a + K * col + row];
		Bsub[row * TS + col] = B[b + N * col + row];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		#pragma unroll
		for (int t = 0; t < TS; t +=4) {
			float4 Avector = (float4)(Asub[t * TS + col], Asub[(t + 1) * TS + col], Asub[(t + 2) * TS + col], Asub[(t + 3) * TS + col]);
			float4 Bvector = (float4)(Bsub[row * TS + t], Bsub[row * TS + (t + 1)], Bsub[row * TS + (t + 2)], Bsub[row * TS + (t + 3)]);
			barrier(CLK_LOCAL_MEM_FENCE);
			inter_sum += Avector * Bvector;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// float4(float x, float y, float z, float w)
	C[global_row * N + global_col] = inter_sum.x + inter_sum.y + inter_sum.z + inter_sum.w;
}