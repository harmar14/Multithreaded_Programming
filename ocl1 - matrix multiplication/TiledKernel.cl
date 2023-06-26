__kernel void Multiplication(const int N, const int K, const int M, const __global float* A, const __global float* B, __global float* C)
{
	// tile is TSxTS elements
	
	// getting work-item IDs with dimension index
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int global_row = TS * get_group_id(0) + row;
	const int global_col = TS * get_group_id(1) + col;
	
	__local float Asub[TS * TS];
	__local float Bsub[TS * TS];
	
	float result_element = 0.0f;
	
	const int num_tiles = K/TS;
	for (int t = 0; t < num_tiles; t++) {
		const int tiled_row = TS * t + row;
		const int tiled_col = TS * t + col;
		Asub[row * TS + col] = A[global_row * K + tiled_col];
		Bsub[row * TS + col] = B[tiled_row * N + global_col];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (int k = 0; k < TS; k++) {
			result_element += Asub[row * TS + k] * Bsub[k * TS + col];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	C[global_row * N + global_col] = result_element;
}