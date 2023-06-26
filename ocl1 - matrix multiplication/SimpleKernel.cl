__kernel void Multiplication(const int N, const int K, const int M, const __global float* A, const __global float* B, __global float* C)
{
	// getting work-item IDs with dimension index
	const int global_row = get_global_id(0);
	const int global_col = get_global_id(1);
	
	//calculating one current element of result_matrix
	float result_element = 0.0f;
	for (int i = 0; i < K; i++) {
		result_element += A[global_row * K + i] * B[i * N + global_col];
	}
	// writing result to result_matrix
	C[global_row * N + global_col] = result_element;
}