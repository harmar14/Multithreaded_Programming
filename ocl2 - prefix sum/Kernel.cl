__kernel void PrefixSum(__global const float* array_in, __global float* array_out, const int n)
{
	// array_in - start array
	// array_out - calculated blocks
	
    const int local_id = get_local_id(0);
	const int global_id = get_global_id(0);
	
    __local float inter_array[MAX_WORK_GR];
	
	inter_array[local_id] = array_in[global_id];

	barrier(CLK_LOCAL_MEM_FENCE);
	if (local_id == 0) {
		for (int i = 1; i < MAX_WORK_GR; i++) {
			inter_array[i] += inter_array[i-1];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	array_out[global_id] = inter_array[local_id];
}

__kernel void AddBlocks(__global const float* array_in, __global float* array_out, const int n)
{
	// array_in - calculated blocks
	// array_out - result array
	
    const int group_id = get_group_id(0);
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	
	array_out[global_id] = array_in[global_id];
	
	if (group_id > 0) {
		float inter_sum = 0;
		
		for (int i = MAX_WORK_GR - 1; i < (global_id-local_id); i += MAX_WORK_GR) {
			inter_sum += array_in[i];
		}
		array_out[global_id] += inter_sum;
	}
	
}