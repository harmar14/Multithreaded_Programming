#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <list>

#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#pragma comment(lib, "opencl.lib")
#endif

using namespace std;

cl_device_id GetDevice(int device) {

	// defining lists for discrete and integrated GPUs and CPUs
	list<cl_device_id> discrete_GPUs;
	list<cl_device_id> integrated_GPUs;
	list<cl_device_id> CPUs;
	list<cl_device_id> devices_res; // resulting list of devices

	cl_int ret;
	cl_uint platform_num;
	cl_uint device_num;
	cl_device_type device_type;
	
	ret = clGetPlatformIDs(0, NULL, &platform_num);
	if (!platform_num)
	{
		cerr << "Number of platforms: 0";
		exit(1);
	}
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platform_num);
	ret = clGetPlatformIDs(platform_num, platforms, NULL);

	for (int i = 0; i < platform_num; i++) {
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);
		cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * device_num);
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_num, devices, NULL);

		for (int j = 0; j < device_num; j++) {
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
			if (device_type == CL_DEVICE_TYPE_GPU) {
				// GPUs need to be checked if they are integrated or discrete
				cl_bool is_integrated;
				ret = clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &is_integrated, NULL);
				if (is_integrated) {
					integrated_GPUs.push_back(devices[j]);
				}
				else {
					discrete_GPUs.push_back(devices[j]);
				}
			}
			else if (device_type == CL_DEVICE_TYPE_CPU) {
				CPUs.push_back(devices[j]); // put an element in the end of list 
			}
		}
		free(devices);
	}
	free(platforms);

	devices_res.insert(devices_res.end(), discrete_GPUs.begin(), discrete_GPUs.end()); // adding discrete_GPUs from the first to the last in the end of devices_res
	devices_res.insert(devices_res.end(), integrated_GPUs.begin(), integrated_GPUs.end());
	devices_res.insert(devices_res.end(), CPUs.begin(), CPUs.end());

	// cleaning temporary lists
	discrete_GPUs.clear();
	integrated_GPUs.clear();
	CPUs.clear();

	// checking the provided number of devices
	if ((device < 0) or (device > (devices_res.size() - 1))) {
		cerr << "Wrong device number";
		exit(1);
	}

	// getting device with provided number
	auto devices_res_front = devices_res.begin();
	advance(devices_res_front, device);
	cl_device_id gotten_device = *devices_res_front;

	// getting its name
	size_t size;
	clGetDeviceInfo(gotten_device, CL_DEVICE_NAME, 0, NULL, &size);
	char* device_name = (char*)malloc(sizeof(char) * size);
	clGetDeviceInfo(gotten_device, CL_DEVICE_NAME, size, device_name, 0);
		
	cout << "Device: " << device_name << "\n";

	return gotten_device;

}

void matrix_out(float* matrix, size_t n, size_t m) {

	cout << "\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++)
			cout << matrix[i * m + j] << " ";
		cout << "\n";
	}

}

// I wrote this function to check results of OpenCL matrix multiplications
float* SimpleMultiplication(float* matrix1, float* matrix2, float* result_matrix, int n, int k, int m) {
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int q = 0; q < k; q++) {
				result_matrix[i * n + j] += matrix1[i * k + q] * matrix2[q * n + j];
			}
		}
	}
	return result_matrix;
}

int main(int argc, char* argv[])
{
	//input example: MTP_info.exe <device_num> input.txt output.txt <realization_num>
	if (argc != 5) {
		cerr << "Wrong number of parameters";
		exit(1);
	}

	int device_num = stoi(argv[1]);
	string file_in = argv[2];
	string file_out = argv[3];
	int realization = stoi(argv[4]);

	if ((realization < 1) or (realization > 3)) {
		cerr << "Wrong realization number";
		exit(1);
	}

	cl_device_id device_id = GetDevice(device_num);

	//opening input file
	ifstream input;
	input.open(file_in);
	if (!input) {
		cerr << "Reading file error";
		exit(1);
	}
	size_t n, k, m;
	input >> n >> k >> m;
	// there are matrices [m x k] and [k x n]

	size_t m_global = m;
	size_t n_global = n;
	size_t k_global = k;
	size_t tile_side_size;

	if (realization == 2 or realization == 3) {
		// getting tile side size according to max work group size
		size_t local_size;
		clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);
		float sqrt_ls = sqrt(local_size);
		if (sqrt_ls == (float)((int)sqrt_ls) ) {
			tile_side_size = sqrt_ls;
		}
		else {
			tile_side_size = sqrt(local_size / 2);
		}

		// I used m_global, n_global and k_global to get number of tiles to avoid creating new variables
		m_global = m / tile_side_size;
		n_global = n / tile_side_size;
		k_global = k / tile_side_size;
		if (m_global * tile_side_size < m) {
			m_global++;
		}
		if (n_global * tile_side_size < n) {
			n_global++;
		}
		if (k_global * tile_side_size < k) {
			k_global++;
		}
		// going back from number of tiles to size
		m_global *= tile_side_size;
		n_global *= tile_side_size;
		k_global *= tile_side_size;
	}

	// making matrix1 and matrix2 able to divide entirely into tiles
	float* matrix1 = new (nothrow) float[m_global * k_global] {0};
	if (matrix1 == nullptr) {
		cerr << "Memory can not be allocated";
		input.close();
		exit(1);
	}
	float* matrix2 = new (nothrow) float[k_global * n_global] {0};
	if (matrix2 == nullptr) {
		cerr << "Memory can not be allocated";
		delete[] matrix1;
		input.close();
		exit(1);
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			input >> matrix1[i * k_global + j];
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++) {
			input >> matrix2[i * n_global + j];
		}
	}
	input.close();

	// matrix_out(matrix1, m_global, k_global);
	// matrix_out(matrix2, k_global, n_global);

	float* result_matrix = new (nothrow) float[m_global * n_global] {0};
	if (result_matrix == nullptr) {
		cerr << "Memory can not be allocated";
		delete[] matrix1;
		delete[] matrix2;
		exit(1);
	}

	// FOR RESULT VALIDATION
	// matrix_out(SimpleMultiplication(matrix1, matrix2, result_matrix, n, k, m), m, n);

	cl_int ret; // ret = 0 is OK, otherwise - error of function execution

	// context creating
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Context creation failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		exit(1);
	}

	// command creating
	// cl_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	// cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, &properties, &ret);
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Command queue creation failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseContext(context);
		exit(1);
	}

	const char* name = "";
	if (realization == 1) {
		name = "SimpleKernel.cl";
	}
	else if (realization == 2) {
		name = "TiledKernel.cl";
	}
	else if (realization == 3) {
		name = "VectorKernel.cl";
	}

	// compiling kernel file
	ifstream kernel_file(name);
	string kernel_string(istreambuf_iterator<char>(kernel_file), (istreambuf_iterator<char>()));
	const char* kernel_code = kernel_string.c_str();

	// program creating
	cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &ret);

	if (ret != CL_SUCCESS) {
		cerr << "Program creation failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	//cout << ("-D TS="+to_string(tile_side_size)).c_str();
	string build_options = "-D TS=" + to_string(tile_side_size);
	// program building
	ret = clBuildProgram(program, 0, NULL, build_options.c_str(), NULL, NULL);

	if (ret != CL_SUCCESS) {
		cerr << "Program building failed";
		cerr << "\n" << ret << "\n";

		
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(stderr, "%s\n", buffer);
		

		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);

		exit(1);
	}

	// connecting to the kernel function
	cl_kernel kernel = clCreateKernel(program, "Multiplication", &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel creating failed";
		// cerr << ret;
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// creating buffers for matrices (global memory)
	cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * m_global * k_global, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Buffer buffer_A creating failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * k_global * n_global, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Buffer buffer_B creating failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * m_global * n_global, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Buffer buffer_C creating failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// creating events with buffers
	cl_event kernel_event, read_event, write_event_A, write_event_B;
	clEnqueueWriteBuffer(command_queue, buffer_A, CL_TRUE, 0, sizeof(float) * m_global * k_global, matrix1, NULL, NULL, &write_event_A);
	clEnqueueWriteBuffer(command_queue, buffer_B, CL_TRUE, 0, sizeof(float)* k_global * n_global, matrix2, NULL, NULL, &write_event_B);

	// setting kernel arguments
	ret = clSetKernelArg(kernel, 0, sizeof(cl_int), &n_global);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_int), &k_global);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), &m_global);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_A);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_B);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_C);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	size_t global_memory_size[] = { m_global, n_global }; // size of global memory
	if (realization == 1) {
		// adding kernel to queue
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_memory_size, NULL, 0, NULL, &kernel_event);
	}
	else if (realization == 2 or realization == 3) {
		size_t local_memory_size[] = { tile_side_size, tile_side_size }; // size of local memory
		// adding kernel to queue
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_memory_size, local_memory_size, 0, NULL, &kernel_event);
	}
	if (ret != CL_SUCCESS) {
		cerr << "Adding kernel to queue failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// waiting for event to be completed
	ret = clWaitForEvents(1, &kernel_event);

	// getting result
	ret = clEnqueueReadBuffer(command_queue, buffer_C, CL_TRUE, 0, sizeof(float) * m_global * n_global, result_matrix, 0, NULL, &read_event);
	if (ret != CL_SUCCESS) {
		cerr << "Reading result from buffer failed";
		delete[] matrix1;
		delete[] matrix2;
		delete[] result_matrix;
		clReleaseMemObject(buffer_A);
		clReleaseMemObject(buffer_B);
		clReleaseMemObject(buffer_C);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	//matrix_out(result_matrix, m_global, n_global);

	// waiting for all enqueued tasks to finish
	clFinish(command_queue);

	// getting kernel execution time
	cl_ulong time_start, time_end, time_start_2, time_end_2;

	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	double kernel_time = time_end - time_start;

	clGetEventProfilingInfo(write_event_A, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start_2, NULL);
	clGetEventProfilingInfo(write_event_A, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end_2, NULL);
	double exec_time = time_end_2 - time_start_2;
	clGetEventProfilingInfo(write_event_B, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start_2, NULL);
	clGetEventProfilingInfo(write_event_B, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end_2, NULL);
	exec_time += time_end_2 - time_start_2;
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start_2, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end_2, NULL);
	exec_time += time_end_2 - time_start_2 + kernel_time;

	cout << "Time: " << kernel_time / 1000000.0 << "\t" << exec_time / 1000000.0 << "\n";
	if (realization == 2) {
		cout << "LOCAL_WORK_SIZE [" << tile_side_size << ", " << tile_side_size << "]" << "\n";
	}
	if (realization == 3) {
		cout << "LOCAL_WORK_SIZE [" << tile_side_size << ", " << tile_side_size << "]" << "\nWI_WORK 4\n";
	}

	delete[] matrix1;
	delete[] matrix2;
	clReleaseMemObject(buffer_A);
	clReleaseMemObject(buffer_B);
	clReleaseMemObject(buffer_C);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	// opening output file
	ofstream output;
	output.open(file_out);
	if (!output) {
		cerr << "Writing file error";
		delete[] result_matrix;
		exit(1);
	}
	// writing result_matrix to file
	output << n << " " << m << "\n";
	
	// teacher's tests ask for 6 digits after point
	output << fixed;
	output.precision(6);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			output << result_matrix[i * n_global + j] << " ";
		}
		output << "\n";
	}

	delete[] result_matrix;
	output.close();
	
	return 0;
}