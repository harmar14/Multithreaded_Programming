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

int main(int argc, char* argv[])
{
	//input example: MTP_ocl2.exe <device_num> input.txt output.txt
	if (argc != 4) {
		cerr << "Wrong number of parameters";
		exit(1);
	}

	int device_num = stoi(argv[1]);
	string file_in = argv[2];
	string file_out = argv[3];

	cl_device_id device_id = GetDevice(device_num);

	//opening input file
	ifstream input;
	input.open(file_in);
	if (!input) {
		cerr << "Reading file error";
		exit(1);
	}
	size_t n;
	input >> n;

	size_t global_size, local_size;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);
	
	//local_size = 3; // for testing

	if (n > local_size) {
		global_size = (n / local_size + 1) * local_size;
	}
	else {
		local_size = n;
		global_size = n;
	}
	string build_options = "-D MAX_WORK_GR=" + to_string(local_size);

	float* array = new (nothrow) float[global_size];
	if (array == nullptr) {
		cerr << "Memory can not be allocated";
		input.close();
		exit(1);
	}
	for (size_t i = 0; i < n; i++) {
		input >> array[i];
	}
	input.close();

	//array_out
	/*
	for (int i = 0; i < n; i++) {
		cout << array[i] << " ";
	}
	cout << "\n";
	*/

	cl_int ret; // error flag

	// context creating
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Context creation failed";
		delete[] array;
		exit(1);
	}

	// command creating
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Command queue creation failed";
		delete[] array;
		clReleaseContext(context);
		exit(1);
	}
	
	// compiling kernel file
	ifstream kernel_file("Kernel.cl");
	string kernel_string(istreambuf_iterator<char>(kernel_file), (istreambuf_iterator<char>()));
	const char* kernel_code = kernel_string.c_str();

	// program creating
	cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Program creation failed";
		delete[] array;
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// program building
	ret = clBuildProgram(program, 0, NULL, build_options.c_str(), NULL, NULL);
	if (ret != CL_SUCCESS) {
		cerr << "Program building failed";
		cerr << "\n" << ret << "\n";
		/*
		// LOGS
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(stderr, "%s\n", buffer);
		*/
		delete[] array;
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);

		exit(1);
	}

	// connecting to the kernel function
	cl_kernel kernel = clCreateKernel(program, "PrefixSum", &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel creating failed";
		delete[] array;
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// creating buffers for arrays (global memory)
	cl_mem buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * global_size, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Buffer creating failed";
		delete[] array;
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	cl_mem buffer_out = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * global_size, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Buffer creating failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// creating events with buffers
	cl_event kernel_event, read_event, write_event;
	clEnqueueWriteBuffer(command_queue, buffer_in, CL_TRUE, 0, sizeof(float) * n, array, NULL, NULL, &write_event);

	// setting kernel arguments
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_out);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), &global_size);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	const size_t global_memory_size[] = { global_size }; // size of global memory
	const size_t local_memory_size[] = { local_size }; // size of local memory
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_memory_size, local_memory_size, 0, NULL, &kernel_event);
	if (ret != CL_SUCCESS) {
		cerr << "Adding kernel to queue failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// waiting for event to be completed
	ret = clWaitForEvents(1, &kernel_event);

	// getting result
	ret = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, sizeof(float) * n, array, 0, NULL, &read_event);
	if (ret != CL_SUCCESS) {
		cerr << "Reading result from buffer failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// waiting for all enqueued tasks to finish
	clFinish(command_queue);

	clReleaseMemObject(buffer_in);
	clReleaseMemObject(buffer_out);
	clReleaseKernel(kernel);

	//array_out
	/*
	for (int i = 0; i < n; i++) {
		cout << array[i] << " ";
	}
	cout << "\n";
	*/

	// connecting to the kernel function
	kernel = clCreateKernel(program, "AddBlocks", &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel creating failed";
		delete[] array;
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	// creating buffers for arrays (global memory)
	buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * global_size, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Buffer creating failed";
		delete[] array;
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	buffer_out = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * global_size, NULL, &ret);
	if (ret != CL_SUCCESS) {
		cerr << "Buffer creating failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	// creating events with buffers
	cl_event kernel_event_add, read_event_add, write_event_add;
	clEnqueueWriteBuffer(command_queue, buffer_in, CL_TRUE, 0, sizeof(float) * n, array, NULL, NULL, &write_event_add);
	// setting kernel arguments
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_out);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), &global_size);
	if (ret != CL_SUCCESS) {
		cerr << "Kernel argument setting failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_memory_size, local_memory_size, 0, NULL, &kernel_event_add);
	if (ret != CL_SUCCESS) {
		cerr << "Adding kernel to queue failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}
	// waiting for event to be completed
	ret = clWaitForEvents(1, &kernel_event_add);
	// getting result
	ret = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, sizeof(float) * n, array, 0, NULL, &read_event_add);
	if (ret != CL_SUCCESS) {
		cerr << "Reading result from buffer failed";
		delete[] array;
		clReleaseMemObject(buffer_in);
		clReleaseMemObject(buffer_out);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		exit(1);
	}

	// waiting for all enqueued tasks to finish
	clFinish(command_queue);

	//array_out
	/*
	for (int i = 0; i < n; i++) {
		cout << array[i] << " ";
	}
	cout << "\n";
	*/

	// getting kernel execution time
	cl_ulong time_start, time_end, time_start_2, time_end_2;
	cl_ulong time_start_add, time_end_add, time_start_add_2, time_end_add_2;

	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	double kernel_time = time_end - time_start;
	clGetEventProfilingInfo(kernel_event_add, CL_PROFILING_COMMAND_START, sizeof(time_start_add), &time_start_add, NULL);
	clGetEventProfilingInfo(kernel_event_add, CL_PROFILING_COMMAND_END, sizeof(time_end_add), &time_end_add, NULL);
	kernel_time += time_end_add - time_start_add;

	clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(time_start_2), &time_start_2, NULL);
	clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(time_end_2), &time_end_2, NULL);
	double exec_time = time_end_2 - time_start_2;
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(time_start_2), &time_start_2, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(time_end_2), &time_end_2, NULL);
	exec_time += time_end_2 - time_start_2;
	clGetEventProfilingInfo(write_event_add, CL_PROFILING_COMMAND_START, sizeof(time_start_add_2), &time_start_add_2, NULL);
	clGetEventProfilingInfo(write_event_add, CL_PROFILING_COMMAND_END, sizeof(time_end_add_2), &time_end_add_2, NULL);
	exec_time += time_end_add_2 - time_start_add_2;
	clGetEventProfilingInfo(read_event_add, CL_PROFILING_COMMAND_START, sizeof(time_start_add_2), &time_start_add_2, NULL);
	clGetEventProfilingInfo(read_event_add, CL_PROFILING_COMMAND_END, sizeof(time_end_add_2), &time_end_add_2, NULL);
	exec_time += time_end_add_2 - time_start_add_2 + kernel_time;

	cout << "Time: " << kernel_time / 1000000.0 << "\t" << exec_time / 1000000.0 << "\n";
	cout << "LOCAL_WORK_SIZE " << local_size << "\n";

	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	// opening output file
	ofstream output;
	output.open(file_out);
	if (!output) {
		cerr << "Writing file error";
		delete[] array;
		exit(1);
	}
	// writing result_matrix to file

	for (int i = 0; i < n; i++) {
		output << array[i] << " ";
	}
	output << "\n";

	delete[] array;
	output.close();

	return 0;
}