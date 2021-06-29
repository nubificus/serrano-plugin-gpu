#include <vaccel.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "minmax.h"
}

#define BLOCKSIZE 32
static const int bc_x = 16;
static const int bc_y = 1;


#define timespec_usec(t) ((double)(t).tv_nsec / 10e3 + (double)(t).tv_sec * 10e6)
#define time_diff(t0, t1) (timespec_usec((t1)) - timespec_usec((t0)))

__global__
void max_min_kernel_bc(
	double *array, double *max, double *min,
	int *mutex, unsigned int n, int bc_x, int bc_y
) {
	unsigned int index = (threadIdx.x * bc_x)
				+ (blockIdx.x * bc_x)
				* blockDim.x;
	
	for (int indexx=0; indexx<bc_x; indexx++) {
		unsigned int stride = gridDim.x*blockDim.x;
		unsigned int offset = 0;

		__shared__ float cache_max[256];
		__shared__ float cache_min[256];

		float temp_max = -1.0;
		float temp_min = 10000;

		while ((index + indexx) + offset < n){
			temp_max = fmaxf(temp_max, array[(index+indexx) + offset]);
			temp_min = fminf(temp_min, array[(index+indexx) + offset]);

			offset += stride;
		}

		cache_max[threadIdx.x] = temp_max;
		cache_min[threadIdx.x] = temp_min;

		__syncthreads();

		// reduction
		unsigned int i = blockDim.x / 2;
		while(i) {
			if (threadIdx.x < i) {
				cache_max[threadIdx.x] =
					fmaxf(cache_max[threadIdx.x], cache_max[threadIdx.x + i]);
				cache_min[threadIdx.x] =
					fminf(cache_min[threadIdx.x], cache_min[threadIdx.x + i]);
			}

			__syncthreads();

			i /= 2;
		}

		if (threadIdx.x == 0) {
			while (atomicCAS(mutex,0,1) != 0);  //lock
			*max = fmaxf(*max, cache_max[0]);
			*min = fminf(*min, cache_min[0]);

			atomicExch(mutex, 0);  //unlock
		}

	}
}





__global__
void normalize_kernel_bc(
	double *indata, double *outdata, int n,
	double *max, double *min,
	int low_th, int high_th, int bc_x, int bc_y)
{
	unsigned int index = (threadIdx.x * bc_x)
				+ (blockIdx.x * bc_x)
				* blockDim.x;
	
	for (int indexx = 0; indexx < bc_x; indexx++) {
		if ((index+indexx) < n) {
			outdata[(index+indexx)] =
				((indata[index + indexx] - *min)
				 / (*max - *min))
				 *(high_th - low_th) + low_th;
		}
	}
}


// void cfun(const void * indatav, int ndata, void * outdatav, int low_th, int high_th) {
int serrano_minmax(
	struct vaccel_session *sess,
	const double *indata, int ndata,
	int low_threshold, int high_threshold,
	double *out_data, double *min, double *max
) {
	double h_max, h_min;
	double *d_max, *d_min;
	int *d_mutex;
	struct timespec t0, t1;

	h_max=0;
	h_min=10000;

	double *indata_gpu;
	double *outdata_gpu;
	cudaMalloc((void**)&d_max, sizeof(double));
	cudaMalloc((void**)&d_min, sizeof(double));
	cudaMalloc((void**)&d_mutex, sizeof(int));

	cudaMemset(d_mutex, 0, sizeof(int));

	cudaMalloc((void **)&indata_gpu, sizeof(double) * ndata);
	cudaMalloc((void **)&outdata_gpu, sizeof(double) * ndata);
	cudaMemcpy(indata_gpu, indata, sizeof(double) * ndata, cudaMemcpyHostToDevice);

	dim3 block(BLOCKSIZE,1);
	dim3 block_bc=block;
	dim3 grid((size_t)(ceil( ((float)ndata)/ ((float)block.x) )),1);
	dim3 grid_bc=grid;

	cudaMemcpy(d_max, &h_max, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_min, &h_min, sizeof(double), cudaMemcpyHostToDevice);

	grid_bc.x = grid.x / bc_x;
	grid_bc.y = grid.y / bc_y;

	vaccel_debug("After block coarsening with bc_x=%d and bc_y=%d. Kernel on %d %d %d %d", bc_x, bc_y, grid_bc.x, grid_bc.y, block_bc.x, block_bc.y);

	if (!grid_bc.x || !grid_bc.y)
		return VACCEL_EINVAL;

	clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
	max_min_kernel_bc<<< grid_bc, block_bc >>>(indata_gpu, d_max, d_min, d_mutex, ndata, bc_x, bc_y);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

	vaccel_debug("max_min_kernel_bc GPU time:%lf msecs", time_diff(t0, t1) / 10e3);
	
	grid_bc.x = grid.x / bc_x;
	grid_bc.y = grid.y / bc_y;
	vaccel_debug("After block coarsening with bc_x=%d and bc_y=%d. Kernel on %d %d %d %d", bc_x, bc_y, grid_bc.x, grid_bc.y, block_bc.x, block_bc.y);

	if (!grid_bc.x || !grid_bc.y)
		return VACCEL_EINVAL;

	clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
	normalize_kernel_bc<<<grid_bc, block_bc>>>(indata_gpu, outdata_gpu,
			ndata, d_max, d_min, low_threshold,
			high_threshold, bc_x, bc_y);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

	vaccel_debug("normalize_kernel_bc GPU time: %lf msecs", time_diff(t0, t1) / 10e3);
	
	cudaMemcpy(out_data, outdata_gpu, sizeof(double) * ndata, cudaMemcpyDeviceToHost);
	cudaMemcpy(max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(min, d_min, sizeof(double), cudaMemcpyDeviceToHost);

	return 0;
}
