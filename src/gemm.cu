#include <vaccel.h>
#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

extern "C" {
#include "gemm.h"
}

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

// Block coarsening factors initialization
static const int bc_x=1;
static const int bc_y=16;

#define GPU_DEVICE 0

void GPU_argv_init(void)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}

__global__
void gemm_kernel_bc(
	int ni, int nj, int nk,
	float alpha, float beta,
	float *a, float *b, float *c,
	int bc_x, int bc_y
) {
	int j = (blockIdx.x * bc_x) * blockDim.x + (threadIdx.x * bc_x);
	int i = (blockIdx.y * bc_y) * blockDim.y + (threadIdx.y * bc_y);
	for (int indexx = 0; indexx < bc_x; indexx++) {
		for (int indexy = 0; indexy < bc_y; indexy++) {
			if (((i+indexy) < ni) && ((j+indexx) < nj)) {	
				c[(i + indexy) * nj + (j + indexx)] *= beta;
				for(int k = 0; k < nk; k++)
					c[(i+indexy) * nj + (j+indexx)] +=
						alpha * a[(i+indexy) * nk + k] *
						b[k * nj +(j+indexx)];
			}
		}
	}
}

int sgemmCuda(
	int m, int n, int k,
	float alpha, float beta,
	float *A, float *B, float *C
) {
	float *A_gpu, *B_gpu, *C_gpu;
	struct timeval t0, t1;
	double t10;

	cudaMalloc((void **)&A_gpu, sizeof(float) * m * k);
	cudaMalloc((void **)&B_gpu, sizeof(float) * k * n);
	cudaMalloc((void **)&C_gpu, sizeof(float) * m * n);

	cudaMemcpy(A_gpu, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 block_bc=block;
	dim3 grid((size_t)(ceil( ((float)m)/ ((float)block.x) )),(size_t)(ceil( ((float)n)/ ((float)block.y) )));
	dim3 grid_bc=grid;

	grid_bc.x = grid.x / bc_x;
	grid_bc.y = grid.y / bc_y;

	fprintf(stdout, "\nAfter block coarsening with bc_x=%d and bc_y=%d. Kernel on %d %d %d %d\n", bc_x, bc_y, grid_bc.x, grid_bc.y, block_bc.x, block_bc.y);

	if ((grid_bc.x == 0) || (grid_bc.y == 0))
		return VACCEL_EINVAL;

	gettimeofday(&t0, NULL);
	gemm_kernel_bc<<<grid_bc, block_bc>>>(
			m, n, k, alpha, beta,
			A_gpu, B_gpu, C_gpu,
			bc_x, bc_y
			);
	cudaDeviceSynchronize();
	gettimeofday(&t1, NULL);

	t10 = (t1.tv_sec * 1000000.0 + t1.tv_usec) -
		(t0.tv_sec * 1000000.0 + t0.tv_usec);
	fprintf(stdout, "gemm_kernel_bc GPU time:\n%lf msecs\n",
			(t10) / 1000.0F);
	fprintf(stdout, "\n-----------------------------\n");

	cudaMemcpy(C, C_gpu, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

	return VACCEL_OK;
}
