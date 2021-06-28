#pragma once

int sgemmCuda(
	int m, int n, int k,
	float alpha, float beta,
	float *a, float *b, float *c
);

void GPU_argv_init(void);
