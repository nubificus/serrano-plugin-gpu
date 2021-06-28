#include <vaccel.h>
#include <stdint.h>

#include "gemm.h"

void GPU_argv_init();

int serrano_sgemm(
	struct vaccel_session *sess,
	int64_t m, int64_t n, int64_t k,
	float alpha,
	float *a, int64_t lda,
	float *b, int64_t ldb,
	float beta,
	float *c, int64_t ldc
) {
	(void)lda;
	(void)ldb;
	(void)ldc;

	if (!sess)
		return VACCEL_EINVAL;

	sgemmCuda(m, n, k, alpha, beta, a, b, c);
}

struct vaccel_op ops[] = {
	VACCEL_OP_INIT(ops[0], VACCEL_BLAS_SGEMM, (void *)serrano_sgemm),
};

int serrano_init(void)
{
	int ret = register_plugin_functions(ops, sizeof(ops) / sizeof(ops[0]));
	if (!ret)
		return ret;

	GPU_argv_init();

	return VACCEL_OK;
}

int serrano_fini(void)
{
	return VACCEL_OK;
}

VACCEL_MODULE(
	.name = "serrano-plugins",
	.version = "0.1",
	.init = serrano_init,
	.fini = serrano_fini,
)
