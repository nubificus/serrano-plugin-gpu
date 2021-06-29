#pragma once

struct vaccel_session;

int serrano_minmax(
	struct vaccel_session *sess,
	const double *indata, int ndata,
	int low_threshold, int high_threshold,
	double *out_data, double *min, double *max
);
