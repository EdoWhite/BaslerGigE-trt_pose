#ifndef PEAK_LOCAL_MAX_H
#define PEAK_LOCAL_MAX_H

#include "tensor.h"
#include "matrix.h"
#include "peak.h"

#ifdef __cplusplus
extern "C" {
#endif

int peak_local_max(matrix_t *m, float threshold, peak_t *peaks, int peaks_capacity);

#ifdef __cplusplus
}
#endif

#endif // PEAK_LOCAL_MAX_H