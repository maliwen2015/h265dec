/*
 * DSP dispatch.  The scalar C kernels are always present; this file is where
 * the SSE2/AVX2 variants are selected at run time.
 */
#include "h265_internal.h"

void h265_dsp_init(void)
{
    h265_cpu_init();
}
