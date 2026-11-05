/*
 * Runtime CPU feature detection.
 */
#include "h265_internal.h"

int h265_have_sse2;
int h265_have_avx2;
int h265_have_neon;
static int h265_cpu_initialized;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define H265_X86 1
#endif

#ifdef H265_X86

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static void cpuid4(unsigned leaf, unsigned subleaf, unsigned *a, unsigned *b,
                   unsigned *c, unsigned *d)
{
    __cpuid_count(leaf, subleaf, *a, *b, *c, *d);
}
#endif

#ifdef _MSC_VER
#include <intrin.h>
static void cpuid4(unsigned leaf, unsigned subleaf, unsigned *a, unsigned *b,
                   unsigned *c, unsigned *d)
{
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *a = (unsigned)regs[0]; *b = (unsigned)regs[1];
    *c = (unsigned)regs[2]; *d = (unsigned)regs[3];
}
#endif

static unsigned long long xgetbv0(void)
{
    unsigned lo, hi;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#else
    lo = (unsigned)_xgetbv(0);
    hi = 0;
#endif
    return ((unsigned long long)hi << 32) | lo;
}

#endif /* H265_X86 */

void h265_cpu_init(void)
{
    const char *disable = getenv("H265_DISABLE_SIMD");

    if (h265_cpu_initialized)
        return;
    h265_cpu_initialized = 1;
    if (disable && disable[0] && disable[0] != '0')
        return;

#ifdef H265_X86
    {
        unsigned a, b, c, d;
        cpuid4(1, 0, &a, &b, &c, &d);
        if (d & (1u << 26))
            h265_have_sse2 = 1;
        if ((c & (1u << 27)) && (c & (1u << 28))) {
            unsigned long long xcr0 = xgetbv0();
            if ((xcr0 & 6) == 6) {
                cpuid4(7, 0, &a, &b, &c, &d);
                if (b & (1u << 5))
                    h265_have_avx2 = 1;
            }
        }
    }
#endif
#ifdef H265_NEON
    /* NEON is mandatory in AArch64. */
    h265_have_neon = 1;
#endif
}
