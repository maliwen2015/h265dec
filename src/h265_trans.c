/*
 * HEVC inverse transforms, dequantisation and residual addition.
 *
 * Derived from FFmpeg's hevc/dsp_template.c (LGPL-2.1+).
 */
#include "h265_internal.h"

#ifdef H265_X86
#include <immintrin.h>
#define IDCT_AVX2 __attribute__((target("avx2")))
#endif
#ifdef H265_NEON
#include <arm_neon.h>
#endif

static const int8_t transform[32][32] = {
    { 64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
      64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64 },
    { 90,  90,  88,  85,  82,  78,  73,  67,  61,  54,  46,  38,  31,  22,  13,   4,
      -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90 },
    { 90,  87,  80,  70,  57,  43,  25,   9,  -9, -25, -43, -57, -70, -80, -87, -90,
     -90, -87, -80, -70, -57, -43, -25,  -9,   9,  25,  43,  57,  70,  80,  87,  90 },
    { 90,  82,  67,  46,  22,  -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13,
      13,  38,  61,  78,  88,  90,  85,  73,  54,  31,   4, -22, -46, -67, -82, -90 },
    { 89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,
      89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89 },
    { 88,  67,  31, -13, -54, -82, -90, -78, -46, -4,   38,  73,  90,  85,  61,  22,
     -22, -61, -85, -90, -73, -38,   4,  46,  78,  90,  82,  54,  13, -31, -67, -88 },
    { 87,  57,   9, -43, -80, -90, -70, -25,  25,  70,  90,  80,  43,  -9, -57, -87,
     -87, -57,  -9,  43,  80,  90,  70,  25, -25, -70, -90, -80, -43,   9,  57,  87 },
    { 85,  46, -13, -67, -90, -73, -22,  38,  82,  88,  54,  -4, -61, -90, -78, -31,
      31,  78,  90,  61,   4, -54, -88, -82, -38,  22,  73,  90,  67,  13, -46, -85 },
    { 83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83,
      83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83 },
    { 82,  22, -54, -90, -61,  13,  78,  85,  31, -46, -90, -67,   4,  73,  88,  38,
     -38, -88, -73,  -4,  67,  90,  46, -31, -85, -78, -13,  61,  90,  54, -22, -82 },
    { 80,   9, -70, -87, -25,  57,  90,  43, -43, -90, -57,  25,  87,  70,  -9, -80,
     -80,  -9,  70,  87,  25, -57, -90, -43,  43,  90,  57, -25, -87, -70,   9,  80 },
    { 78,  -4, -82, -73,  13,  85,  67, -22, -88, -61,  31,  90,  54, -38, -90, -46,
      46,  90,  38, -54, -90, -31,  61,  88,  22, -67, -85, -13,  73,  82,   4, -78 },
    { 75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,
      75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75 },
    { 73, -31, -90, -22,  78,  67, -38, -90, -13,  82,  61, -46, -88,  -4,  85,  54,
     -54, -85,   4,  88,  46, -61, -82,  13,  90,  38, -67, -78,  22,  90,  31, -73 },
    { 70, -43, -87,   9,  90,  25, -80, -57,  57,  80, -25, -90,  -9,  87,  43, -70,
     -70,  43,  87,  -9, -90, -25,  80,  57, -57, -80,  25,  90,   9, -87, -43,  70 },
    { 67, -54, -78,  38,  85, -22, -90,   4,  90,  13, -88, -31,  82,  46, -73, -61,
      61,  73, -46, -82,  31,  88, -13, -90,  -4,  90,  22, -85, -38,  78,  54, -67 },
    { 64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,
      64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64 },
    { 61, -73, -46,  82,  31, -88, -13,  90,  -4, -90,  22,  85, -38, -78,  54,  67,
     -67, -54,  78,  38, -85, -22,  90,   4, -90,  13,  88, -31, -82,  46,  73, -61 },
    { 57, -80, -25,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  25,  80, -57,
     -57,  80,  25, -90,   9,  87, -43, -70,  70,  43, -87,  -9,  90, -25, -80,  57 },
    { 54, -85,  -4,  88, -46, -61,  82,  13, -90,  38,  67, -78, -22,  90, -31, -73,
      73,  31, -90,  22,  78, -67, -38,  90, -13, -82,  61,  46, -88,   4,  85, -54 },
    { 50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,
      50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50 },
    { 46, -90,  38,  54, -90,  31,  61, -88,  22,  67, -85,  13,  73, -82,   4,  78,
     -78,  -4,  82, -73, -13,  85, -67, -22,  88, -61, -31,  90, -54, -38,  90, -46 },
    { 43, -90,  57,  25, -87,  70,   9, -80,  80,  -9, -70,  87, -25, -57,  90, -43,
     -43,  90, -57, -25,  87, -70,  -9,  80, -80,   9,  70, -87,  25,  57, -90,  43 },
    { 38, -88,  73,  -4, -67,  90, -46, -31,  85, -78,  13,  61, -90,  54,  22, -82,
      82, -22, -54,  90, -61, -13,  78, -85,  31,  46, -90,  67,   4, -73,  88, -38 },
    { 36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36,
      36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36 },
    { 31, -78,  90, -61,   4,  54, -88,  82, -38, -22,  73, -90,  67, -13, -46,  85,
     -85,  46,  13, -67,  90, -73,  22,  38, -82,  88, -54,  -4,  61, -90,  78, -31 },
    { 25, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -25,
     -25,  70, -90,  80, -43,  -9,  57, -87,  87, -57,   9,  43, -80,  90, -70,  25 },
    { 22, -61,  85, -90,  73, -38,  -4,  46, -78,  90, -82,  54, -13, -31,  67, -88,
      88, -67,  31,  13, -54,  82, -90,  78, -46,   4,  38, -73,  90, -85,  61, -22 },
    { 18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,
      18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18 },
    { 13, -38,  61, -78,  88, -90,  85, -73,  54, -31,   4,  22, -46,  67, -82,  90,
     -90,  82, -67,  46, -22,  -4,  31, -54,  73, -85,  90, -88,  78, -61,  38, -13 },
    {  9, -25,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  25, -9,
      -9,  25, -43,  57, -70,  80, -87,  90, -90,  87, -80,  70, -57,  43, -25,   9 },
    {  4, -13,  22, -31,  38, -46,  54, -61,  67, -73,  78, -82,  85, -88,  90, -90,
      90, -90,  88, -85,  82, -78,  73, -67,  61, -54,  46, -38,  31, -22,  13,  -4 },
};


H265_INLINE int16_t clip_int16(int a)
{
    if (a < -32768)
        return -32768;
    if (a > 32767)
        return 32767;
    return (int16_t)a;
}

#ifdef H265_X86
/* Four columns are transformed in parallel.  This is the same partial-butterfly
 * arithmetic as the scalar macros below, with the four columns held in the
 * four 32-bit lanes of an SSE vector. */
IDCT_AVX2 static __m128i idct_load4(const int16_t *p)
{
    return _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)p));
}

IDCT_AVX2 static void idct_store4(int16_t *p, __m128i v)
{
    __m128i packed = _mm_packs_epi32(v, v);
    _mm_storel_epi64((__m128i *)p, packed);
}

IDCT_AVX2 static __m128i idct_scale4(__m128i v, int shift, int add)
{
    v = _mm_add_epi32(v, _mm_set1_epi32(add));
    v = _mm_sra_epi32(v, _mm_cvtsi32_si128(shift));
    v = _mm_max_epi32(v, _mm_set1_epi32(-32768));
    v = _mm_min_epi32(v, _mm_set1_epi32(32767));
    return v;
}

IDCT_AVX2 static void idct_tr4_set(__m128i *dst, const __m128i *src)
{
    __m128i s0 = src[0], s1 = src[1], s2 = src[2], s3 = src[3];
    __m128i e0 = _mm_mullo_epi32(_mm_add_epi32(s0, s2), _mm_set1_epi32(64));
    __m128i e1 = _mm_mullo_epi32(_mm_sub_epi32(s0, s2), _mm_set1_epi32(64));
    __m128i o0 = _mm_add_epi32(_mm_mullo_epi32(s1, _mm_set1_epi32(83)),
                               _mm_mullo_epi32(s3, _mm_set1_epi32(36)));
    __m128i o1 = _mm_sub_epi32(_mm_mullo_epi32(s1, _mm_set1_epi32(36)),
                               _mm_mullo_epi32(s3, _mm_set1_epi32(83)));

    dst[0] = _mm_add_epi32(e0, o0);
    dst[1] = _mm_add_epi32(e1, o1);
    dst[2] = _mm_sub_epi32(e1, o1);
    dst[3] = _mm_sub_epi32(e0, o0);
}

IDCT_AVX2 static void idct_tr4_scale(__m128i *dst, const __m128i *src,
                                     int shift, int add)
{
    __m128i tmp[4];
    idct_tr4_set(tmp, src);
    dst[0] = idct_scale4(tmp[0], shift, add);
    dst[1] = idct_scale4(tmp[1], shift, add);
    dst[2] = idct_scale4(tmp[2], shift, add);
    dst[3] = idct_scale4(tmp[3], shift, add);
}

IDCT_AVX2 static void idct_tr8_set(__m128i *dst, const __m128i *src, int end)
{
    __m128i e[4], o[4], even[4];
    int i, j;
    for (i = 0; i < 4; i++)
        o[i] = _mm_setzero_si128();
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 4; i++)
            o[i] = _mm_add_epi32(o[i],
                _mm_mullo_epi32(src[j], _mm_set1_epi32(transform[4 * j][i])));
    for (i = 0; i < 4; i++)
        even[i] = src[2 * i];
    idct_tr4_set(e, even);
    for (i = 0; i < 4; i++) {
        dst[i] = _mm_add_epi32(e[i], o[i]);
        dst[7 - i] = _mm_sub_epi32(e[i], o[i]);
    }
}

IDCT_AVX2 static void idct_tr8_scale(__m128i *dst, const __m128i *src,
                                     int end, int shift, int add)
{
    __m128i tmp[8];
    int i;
    idct_tr8_set(tmp, src, end);
    for (i = 0; i < 8; i++)
        dst[i] = idct_scale4(tmp[i], shift, add);
}

IDCT_AVX2 static void idct_tr16_set(__m128i *dst, const __m128i *src, int end)
{
    __m128i e[8], o[8], even[8];
    int i, j;
    for (i = 0; i < 8; i++)
        o[i] = _mm_setzero_si128();
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 8; i++)
            o[i] = _mm_add_epi32(o[i],
                _mm_mullo_epi32(src[j], _mm_set1_epi32(transform[2 * j][i])));
    for (i = 0; i < 8; i++)
        even[i] = src[2 * i];
    idct_tr8_set(e, even, 8);
    for (i = 0; i < 8; i++) {
        dst[i] = _mm_add_epi32(e[i], o[i]);
        dst[15 - i] = _mm_sub_epi32(e[i], o[i]);
    }
}

IDCT_AVX2 static void idct_tr16_scale(__m128i *dst, const __m128i *src,
                                      int end, int shift, int add)
{
    __m128i tmp[16];
    int i;
    idct_tr16_set(tmp, src, end);
    for (i = 0; i < 16; i++)
        dst[i] = idct_scale4(tmp[i], shift, add);
}

IDCT_AVX2 static void idct_tr32_set(__m128i *dst, const __m128i *src, int end)
{
    __m128i e[16], o[16], even[16];
    int i, j;
    for (i = 0; i < 16; i++)
        o[i] = _mm_setzero_si128();
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 16; i++)
            o[i] = _mm_add_epi32(o[i],
                _mm_mullo_epi32(src[j], _mm_set1_epi32(transform[j][i])));
    for (i = 0; i < 16; i++)
        even[i] = src[2 * i];
    idct_tr16_set(e, even, end / 2);
    for (i = 0; i < 16; i++) {
        dst[i] = _mm_add_epi32(e[i], o[i]);
        dst[31 - i] = _mm_sub_epi32(e[i], o[i]);
    }
}

IDCT_AVX2 static void idct_tr32_scale(__m128i *dst, const __m128i *src,
                                      int end, int shift, int add)
{
    __m128i tmp[32];
    int i;
    idct_tr32_set(tmp, src, end);
    for (i = 0; i < 32; i++)
        dst[i] = idct_scale4(tmp[i], shift, add);
}

/* Eight columns at a time (AVX2 256-bit lanes).  Used when all columns of
 * the transform share the same coefficient limit. */
IDCT_AVX2 static __m256i idct_load8(const int16_t *p)
{
    return _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)p));
}

IDCT_AVX2 static void idct_store8(int16_t *p, __m256i v)
{
    __m256i pk = _mm256_packs_epi32(v, v);
    __m128i lo = _mm256_castsi256_si128(pk);
    __m128i hi = _mm256_extracti128_si256(pk, 1);
    _mm_storeu_si128((__m128i *)p, _mm_unpacklo_epi64(lo, hi));
}

IDCT_AVX2 static __m256i idct_scale8(__m256i v, int shift, int add)
{
    v = _mm256_add_epi32(v, _mm256_set1_epi32(add));
    v = _mm256_srav_epi32(v, _mm256_set1_epi32(shift));
    v = _mm256_max_epi32(v, _mm256_set1_epi32(-32768));
    v = _mm256_min_epi32(v, _mm256_set1_epi32(32767));
    return v;
}

IDCT_AVX2 static void idct_tr4_set8(__m256i *dst, const __m256i *src)
{
    __m256i s0 = src[0], s1 = src[1], s2 = src[2], s3 = src[3];
    __m256i e0 = _mm256_mullo_epi32(_mm256_add_epi32(s0, s2), _mm256_set1_epi32(64));
    __m256i e1 = _mm256_mullo_epi32(_mm256_sub_epi32(s0, s2), _mm256_set1_epi32(64));
    __m256i o0 = _mm256_add_epi32(_mm256_mullo_epi32(s1, _mm256_set1_epi32(83)),
                                  _mm256_mullo_epi32(s3, _mm256_set1_epi32(36)));
    __m256i o1 = _mm256_sub_epi32(_mm256_mullo_epi32(s1, _mm256_set1_epi32(36)),
                                  _mm256_mullo_epi32(s3, _mm256_set1_epi32(83)));

    dst[0] = _mm256_add_epi32(e0, o0);
    dst[1] = _mm256_add_epi32(e1, o1);
    dst[2] = _mm256_sub_epi32(e1, o1);
    dst[3] = _mm256_sub_epi32(e0, o0);
}

IDCT_AVX2 static void idct_tr8_set8(__m256i *dst, const __m256i *src, int end)
{
    __m256i e[4], o[4], even[4];
    int i, j;
    for (i = 0; i < 4; i++)
        o[i] = _mm256_setzero_si256();
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 4; i++)
            o[i] = _mm256_add_epi32(o[i], _mm256_mullo_epi32(
                src[j], _mm256_set1_epi32(transform[4 * j][i])));
    for (i = 0; i < 4; i++)
        even[i] = src[2 * i];
    idct_tr4_set8(e, even);
    for (i = 0; i < 4; i++) {
        dst[i] = _mm256_add_epi32(e[i], o[i]);
        dst[7 - i] = _mm256_sub_epi32(e[i], o[i]);
    }
}

IDCT_AVX2 static void idct_tr8_scale8(__m256i *dst, const __m256i *src,
                                      int end, int shift, int add)
{
    __m256i tmp[8];
    int i;
    idct_tr8_set8(tmp, src, end);
    for (i = 0; i < 8; i++)
        dst[i] = idct_scale8(tmp[i], shift, add);
}

IDCT_AVX2 static void idct_tr16_set8(__m256i *dst, const __m256i *src, int end)
{
    __m256i e[8], o[8], even[8];
    int i, j;
    for (i = 0; i < 8; i++)
        o[i] = _mm256_setzero_si256();
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 8; i++)
            o[i] = _mm256_add_epi32(o[i], _mm256_mullo_epi32(
                src[j], _mm256_set1_epi32(transform[2 * j][i])));
    for (i = 0; i < 8; i++)
        even[i] = src[2 * i];
    idct_tr8_set8(e, even, 8);
    for (i = 0; i < 8; i++) {
        dst[i] = _mm256_add_epi32(e[i], o[i]);
        dst[15 - i] = _mm256_sub_epi32(e[i], o[i]);
    }
}

IDCT_AVX2 static void idct_tr16_scale8(__m256i *dst, const __m256i *src,
                                       int end, int shift, int add)
{
    __m256i tmp[16];
    int i;
    idct_tr16_set8(tmp, src, end);
    for (i = 0; i < 16; i++)
        dst[i] = idct_scale8(tmp[i], shift, add);
}

IDCT_AVX2 static void idct_tr32_set8(__m256i *dst, const __m256i *src, int end)
{
    __m256i e[16], o[16], even[16];
    int i, j;
    for (i = 0; i < 16; i++)
        o[i] = _mm256_setzero_si256();
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 16; i++)
            o[i] = _mm256_add_epi32(o[i], _mm256_mullo_epi32(
                src[j], _mm256_set1_epi32(transform[j][i])));
    for (i = 0; i < 16; i++)
        even[i] = src[2 * i];
    idct_tr16_set8(e, even, end / 2);
    for (i = 0; i < 16; i++) {
        dst[i] = _mm256_add_epi32(e[i], o[i]);
        dst[31 - i] = _mm256_sub_epi32(e[i], o[i]);
    }
}

IDCT_AVX2 static void idct_tr32_scale8(__m256i *dst, const __m256i *src,
                                       int end, int shift, int add)
{
    __m256i tmp[32];
    int i;
    idct_tr32_set8(tmp, src, end);
    for (i = 0; i < 32; i++)
        dst[i] = idct_scale8(tmp[i], shift, add);
}

IDCT_AVX2 static void idct_cols_pass8_avx2(int16_t *coeffs, int H, int limit,
                                           int shift, int add)
{
    int c0, i, j;
    for (c0 = 0; c0 + 8 <= H; c0 += 8) {
        __m256i src[32], dst[32];
        for (j = 0; j < H; j++)
            src[j] = idct_load8(&coeffs[j * H + c0]);
        switch (H) {
        case 8:  idct_tr8_scale8(dst, src, limit, shift, add); break;
        case 16: idct_tr16_scale8(dst, src, limit, shift, add); break;
        default: idct_tr32_scale8(dst, src, limit, shift, add); break;
        }
        for (i = 0; i < H; i++)
            idct_store8(&coeffs[i * H + c0], dst[i]);
    }
}

static void idct_col_scalar(int16_t *col, int H, int limit2);

IDCT_AVX2 static void idct_cols_avx2(int16_t *coeffs, int H, int limit2)
{
    int c0, i, j;
    int lim = limit2;
    int do_dec = limit2 < H;
    if (!do_dec && H >= 8) {
        idct_cols_pass8_avx2(coeffs, H, limit2, 7, 64);
        return;
    }
    for (c0 = 0; c0 < H; c0 += 4) {
        __m128i src[32], dst[32];
        int vec_lim = (c0 == 0 || !do_dec) ? lim : lim - 4;
        int16_t fixed_col[32];
        int fix_col = (c0 > 0 && do_dec);
        if (fix_col) {
            for (j = 0; j < H; j++)
                fixed_col[j] = coeffs[j * H + c0];
            idct_col_scalar(fixed_col, H, lim);
        }
        for (j = 0; j < H; j++)
            src[j] = idct_load4(&coeffs[j * H + c0]);
        switch (H) {
        case 4:  idct_tr4_scale(dst, src, 7, 64); break;
        case 8:  idct_tr8_scale(dst, src, vec_lim, 7, 64); break;
        case 16: idct_tr16_scale(dst, src, vec_lim, 7, 64); break;
        default: idct_tr32_scale(dst, src, vec_lim, 7, 64); break;
        }
        for (i = 0; i < H; i++)
            idct_store4(&coeffs[i * H + c0], dst[i]);
        if (fix_col) {
            for (j = 0; j < H; j++)
                coeffs[j * H + c0] = fixed_col[j];
            lim -= 4;
        }
    }
}

IDCT_AVX2 static void idct_cols_pass_avx2(int16_t *coeffs, int H, int limit,
                                          int shift, int add)
{
    int c0, i, j;
    for (c0 = 0; c0 < H; c0 += 4) {
        __m128i src[32], dst[32];
        for (j = 0; j < H; j++)
            src[j] = idct_load4(&coeffs[j * H + c0]);
        switch (H) {
        case 4:  idct_tr4_scale(dst, src, shift, add); break;
        case 8:  idct_tr8_scale(dst, src, limit, shift, add); break;
        case 16: idct_tr16_scale(dst, src, limit, shift, add); break;
        default: idct_tr32_scale(dst, src, limit, shift, add); break;
        }
        for (i = 0; i < H; i++)
            idct_store4(&coeffs[i * H + c0], dst[i]);
    }
}

IDCT_AVX2 static void idct_transpose8(int16_t *dst, int dst_stride,
                                      const int16_t *src, int src_stride)
{
    __m128i r0 = _mm_loadu_si128((const __m128i *)(src + 0 * src_stride));
    __m128i r1 = _mm_loadu_si128((const __m128i *)(src + 1 * src_stride));
    __m128i r2 = _mm_loadu_si128((const __m128i *)(src + 2 * src_stride));
    __m128i r3 = _mm_loadu_si128((const __m128i *)(src + 3 * src_stride));
    __m128i r4 = _mm_loadu_si128((const __m128i *)(src + 4 * src_stride));
    __m128i r5 = _mm_loadu_si128((const __m128i *)(src + 5 * src_stride));
    __m128i r6 = _mm_loadu_si128((const __m128i *)(src + 6 * src_stride));
    __m128i r7 = _mm_loadu_si128((const __m128i *)(src + 7 * src_stride));
    __m128i t0 = _mm_unpacklo_epi16(r0, r1);
    __m128i t1 = _mm_unpackhi_epi16(r0, r1);
    __m128i t2 = _mm_unpacklo_epi16(r2, r3);
    __m128i t3 = _mm_unpackhi_epi16(r2, r3);
    __m128i t4 = _mm_unpacklo_epi16(r4, r5);
    __m128i t5 = _mm_unpackhi_epi16(r4, r5);
    __m128i t6 = _mm_unpacklo_epi16(r6, r7);
    __m128i t7 = _mm_unpackhi_epi16(r6, r7);
    __m128i u0 = _mm_unpacklo_epi32(t0, t2);
    __m128i u1 = _mm_unpackhi_epi32(t0, t2);
    __m128i u2 = _mm_unpacklo_epi32(t1, t3);
    __m128i u3 = _mm_unpackhi_epi32(t1, t3);
    __m128i u4 = _mm_unpacklo_epi32(t4, t6);
    __m128i u5 = _mm_unpackhi_epi32(t4, t6);
    __m128i u6 = _mm_unpacklo_epi32(t5, t7);
    __m128i u7 = _mm_unpackhi_epi32(t5, t7);
    __m128i v0 = _mm_unpacklo_epi64(u0, u4);
    __m128i v1 = _mm_unpackhi_epi64(u0, u4);
    __m128i v2 = _mm_unpacklo_epi64(u1, u5);
    __m128i v3 = _mm_unpackhi_epi64(u1, u5);
    __m128i v4 = _mm_unpacklo_epi64(u2, u6);
    __m128i v5 = _mm_unpackhi_epi64(u2, u6);
    __m128i v6 = _mm_unpacklo_epi64(u3, u7);
    __m128i v7 = _mm_unpackhi_epi64(u3, u7);
    _mm_storeu_si128((__m128i *)(dst + 0 * dst_stride), v0);
    _mm_storeu_si128((__m128i *)(dst + 1 * dst_stride), v1);
    _mm_storeu_si128((__m128i *)(dst + 2 * dst_stride), v2);
    _mm_storeu_si128((__m128i *)(dst + 3 * dst_stride), v3);
    _mm_storeu_si128((__m128i *)(dst + 4 * dst_stride), v4);
    _mm_storeu_si128((__m128i *)(dst + 5 * dst_stride), v5);
    _mm_storeu_si128((__m128i *)(dst + 6 * dst_stride), v6);
    _mm_storeu_si128((__m128i *)(dst + 7 * dst_stride), v7);
}

IDCT_AVX2 static void idct_transpose_avx2(int16_t *a, int H)
{
    if (H == 4) {
        int i, j;
        for (i = 0; i < 4; i++)
            for (j = i + 1; j < 4; j++) {
                int16_t t = a[i * 4 + j];
                a[i * 4 + j] = a[j * 4 + i];
                a[j * 4 + i] = t;
            }
    } else {
        int n = H / 8;
        int ti, tj;
        int16_t tmp[64];
        for (ti = 0; ti < n; ti++) {
            for (tj = ti; tj < n; tj++) {
                int16_t *A = a + (ti * 8) * H + tj * 8;
                int16_t *B = a + (tj * 8) * H + ti * 8;
                if (ti == tj) {
                    int i;
                    idct_transpose8(tmp, 8, A, H);
                    for (i = 0; i < 8; i++)
                        memcpy(A + i * H, tmp + i * 8,
                               8 * sizeof(int16_t));
                } else {
                    int16_t tb[64];
                    int i;
                    idct_transpose8(tb, 8, B, H);
                    idct_transpose8(B, H, A, H);
                    for (i = 0; i < 8; i++)
                        memcpy(A + i * H, tb + i * 8,
                               8 * sizeof(int16_t));
                }
            }
        }
    }
}
#endif

#ifdef H265_NEON
static void idct_col_scalar(int16_t *col, int H, int limit2);

/* Four columns are transformed in parallel with NEON.  This mirrors the SSE
 * 4-lane path above, using int32x4_t lanes. */
static int32x4_t idct_load4_neon(const int16_t *p)
{
    return vmovl_s16(vld1_s16(p));
}

static void idct_store4_neon(int16_t *p, int32x4_t v)
{
    vst1_s16(p, vqmovn_s32(v));
}

static int32x4_t idct_scale4_neon(int32x4_t v, int shift, int add)
{
    v = vaddq_s32(v, vdupq_n_s32(add));
    v = vshlq_s32(v, vdupq_n_s32(-shift));
    v = vmaxq_s32(v, vdupq_n_s32(-32768));
    v = vminq_s32(v, vdupq_n_s32(32767));
    return v;
}

static void idct_tr4_set_neon(int32x4_t *dst, const int32x4_t *src)
{
    int32x4_t s0 = src[0], s1 = src[1], s2 = src[2], s3 = src[3];
    int32x4_t c64 = vdupq_n_s32(64);
    int32x4_t e0 = vmulq_s32(vaddq_s32(s0, s2), c64);
    int32x4_t e1 = vmulq_s32(vsubq_s32(s0, s2), c64);
    int32x4_t o0 = vaddq_s32(vmulq_s32(s1, vdupq_n_s32(83)),
                             vmulq_s32(s3, vdupq_n_s32(36)));
    int32x4_t o1 = vsubq_s32(vmulq_s32(s1, vdupq_n_s32(36)),
                             vmulq_s32(s3, vdupq_n_s32(83)));

    dst[0] = vaddq_s32(e0, o0);
    dst[1] = vaddq_s32(e1, o1);
    dst[2] = vsubq_s32(e1, o1);
    dst[3] = vsubq_s32(e0, o0);
}

static void idct_tr4_scale_neon(int32x4_t *dst, const int32x4_t *src,
                                int shift, int add)
{
    int32x4_t tmp[4];
    int i;
    idct_tr4_set_neon(tmp, src);
    for (i = 0; i < 4; i++)
        dst[i] = idct_scale4_neon(tmp[i], shift, add);
}

static void idct_tr8_set_neon(int32x4_t *dst, const int32x4_t *src, int end)
{
    int32x4_t e[4], o[4], even[4];
    int i, j;
    for (i = 0; i < 4; i++)
        o[i] = vdupq_n_s32(0);
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 4; i++)
            o[i] = vmlaq_s32(o[i], src[j],
                             vdupq_n_s32(transform[4 * j][i]));
    for (i = 0; i < 4; i++)
        even[i] = src[2 * i];
    idct_tr4_set_neon(e, even);
    for (i = 0; i < 4; i++) {
        dst[i] = vaddq_s32(e[i], o[i]);
        dst[7 - i] = vsubq_s32(e[i], o[i]);
    }
}

static void idct_tr8_scale_neon(int32x4_t *dst, const int32x4_t *src,
                                int end, int shift, int add)
{
    int32x4_t tmp[8];
    int i;
    idct_tr8_set_neon(tmp, src, end);
    for (i = 0; i < 8; i++)
        dst[i] = idct_scale4_neon(tmp[i], shift, add);
}

static void idct_tr16_set_neon(int32x4_t *dst, const int32x4_t *src, int end)
{
    int32x4_t e[8], o[8], even[8];
    int i, j;
    for (i = 0; i < 8; i++)
        o[i] = vdupq_n_s32(0);
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 8; i++)
            o[i] = vmlaq_s32(o[i], src[j],
                             vdupq_n_s32(transform[2 * j][i]));
    for (i = 0; i < 8; i++)
        even[i] = src[2 * i];
    idct_tr8_set_neon(e, even, 8);
    for (i = 0; i < 8; i++) {
        dst[i] = vaddq_s32(e[i], o[i]);
        dst[15 - i] = vsubq_s32(e[i], o[i]);
    }
}

static void idct_tr16_scale_neon(int32x4_t *dst, const int32x4_t *src,
                                 int end, int shift, int add)
{
    int32x4_t tmp[16];
    int i;
    idct_tr16_set_neon(tmp, src, end);
    for (i = 0; i < 16; i++)
        dst[i] = idct_scale4_neon(tmp[i], shift, add);
}

static void idct_tr32_set_neon(int32x4_t *dst, const int32x4_t *src, int end)
{
    int32x4_t e[16], o[16], even[16];
    int i, j;
    for (i = 0; i < 16; i++)
        o[i] = vdupq_n_s32(0);
    for (j = 1; j < end; j += 2)
        for (i = 0; i < 16; i++)
            o[i] = vmlaq_s32(o[i], src[j],
                             vdupq_n_s32(transform[j][i]));
    for (i = 0; i < 16; i++)
        even[i] = src[2 * i];
    idct_tr16_set_neon(e, even, end / 2);
    for (i = 0; i < 16; i++) {
        dst[i] = vaddq_s32(e[i], o[i]);
        dst[31 - i] = vsubq_s32(e[i], o[i]);
    }
}

static void idct_tr32_scale_neon(int32x4_t *dst, const int32x4_t *src,
                                 int end, int shift, int add)
{
    int32x4_t tmp[32];
    int i;
    idct_tr32_set_neon(tmp, src, end);
    for (i = 0; i < 32; i++)
        dst[i] = idct_scale4_neon(tmp[i], shift, add);
}

static void idct_cols_neon(int16_t *coeffs, int H, int limit2)
{
    int c0, i, j;
    int lim = limit2;
    int do_dec = limit2 < H;
    for (c0 = 0; c0 < H; c0 += 4) {
        int32x4_t src[32], dst[32];
        int vec_lim = (c0 == 0 || !do_dec) ? lim : lim - 4;
        int16_t fixed_col[32];
        int fix_col = (c0 > 0 && do_dec);
        if (fix_col) {
            for (j = 0; j < H; j++)
                fixed_col[j] = coeffs[j * H + c0];
            idct_col_scalar(fixed_col, H, lim);
        }
        for (j = 0; j < H; j++)
            src[j] = idct_load4_neon(&coeffs[j * H + c0]);
        switch (H) {
        case 4:  idct_tr4_scale_neon(dst, src, 7, 64); break;
        case 8:  idct_tr8_scale_neon(dst, src, vec_lim, 7, 64); break;
        case 16: idct_tr16_scale_neon(dst, src, vec_lim, 7, 64); break;
        default: idct_tr32_scale_neon(dst, src, vec_lim, 7, 64); break;
        }
        for (i = 0; i < H; i++)
            idct_store4_neon(&coeffs[i * H + c0], dst[i]);
        if (fix_col) {
            for (j = 0; j < H; j++)
                coeffs[j * H + c0] = fixed_col[j];
            lim -= 4;
        }
    }
}

static void idct_cols_pass_neon(int16_t *coeffs, int H, int limit,
                                int shift, int add)
{
    int c0, i, j;
    for (c0 = 0; c0 < H; c0 += 4) {
        int32x4_t src[32], dst[32];
        for (j = 0; j < H; j++)
            src[j] = idct_load4_neon(&coeffs[j * H + c0]);
        switch (H) {
        case 4:  idct_tr4_scale_neon(dst, src, shift, add); break;
        case 8:  idct_tr8_scale_neon(dst, src, limit, shift, add); break;
        case 16: idct_tr16_scale_neon(dst, src, limit, shift, add); break;
        default: idct_tr32_scale_neon(dst, src, limit, shift, add); break;
        }
        for (i = 0; i < H; i++)
            idct_store4_neon(&coeffs[i * H + c0], dst[i]);
    }
}

static void idct_transpose8_neon(int16_t *dst, int dst_stride,
                                 const int16_t *src, int src_stride)
{
    int16x8_t r0 = vld1q_s16(src + 0 * src_stride);
    int16x8_t r1 = vld1q_s16(src + 1 * src_stride);
    int16x8_t r2 = vld1q_s16(src + 2 * src_stride);
    int16x8_t r3 = vld1q_s16(src + 3 * src_stride);
    int16x8_t r4 = vld1q_s16(src + 4 * src_stride);
    int16x8_t r5 = vld1q_s16(src + 5 * src_stride);
    int16x8_t r6 = vld1q_s16(src + 6 * src_stride);
    int16x8_t r7 = vld1q_s16(src + 7 * src_stride);
    int16x8_t t0 = vzip1q_s16(r0, r1);
    int16x8_t t1 = vzip2q_s16(r0, r1);
    int16x8_t t2 = vzip1q_s16(r2, r3);
    int16x8_t t3 = vzip2q_s16(r2, r3);
    int16x8_t t4 = vzip1q_s16(r4, r5);
    int16x8_t t5 = vzip2q_s16(r4, r5);
    int16x8_t t6 = vzip1q_s16(r6, r7);
    int16x8_t t7 = vzip2q_s16(r6, r7);
    int16x8_t u0 = vreinterpretq_s16_s32(vzip1q_s32(
        vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2)));
    int16x8_t u1 = vreinterpretq_s16_s32(vzip2q_s32(
        vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2)));
    int16x8_t u2 = vreinterpretq_s16_s32(vzip1q_s32(
        vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3)));
    int16x8_t u3 = vreinterpretq_s16_s32(vzip2q_s32(
        vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3)));
    int16x8_t u4 = vreinterpretq_s16_s32(vzip1q_s32(
        vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6)));
    int16x8_t u5 = vreinterpretq_s16_s32(vzip2q_s32(
        vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6)));
    int16x8_t u6 = vreinterpretq_s16_s32(vzip1q_s32(
        vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7)));
    int16x8_t u7 = vreinterpretq_s16_s32(vzip2q_s32(
        vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7)));
    int16x8_t v0 = vreinterpretq_s16_s64(vzip1q_s64(
        vreinterpretq_s64_s16(u0), vreinterpretq_s64_s16(u4)));
    int16x8_t v1 = vreinterpretq_s16_s64(vzip2q_s64(
        vreinterpretq_s64_s16(u0), vreinterpretq_s64_s16(u4)));
    int16x8_t v2 = vreinterpretq_s16_s64(vzip1q_s64(
        vreinterpretq_s64_s16(u1), vreinterpretq_s64_s16(u5)));
    int16x8_t v3 = vreinterpretq_s16_s64(vzip2q_s64(
        vreinterpretq_s64_s16(u1), vreinterpretq_s64_s16(u5)));
    int16x8_t v4 = vreinterpretq_s16_s64(vzip1q_s64(
        vreinterpretq_s64_s16(u2), vreinterpretq_s64_s16(u6)));
    int16x8_t v5 = vreinterpretq_s16_s64(vzip2q_s64(
        vreinterpretq_s64_s16(u2), vreinterpretq_s64_s16(u6)));
    int16x8_t v6 = vreinterpretq_s16_s64(vzip1q_s64(
        vreinterpretq_s64_s16(u3), vreinterpretq_s64_s16(u7)));
    int16x8_t v7 = vreinterpretq_s16_s64(vzip2q_s64(
        vreinterpretq_s64_s16(u3), vreinterpretq_s64_s16(u7)));
    vst1q_s16(dst + 0 * dst_stride, v0);
    vst1q_s16(dst + 1 * dst_stride, v1);
    vst1q_s16(dst + 2 * dst_stride, v2);
    vst1q_s16(dst + 3 * dst_stride, v3);
    vst1q_s16(dst + 4 * dst_stride, v4);
    vst1q_s16(dst + 5 * dst_stride, v5);
    vst1q_s16(dst + 6 * dst_stride, v6);
    vst1q_s16(dst + 7 * dst_stride, v7);
}

static void idct_transpose_neon(int16_t *a, int H)
{
    if (H == 4) {
        int i, j;
        for (i = 0; i < 4; i++)
            for (j = i + 1; j < 4; j++) {
                int16_t t = a[i * 4 + j];
                a[i * 4 + j] = a[j * 4 + i];
                a[j * 4 + i] = t;
            }
    } else {
        int n = H / 8;
        int ti, tj;
        int16_t tmp[64];
        for (ti = 0; ti < n; ti++) {
            for (tj = ti; tj < n; tj++) {
                int16_t *A = a + (ti * 8) * H + tj * 8;
                int16_t *B = a + (tj * 8) * H + ti * 8;
                if (ti == tj) {
                    int i;
                    idct_transpose8_neon(tmp, 8, A, H);
                    for (i = 0; i < 8; i++)
                        memcpy(A + i * H, tmp + i * 8,
                               8 * sizeof(int16_t));
                } else {
                    int16_t tb[64];
                    int i;
                    idct_transpose8_neon(tb, 8, B, H);
                    idct_transpose8_neon(B, H, A, H);
                    for (i = 0; i < 8; i++)
                        memcpy(A + i * H, tb + i * 8,
                               8 * sizeof(int16_t));
                }
            }
        }
    }
}
#endif /* H265_NEON */

void h265_dequant(int16_t *coeffs, int log2_size, int bit_depth)
{
    int shift = 15 - bit_depth - log2_size;
    int size = 1 << log2_size;
    int x, y;

    if (shift > 0) {
        int offset = 1 << (shift - 1);
        for (y = 0; y < size; y++)
            for (x = 0; x < size; x++) {
                *coeffs = (int16_t)((*coeffs + offset) >> shift);
                coeffs++;
            }
    } else if (shift < 0) {
        for (y = 0; y < size; y++)
            for (x = 0; x < size; x++) {
                *coeffs = (int16_t)(*(uint16_t *)coeffs << -shift);
                coeffs++;
            }
    }
}

#define SET(dst, x)   (dst) = (x)
#define SCALE(dst, x) (dst) = clip_int16(((x) + add) >> shift)

#define TR_4x4_LUMA(dst, src, step, assign)                             \
    do {                                                                \
        int c0 = src[0 * step] + src[2 * step];                         \
        int c1 = src[2 * step] + src[3 * step];                         \
        int c2 = src[0 * step] - src[3 * step];                         \
        int c3 = 74 * src[1 * step];                                    \
                                                                        \
        assign(dst[2 * step], 74 * (src[0 * step] -                     \
                                    src[2 * step] +                     \
                                    src[3 * step]));                    \
        assign(dst[0 * step], 29 * c0 + 55 * c1 + c3);                  \
        assign(dst[1 * step], 55 * c2 - 29 * c1 + c3);                  \
        assign(dst[3 * step], 55 * c0 + 29 * c2 - c3);                  \
    } while (0)

void h265_transform_4x4_luma(int16_t *coeffs, int bit_depth)
{
    int i;
    int shift = 7;
    int add = 1 << (shift - 1);
    int16_t *src = coeffs;

    for (i = 0; i < 4; i++) {
        TR_4x4_LUMA(src, src, 4, SCALE);
        src++;
    }
    shift = 20 - bit_depth;
    add = 1 << (shift - 1);
    for (i = 0; i < 4; i++) {
        TR_4x4_LUMA(coeffs, coeffs, 1, SCALE);
        coeffs += 4;
    }
}

#undef TR_4x4_LUMA

#define TR_4(dst, src, dstep, sstep, assign, end)                 \
    do {                                                          \
        const int e0 = 64 * src[0 * sstep] + 64 * src[2 * sstep]; \
        const int e1 = 64 * src[0 * sstep] - 64 * src[2 * sstep]; \
        const int o0 = 83 * src[1 * sstep] + 36 * src[3 * sstep]; \
        const int o1 = 36 * src[1 * sstep] - 83 * src[3 * sstep]; \
                                                                  \
        assign(dst[0 * dstep], e0 + o0);                          \
        assign(dst[1 * dstep], e1 + o1);                          \
        assign(dst[2 * dstep], e1 - o1);                          \
        assign(dst[3 * dstep], e0 - o0);                          \
    } while (0)

#define TR_8(dst, src, dstep, sstep, assign, end)                 \
    do {                                                          \
        int i, j;                                                 \
        int e_8[4];                                               \
        int o_8[4] = { 0 };                                       \
        for (i = 0; i < 4; i++)                                   \
            for (j = 1; j < end; j += 2)                          \
                o_8[i] += transform[4 * j][i] * src[j * sstep];   \
        TR_4(e_8, src, 1, 2 * sstep, SET, 4);                     \
                                                                  \
        for (i = 0; i < 4; i++) {                                 \
            assign(dst[i * dstep], e_8[i] + o_8[i]);              \
            assign(dst[(7 - i) * dstep], e_8[i] - o_8[i]);        \
        }                                                         \
    } while (0)

#define TR_16(dst, src, dstep, sstep, assign, end)                \
    do {                                                          \
        int i, j;                                                 \
        int e_16[8];                                              \
        int o_16[8] = { 0 };                                      \
        for (i = 0; i < 8; i++)                                   \
            for (j = 1; j < end; j += 2)                          \
                o_16[i] += transform[2 * j][i] * src[j * sstep];  \
        TR_8(e_16, src, 1, 2 * sstep, SET, 8);                    \
                                                                  \
        for (i = 0; i < 8; i++) {                                 \
            assign(dst[i * dstep], e_16[i] + o_16[i]);            \
            assign(dst[(15 - i) * dstep], e_16[i] - o_16[i]);     \
        }                                                         \
    } while (0)

#define TR_32(dst, src, dstep, sstep, assign, end)                \
    do {                                                          \
        int i, j;                                                 \
        int e_32[16];                                             \
        int o_32[16] = { 0 };                                     \
        for (i = 0; i < 16; i++)                                  \
            for (j = 1; j < end; j += 2)                          \
                o_32[i] += transform[j][i] * src[j * sstep];      \
        TR_16(e_32, src, 1, 2 * sstep, SET, end / 2);             \
                                                                  \
        for (i = 0; i < 16; i++) {                                \
            assign(dst[i * dstep], e_32[i] + o_32[i]);            \
            assign(dst[(31 - i) * dstep], e_32[i] - o_32[i]);     \
        }                                                         \
    } while (0)

#define IDCT_VAR4(H)                                              \
    int limit = H;                                                \
    int limit2 = H265_MIN(col_limit + 4, H)
#define IDCT_VAR8(H)                                              \
    int limit  = H265_MIN(col_limit, H);                          \
    int limit2 = H265_MIN(col_limit + 4, H)
#define IDCT_VAR16(H)   IDCT_VAR8(H)
#define IDCT_VAR32(H)   IDCT_VAR8(H)

#define IDCT_COLS(H)                                              \
static void idct_ ## H ## x ## H ## _cols(int16_t *coeffs,        \
                                          int limit2)             \
{                                                                 \
    int i;                                                        \
    int shift = 7;                                                \
    int add = 1 << (shift - 1);                                   \
    int16_t *src = coeffs;                                        \
    for (i = 0; i < H; i++) {                                     \
        TR_ ## H(src, src, H, H, SCALE, limit2);                  \
        if (limit2 < H && i % 4 == 0 && !!i)                      \
            limit2 -= 4;                                          \
        src++;                                                    \
    }                                                             \
}

IDCT_COLS( 4)
IDCT_COLS( 8)
IDCT_COLS(16)
IDCT_COLS(32)

static void idct_col_scalar(int16_t *col, int H, int limit2)
{
    int shift = 7;
    int add = 1 << (shift - 1);
    switch (H) {
    case 4:  TR_4(col, col, 1, 1, SCALE, limit2); break;
    case 8:  TR_8(col, col, 1, 1, SCALE, limit2); break;
    case 16: TR_16(col, col, 1, 1, SCALE, limit2); break;
    default: TR_32(col, col, 1, 1, SCALE, limit2); break;
    }
}

#define IDCT_ROWS(H)                                              \
static void idct_ ## H ## x ## H ## _rows(int16_t *coeffs,        \
                                           int limit, int shift,  \
                                           int add)               \
{                                                                 \
    int i;                                                        \
    for (i = 0; i < H; i++) {                                     \
        TR_ ## H(coeffs, coeffs, 1, 1, SCALE, limit);             \
        coeffs += H;                                              \
    }                                                             \
}

IDCT_ROWS( 4)
IDCT_ROWS( 8)
IDCT_ROWS(16)
IDCT_ROWS(32)

static void idct_rows_dispatch(int16_t *coeffs, int H, int limit,
                               int shift, int add)
{
#ifdef H265_X86
    if (h265_have_avx2) {
        idct_transpose_avx2(coeffs, H);
        if (H >= 8)
            idct_cols_pass8_avx2(coeffs, H, limit, shift, add);
        else
            idct_cols_pass_avx2(coeffs, H, limit, shift, add);
        idct_transpose_avx2(coeffs, H);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        idct_transpose_neon(coeffs, H);
        idct_cols_pass_neon(coeffs, H, limit, shift, add);
        idct_transpose_neon(coeffs, H);
        return;
    }
#endif
    switch (H) {
    case 4:  idct_4x4_rows(coeffs, limit, shift, add); break;
    case 8:  idct_8x8_rows(coeffs, limit, shift, add); break;
    case 16: idct_16x16_rows(coeffs, limit, shift, add); break;
    default: idct_32x32_rows(coeffs, limit, shift, add); break;
    }
}

static void idct_cols_dispatch(int16_t *coeffs, int H, int limit2)
{
#ifdef H265_X86
    if (h265_have_avx2) {
        idct_cols_avx2(coeffs, H, limit2);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        idct_cols_neon(coeffs, H, limit2);
        return;
    }
#endif
    switch (H) {
    case 4:  idct_4x4_cols(coeffs, limit2); break;
    case 8:  idct_8x8_cols(coeffs, limit2); break;
    case 16: idct_16x16_cols(coeffs, limit2); break;
    default: idct_32x32_cols(coeffs, limit2); break;
    }
}

#define IDCT(H)                                                   \
static void idct_ ## H ## x ## H(int16_t *coeffs, int col_limit,  \
                                 int bit_depth)                   \
{                                                                 \
    int i;                                                        \
    int shift;                                                    \
    int add;                                                      \
    IDCT_VAR ## H(H);                                             \
                                                                  \
    /* The column pass is four-wide SIMD; the row pass stays       \
     * scalar and preserves the per-row rounding order. */        \
    idct_cols_dispatch(coeffs, H, limit2);                        \
                                                                  \
    shift = 20 - bit_depth;                                       \
    add = 1 << (shift - 1);                                       \
    idct_rows_dispatch(coeffs, H, limit, shift, add);             \
    (void)i;                                                      \
}

IDCT( 4)
IDCT( 8)
IDCT(16)
IDCT(32)

#define IDCT_DC(H)                                                \
static void idct_ ## H ## x ## H ## _dc(int16_t *coeffs, int bit_depth) \
{                                                                 \
    int i, j;                                                     \
    int shift = 14 - bit_depth;                                   \
    int add = 1 << (shift - 1);                                   \
    int coeff = (((coeffs[0] + 1) >> 1) + add) >> shift;          \
                                                                  \
    for (j = 0; j < H; j++)                                       \
        for (i = 0; i < H; i++)                                   \
            coeffs[i + j * H] = (int16_t)coeff;                   \
}

IDCT_DC( 4)
IDCT_DC( 8)
IDCT_DC(16)
IDCT_DC(32)

void h265_idct(int16_t *coeffs, int log2_size, int col_limit, int bit_depth)
{
    switch (log2_size) {
    case 2: idct_4x4(coeffs, col_limit, bit_depth); break;
    case 3: idct_8x8(coeffs, col_limit, bit_depth); break;
    case 4: idct_16x16(coeffs, col_limit, bit_depth); break;
    default: idct_32x32(coeffs, col_limit, bit_depth); break;
    }
}

void h265_idct_dc(int16_t *coeffs, int log2_size, int bit_depth)
{
    switch (log2_size) {
    case 2: idct_4x4_dc(coeffs, bit_depth); break;
    case 3: idct_8x8_dc(coeffs, bit_depth); break;
    case 4: idct_16x16_dc(coeffs, bit_depth); break;
    default: idct_32x32_dc(coeffs, bit_depth); break;
    }
}

void h265_transform_rdpcm(int16_t *coeffs, int log2_size, int mode)
{
    int x, y;
    int size = 1 << log2_size;

    if (mode) {
        coeffs += size;
        for (y = 0; y < size - 1; y++) {
            for (x = 0; x < size; x++)
                coeffs[x] = (int16_t)(coeffs[x] + coeffs[x - size]);
            coeffs += size;
        }
    } else {
        for (y = 0; y < size; y++) {
            for (x = 1; x < size; x++)
                coeffs[x] = (int16_t)(coeffs[x] + coeffs[x - 1]);
            coeffs += size;
        }
    }
}

void h265_add_residual(uint8_t *dst8, const int16_t *res, ptrdiff_t stride,
                       int size, int pixel_shift, int bit_depth)
{
    int x, y;
    int max = (1 << bit_depth) - 1;

#ifdef H265_X86
    if (bit_depth <= 12) {
        if (h265_have_avx2) {
            h265_add_residual_avx2(dst8, res, stride, size, pixel_shift, bit_depth);
            return;
        }
        if (h265_have_sse2) {
            h265_add_residual_sse2(dst8, res, stride, size, pixel_shift, bit_depth);
            return;
        }
    }
#endif
#ifdef H265_NEON
    if (bit_depth <= 12 && h265_have_neon) {
        h265_add_residual_neon(dst8, res, stride, size, pixel_shift, bit_depth);
        return;
    }
#endif

    if (!pixel_shift) {
        for (y = 0; y < size; y++) {
            uint8_t *dst = dst8;
            for (x = 0; x < size; x++) {
                int v = dst[x] + res[x];
                dst[x] = (uint8_t)(v < 0 ? 0 : v > max ? max : v);
            }
            dst8 += stride;
            res += size;
        }
    } else {
        for (y = 0; y < size; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            for (x = 0; x < size; x++) {
                int v = dst[x] + res[x];
                dst[x] = (uint16_t)(v < 0 ? 0 : v > max ? max : v);
            }
            dst8 += stride;
            res += size;
        }
    }
}

void h265_put_pcm(uint8_t *dst8, ptrdiff_t stride, int width, int height,
                  H265BitReader *gb, int pcm_bit_depth, int pixel_shift,
                  int bit_depth)
{
    int x, y;
    int left = bit_depth - pcm_bit_depth;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            for (x = 0; x < width; x++)
                dst[x] = (uint8_t)(h265_br_get_bits(gb, pcm_bit_depth) << left);
            dst8 += stride;
        }
    } else {
        for (y = 0; y < height; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            for (x = 0; x < width; x++)
                dst[x] = (uint16_t)(h265_br_get_bits(gb, pcm_bit_depth) << left);
            dst8 += stride;
        }
    }
}
