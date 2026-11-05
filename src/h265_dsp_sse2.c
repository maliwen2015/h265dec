/*
 * SSE2 kernels for HEVC reconstruction.
 */
#include "h265_internal.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>

#define SSE2_TARGET __attribute__((target("sse2")))

SSE2_TARGET void h265_add_residual_sse2(uint8_t *dst8, const int16_t *res,
                                        ptrdiff_t stride, int size,
                                        int pixel_shift, int bit_depth)
{
    int x, y;

    if (!pixel_shift) {
        for (y = 0; y < size; y++) {
            uint8_t *dst = dst8;
            x = 0;
            for (; x + 16 <= size; x += 16) {
                __m128i d = _mm_loadu_si128((const __m128i *)(dst + x));
                __m128i r0 = _mm_loadu_si128((const __m128i *)(res + x));
                __m128i r1 = _mm_loadu_si128((const __m128i *)(res + x + 8));
                __m128i lo = _mm_adds_epi16(_mm_unpacklo_epi8(d, _mm_setzero_si128()), r0);
                __m128i hi = _mm_adds_epi16(_mm_unpackhi_epi8(d, _mm_setzero_si128()), r1);
                lo = _mm_max_epi16(lo, _mm_setzero_si128());
                hi = _mm_max_epi16(hi, _mm_setzero_si128());
                d = _mm_packus_epi16(lo, hi);
                _mm_storeu_si128((__m128i *)(dst + x), d);
            }
            for (; x < size; x++) {
                int v = dst[x] + res[x];
                dst[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
            dst8 += stride;
            res += size;
        }
    } else if (bit_depth <= 12) {
        const __m128i zero = _mm_setzero_si128();
        const __m128i maxv = _mm_set1_epi16((short)((1 << bit_depth) - 1));
        for (y = 0; y < size; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            x = 0;
            for (; x + 8 <= size; x += 8) {
                __m128i d = _mm_loadu_si128((const __m128i *)(dst + x));
                __m128i r = _mm_loadu_si128((const __m128i *)(res + x));
                d = _mm_adds_epi16(d, r);
                d = _mm_max_epi16(d, zero);
                d = _mm_min_epi16(d, maxv);
                _mm_storeu_si128((__m128i *)(dst + x), d);
            }
            for (; x < size; x++) {
                int v = dst[x] + res[x];
                int mx = (1 << bit_depth) - 1;
                dst[x] = (uint16_t)(v < 0 ? 0 : v > mx ? mx : v);
            }
            dst8 += stride;
            res += size;
        }
    }
    /* bit_depth > 10 is handled by the scalar dispatch. */
}

SSE2_TARGET void h265_avg_pixels_sse2(uint8_t *dst, ptrdiff_t dst_stride,
                                      const uint8_t *src1, ptrdiff_t src1_stride,
                                      const uint8_t *src2, ptrdiff_t src2_stride,
                                      int width, int height, int pixel_shift)
{
    int x, y;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            x = 0;
            for (; x + 16 <= width; x += 16) {
                __m128i a = _mm_loadu_si128((const __m128i *)(src1 + x));
                __m128i b = _mm_loadu_si128((const __m128i *)(src2 + x));
                _mm_storeu_si128((__m128i *)(dst + x), _mm_avg_epu8(a, b));
            }
            for (; x < width; x++)
                dst[x] = (uint8_t)((src1[x] + src2[x] + 1) >> 1);
            dst += dst_stride;
            src1 += src1_stride;
            src2 += src2_stride;
        }
    } else {
        for (y = 0; y < height; y++) {
            const uint16_t *s1 = (const uint16_t *)src1;
            const uint16_t *s2 = (const uint16_t *)src2;
            uint16_t *d = (uint16_t *)dst;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                __m128i a = _mm_loadu_si128((const __m128i *)(s1 + x));
                __m128i b = _mm_loadu_si128((const __m128i *)(s2 + x));
                _mm_storeu_si128((__m128i *)(d + x), _mm_avg_epu16(a, b));
            }
            for (; x < width; x++)
                d[x] = (uint16_t)((s1[x] + s2[x] + 1) >> 1);
            dst += dst_stride;
            src1 += src1_stride;
            src2 += src2_stride;
        }
    }
}

SSE2_TARGET void h265_copy_pixels_sse2(uint8_t *dst, ptrdiff_t dst_stride,
                                       const uint8_t *src, ptrdiff_t src_stride,
                                       int width, int height, int pixel_shift)
{
    int y;
    int bytes = width << pixel_shift;
    for (y = 0; y < height; y++) {
        int x = 0;
        for (; x + 16 <= bytes; x += 16)
            _mm_storeu_si128((__m128i *)(dst + x),
                             _mm_loadu_si128((const __m128i *)(src + x)));
        for (; x < bytes; x++)
            dst[x] = src[x];
        dst += dst_stride;
        src += src_stride;
    }
}

#endif
