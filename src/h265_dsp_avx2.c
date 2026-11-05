/*
 * AVX2 kernels for HEVC reconstruction.
 */
#include "h265_internal.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>

#define AVX2_TARGET __attribute__((target("avx2")))

AVX2_TARGET void h265_add_residual_avx2(uint8_t *dst8, const int16_t *res,
                                        ptrdiff_t stride, int size,
                                        int pixel_shift, int bit_depth)
{
    int x, y;

    if (!pixel_shift) {
        for (y = 0; y < size; y++) {
            uint8_t *dst = dst8;
            x = 0;
            for (; x + 32 <= size; x += 32) {
                __m256i d = _mm256_loadu_si256((const __m256i *)(dst + x));
                __m256i r0 = _mm256_loadu_si256((const __m256i *)(res + x));
                __m256i r1 = _mm256_loadu_si256((const __m256i *)(res + x + 16));
                __m256i zero = _mm256_setzero_si256();
                __m256i rl = _mm256_permute2x128_si256(r0, r1, 0x20);
                __m256i rh = _mm256_permute2x128_si256(r0, r1, 0x31);
                __m256i lo = _mm256_adds_epi16(_mm256_unpacklo_epi8(d, zero), rl);
                __m256i hi = _mm256_adds_epi16(_mm256_unpackhi_epi8(d, zero), rh);
                lo = _mm256_max_epi16(lo, zero);
                hi = _mm256_max_epi16(hi, zero);
                d = _mm256_packus_epi16(lo, hi);
                _mm256_storeu_si256((__m256i *)(dst + x), d);
            }
            for (; x < size; x++) {
                int v = dst[x] + res[x];
                dst[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
            dst8 += stride;
            res += size;
        }
    } else if (bit_depth <= 12) {
        const __m256i zero = _mm256_setzero_si256();
        const __m256i maxv = _mm256_set1_epi16((short)((1 << bit_depth) - 1));
        for (y = 0; y < size; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            x = 0;
            for (; x + 16 <= size; x += 16) {
                __m256i d = _mm256_loadu_si256((const __m256i *)(dst + x));
                __m256i r = _mm256_loadu_si256((const __m256i *)(res + x));
                d = _mm256_adds_epi16(d, r);
                d = _mm256_max_epi16(d, zero);
                d = _mm256_min_epi16(d, maxv);
                _mm256_storeu_si256((__m256i *)(dst + x), d);
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

AVX2_TARGET void h265_avg_pixels_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                                      const uint8_t *src1, ptrdiff_t src1_stride,
                                      const uint8_t *src2, ptrdiff_t src2_stride,
                                      int width, int height, int pixel_shift)
{
    int x, y;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            x = 0;
            for (; x + 32 <= width; x += 32) {
                __m256i a = _mm256_loadu_si256((const __m256i *)(src1 + x));
                __m256i b = _mm256_loadu_si256((const __m256i *)(src2 + x));
                _mm256_storeu_si256((__m256i *)(dst + x), _mm256_avg_epu8(a, b));
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
            for (; x + 16 <= width; x += 16) {
                __m256i a = _mm256_loadu_si256((const __m256i *)(s1 + x));
                __m256i b = _mm256_loadu_si256((const __m256i *)(s2 + x));
                _mm256_storeu_si256((__m256i *)(d + x), _mm256_avg_epu16(a, b));
            }
            for (; x < width; x++)
                d[x] = (uint16_t)((s1[x] + s2[x] + 1) >> 1);
            dst += dst_stride;
            src1 += src1_stride;
            src2 += src2_stride;
        }
    }
}

AVX2_TARGET void h265_copy_pixels_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                                       const uint8_t *src, ptrdiff_t src_stride,
                                       int width, int height, int pixel_shift)
{
    int y;
    int bytes = width << pixel_shift;
    for (y = 0; y < height; y++) {
        int x = 0;
        for (; x + 32 <= bytes; x += 32)
            _mm256_storeu_si256((__m256i *)(dst + x),
                                _mm256_loadu_si256((const __m256i *)(src + x)));
        for (; x < bytes; x++)
            dst[x] = src[x];
        dst += dst_stride;
        src += src_stride;
    }
}

AVX2_TARGET H265_INLINE __m128i pack8_i16(__m256i v)
{
    __m256i p = _mm256_packs_epi32(v, v);
    __m128i lo = _mm256_castsi256_si128(p);
    __m128i hi = _mm256_extracti128_si256(p, 1);
    return _mm_unpacklo_epi64(lo, hi);
}

/* Write 8 clipped 32-bit lanes to 8-bit or 16-bit pixels. */
AVX2_TARGET H265_INLINE void write_pixels8(uint8_t *dst, __m256i v,
                                           int pixel_shift)
{
    __m128i p16 = pack8_i16(v);
    if (!pixel_shift)
        _mm_storel_epi64((__m128i *)dst, _mm_packus_epi16(p16, p16));
    else
        _mm_storeu_si128((__m128i *)dst, p16);
}

AVX2_TARGET H265_INLINE void write_pixels4(uint8_t *dst, __m128i v,
                                           int pixel_shift)
{
    __m128i p16 = _mm_packs_epi32(v, v);
    if (!pixel_shift)
        _mm_storeu_si32(dst, _mm_packus_epi16(p16, p16));
    else
        _mm_storel_epi64((__m128i *)dst, p16);
}

AVX2_TARGET H265_INLINE void store8_i16(int16_t *dst, __m256i v)
{
    _mm_storeu_si128((__m128i *)dst, pack8_i16(v));
}

/* Convert the 14-bit motion-compensation intermediate to clipped pixels.
 * Arithmetic matches the scalar uni_write() exactly. */
AVX2_TARGET void h265_put_pixels_avx2(int16_t *dst, int dst_stride,
                                      const uint8_t *src, ptrdiff_t src_stride,
                                      int width, int height, int shift,
                                      int pixel_shift)
{
    int x, y;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            x = 0;
            for (; x + 16 <= width; x += 16) {
                __m128i b = _mm_loadu_si128((const __m128i *)(src + x));
                __m256i v = _mm256_cvtepu8_epi16(b);
                _mm256_storeu_si256((__m256i *)(dst + x),
                                    _mm256_slli_epi16(v, shift));
            }
            for (; x + 8 <= width; x += 8) {
                __m128i b = _mm_loadl_epi64((const __m128i *)(src + x));
                __m128i v = _mm_cvtepu8_epi16(b);
                _mm_storeu_si128((__m128i *)(dst + x), _mm_slli_epi16(v, shift));
            }
            for (; x < width; x++)
                dst[x] = (int16_t)(src[x] << shift);
            src += src_stride;
            dst += dst_stride;
        }
    } else {
        for (y = 0; y < height; y++) {
            const uint16_t *s = (const uint16_t *)src;
            x = 0;
            for (; x + 16 <= width; x += 16) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(s + x));
                _mm256_storeu_si256((__m256i *)(dst + x),
                                    _mm256_slli_epi16(v, shift));
            }
            for (; x + 8 <= width; x += 8) {
                __m128i v = _mm_loadu_si128((const __m128i *)(s + x));
                _mm_storeu_si128((__m128i *)(dst + x), _mm_slli_epi16(v, shift));
            }
            for (; x < width; x++)
                dst[x] = (int16_t)(s[x] << shift);
            src += src_stride;
            dst += dst_stride;
        }
    }
}

AVX2_TARGET void h265_mc_uni_write_avx2(uint8_t *dst8, ptrdiff_t dst_stride,
                                        const int16_t *src, int src_stride,
                                        int width, int height, int shift,
                                        int add, int bit_depth, int wx, int ox)
{
    const int pixel_shift = bit_depth > 8;
    const int maxv = (1 << bit_depth) - 1;
    int x, y;

    if (!wx) {
        const __m256i vadd = _mm256_set1_epi16((short)add);
        const __m256i vmax = _mm256_set1_epi16((short)maxv);
        const __m256i vzero = _mm256_setzero_si256();
        const __m128i vsh = _mm_cvtsi32_si128(shift);

        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            const int16_t *s = src;
            x = 0;
            for (; x + 16 <= width; x += 16) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(s + x));
                v = _mm256_adds_epi16(v, vadd);
                v = _mm256_sra_epi16(v, vsh);
                v = _mm256_max_epi16(v, vzero);
                v = _mm256_min_epi16(v, vmax);
                if (!pixel_shift) {
                    __m128i lo = _mm256_castsi256_si128(v);
                    __m128i hi = _mm256_extracti128_si256(v, 1);
                    _mm_storeu_si128((__m128i *)(dst + x),
                                     _mm_packus_epi16(lo, hi));
                } else {
                    _mm256_storeu_si256((__m256i *)((uint16_t *)dst + x), v);
                }
            }
            for (; x + 8 <= width; x += 8) {
                __m128i v = _mm_loadu_si128((const __m128i *)(s + x));
                v = _mm_adds_epi16(v, _mm256_castsi256_si128(vadd));
                v = _mm_sra_epi16(v, vsh);
                v = _mm_max_epi16(v, _mm256_castsi256_si128(vzero));
                v = _mm_min_epi16(v, _mm256_castsi256_si128(vmax));
                if (!pixel_shift)
                    _mm_storel_epi64((__m128i *)(dst + x),
                                     _mm_packus_epi16(v, v));
                else
                    _mm_storeu_si128((__m128i *)((uint16_t *)dst + x), v);
            }
            for (; x < width; x++) {
                int v = (s[x] + add) >> shift;
                if (v < 0)
                    v = 0;
                else if (v > maxv)
                    v = maxv;
                if (!pixel_shift)
                    dst[x] = (uint8_t)v;
                else
                    ((uint16_t *)dst)[x] = (uint16_t)v;
            }
            dst8 += dst_stride;
            src += src_stride;
        }
        return;
    }

    {
        const __m256i vzero = _mm256_setzero_si256();
        const __m256i vmax = _mm256_set1_epi32(maxv);
        const __m256i vadd = _mm256_set1_epi32(add);
        const __m256i vwx = _mm256_set1_epi32(wx);
        const __m256i vox = _mm256_set1_epi32(ox);
        const __m256i vsh = _mm256_set1_epi32(shift);
        const __m128i vzero128 = _mm_setzero_si128();
        const __m128i vmax128 = _mm_set1_epi32(maxv);
        const __m128i vadd128 = _mm_set1_epi32(add);
        const __m128i vwx128 = _mm_set1_epi32(wx);
        const __m128i vox128 = _mm_set1_epi32(ox);
        const __m128i vsh128 = _mm_set1_epi32(shift);

        if (width == 4) {
            for (y = 0; y + 1 < height; y += 2) {
                __m128i a = _mm_cvtepi16_epi32(
                    _mm_loadl_epi64((const __m128i *)src));
                __m128i b = _mm_cvtepi16_epi32(
                    _mm_loadl_epi64((const __m128i *)(src + src_stride)));
                __m256i v = _mm256_set_m128i(b, a);
                v = _mm256_mullo_epi32(v, vwx);
                v = _mm256_add_epi32(v, vadd);
                v = _mm256_srav_epi32(v, vsh);
                v = _mm256_add_epi32(v, vox);
                v = _mm256_max_epi32(v, vzero);
                v = _mm256_min_epi32(v, vmax);
                write_pixels4(dst8, _mm256_castsi256_si128(v), pixel_shift);
                write_pixels4(dst8 + dst_stride,
                              _mm256_extracti128_si256(v, 1), pixel_shift);
                dst8 += 2 * dst_stride;
                src += 2 * src_stride;
            }
            for (; y < height; y++) {
                __m128i v = _mm_cvtepi16_epi32(
                    _mm_loadl_epi64((const __m128i *)src));
                v = _mm_mullo_epi32(v, vwx128);
                v = _mm_add_epi32(v, vadd128);
                v = _mm_srav_epi32(v, vsh128);
                v = _mm_add_epi32(v, vox128);
                v = _mm_max_epi32(v, vzero128);
                v = _mm_min_epi32(v, vmax128);
                write_pixels4(dst8, v, pixel_shift);
                dst8 += dst_stride;
                src += src_stride;
            }
            return;
        }

        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            const int16_t *s = src;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                __m256i v = _mm256_cvtepi16_epi32(
                    _mm_loadu_si128((const __m128i *)(s + x)));
                v = _mm256_mullo_epi32(v, vwx);
                v = _mm256_add_epi32(v, vadd);
                v = _mm256_srav_epi32(v, vsh);
                v = _mm256_add_epi32(v, vox);
                v = _mm256_max_epi32(v, vzero);
                v = _mm256_min_epi32(v, vmax);
                write_pixels8(dst + (x << pixel_shift), v, pixel_shift);
            }
            for (; x + 4 <= width; x += 4) {
                __m128i v = _mm_cvtepi16_epi32(
                    _mm_loadl_epi64((const __m128i *)(s + x)));
                v = _mm_mullo_epi32(v, vwx128);
                v = _mm_add_epi32(v, vadd128);
                v = _mm_srav_epi32(v, vsh128);
                v = _mm_add_epi32(v, vox128);
                v = _mm_max_epi32(v, vzero128);
                v = _mm_min_epi32(v, vmax128);
                write_pixels4(dst + (x << pixel_shift), v, pixel_shift);
            }
            for (; x < width; x++) {
                int v = ((s[x] * wx + add) >> shift) + ox;
                if (v < 0)
                    v = 0;
                else if (v > maxv)
                    v = maxv;
                if (!pixel_shift)
                    dst[x] = (uint8_t)v;
                else
                    ((uint16_t *)dst)[x] = (uint16_t)v;
            }
            dst8 += dst_stride;
            src += src_stride;
        }
    }
}

/* Weighted prediction straight from integer-MV reference pixels.  This is
 * algebraically identical to put_pixels() + weighted uni_write():
 *   ((src << (14-bd)) * w + (1 << (denom+13-bd))) >> (denom+14-bd)
 *   == (src * w + (denom ? 1 << (denom-1) : 0)) >> denom
 */
AVX2_TARGET void h265_mc_weighted_int_avx2(uint8_t *dst8, ptrdiff_t dst_stride,
                                           const uint8_t *src,
                                           ptrdiff_t src_stride, int width,
                                           int height, int wx, int ox,
                                           int denom, int bit_depth,
                                           int pixel_shift)
{
    const int maxv = (1 << bit_depth) - 1;
    const int rnd = denom ? 1 << (denom - 1) : 0;
    const __m256i vwx = _mm256_set1_epi32(wx);
    const __m256i vrnd = _mm256_set1_epi32(rnd);
    const __m256i vox = _mm256_set1_epi32(ox);
    const __m256i vsh = _mm256_set1_epi32(denom);
    const __m256i vzero = _mm256_setzero_si256();
    const __m256i vmax = _mm256_set1_epi32(maxv);
    const __m128i vwx128 = _mm_set1_epi32(wx);
    const __m128i vrnd128 = _mm_set1_epi32(rnd);
    const __m128i vox128 = _mm_set1_epi32(ox);
    const __m128i vsh128 = _mm_set1_epi32(denom);
    const __m128i vzero128 = _mm_setzero_si128();
    const __m128i vmax128 = _mm_set1_epi32(maxv);
    int x, y;

    for (y = 0; y < height; y++) {
        uint8_t *dst = dst8;
        const uint8_t *s = src;
        x = 0;
        for (; x + 8 <= width; x += 8) {
            __m256i v = pixel_shift
                ? _mm256_cvtepu16_epi32(
                      _mm_loadu_si128((const __m128i *)((const uint16_t *)s + x)))
                : _mm256_cvtepu8_epi32(
                      _mm_loadl_epi64((const __m128i *)(s + x)));
            v = _mm256_add_epi32(_mm256_mullo_epi32(v, vwx), vrnd);
            v = _mm256_srav_epi32(v, vsh);
            v = _mm256_add_epi32(v, vox);
            v = _mm256_max_epi32(v, vzero);
            v = _mm256_min_epi32(v, vmax);
            write_pixels8(dst + (x << pixel_shift), v, pixel_shift);
        }
        for (; x + 4 <= width; x += 4) {
            __m128i v = pixel_shift
                ? _mm_cvtepu16_epi32(
                      _mm_loadl_epi64((const __m128i *)((const uint16_t *)s + x)))
                : _mm_cvtepu8_epi32(
                      _mm_cvtsi32_si128(*(const int32_t *)(s + x)));
            v = _mm_add_epi32(_mm_mullo_epi32(v, vwx128), vrnd128);
            v = _mm_srav_epi32(v, vsh128);
            v = _mm_add_epi32(v, vox128);
            v = _mm_max_epi32(v, vzero128);
            v = _mm_min_epi32(v, vmax128);
            write_pixels4(dst + (x << pixel_shift), v, pixel_shift);
        }
        for (; x < width; x++) {
            int v = pixel_shift ? ((const uint16_t *)s)[x] : s[x];
            v = ((v * wx + rnd) >> denom) + ox;
            if (v < 0)
                v = 0;
            else if (v > maxv)
                v = maxv;
            if (!pixel_shift)
                dst[x] = (uint8_t)v;
            else
                ((uint16_t *)dst)[x] = (uint16_t)v;
        }
        dst8 += dst_stride;
        src += src_stride;
    }
}

/* Combine two 14-bit motion-compensation intermediates (bi-prediction). */
AVX2_TARGET void h265_mc_combine_bi_avx2(uint8_t *dst8, ptrdiff_t dst_stride,
                                         const int16_t *first,
                                         const int16_t *second, int src_stride,
                                         int width, int height, int shift,
                                         int add, int bit_depth, int weighted,
                                         int wx0, int wx1)
{
    const int pixel_shift = bit_depth > 8;
    const int maxv = (1 << bit_depth) - 1;
    const __m256i vzero = _mm256_setzero_si256();
    const __m256i vmax = _mm256_set1_epi32(maxv);
    const __m256i vadd = _mm256_set1_epi32(add);
    const __m256i vwx0 = _mm256_set1_epi32(wx0);
    const __m256i vwx1 = _mm256_set1_epi32(wx1);
    const __m256i vsh = _mm256_set1_epi32(shift);
    const __m128i vzero128 = _mm_setzero_si128();
    const __m128i vmax128 = _mm_set1_epi32(maxv);
    const __m128i vadd128 = _mm_set1_epi32(add);
    const __m128i vwx0128 = _mm_set1_epi32(wx0);
    const __m128i vwx1128 = _mm_set1_epi32(wx1);
    const __m128i vsh128 = _mm_set1_epi32(shift);
    int x, y;

    for (y = 0; y < height; y++) {
        uint8_t *dst = dst8;
        x = 0;
        for (; x + 8 <= width; x += 8) {
            __m256i a = _mm256_cvtepi16_epi32(
                _mm_loadu_si128((const __m128i *)(first + x)));
            __m256i b = _mm256_cvtepi16_epi32(
                _mm_loadu_si128((const __m128i *)(second + x)));
            __m256i v;
            if (weighted)
                v = _mm256_add_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(a, vwx0),
                                     _mm256_mullo_epi32(b, vwx1)), vadd);
            else
                v = _mm256_add_epi32(_mm256_add_epi32(a, b), vadd);
            v = _mm256_srav_epi32(v, vsh);
            v = _mm256_max_epi32(v, vzero);
            v = _mm256_min_epi32(v, vmax);
            write_pixels8(dst + (x << pixel_shift), v, pixel_shift);
        }
        for (; x + 4 <= width; x += 4) {
            __m128i a = _mm_cvtepi16_epi32(
                _mm_loadl_epi64((const __m128i *)(first + x)));
            __m128i b = _mm_cvtepi16_epi32(
                _mm_loadl_epi64((const __m128i *)(second + x)));
            __m128i v;
            if (weighted)
                v = _mm_add_epi32(
                    _mm_add_epi32(_mm_mullo_epi32(a, vwx0128),
                                  _mm_mullo_epi32(b, vwx1128)), vadd128);
            else
                v = _mm_add_epi32(_mm_add_epi32(a, b), vadd128);
            v = _mm_srav_epi32(v, vsh128);
            v = _mm_max_epi32(v, vzero128);
            v = _mm_min_epi32(v, vmax128);
            write_pixels4(dst + (x << pixel_shift), v, pixel_shift);
        }
        for (; x < width; x++) {
            int v;
            if (weighted)
                v = (second[x] * wx1 + first[x] * wx0 + add) >> shift;
            else
                v = (first[x] + second[x] + add) >> shift;
            if (v < 0)
                v = 0;
            else if (v > maxv)
                v = maxv;
            if (!pixel_shift)
                dst[x] = (uint8_t)v;
            else
                ((uint16_t *)dst)[x] = (uint16_t)v;
        }
        dst8 += dst_stride;
        first += src_stride;
        second += src_stride;
    }
}

AVX2_TARGET H265_INLINE int16_t mc_pair8(const int8_t *f, int i)
{
    return (int16_t)((uint16_t)(uint8_t)f[i] |
                     ((uint16_t)(uint8_t)f[i + 1] << 8));
}

AVX2_TARGET H265_INLINE int mc_pair16(const int8_t *f, int i)
{
    return (int)(uint16_t)(int16_t)f[i] |
           ((int)(uint16_t)(int16_t)f[i + 1] << 16);
}

AVX2_TARGET void h265_mc_filter_h_avx2(int16_t *dst, int dst_stride,
                                       const uint8_t *src, ptrdiff_t src_stride,
                                       int height, int width,
                                       const int8_t *filter, int taps,
                                       int shift, int pixel_shift)
{
    int x, y, k;
    int start = taps / 2 - 1;

    if (!pixel_shift && (taps == 4 || taps == 8)) {
        const __m128i c01 = _mm_set1_epi16(mc_pair8(filter, 0));
        const __m128i c23 = _mm_set1_epi16(mc_pair8(filter, 2));
        const __m128i c45 = _mm_set1_epi16(mc_pair8(filter, 4));
        const __m128i c67 = _mm_set1_epi16(mc_pair8(filter, 6));
        const __m128i vsh = _mm_cvtsi32_si128(shift);

        for (y = 0; y < height; y++) {
            const uint8_t *row = src + (size_t)y * src_stride;
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 16 <= width; x += 16) {
                const uint8_t *p = row + x - start;
                __m128i t0 = _mm_loadu_si128((const __m128i *)(p + 0));
                __m128i t1 = _mm_loadu_si128((const __m128i *)(p + 1));
                __m128i t2 = _mm_loadu_si128((const __m128i *)(p + 2));
                __m128i t3 = _mm_loadu_si128((const __m128i *)(p + 3));
                __m128i lo = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t0, t1), c01),
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t2, t3), c23));
                __m128i hi = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(t0, t1), c01),
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(t2, t3), c23));
                if (taps == 8) {
                    __m128i t4 = _mm_loadu_si128((const __m128i *)(p + 4));
                    __m128i t5 = _mm_loadu_si128((const __m128i *)(p + 5));
                    __m128i t6 = _mm_loadu_si128((const __m128i *)(p + 6));
                    __m128i t7 = _mm_loadu_si128((const __m128i *)(p + 7));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t4, t5), c45));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t6, t7), c67));
                    hi = _mm_add_epi16(hi, _mm_maddubs_epi16(
                        _mm_unpackhi_epi8(t4, t5), c45));
                    hi = _mm_add_epi16(hi, _mm_maddubs_epi16(
                        _mm_unpackhi_epi8(t6, t7), c67));
                }
                if (shift) {
                    lo = _mm_sra_epi16(lo, vsh);
                    hi = _mm_sra_epi16(hi, vsh);
                }
                _mm256_storeu_si256((__m256i *)(d + x),
                                    _mm256_set_m128i(hi, lo));
            }
            for (; x + 8 <= width; x += 8) {
                const uint8_t *p = row + x - start;
                __m128i t0 = _mm_loadl_epi64((const __m128i *)(p + 0));
                __m128i t1 = _mm_loadl_epi64((const __m128i *)(p + 1));
                __m128i t2 = _mm_loadl_epi64((const __m128i *)(p + 2));
                __m128i t3 = _mm_loadl_epi64((const __m128i *)(p + 3));
                __m128i lo = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t0, t1), c01),
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t2, t3), c23));
                if (taps == 8) {
                    __m128i t4 = _mm_loadl_epi64((const __m128i *)(p + 4));
                    __m128i t5 = _mm_loadl_epi64((const __m128i *)(p + 5));
                    __m128i t6 = _mm_loadl_epi64((const __m128i *)(p + 6));
                    __m128i t7 = _mm_loadl_epi64((const __m128i *)(p + 7));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t4, t5), c45));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t6, t7), c67));
                }
                if (shift)
                    lo = _mm_sra_epi16(lo, vsh);
                _mm_storeu_si128((__m128i *)(d + x), lo);
            }
            for (; x < width; x++) {
                int sum = 0;
                for (k = 0; k < taps; k++)
                    sum += filter[k] * row[x - start + k];
                d[x] = (int16_t)(sum >> shift);
            }
        }
    } else if (pixel_shift && (taps == 4 || taps == 8)) {
        const __m256i c01 = _mm256_set1_epi32(mc_pair16(filter, 0));
        const __m256i c23 = _mm256_set1_epi32(mc_pair16(filter, 2));
        const __m256i c45 = _mm256_set1_epi32(mc_pair16(filter, 4));
        const __m256i c67 = _mm256_set1_epi32(mc_pair16(filter, 6));
        const __m128i c01s = _mm_set1_epi32(mc_pair16(filter, 0));
        const __m128i c23s = _mm_set1_epi32(mc_pair16(filter, 2));
        const __m128i c45s = _mm_set1_epi32(mc_pair16(filter, 4));
        const __m128i c67s = _mm_set1_epi32(mc_pair16(filter, 6));
        const __m256i vsh = _mm256_set1_epi32(shift);
        const __m128i vshs = _mm_set1_epi32(shift);

        for (y = 0; y < height; y++) {
            const uint16_t *row = (const uint16_t *)
                (src + (size_t)y * src_stride);
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 16 <= width; x += 16) {
                const uint16_t *p = row + x - start;
                __m256i t0 = _mm256_loadu_si256((const __m256i *)(p + 0));
                __m256i t1 = _mm256_loadu_si256((const __m256i *)(p + 1));
                __m256i t2 = _mm256_loadu_si256((const __m256i *)(p + 2));
                __m256i t3 = _mm256_loadu_si256((const __m256i *)(p + 3));
                __m256i lo = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(t0, t1), c01),
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(t2, t3), c23));
                __m256i hi = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(t0, t1), c01),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(t2, t3), c23));
                if (taps == 8) {
                    __m256i t4 = _mm256_loadu_si256((const __m256i *)(p + 4));
                    __m256i t5 = _mm256_loadu_si256((const __m256i *)(p + 5));
                    __m256i t6 = _mm256_loadu_si256((const __m256i *)(p + 6));
                    __m256i t7 = _mm256_loadu_si256((const __m256i *)(p + 7));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(
                        _mm256_unpacklo_epi16(t4, t5), c45));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(
                        _mm256_unpacklo_epi16(t6, t7), c67));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(
                        _mm256_unpackhi_epi16(t4, t5), c45));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(
                        _mm256_unpackhi_epi16(t6, t7), c67));
                }
                if (shift) {
                    lo = _mm256_srav_epi32(lo, vsh);
                    hi = _mm256_srav_epi32(hi, vsh);
                }
                _mm256_storeu_si256((__m256i *)(d + x),
                                    _mm256_packs_epi32(lo, hi));
            }
            for (; x + 8 <= width; x += 8) {
                const uint16_t *p = row + x - start;
                __m128i t0 = _mm_loadu_si128((const __m128i *)(p + 0));
                __m128i t1 = _mm_loadu_si128((const __m128i *)(p + 1));
                __m128i t2 = _mm_loadu_si128((const __m128i *)(p + 2));
                __m128i t3 = _mm_loadu_si128((const __m128i *)(p + 3));
                __m128i lo = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpacklo_epi16(t0, t1), c01s),
                    _mm_madd_epi16(_mm_unpacklo_epi16(t2, t3), c23s));
                __m128i hi = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpackhi_epi16(t0, t1), c01s),
                    _mm_madd_epi16(_mm_unpackhi_epi16(t2, t3), c23s));
                if (taps == 8) {
                    __m128i t4 = _mm_loadu_si128((const __m128i *)(p + 4));
                    __m128i t5 = _mm_loadu_si128((const __m128i *)(p + 5));
                    __m128i t6 = _mm_loadu_si128((const __m128i *)(p + 6));
                    __m128i t7 = _mm_loadu_si128((const __m128i *)(p + 7));
                    lo = _mm_add_epi32(lo, _mm_madd_epi16(
                        _mm_unpacklo_epi16(t4, t5), c45s));
                    lo = _mm_add_epi32(lo, _mm_madd_epi16(
                        _mm_unpacklo_epi16(t6, t7), c67s));
                    hi = _mm_add_epi32(hi, _mm_madd_epi16(
                        _mm_unpackhi_epi16(t4, t5), c45s));
                    hi = _mm_add_epi32(hi, _mm_madd_epi16(
                        _mm_unpackhi_epi16(t6, t7), c67s));
                }
                if (shift) {
                    lo = _mm_srav_epi32(lo, vshs);
                    hi = _mm_srav_epi32(hi, vshs);
                }
                _mm_storeu_si128((__m128i *)(d + x),
                                 _mm_packs_epi32(lo, hi));
            }
            for (; x < width; x++) {
                int sum = 0;
                for (k = 0; k < taps; k++)
                    sum += filter[k] * row[x - start + k];
                d[x] = (int16_t)(sum >> shift);
            }
        }
    } else {
        for (y = 0; y < height; y++) {
            const uint8_t *row = src + (size_t)y * src_stride;
            int16_t *d = dst + (size_t)y * dst_stride;
            if (!pixel_shift) {
                for (x = 0; x < width; x++) {
                    int sum = 0;
                    for (k = 0; k < taps; k++)
                        sum += filter[k] * row[x - start + k];
                    d[x] = (int16_t)(sum >> shift);
                }
            } else {
                const uint16_t *row16 = (const uint16_t *)row;
                for (x = 0; x < width; x++) {
                    int sum = 0;
                    for (k = 0; k < taps; k++)
                        sum += filter[k] * row16[x - start + k];
                    d[x] = (int16_t)(sum >> shift);
                }
            }
        }
    }
}

AVX2_TARGET void h265_mc_filter_v_avx2(int16_t *dst, int dst_stride,
                                       const uint8_t *src, ptrdiff_t src_stride,
                                       int height, int width,
                                       const int8_t *filter, int taps,
                                       int shift, int pixel_shift,
                                       int signed_src)
{
    int x, y, k;
    int start = taps / 2 - 1;

    if (!pixel_shift && (taps == 4 || taps == 8)) {
        const __m128i c01 = _mm_set1_epi16(mc_pair8(filter, 0));
        const __m128i c23 = _mm_set1_epi16(mc_pair8(filter, 2));
        const __m128i c45 = _mm_set1_epi16(mc_pair8(filter, 4));
        const __m128i c67 = _mm_set1_epi16(mc_pair8(filter, 6));
        const __m128i vsh = _mm_cvtsi32_si128(shift);

        for (y = 0; y < height; y++) {
            const uint8_t *p = src + (ptrdiff_t)(y - start) * src_stride;
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 16 <= width; x += 16) {
                __m128i t0 = _mm_loadu_si128((const __m128i *)(p + x));
                __m128i t1 = _mm_loadu_si128((const __m128i *)(p + src_stride + x));
                __m128i t2 = _mm_loadu_si128((const __m128i *)(p + 2 * src_stride + x));
                __m128i t3 = _mm_loadu_si128((const __m128i *)(p + 3 * src_stride + x));
                __m128i lo = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t0, t1), c01),
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t2, t3), c23));
                __m128i hi = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(t0, t1), c01),
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(t2, t3), c23));
                if (taps == 8) {
                    __m128i t4 = _mm_loadu_si128((const __m128i *)(p + 4 * src_stride + x));
                    __m128i t5 = _mm_loadu_si128((const __m128i *)(p + 5 * src_stride + x));
                    __m128i t6 = _mm_loadu_si128((const __m128i *)(p + 6 * src_stride + x));
                    __m128i t7 = _mm_loadu_si128((const __m128i *)(p + 7 * src_stride + x));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t4, t5), c45));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t6, t7), c67));
                    hi = _mm_add_epi16(hi, _mm_maddubs_epi16(
                        _mm_unpackhi_epi8(t4, t5), c45));
                    hi = _mm_add_epi16(hi, _mm_maddubs_epi16(
                        _mm_unpackhi_epi8(t6, t7), c67));
                }
                if (shift) {
                    lo = _mm_sra_epi16(lo, vsh);
                    hi = _mm_sra_epi16(hi, vsh);
                }
                _mm256_storeu_si256((__m256i *)(d + x),
                                    _mm256_set_m128i(hi, lo));
            }
            for (; x + 8 <= width; x += 8) {
                __m128i t0 = _mm_loadl_epi64((const __m128i *)(p + x));
                __m128i t1 = _mm_loadl_epi64((const __m128i *)(p + src_stride + x));
                __m128i t2 = _mm_loadl_epi64((const __m128i *)(p + 2 * src_stride + x));
                __m128i t3 = _mm_loadl_epi64((const __m128i *)(p + 3 * src_stride + x));
                __m128i lo = _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t0, t1), c01),
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(t2, t3), c23));
                if (taps == 8) {
                    __m128i t4 = _mm_loadl_epi64((const __m128i *)(p + 4 * src_stride + x));
                    __m128i t5 = _mm_loadl_epi64((const __m128i *)(p + 5 * src_stride + x));
                    __m128i t6 = _mm_loadl_epi64((const __m128i *)(p + 6 * src_stride + x));
                    __m128i t7 = _mm_loadl_epi64((const __m128i *)(p + 7 * src_stride + x));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t4, t5), c45));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(
                        _mm_unpacklo_epi8(t6, t7), c67));
                }
                if (shift)
                    lo = _mm_sra_epi16(lo, vsh);
                _mm_storeu_si128((__m128i *)(d + x), lo);
            }
            for (; x < width; x++) {
                int sum = 0;
                for (k = 0; k < taps; k++)
                    sum += filter[k] * p[k * src_stride + x];
                d[x] = (int16_t)(sum >> shift);
            }
        }
    } else if (pixel_shift && (taps == 4 || taps == 8)) {
        const __m256i c01 = _mm256_set1_epi32(mc_pair16(filter, 0));
        const __m256i c23 = _mm256_set1_epi32(mc_pair16(filter, 2));
        const __m256i c45 = _mm256_set1_epi32(mc_pair16(filter, 4));
        const __m256i c67 = _mm256_set1_epi32(mc_pair16(filter, 6));
        const __m128i c01s = _mm_set1_epi32(mc_pair16(filter, 0));
        const __m128i c23s = _mm_set1_epi32(mc_pair16(filter, 2));
        const __m128i c45s = _mm_set1_epi32(mc_pair16(filter, 4));
        const __m128i c67s = _mm_set1_epi32(mc_pair16(filter, 6));
        const __m256i vsh = _mm256_set1_epi32(shift);
        const __m128i vshs = _mm_set1_epi32(shift);

        for (y = 0; y < height; y++) {
            const uint8_t *p = src + (ptrdiff_t)(y - start) * src_stride;
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 16 <= width; x += 16) {
                const uint16_t *p0 = (const uint16_t *)p + x;
                const uint16_t *p1 = (const uint16_t *)(p + src_stride) + x;
                const uint16_t *p2 = (const uint16_t *)(p + 2 * src_stride) + x;
                const uint16_t *p3 = (const uint16_t *)(p + 3 * src_stride) + x;
                __m256i t0 = _mm256_loadu_si256((const __m256i *)p0);
                __m256i t1 = _mm256_loadu_si256((const __m256i *)p1);
                __m256i t2 = _mm256_loadu_si256((const __m256i *)p2);
                __m256i t3 = _mm256_loadu_si256((const __m256i *)p3);
                __m256i lo = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(t0, t1), c01),
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(t2, t3), c23));
                __m256i hi = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(t0, t1), c01),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(t2, t3), c23));
                if (taps == 8) {
                    const uint16_t *p4 = (const uint16_t *)(p + 4 * src_stride) + x;
                    const uint16_t *p5 = (const uint16_t *)(p + 5 * src_stride) + x;
                    const uint16_t *p6 = (const uint16_t *)(p + 6 * src_stride) + x;
                    const uint16_t *p7 = (const uint16_t *)(p + 7 * src_stride) + x;
                    __m256i t4 = _mm256_loadu_si256((const __m256i *)p4);
                    __m256i t5 = _mm256_loadu_si256((const __m256i *)p5);
                    __m256i t6 = _mm256_loadu_si256((const __m256i *)p6);
                    __m256i t7 = _mm256_loadu_si256((const __m256i *)p7);
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(
                        _mm256_unpacklo_epi16(t4, t5), c45));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(
                        _mm256_unpacklo_epi16(t6, t7), c67));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(
                        _mm256_unpackhi_epi16(t4, t5), c45));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(
                        _mm256_unpackhi_epi16(t6, t7), c67));
                }
                if (shift) {
                    lo = _mm256_srav_epi32(lo, vsh);
                    hi = _mm256_srav_epi32(hi, vsh);
                }
                _mm256_storeu_si256((__m256i *)(d + x),
                                    _mm256_packs_epi32(lo, hi));
            }
            for (; x + 8 <= width; x += 8) {
                const uint16_t *p0 = (const uint16_t *)p + x;
                const uint16_t *p1 = (const uint16_t *)(p + src_stride) + x;
                const uint16_t *p2 = (const uint16_t *)(p + 2 * src_stride) + x;
                const uint16_t *p3 = (const uint16_t *)(p + 3 * src_stride) + x;
                __m128i t0 = _mm_loadu_si128((const __m128i *)p0);
                __m128i t1 = _mm_loadu_si128((const __m128i *)p1);
                __m128i t2 = _mm_loadu_si128((const __m128i *)p2);
                __m128i t3 = _mm_loadu_si128((const __m128i *)p3);
                __m128i lo = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpacklo_epi16(t0, t1), c01s),
                    _mm_madd_epi16(_mm_unpacklo_epi16(t2, t3), c23s));
                __m128i hi = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpackhi_epi16(t0, t1), c01s),
                    _mm_madd_epi16(_mm_unpackhi_epi16(t2, t3), c23s));
                if (taps == 8) {
                    const uint16_t *p4 = (const uint16_t *)(p + 4 * src_stride) + x;
                    const uint16_t *p5 = (const uint16_t *)(p + 5 * src_stride) + x;
                    const uint16_t *p6 = (const uint16_t *)(p + 6 * src_stride) + x;
                    const uint16_t *p7 = (const uint16_t *)(p + 7 * src_stride) + x;
                    __m128i t4 = _mm_loadu_si128((const __m128i *)p4);
                    __m128i t5 = _mm_loadu_si128((const __m128i *)p5);
                    __m128i t6 = _mm_loadu_si128((const __m128i *)p6);
                    __m128i t7 = _mm_loadu_si128((const __m128i *)p7);
                    lo = _mm_add_epi32(lo, _mm_madd_epi16(
                        _mm_unpacklo_epi16(t4, t5), c45s));
                    lo = _mm_add_epi32(lo, _mm_madd_epi16(
                        _mm_unpacklo_epi16(t6, t7), c67s));
                    hi = _mm_add_epi32(hi, _mm_madd_epi16(
                        _mm_unpackhi_epi16(t4, t5), c45s));
                    hi = _mm_add_epi32(hi, _mm_madd_epi16(
                        _mm_unpackhi_epi16(t6, t7), c67s));
                }
                if (shift) {
                    lo = _mm_srav_epi32(lo, vshs);
                    hi = _mm_srav_epi32(hi, vshs);
                }
                _mm_storeu_si128((__m128i *)(d + x),
                                 _mm_packs_epi32(lo, hi));
            }
            for (; x < width; x++) {
                int sum = 0;
                for (k = 0; k < taps; k++) {
                    const uint8_t *q = p + (ptrdiff_t)k * src_stride;
                    sum += filter[k] * (signed_src ? ((const int16_t *)q)[x]
                                                   : ((const uint16_t *)q)[x]);
                }
                d[x] = (int16_t)(sum >> shift);
            }
        }
    } else {
        for (y = 0; y < height; y++) {
            int16_t *d = dst + (size_t)y * dst_stride;
            const uint8_t *p = src + (ptrdiff_t)(y - start) * src_stride;
            if (!pixel_shift) {
                for (x = 0; x < width; x++) {
                    int sum = 0;
                    for (k = 0; k < taps; k++)
                        sum += filter[k] * p[k * src_stride + x];
                    d[x] = (int16_t)(sum >> shift);
                }
            } else {
                for (x = 0; x < width; x++) {
                    int sum = 0;
                    for (k = 0; k < taps; k++) {
                        const uint8_t *q = p + (ptrdiff_t)k * src_stride;
                        sum += filter[k] * (signed_src ? ((const int16_t *)q)[x]
                                                       : ((const uint16_t *)q)[x]);
                    }
                    d[x] = (int16_t)(sum >> shift);
                }
            }
        }
    }
}

AVX2_TARGET void h265_sao_edge_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                                    const uint8_t *src, ptrdiff_t src_stride,
                                    int a_stride, int b_stride,
                                    const int16_t *sao_offset_val,
                                    int width, int height, int pixel_shift,
                                    int bit_depth)
{
    static const uint8_t edge_idx[5] = { 1, 2, 0, 3, 4 };
    int8_t lutv[16] = { 0 };
    __m128i lut;
    __m128i one16 = _mm_set1_epi16(1);
    __m128i two16 = _mm_set1_epi16(2);
    int maxval = (1 << bit_depth) - 1;
    __m128i zero = _mm_setzero_si128();
    __m128i maxv = _mm_set1_epi16((short)maxval);
    int x, y, i;

    for (i = 0; i < 5; i++) {
        int v = sao_offset_val[edge_idx[i]];
        lutv[i] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
    }
    lut = _mm_loadu_si128((const __m128i *)lutv);

    for (y = 0; y < height; y++) {
        if (!pixel_shift) {
            for (x = 0; x + 16 <= width; x += 16) {
                __m256i s = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((const __m128i *)(src + x)));
                __m256i a = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((const __m128i *)(src + x + a_stride)));
                __m256i b = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((const __m128i *)(src + x + b_stride)));
                __m256i one = _mm256_set1_epi16(1);
                __m256i gt0 = _mm256_cmpgt_epi16(s, a);
                __m256i lt0 = _mm256_cmpgt_epi16(a, s);
                __m256i gt1 = _mm256_cmpgt_epi16(s, b);
                __m256i lt1 = _mm256_cmpgt_epi16(b, s);
                __m256i diff0 = _mm256_sub_epi16(_mm256_and_si256(gt0, one),
                                                 _mm256_and_si256(lt0, one));
                __m256i diff1 = _mm256_sub_epi16(_mm256_and_si256(gt1, one),
                                                 _mm256_and_si256(lt1, one));
                __m256i idx = _mm256_add_epi16(
                    _mm256_set1_epi16(2),
                    _mm256_add_epi16(diff0, diff1));
                __m128i packed = _mm_packus_epi16(
                    _mm256_castsi256_si128(idx),
                    _mm256_extracti128_si256(idx, 1));
                __m256i off = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(lut, packed));
                __m256i v = _mm256_add_epi16(s, off);
                v = _mm256_max_epi16(v, _mm256_setzero_si256());
                v = _mm256_min_epi16(
                    v, _mm256_set1_epi16((short)maxval));
                _mm_storeu_si128((__m128i *)(dst + x),
                    _mm_packus_epi16(_mm256_castsi256_si128(v),
                                     _mm256_extracti128_si256(v, 1)));
            }
            for (; x + 8 <= width; x += 8) {
                __m128i s = _mm_cvtepu8_epi16(
                    _mm_loadl_epi64((const __m128i *)(src + x)));
                __m128i a = _mm_cvtepu8_epi16(
                    _mm_loadl_epi64((const __m128i *)(src + x + a_stride)));
                __m128i b = _mm_cvtepu8_epi16(
                    _mm_loadl_epi64((const __m128i *)(src + x + b_stride)));
                __m128i gt0 = _mm_cmpgt_epi16(s, a);
                __m128i lt0 = _mm_cmpgt_epi16(a, s);
                __m128i gt1 = _mm_cmpgt_epi16(s, b);
                __m128i lt1 = _mm_cmpgt_epi16(b, s);
                __m128i diff0 = _mm_sub_epi16(_mm_and_si128(gt0, one16),
                                              _mm_and_si128(lt0, one16));
                __m128i diff1 = _mm_sub_epi16(_mm_and_si128(gt1, one16),
                                              _mm_and_si128(lt1, one16));
                __m128i idx = _mm_add_epi16(two16,
                    _mm_add_epi16(diff0, diff1));
                __m128i packed = _mm_packus_epi16(idx, idx);
                __m128i off = _mm_cvtepi8_epi16(
                    _mm_shuffle_epi8(lut, packed));
                __m128i v = _mm_add_epi16(s, off);
                v = _mm_max_epi16(v, zero);
                v = _mm_min_epi16(v, maxv);
                _mm_storel_epi64((__m128i *)(dst + x),
                                 _mm_packus_epi16(v, v));
            }
            for (; x < width; x++) {
                static const uint8_t edge_idx[5] = { 1, 2, 0, 3, 4 };
                int a = src[x + a_stride], b = src[x + b_stride];
                int diff0 = (src[x] > a) - (src[x] < a);
                int diff1 = (src[x] > b) - (src[x] < b);
                int v = src[x] + sao_offset_val[edge_idx[2 + diff0 + diff1]];
                dst[x] = (uint8_t)(v < 0 ? 0 : v > maxval ? maxval : v);
            }
        } else {
            const uint16_t *s16 = (const uint16_t *)src;
            uint16_t *d16 = (uint16_t *)dst;
            for (x = 0; x + 16 <= width; x += 16) {
                __m256i s = _mm256_loadu_si256((const __m256i *)(s16 + x));
                __m256i a = _mm256_loadu_si256((const __m256i *)
                    ((const uint8_t *)(s16 + x) + a_stride));
                __m256i b = _mm256_loadu_si256((const __m256i *)
                    ((const uint8_t *)(s16 + x) + b_stride));
                __m256i one = _mm256_set1_epi16(1);
                __m256i gt0 = _mm256_cmpgt_epi16(s, a);
                __m256i lt0 = _mm256_cmpgt_epi16(a, s);
                __m256i gt1 = _mm256_cmpgt_epi16(s, b);
                __m256i lt1 = _mm256_cmpgt_epi16(b, s);
                __m256i diff0 = _mm256_sub_epi16(_mm256_and_si256(gt0, one),
                                                 _mm256_and_si256(lt0, one));
                __m256i diff1 = _mm256_sub_epi16(_mm256_and_si256(gt1, one),
                                                 _mm256_and_si256(lt1, one));
                __m256i idx = _mm256_add_epi16(
                    _mm256_set1_epi16(2),
                    _mm256_add_epi16(diff0, diff1));
                __m128i packed = _mm_packus_epi16(
                    _mm256_castsi256_si128(idx),
                    _mm256_extracti128_si256(idx, 1));
                __m256i off = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(lut, packed));
                __m256i v = _mm256_add_epi16(s, off);
                v = _mm256_max_epi16(v, _mm256_setzero_si256());
                v = _mm256_min_epi16(
                    v, _mm256_set1_epi16((short)maxval));
                _mm256_storeu_si256((__m256i *)(d16 + x), v);
            }
            for (; x + 8 <= width; x += 8) {
                __m128i s = _mm_loadu_si128((const __m128i *)(s16 + x));
                __m128i a = _mm_loadu_si128(
                    (const __m128i *)((const uint8_t *)(s16 + x) + a_stride));
                __m128i b = _mm_loadu_si128(
                    (const __m128i *)((const uint8_t *)(s16 + x) + b_stride));
                __m128i gt0 = _mm_cmpgt_epi16(s, a);
                __m128i lt0 = _mm_cmpgt_epi16(a, s);
                __m128i gt1 = _mm_cmpgt_epi16(s, b);
                __m128i lt1 = _mm_cmpgt_epi16(b, s);
                __m128i diff0 = _mm_sub_epi16(_mm_and_si128(gt0, one16),
                                              _mm_and_si128(lt0, one16));
                __m128i diff1 = _mm_sub_epi16(_mm_and_si128(gt1, one16),
                                              _mm_and_si128(lt1, one16));
                __m128i idx = _mm_add_epi16(two16,
                    _mm_add_epi16(diff0, diff1));
                __m128i packed = _mm_packus_epi16(idx, idx);
                __m128i off = _mm_cvtepi8_epi16(
                    _mm_shuffle_epi8(lut, packed));
                __m128i v = _mm_add_epi16(s, off);
                v = _mm_max_epi16(v, zero);
                v = _mm_min_epi16(v, maxv);
                _mm_storeu_si128((__m128i *)(d16 + x), v);
            }
            for (; x < width; x++) {
                static const uint8_t edge_idx[5] = { 1, 2, 0, 3, 4 };
                int a = s16[x + (a_stride >> 1)];
                int b = s16[x + (b_stride >> 1)];
                int diff0 = (s16[x] > a) - (s16[x] < a);
                int diff1 = (s16[x] > b) - (s16[x] < b);
                int v = s16[x] + sao_offset_val[edge_idx[2 + diff0 + diff1]];
                d16[x] = (uint16_t)(v < 0 ? 0 : v > maxval ? maxval : v);
            }
        }
        dst += dst_stride;
        src += src_stride;
    }
}

/* SAO band offset: class = (pixel >> (bit_depth - 5)) & 31.  The 32-entry
 * offset table is split into two 16-byte LUTs so vpshufb can handle the full
 * 5-bit class on each 128-bit lane. */
AVX2_TARGET void h265_sao_band_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                                    const uint8_t *src, ptrdiff_t src_stride,
                                    const int *offset_table, int width,
                                    int height, int pixel_shift, int bit_depth)
{
    int8_t lo[16], hi[16];
    __m128i lut_lo, lut_hi, sh, zero, maxv;
    int shift = bit_depth - 5;
    int maxval = (1 << bit_depth) - 1;
    int x, y;

    for (x = 0; x < 16; x++) {
        int a = offset_table[x];
        int b = offset_table[x + 16];
        lo[x] = (int8_t)(a < -127 ? -127 : a > 127 ? 127 : a);
        hi[x] = (int8_t)(b < -127 ? -127 : b > 127 ? 127 : b);
    }
    lut_lo = _mm_loadu_si128((const __m128i *)lo);
    lut_hi = _mm_loadu_si128((const __m128i *)hi);
    sh = _mm_cvtsi32_si128(shift);
    zero = _mm_setzero_si128();
    maxv = _mm_set1_epi16((short)maxval);

    for (y = 0; y < height; y++) {
        const uint8_t *s = src + (size_t)y * src_stride;
        uint8_t *d = dst + (size_t)y * dst_stride;

        if (!pixel_shift) {
            for (x = 0; x + 16 <= width; x += 16) {
                __m128i b = _mm_loadu_si128((const __m128i *)(s + x));
                __m256i v = _mm256_cvtepu8_epi16(b);
                __m256i cls = _mm256_and_si256(_mm256_srl_epi16(v, sh),
                                               _mm256_set1_epi16(31));
                __m128i idx = _mm_packus_epi16(
                    _mm256_castsi256_si128(cls),
                    _mm256_extracti128_si256(cls, 1));
                __m128i off_lo = _mm_shuffle_epi8(lut_lo, idx);
                __m128i off_hi = _mm_shuffle_epi8(lut_hi, idx);
                __m128i mask = _mm_cmpgt_epi8(idx, _mm_set1_epi8(15));
                __m128i off8 = _mm_blendv_epi8(off_lo, off_hi, mask);
                __m256i off16 = _mm256_cvtepi8_epi16(off8);
                v = _mm256_add_epi16(v, off16);
                v = _mm256_max_epi16(v, _mm256_setzero_si256());
                v = _mm256_min_epi16(v, _mm256_set1_epi16((short)maxval));
                _mm_storeu_si128((__m128i *)(d + x),
                    _mm_packus_epi16(_mm256_castsi256_si128(v),
                                     _mm256_extracti128_si256(v, 1)));
            }
            for (; x + 8 <= width; x += 8) {
                __m128i v = _mm_cvtepu8_epi16(
                    _mm_loadl_epi64((const __m128i *)(s + x)));
                __m128i cls = _mm_and_si128(_mm_srl_epi16(v, sh),
                                            _mm_set1_epi16(31));
                __m128i idx = _mm_packus_epi16(cls, cls);
                __m128i off_lo = _mm_shuffle_epi8(lut_lo, idx);
                __m128i off_hi = _mm_shuffle_epi8(lut_hi, idx);
                __m128i mask = _mm_cmpgt_epi8(idx, _mm_set1_epi8(15));
                __m128i off8 = _mm_blendv_epi8(off_lo, off_hi, mask);
                __m128i off16 = _mm_cvtepi8_epi16(off8);
                v = _mm_add_epi16(v, off16);
                v = _mm_max_epi16(v, zero);
                v = _mm_min_epi16(v, maxv);
                _mm_storel_epi64((__m128i *)(d + x), _mm_packus_epi16(v, v));
            }
            for (; x < width; x++) {
                int v = s[x] + offset_table[(s[x] >> shift) & 31];
                d[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        } else {
            const uint16_t *s16 = (const uint16_t *)s;
            uint16_t *d16 = (uint16_t *)d;
            for (x = 0; x + 16 <= width; x += 16) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(s16 + x));
                __m256i cls = _mm256_and_si256(_mm256_srl_epi16(v, sh),
                                               _mm256_set1_epi16(31));
                __m128i idx = _mm_packus_epi16(
                    _mm256_castsi256_si128(cls),
                    _mm256_extracti128_si256(cls, 1));
                __m128i off_lo = _mm_shuffle_epi8(lut_lo, idx);
                __m128i off_hi = _mm_shuffle_epi8(lut_hi, idx);
                __m128i mask = _mm_cmpgt_epi8(idx, _mm_set1_epi8(15));
                __m128i off8 = _mm_blendv_epi8(off_lo, off_hi, mask);
                __m256i off16 = _mm256_cvtepi8_epi16(off8);
                v = _mm256_add_epi16(v, off16);
                v = _mm256_max_epi16(v, _mm256_setzero_si256());
                v = _mm256_min_epi16(v, _mm256_set1_epi16((short)maxval));
                _mm256_storeu_si256((__m256i *)(d16 + x), v);
            }
            for (; x + 8 <= width; x += 8) {
                __m128i v = _mm_loadu_si128((const __m128i *)(s16 + x));
                __m128i cls = _mm_and_si128(_mm_srl_epi16(v, sh),
                                            _mm_set1_epi16(31));
                __m128i idx = _mm_packus_epi16(cls, cls);
                __m128i off_lo = _mm_shuffle_epi8(lut_lo, idx);
                __m128i off_hi = _mm_shuffle_epi8(lut_hi, idx);
                __m128i mask = _mm_cmpgt_epi8(idx, _mm_set1_epi8(15));
                __m128i off8 = _mm_blendv_epi8(off_lo, off_hi, mask);
                __m128i off16 = _mm_cvtepi8_epi16(off8);
                v = _mm_add_epi16(v, off16);
                v = _mm_max_epi16(v, zero);
                v = _mm_min_epi16(v, maxv);
                _mm_storeu_si128((__m128i *)(d16 + x), v);
            }
            for (; x < width; x++) {
                int v = s16[x] + offset_table[(s16[x] >> shift) & 31];
                d16[x] = (uint16_t)(v < 0 ? 0 : v > maxval ? maxval : v);
            }
        }
    }
}

#endif
