/*
 * ARM64 NEON kernels for HEVC reconstruction.
 *
 * Mirrors the SSE2/AVX2 kernels in h265_dsp_sse2.c / h265_dsp_avx2.c and is
 * selected at run time on AArch64.  NEON is mandatory in AArch64, so the
 * runtime flag is always set unless H265_DISABLE_SIMD is given.
 */
#include "h265_internal.h"

#if defined(H265_NEON)

#include <arm_neon.h>

/* ------------------------------------------------------------------ */
/* residual addition                                                   */
/* ------------------------------------------------------------------ */
void h265_add_residual_neon(uint8_t *dst8, const int16_t *res,
                            ptrdiff_t stride, int size, int pixel_shift,
                            int bit_depth)
{
    int x, y;

    if (!pixel_shift) {
        const int16x8_t zero = vdupq_n_s16(0);
        for (y = 0; y < size; y++) {
            uint8_t *dst = dst8;
            x = 0;
            for (; x + 8 <= size; x += 8) {
                int16x8_t d = vreinterpretq_s16_u16(
                    vmovl_u8(vld1_u8(dst + x)));
                int16x8_t r = vld1q_s16(res + x);
                d = vqaddq_s16(d, r);
                d = vmaxq_s16(d, zero);
                vst1_u8(dst + x, vqmovun_s16(d));
            }
            for (; x < size; x++) {
                int v = dst[x] + res[x];
                dst[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
            dst8 += stride;
            res += size;
        }
    } else if (bit_depth <= 12) {
        const int16x8_t zero = vdupq_n_s16(0);
        const int16x8_t maxv = vdupq_n_s16((short)((1 << bit_depth) - 1));
        for (y = 0; y < size; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            x = 0;
            for (; x + 8 <= size; x += 8) {
                int16x8_t d = vreinterpretq_s16_u16(vld1q_u16(dst + x));
                int16x8_t r = vld1q_s16(res + x);
                d = vqaddq_s16(d, r);
                d = vmaxq_s16(d, zero);
                d = vminq_s16(d, maxv);
                vst1q_u16(dst + x, vreinterpretq_u16_s16(d));
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
}

/* ------------------------------------------------------------------ */
/* pixel copy / average                                                */
/* ------------------------------------------------------------------ */
void h265_copy_pixels_neon(uint8_t *dst, ptrdiff_t dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int width, int height, int pixel_shift)
{
    int bytes = width << pixel_shift;
    int x, y;

    for (y = 0; y < height; y++) {
        x = 0;
        for (; x + 16 <= bytes; x += 16)
            vst1q_u8(dst + x, vld1q_u8(src + x));
        for (; x < bytes; x++)
            dst[x] = src[x];
        dst += dst_stride;
        src += src_stride;
    }
}

void h265_avg_pixels_neon(uint8_t *dst, ptrdiff_t dst_stride,
                          const uint8_t *src1, ptrdiff_t src1_stride,
                          const uint8_t *src2, ptrdiff_t src2_stride,
                          int width, int height, int pixel_shift)
{
    int x, y;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            x = 0;
            for (; x + 16 <= width; x += 16)
                vst1q_u8(dst + x, vrhaddq_u8(vld1q_u8(src1 + x),
                                             vld1q_u8(src2 + x)));
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
            for (; x + 8 <= width; x += 8)
                vst1q_u16(d + x, vrhaddq_u16(vld1q_u16(s1 + x),
                                             vld1q_u16(s2 + x)));
            for (; x < width; x++)
                d[x] = (uint16_t)((s1[x] + s2[x] + 1) >> 1);
            dst += dst_stride;
            src1 += src1_stride;
            src2 += src2_stride;
        }
    }
}

/* ------------------------------------------------------------------ */
/* integer motion compensation                                         */
/* ------------------------------------------------------------------ */
void h265_put_pixels_neon(int16_t *dst, int dst_stride, const uint8_t *src,
                          ptrdiff_t src_stride, int width, int height,
                          int shift, int pixel_shift)
{
    const int16x8_t vsh = vdupq_n_s16((short)shift);
    int x, y;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            x = 0;
            for (; x + 8 <= width; x += 8) {
                int16x8_t v = vreinterpretq_s16_u16(
                    vmovl_u8(vld1_u8(src + x)));
                vst1q_s16(dst + x, vshlq_s16(v, vsh));
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
            for (; x + 8 <= width; x += 8) {
                int16x8_t v = vreinterpretq_s16_u16(vld1q_u16(s + x));
                vst1q_s16(dst + x, vshlq_s16(v, vsh));
            }
            for (; x < width; x++)
                dst[x] = (int16_t)(s[x] << shift);
            src += src_stride;
            dst += dst_stride;
        }
    }
}

/* ------------------------------------------------------------------ */
/* sub-pixel interpolation                                             */
/* ------------------------------------------------------------------ */
void h265_mc_filter_h_neon(int16_t *dst, int dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int height, int width, const int8_t *filter,
                           int taps, int shift, int pixel_shift)
{
    int start = taps / 2 - 1;
    int16x8_t vsh = vdupq_n_s16((short)-shift);
    int x, y, k;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            const uint8_t *row = src + (size_t)y * src_stride;
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                int16x8_t acc = vdupq_n_s16(0);
                for (k = 0; k < taps; k++) {
                    int16x8_t p = vreinterpretq_s16_u16(
                        vmovl_u8(vld1_u8(row + x - start + k)));
                    acc = vmlaq_s16(acc, p, vdupq_n_s16(filter[k]));
                }
                if (shift)
                    acc = vshlq_s16(acc, vsh);
                vst1q_s16(d + x, acc);
            }
            for (; x < width; x++) {
                int sum = 0;
                for (k = 0; k < taps; k++)
                    sum += filter[k] * row[x - start + k];
                d[x] = (int16_t)(sum >> shift);
            }
        }
    } else {
        int32x4_t vsh32 = vdupq_n_s32(-shift);
        for (y = 0; y < height; y++) {
            const uint16_t *row = (const uint16_t *)
                (src + (size_t)y * src_stride);
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                int32x4_t lo = vdupq_n_s32(0), hi = vdupq_n_s32(0);
                for (k = 0; k < taps; k++) {
                    int16x8_t p = vreinterpretq_s16_u16(
                        vld1q_u16(row + x - start + k));
                    int16x4_t c = vdup_n_s16(filter[k]);
                    lo = vmlal_s16(lo, vget_low_s16(p), c);
                    hi = vmlal_s16(hi, vget_high_s16(p), c);
                }
                if (shift) {
                    lo = vshlq_s32(lo, vsh32);
                    hi = vshlq_s32(hi, vsh32);
                }
                vst1q_s16(d + x, vcombine_s16(vqmovn_s32(lo),
                                              vqmovn_s32(hi)));
            }
            for (; x < width; x++) {
                int sum = 0;
                for (k = 0; k < taps; k++)
                    sum += filter[k] * row[x - start + k];
                d[x] = (int16_t)(sum >> shift);
            }
        }
    }
}

void h265_mc_filter_v_neon(int16_t *dst, int dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int height, int width, const int8_t *filter,
                           int taps, int shift, int pixel_shift,
                           int signed_src)
{
    int start = taps / 2 - 1;
    int16x8_t vsh = vdupq_n_s16((short)-shift);
    int x, y, k;

    if (!pixel_shift) {
        for (y = 0; y < height; y++) {
            const uint8_t *p = src + (ptrdiff_t)(y - start) * src_stride;
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                int16x8_t acc = vdupq_n_s16(0);
                for (k = 0; k < taps; k++) {
                    int16x8_t q = vreinterpretq_s16_u16(
                        vmovl_u8(vld1_u8(p + (ptrdiff_t)k * src_stride + x)));
                    acc = vmlaq_s16(acc, q, vdupq_n_s16(filter[k]));
                }
                if (shift)
                    acc = vshlq_s16(acc, vsh);
                vst1q_s16(d + x, acc);
            }
            for (; x < width; x++) {
                int sum = 0;
                for (k = 0; k < taps; k++)
                    sum += filter[k] * p[k * src_stride + x];
                d[x] = (int16_t)(sum >> shift);
            }
        }
    } else {
        int32x4_t vsh32 = vdupq_n_s32(-shift);
        for (y = 0; y < height; y++) {
            const uint8_t *p = src + (ptrdiff_t)(y - start) * src_stride;
            int16_t *d = dst + (size_t)y * dst_stride;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                int32x4_t lo = vdupq_n_s32(0), hi = vdupq_n_s32(0);
                for (k = 0; k < taps; k++) {
                    const uint8_t *q = p + (ptrdiff_t)k * src_stride;
                    int16x8_t px;
                    if (signed_src)
                        px = vld1q_s16((const int16_t *)q + x);
                    else
                        px = vreinterpretq_s16_u16(
                            vld1q_u16((const uint16_t *)q + x));
                    int16x4_t c = vdup_n_s16(filter[k]);
                    lo = vmlal_s16(lo, vget_low_s16(px), c);
                    hi = vmlal_s16(hi, vget_high_s16(px), c);
                }
                if (shift) {
                    lo = vshlq_s32(lo, vsh32);
                    hi = vshlq_s32(hi, vsh32);
                }
                vst1q_s16(d + x, vcombine_s16(vqmovn_s32(lo),
                                              vqmovn_s32(hi)));
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
    }
}

/* ------------------------------------------------------------------ */
/* motion-compensation output                                          */
/* ------------------------------------------------------------------ */
H265_INLINE uint32_t neon_pack_u8x4(int16x4_t v)
{
    uint8x8_t b = vqmovun_s16(vcombine_s16(v, v));
    return vget_lane_u32(vreinterpret_u32_u8(b), 0);
}

void h265_mc_uni_write_neon(uint8_t *dst8, ptrdiff_t dst_stride,
                            const int16_t *src, int src_stride,
                            int width, int height, int shift, int add,
                            int bit_depth, int wx, int ox)
{
    const int pixel_shift = bit_depth > 8;
    const int maxv = (1 << bit_depth) - 1;
    int x, y;

    if (!wx) {
        int16x8_t vadd = vdupq_n_s16((short)add);
        int16x8_t vmax = vdupq_n_s16((short)maxv);
        int16x8_t vsh = vdupq_n_s16((short)-shift);
        int16x4_t vadd4 = vdup_n_s16((short)add);
        int16x4_t vmax4 = vdup_n_s16((short)maxv);
        int16x4_t vsh4 = vdup_n_s16((short)-shift);
        int16x8_t zero = vdupq_n_s16(0);

        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            const int16_t *s = src;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                int16x8_t v = vld1q_s16(s + x);
                v = vqaddq_s16(v, vadd);
                v = vshlq_s16(v, vsh);
                v = vmaxq_s16(v, zero);
                v = vminq_s16(v, vmax);
                if (!pixel_shift)
                    vst1_u8(dst + x, vqmovun_s16(v));
                else
                    vst1q_u16((uint16_t *)dst + x, vreinterpretq_u16_s16(v));
            }
            for (; x + 4 <= width; x += 4) {
                int16x4_t v = vld1_s16(s + x);
                v = vqadd_s16(v, vadd4);
                v = vshl_s16(v, vsh4);
                v = vmax_s16(v, vget_low_s16(zero));
                v = vmin_s16(v, vmax4);
                if (!pixel_shift) {
                    uint32_t w = neon_pack_u8x4(v);
                    memcpy(dst + x, &w, 4);
                } else {
                    vst1_s16((int16_t *)dst + x, v);
                }
            }
            for (; x < width; x++) {
                int v = (s[x] + add) >> shift;
                v = v < 0 ? 0 : v > maxv ? maxv : v;
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
        int32x4_t vwx = vdupq_n_s32(wx);
        int32x4_t vadd = vdupq_n_s32(add);
        int32x4_t vox = vdupq_n_s32(ox);
        int32x4_t vsh = vdupq_n_s32(-shift);
        int32x4_t zero = vdupq_n_s32(0);
        int32x4_t vmax = vdupq_n_s32(maxv);

        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            const int16_t *s = src;
            x = 0;
            for (; x + 8 <= width; x += 8) {
                int16x8_t p = vld1q_s16(s + x);
                int32x4_t lo = vmulq_s32(vmovl_s16(vget_low_s16(p)), vwx);
                int32x4_t hi = vmulq_s32(vmovl_s16(vget_high_s16(p)), vwx);
                lo = vshlq_s32(vaddq_s32(lo, vadd), vsh);
                hi = vshlq_s32(vaddq_s32(hi, vadd), vsh);
                lo = vaddq_s32(lo, vox);
                hi = vaddq_s32(hi, vox);
                lo = vmaxq_s32(vminq_s32(lo, vmax), zero);
                hi = vmaxq_s32(vminq_s32(hi, vmax), zero);
                if (!pixel_shift)
                    vst1_u8(dst + x, vqmovun_s16(vcombine_s16(vqmovn_s32(lo),
                                                              vqmovn_s32(hi))));
                else
                    vst1q_u16((uint16_t *)dst + x,
                              vreinterpretq_u16_s16(
                                  vcombine_s16(vqmovn_s32(lo),
                                               vqmovn_s32(hi))));
            }
            for (; x + 4 <= width; x += 4) {
                int32x4_t v = vmulq_s32(vmovl_s16(vld1_s16(s + x)), vwx);
                v = vshlq_s32(vaddq_s32(v, vadd), vsh);
                v = vaddq_s32(v, vox);
                v = vmaxq_s32(vminq_s32(v, vmax), zero);
                if (!pixel_shift) {
                    uint32_t w = neon_pack_u8x4(vqmovn_s32(v));
                    memcpy(dst + x, &w, 4);
                } else {
                    vst1_s16((int16_t *)dst + x, vqmovn_s32(v));
                }
            }
            for (; x < width; x++) {
                int v = ((s[x] * wx + add) >> shift) + ox;
                v = v < 0 ? 0 : v > maxv ? maxv : v;
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

void h265_mc_combine_bi_neon(uint8_t *dst8, ptrdiff_t dst_stride,
                             const int16_t *first, const int16_t *second,
                             int src_stride, int width, int height, int shift,
                             int add, int bit_depth, int weighted,
                             int wx0, int wx1)
{
    const int pixel_shift = bit_depth > 8;
    const int maxv = (1 << bit_depth) - 1;
    int32x4_t vadd = vdupq_n_s32(add);
    int32x4_t vsh = vdupq_n_s32(-shift);
    int32x4_t vox0 = vdupq_n_s32(wx0);
    int32x4_t vox1 = vdupq_n_s32(wx1);
    int32x4_t zero = vdupq_n_s32(0);
    int32x4_t vmax = vdupq_n_s32(maxv);
    int x, y;

    for (y = 0; y < height; y++) {
        uint8_t *dst = dst8;
        x = 0;
        for (; x + 8 <= width; x += 8) {
            int16x8_t a16 = vld1q_s16(first + x);
            int16x8_t b16 = vld1q_s16(second + x);
            int32x4_t alo = vmovl_s16(vget_low_s16(a16));
            int32x4_t ahi = vmovl_s16(vget_high_s16(a16));
            int32x4_t blo = vmovl_s16(vget_low_s16(b16));
            int32x4_t bhi = vmovl_s16(vget_high_s16(b16));
            int32x4_t lo, hi;
            if (weighted) {
                lo = vaddq_s32(vaddq_s32(vmulq_s32(alo, vox0),
                                         vmulq_s32(blo, vox1)), vadd);
                hi = vaddq_s32(vaddq_s32(vmulq_s32(ahi, vox0),
                                         vmulq_s32(bhi, vox1)), vadd);
            } else {
                lo = vaddq_s32(vaddq_s32(alo, blo), vadd);
                hi = vaddq_s32(vaddq_s32(ahi, bhi), vadd);
            }
            lo = vshlq_s32(lo, vsh);
            hi = vshlq_s32(hi, vsh);
            lo = vmaxq_s32(vminq_s32(lo, vmax), zero);
            hi = vmaxq_s32(vminq_s32(hi, vmax), zero);
            if (!pixel_shift)
                vst1_u8(dst + x, vqmovun_s16(vcombine_s16(vqmovn_s32(lo),
                                                          vqmovn_s32(hi))));
            else
                vst1q_u16((uint16_t *)dst + x,
                          vreinterpretq_u16_s16(
                              vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))));
        }
        for (; x + 4 <= width; x += 4) {
            int32x4_t a = vmovl_s16(vld1_s16(first + x));
            int32x4_t b = vmovl_s16(vld1_s16(second + x));
            int32x4_t v;
            if (weighted)
                v = vaddq_s32(vaddq_s32(vmulq_s32(a, vox0),
                                        vmulq_s32(b, vox1)), vadd);
            else
                v = vaddq_s32(vaddq_s32(a, b), vadd);
            v = vshlq_s32(v, vsh);
            v = vmaxq_s32(vminq_s32(v, vmax), zero);
            if (!pixel_shift) {
                uint32_t w = neon_pack_u8x4(vqmovn_s32(v));
                memcpy(dst + x, &w, 4);
            } else {
                vst1_s16((int16_t *)dst + x, vqmovn_s32(v));
            }
        }
        for (; x < width; x++) {
            int v;
            if (weighted)
                v = (second[x] * wx1 + first[x] * wx0 + add) >> shift;
            else
                v = (first[x] + second[x] + add) >> shift;
            v = v < 0 ? 0 : v > maxv ? maxv : v;
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

void h265_mc_weighted_int_neon(uint8_t *dst8, ptrdiff_t dst_stride,
                               const uint8_t *src, ptrdiff_t src_stride,
                               int width, int height, int wx, int ox,
                               int denom, int bit_depth, int pixel_shift)
{
    const int maxv = (1 << bit_depth) - 1;
    const int rnd = denom ? 1 << (denom - 1) : 0;
    int32x4_t vwx = vdupq_n_s32(wx);
    int32x4_t vrnd = vdupq_n_s32(rnd);
    int32x4_t vox = vdupq_n_s32(ox);
    int32x4_t vsh = vdupq_n_s32(-denom);
    int32x4_t zero = vdupq_n_s32(0);
    int32x4_t vmax = vdupq_n_s32(maxv);
    int x, y;

    for (y = 0; y < height; y++) {
        uint8_t *dst = dst8;
        const uint8_t *s = src;
        x = 0;
        for (; x + 8 <= width; x += 8) {
            int16x8_t p;
            if (pixel_shift)
                p = vreinterpretq_s16_u16(
                    vld1q_u16((const uint16_t *)s + x));
            else
                p = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s + x)));
            {
                int32x4_t lo = vmlaq_s32(vrnd,
                    vmovl_s16(vget_low_s16(p)), vwx);
                int32x4_t hi = vmlaq_s32(vrnd,
                    vmovl_s16(vget_high_s16(p)), vwx);
                lo = vaddq_s32(vshlq_s32(lo, vsh), vox);
                hi = vaddq_s32(vshlq_s32(hi, vsh), vox);
                lo = vmaxq_s32(vminq_s32(lo, vmax), zero);
                hi = vmaxq_s32(vminq_s32(hi, vmax), zero);
                if (!pixel_shift)
                    vst1_u8(dst + x,
                            vqmovun_s16(vcombine_s16(vqmovn_s32(lo),
                                                     vqmovn_s32(hi))));
                else
                    vst1q_u16((uint16_t *)dst + x,
                              vreinterpretq_u16_s16(
                                  vcombine_s16(vqmovn_s32(lo),
                                               vqmovn_s32(hi))));
            }
        }
        for (; x + 4 <= width; x += 4) {
            int32x4_t v;
            if (pixel_shift)
                v = vreinterpretq_s32_u32(
                    vmovl_u16(vld1_u16((const uint16_t *)s + x)));
            else {
                uint32_t w;
                memcpy(&w, s + x, 4);
                uint8x8_t b = vreinterpret_u8_u32(vdup_n_u32(w));
                v = vreinterpretq_s32_u32(
                    vmovl_u16(vget_low_u16(vmovl_u8(b))));
            }
            v = vmlaq_s32(vrnd, v, vwx);
            v = vaddq_s32(vshlq_s32(v, vsh), vox);
            v = vmaxq_s32(vminq_s32(v, vmax), zero);
            if (!pixel_shift) {
                uint32_t w = neon_pack_u8x4(vqmovn_s32(v));
                memcpy(dst + x, &w, 4);
            } else {
                vst1_s16((int16_t *)dst + x, vqmovn_s32(v));
            }
        }
        for (; x < width; x++) {
            int v = pixel_shift ? ((const uint16_t *)s)[x] : s[x];
            v = ((v * wx + rnd) >> denom) + ox;
            v = v < 0 ? 0 : v > maxv ? maxv : v;
            if (!pixel_shift)
                dst[x] = (uint8_t)v;
            else
                ((uint16_t *)dst)[x] = (uint16_t)v;
        }
        dst8 += dst_stride;
        src += src_stride;
    }
}

/* ------------------------------------------------------------------ */
/* SAO                                                                 */
/* ------------------------------------------------------------------ */
void h265_sao_band_neon(uint8_t *dst, ptrdiff_t dst_stride,
                        const uint8_t *src, ptrdiff_t src_stride,
                        const int *offset_table, int width, int height,
                        int pixel_shift, int bit_depth)
{
    int8_t lo[16], hi[16];
    uint8x16_t lut_lo, lut_hi;
    int16x8_t vsh, vmax;
    int shift = bit_depth - 5;
    int maxval = (1 << bit_depth) - 1;
    int x, y;

    for (x = 0; x < 16; x++) {
        int a = offset_table[x];
        int b = offset_table[x + 16];
        lo[x] = (int8_t)(a < -127 ? -127 : a > 127 ? 127 : a);
        hi[x] = (int8_t)(b < -127 ? -127 : b > 127 ? 127 : b);
    }
    lut_lo = vld1q_u8((const uint8_t *)lo);
    lut_hi = vld1q_u8((const uint8_t *)hi);
    vsh = vdupq_n_s16(-shift);
    vmax = vdupq_n_s16((short)maxval);

    for (y = 0; y < height; y++) {
        const uint8_t *s = src + (size_t)y * src_stride;
        uint8_t *d = dst + (size_t)y * dst_stride;
        if (!pixel_shift) {
            for (x = 0; x + 8 <= width; x += 8) {
                int16x8_t v = vreinterpretq_s16_u16(
                    vmovl_u8(vld1_u8(s + x)));
                int16x8_t cls = vandq_s16(vshlq_s16(v, vsh),
                                          vdupq_n_s16(31));
                uint8x8_t idx8 = vqmovun_s16(cls);
                uint8x16_t idx = vcombine_u8(idx8, idx8);
                uint8x16_t off_lo = vqtbl1q_u8(lut_lo, idx);
                uint8x16_t off_hi = vqtbl1q_u8(
                    lut_hi, vsubq_u8(idx, vdupq_n_u8(16)));
                uint8x16_t mask = vcgeq_u8(idx, vdupq_n_u8(16));
                uint8x16_t off8 = vbslq_u8(mask, off_hi, off_lo);
                int16x8_t off = vmovl_s8(
                    vreinterpret_s8_u8(vget_low_u8(off8)));
                v = vaddq_s16(v, off);
                v = vmaxq_s16(v, vdupq_n_s16(0));
                v = vminq_s16(v, vmax);
                vst1_u8(d + x, vqmovun_s16(v));
            }
            for (; x < width; x++) {
                int v = s[x] + offset_table[(s[x] >> shift) & 31];
                d[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        } else {
            const uint16_t *s16 = (const uint16_t *)s;
            uint16_t *d16 = (uint16_t *)d;
            for (x = 0; x + 8 <= width; x += 8) {
                int16x8_t v = vreinterpretq_s16_u16(vld1q_u16(s16 + x));
                int16x8_t cls = vandq_s16(vshlq_s16(v, vsh),
                                          vdupq_n_s16(31));
                uint8x8_t idx8 = vqmovun_s16(cls);
                uint8x16_t idx = vcombine_u8(idx8, idx8);
                uint8x16_t off_lo = vqtbl1q_u8(lut_lo, idx);
                uint8x16_t off_hi = vqtbl1q_u8(
                    lut_hi, vsubq_u8(idx, vdupq_n_u8(16)));
                uint8x16_t mask = vcgeq_u8(idx, vdupq_n_u8(16));
                uint8x16_t off8 = vbslq_u8(mask, off_hi, off_lo);
                int16x8_t off = vmovl_s8(
                    vreinterpret_s8_u8(vget_low_u8(off8)));
                v = vaddq_s16(v, off);
                v = vmaxq_s16(v, vdupq_n_s16(0));
                v = vminq_s16(v, vmax);
                vst1q_u16(d16 + x, vreinterpretq_u16_s16(v));
            }
            for (; x < width; x++) {
                int v = s16[x] + offset_table[(s16[x] >> shift) & 31];
                d16[x] = (uint16_t)(v < 0 ? 0 : v > maxval ? maxval : v);
            }
        }
    }
}

void h265_sao_edge_neon(uint8_t *dst, ptrdiff_t dst_stride,
                        const uint8_t *src, ptrdiff_t src_stride,
                        int a_stride, int b_stride,
                        const int16_t *sao_offset_val, int width, int height,
                        int pixel_shift, int bit_depth)
{
    static const uint8_t edge_idx[5] = { 1, 2, 0, 3, 4 };
    int8_t lutv[16] = { 0 };
    uint8x16_t lut;
    int16x8_t one = vdupq_n_s16(1);
    int16x8_t two = vdupq_n_s16(2);
    int16x8_t zero = vdupq_n_s16(0);
    int16x8_t vmax = vdupq_n_s16((short)((1 << bit_depth) - 1));
    int x, y, i;

    for (i = 0; i < 5; i++) {
        int v = sao_offset_val[edge_idx[i]];
        lutv[i] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
    }
    lut = vld1q_u8((const uint8_t *)lutv);

    for (y = 0; y < height; y++) {
        if (!pixel_shift) {
            const uint8_t *s = src + (size_t)y * src_stride;
            uint8_t *d = dst + (size_t)y * dst_stride;
            for (x = 0; x + 8 <= width; x += 8) {
                int16x8_t v = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s + x)));
                int16x8_t a = vreinterpretq_s16_u16(
                    vmovl_u8(vld1_u8(s + x + a_stride)));
                int16x8_t b = vreinterpretq_s16_u16(
                    vmovl_u8(vld1_u8(s + x + b_stride)));
                int16x8_t gt0 = vreinterpretq_s16_u16(vcgtq_s16(v, a));
                int16x8_t lt0 = vreinterpretq_s16_u16(vcgtq_s16(a, v));
                int16x8_t gt1 = vreinterpretq_s16_u16(vcgtq_s16(v, b));
                int16x8_t lt1 = vreinterpretq_s16_u16(vcgtq_s16(b, v));
                int16x8_t diff0 = vsubq_s16(vandq_s16(gt0, one),
                                            vandq_s16(lt0, one));
                int16x8_t diff1 = vsubq_s16(vandq_s16(gt1, one),
                                            vandq_s16(lt1, one));
                int16x8_t idx16 = vaddq_s16(two,
                                            vaddq_s16(diff0, diff1));
                uint8x8_t idx8 = vqmovun_s16(idx16);
                uint8x16_t idx = vcombine_u8(idx8, idx8);
                uint8x16_t off8 = vqtbl1q_u8(lut, idx);
                int16x8_t off = vmovl_s8(
                    vreinterpret_s8_u8(vget_low_u8(off8)));
                v = vaddq_s16(v, off);
                v = vmaxq_s16(v, zero);
                v = vminq_s16(v, vmax);
                vst1_u8(d + x, vqmovun_s16(v));
            }
            for (; x < width; x++) {
                int a = s[x + a_stride], b = s[x + b_stride];
                int diff0 = (s[x] > a) - (s[x] < a);
                int diff1 = (s[x] > b) - (s[x] < b);
                int v = s[x] + sao_offset_val[edge_idx[2 + diff0 + diff1]];
                d[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        } else {
            const uint16_t *s16 = (const uint16_t *)
                (src + (size_t)y * src_stride);
            uint16_t *d16 = (uint16_t *)(dst + (size_t)y * dst_stride);
            for (x = 0; x + 8 <= width; x += 8) {
                int16x8_t v = vreinterpretq_s16_u16(vld1q_u16(s16 + x));
                int16x8_t a = vreinterpretq_s16_u16(vld1q_u16(
                    (const uint16_t *)((const uint8_t *)(s16 + x) + a_stride)));
                int16x8_t b = vreinterpretq_s16_u16(vld1q_u16(
                    (const uint16_t *)((const uint8_t *)(s16 + x) + b_stride)));
                int16x8_t gt0 = vreinterpretq_s16_u16(vcgtq_s16(v, a));
                int16x8_t lt0 = vreinterpretq_s16_u16(vcgtq_s16(a, v));
                int16x8_t gt1 = vreinterpretq_s16_u16(vcgtq_s16(v, b));
                int16x8_t lt1 = vreinterpretq_s16_u16(vcgtq_s16(b, v));
                int16x8_t diff0 = vsubq_s16(vandq_s16(gt0, one),
                                            vandq_s16(lt0, one));
                int16x8_t diff1 = vsubq_s16(vandq_s16(gt1, one),
                                            vandq_s16(lt1, one));
                int16x8_t idx16 = vaddq_s16(two,
                                            vaddq_s16(diff0, diff1));
                uint8x8_t idx8 = vqmovun_s16(idx16);
                uint8x16_t idx = vcombine_u8(idx8, idx8);
                uint8x16_t off8 = vqtbl1q_u8(lut, idx);
                int16x8_t off = vmovl_s8(
                    vreinterpret_s8_u8(vget_low_u8(off8)));
                v = vaddq_s16(v, off);
                v = vmaxq_s16(v, zero);
                v = vminq_s16(v, vmax);
                vst1q_u16(d16 + x, vreinterpretq_u16_s16(v));
            }
            for (; x < width; x++) {
                int a = s16[x + (a_stride >> 1)];
                int b = s16[x + (b_stride >> 1)];
                int diff0 = (s16[x] > a) - (s16[x] < a);
                int diff1 = (s16[x] > b) - (s16[x] < b);
                int v = s16[x] + sao_offset_val[edge_idx[2 + diff0 + diff1]];
                d16[x] = (uint16_t)(v < 0 ? 0 : v > ((1 << bit_depth) - 1)
                                                   ? ((1 << bit_depth) - 1)
                                                   : v);
            }
        }
    }
}

#endif /* H265_NEON */
