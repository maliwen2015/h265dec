/*
 * HEVC motion compensation: 8-tap luma, 4-tap chroma, uni/bi and weighted
 * prediction, plus edge emulation.
 *
 * Derived from FFmpeg's hevc/dsp_template.c and h26x/h2656_inter_template.c.
 */
#include "h265_internal.h"

#define MAX_PB H265_MAX_PB_SIZE

void h265_emulated_edge_mc(uint8_t *dst, const uint8_t *src, ptrdiff_t dst_stride,
                           ptrdiff_t src_stride, int block_w, int block_h,
                           int src_x, int src_y, int w, int h, int pixel_shift)
{
    int x, y;

    for (y = 0; y < block_h; y++) {
        int sy = src_y + y;
        const uint8_t *srow;
        if (sy < 0)
            sy = 0;
        else if (sy >= h)
            sy = h - 1;
        srow = src + (ptrdiff_t)(sy - src_y) * src_stride;
        if (src_x >= 0 && src_x + block_w <= w) {
            if (!pixel_shift)
                memcpy(dst + y * dst_stride, srow, (size_t)block_w);
            else
                memcpy(dst + y * dst_stride, srow,
                       (size_t)block_w * sizeof(uint16_t));
            continue;
        }
        {
            int left = src_x < 0 ? -src_x : 0;
            int right = src_x + block_w > w ? src_x + block_w - w : 0;
            int interior = block_w - left - right;
            uint8_t *d = dst + y * dst_stride;
            const uint16_t *s16 = (const uint16_t *)srow;
            uint16_t *d16 = (uint16_t *)d;

            if (src_x + block_w <= 0) {
                int idx = -src_x;
                if (!pixel_shift) {
                    memset(d, srow[idx], (size_t)block_w);
                } else {
                    uint16_t v = s16[idx];
                    for (x = 0; x < block_w; x++)
                        d16[x] = v;
                }
                continue;
            }
            if (src_x >= w) {
                int idx = w - 1 - src_x;
                if (!pixel_shift) {
                    memset(d, srow[idx], (size_t)block_w);
                } else {
                    uint16_t v = s16[idx];
                    for (x = 0; x < block_w; x++)
                        d16[x] = v;
                }
                continue;
            }
            if (left > 0) {
                if (!pixel_shift) {
                    memset(d, srow[left], (size_t)left);
                } else {
                    uint16_t v = s16[left];
                    for (x = 0; x < left; x++)
                        d16[x] = v;
                }
            }
            if (interior > 0) {
                if (!pixel_shift)
                    memcpy(d + left, srow + left, (size_t)interior);
                else
                    memcpy(d16 + left, s16 + left,
                           (size_t)interior * sizeof(uint16_t));
            }
            if (right > 0) {
                int pos = left + interior;
                if (!pixel_shift) {
                    memset(d + pos, pos > 0 ? srow[pos - 1] : srow[0],
                           (size_t)right);
                } else {
                    uint16_t v = s16[pos > 0 ? pos - 1 : 0];
                    for (x = pos; x < block_w; x++)
                        d16[x] = v;
                }
            }
            continue;
        }
    }
}

/* ------------------------------------------------------------------ */
/* interpolation kernels                                               */
/* ------------------------------------------------------------------ */
#define LUMA_FILTER(src, stride)                                        \
    (filter[0] * src[x - 3 * (stride)] +                                \
     filter[1] * src[x - 2 * (stride)] +                                \
     filter[2] * src[x -     (stride)] +                                \
     filter[3] * src[x                 ] +                              \
     filter[4] * src[x +     (stride)] +                                \
     filter[5] * src[x + 2 * (stride)] +                                \
     filter[6] * src[x + 3 * (stride)] +                                \
     filter[7] * src[x + 4 * (stride)])

#define CHROMA_FILTER(src, stride)                                      \
    (filter[0] * src[x - (stride)] +                                    \
     filter[1] * src[x            ] +                                   \
     filter[2] * src[x + (stride)] +                                    \
     filter[3] * src[x + 2 * (stride)])

static void put_pixels(int16_t *dst, const uint8_t *_src,
                       ptrdiff_t _src_stride, int height, int width,
                       int bit_depth)
{
    int x, y;
#ifdef H265_X86
    if (h265_have_avx2) {
        h265_put_pixels_avx2(dst, MAX_PB, _src, _src_stride, width, height,
                             14 - bit_depth, bit_depth > 8);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_put_pixels_neon(dst, MAX_PB, _src, _src_stride, width, height,
                             14 - bit_depth, bit_depth > 8);
        return;
    }
#endif
    if (bit_depth <= 8) {
        const uint8_t *src = _src;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(src[x] << (14 - bit_depth));
            src += _src_stride;
            dst += MAX_PB;
        }
    } else {
        const uint16_t *src = (const uint16_t *)_src;
        ptrdiff_t stride = _src_stride >> 1;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(src[x] << (14 - bit_depth));
            src += stride;
            dst += MAX_PB;
        }
    }
}

static void put_luma_h(int16_t *dst, const uint8_t *_src, ptrdiff_t _src_stride,
                       int height, int width, const int8_t *filter,
                       int bit_depth)
{
    int x, y;
#ifdef H265_X86
    if (h265_have_avx2) {
        h265_mc_filter_h_avx2(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 8, bit_depth - 8, bit_depth > 8);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_mc_filter_h_neon(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 8, bit_depth - 8, bit_depth > 8);
        return;
    }
#endif
    if (bit_depth <= 8) {
        const uint8_t *src = _src;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(LUMA_FILTER(src, 1) >> (bit_depth - 8));
            src += _src_stride;
            dst += MAX_PB;
        }
    } else {
        const uint16_t *src = (const uint16_t *)_src;
        ptrdiff_t stride = _src_stride >> 1;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(LUMA_FILTER(src, 1) >> (bit_depth - 8));
            src += stride;
            dst += MAX_PB;
        }
    }
}

static void put_luma_v(int16_t *dst, const uint8_t *_src, ptrdiff_t _src_stride,
                       int height, int width, const int8_t *filter,
                       int bit_depth)
{
    int x, y;
#ifdef H265_X86
    if (h265_have_avx2) {
        h265_mc_filter_v_avx2(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 8, bit_depth - 8, bit_depth > 8, 0);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_mc_filter_v_neon(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 8, bit_depth - 8, bit_depth > 8, 0);
        return;
    }
#endif
    if (bit_depth <= 8) {
        const uint8_t *src = _src;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(LUMA_FILTER(src, _src_stride) >> (bit_depth - 8));
            src += _src_stride;
            dst += MAX_PB;
        }
    } else {
        const uint16_t *src = (const uint16_t *)_src;
        ptrdiff_t stride = _src_stride >> 1;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(LUMA_FILTER(src, stride) >> (bit_depth - 8));
            src += stride;
            dst += MAX_PB;
        }
    }
}

static void put_luma_hv(int16_t *dst, const uint8_t *_src, ptrdiff_t _src_stride,
                        int height, int width, const int8_t *hf,
                        const int8_t *vf, int bit_depth)
{
    int x, y;
    int16_t tmp_array[(MAX_PB + H265_QPEL_EXTRA) * MAX_PB];
    int16_t *tmp = tmp_array;
    const int8_t *filter = hf;

#ifdef H265_X86
    if (h265_have_avx2) {
        const uint8_t *src = _src - H265_QPEL_EXTRA_BEFORE * _src_stride;
        h265_mc_filter_h_avx2(tmp_array, MAX_PB, src, _src_stride,
                              height + H265_QPEL_EXTRA, width, hf, 8,
                              bit_depth - 8, bit_depth > 8);
        h265_mc_filter_v_avx2(dst, MAX_PB,
                              (const uint8_t *)(tmp_array +
                                  H265_QPEL_EXTRA_BEFORE * MAX_PB),
                              MAX_PB * (ptrdiff_t)sizeof(int16_t),
                              height, width, vf, 8, 6, 1, 1);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        const uint8_t *src = _src - H265_QPEL_EXTRA_BEFORE * _src_stride;
        h265_mc_filter_h_neon(tmp_array, MAX_PB, src, _src_stride,
                              height + H265_QPEL_EXTRA, width, hf, 8,
                              bit_depth - 8, bit_depth > 8);
        h265_mc_filter_v_neon(dst, MAX_PB,
                              (const uint8_t *)(tmp_array +
                                  H265_QPEL_EXTRA_BEFORE * MAX_PB),
                              MAX_PB * (ptrdiff_t)sizeof(int16_t),
                              height, width, vf, 8, 6, 1, 1);
        return;
    }
#endif

    if (bit_depth <= 8) {
        const uint8_t *src = _src - H265_QPEL_EXTRA_BEFORE * _src_stride;
        for (y = 0; y < height + H265_QPEL_EXTRA; y++) {
            for (x = 0; x < width; x++)
                tmp[x] = (int16_t)(LUMA_FILTER(src, 1) >> (bit_depth - 8));
            src += _src_stride;
            tmp += MAX_PB;
        }
        tmp = tmp_array + H265_QPEL_EXTRA_BEFORE * MAX_PB;
        filter = vf;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(LUMA_FILTER(tmp, MAX_PB) >> 6);
            tmp += MAX_PB;
            dst += MAX_PB;
        }
    } else {
        ptrdiff_t stride = _src_stride >> 1;
        const uint16_t *src = (const uint16_t *)_src - H265_QPEL_EXTRA_BEFORE * stride;
        for (y = 0; y < height + H265_QPEL_EXTRA; y++) {
            for (x = 0; x < width; x++)
                tmp[x] = (int16_t)(LUMA_FILTER(src, 1) >> (bit_depth - 8));
            src += stride;
            tmp += MAX_PB;
        }
        tmp = tmp_array + H265_QPEL_EXTRA_BEFORE * MAX_PB;
        filter = vf;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(LUMA_FILTER(tmp, MAX_PB) >> 6);
            tmp += MAX_PB;
            dst += MAX_PB;
        }
    }
}

static void put_chroma_h(int16_t *dst, const uint8_t *_src, ptrdiff_t _src_stride,
                         int height, int width, const int8_t *filter,
                         int bit_depth)
{
    int x, y;
#ifdef H265_X86
    if (h265_have_avx2) {
        h265_mc_filter_h_avx2(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 4, bit_depth - 8, bit_depth > 8);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_mc_filter_h_neon(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 4, bit_depth - 8, bit_depth > 8);
        return;
    }
#endif
    if (bit_depth <= 8) {
        const uint8_t *src = _src;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(CHROMA_FILTER(src, 1) >> (bit_depth - 8));
            src += _src_stride;
            dst += MAX_PB;
        }
    } else {
        const uint16_t *src = (const uint16_t *)_src;
        ptrdiff_t stride = _src_stride >> 1;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(CHROMA_FILTER(src, 1) >> (bit_depth - 8));
            src += stride;
            dst += MAX_PB;
        }
    }
}

static void put_chroma_v(int16_t *dst, const uint8_t *_src, ptrdiff_t _src_stride,
                         int height, int width, const int8_t *filter,
                         int bit_depth)
{
    int x, y;
#ifdef H265_X86
    if (h265_have_avx2) {
        h265_mc_filter_v_avx2(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 4, bit_depth - 8, bit_depth > 8, 0);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_mc_filter_v_neon(dst, MAX_PB, _src, _src_stride, height, width,
                              filter, 4, bit_depth - 8, bit_depth > 8, 0);
        return;
    }
#endif
    if (bit_depth <= 8) {
        const uint8_t *src = _src;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(CHROMA_FILTER(src, _src_stride) >> (bit_depth - 8));
            src += _src_stride;
            dst += MAX_PB;
        }
    } else {
        const uint16_t *src = (const uint16_t *)_src;
        ptrdiff_t stride = _src_stride >> 1;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(CHROMA_FILTER(src, stride) >> (bit_depth - 8));
            src += stride;
            dst += MAX_PB;
        }
    }
}

static void put_chroma_hv(int16_t *dst, const uint8_t *_src, ptrdiff_t _src_stride,
                          int height, int width, const int8_t *hf,
                          const int8_t *vf, int bit_depth)
{
    int x, y;
    int16_t tmp_array[(MAX_PB + H265_EPEL_EXTRA) * MAX_PB];
    int16_t *tmp = tmp_array;
    const int8_t *filter = hf;

#ifdef H265_X86
    if (h265_have_avx2) {
        const uint8_t *src = _src - H265_EPEL_EXTRA_BEFORE * _src_stride;
        h265_mc_filter_h_avx2(tmp_array, MAX_PB, src, _src_stride,
                              height + H265_EPEL_EXTRA, width, hf, 4,
                              bit_depth - 8, bit_depth > 8);
        h265_mc_filter_v_avx2(dst, MAX_PB,
                              (const uint8_t *)(tmp_array +
                                  H265_EPEL_EXTRA_BEFORE * MAX_PB),
                              MAX_PB * (ptrdiff_t)sizeof(int16_t),
                              height, width, vf, 4, 6, 1, 1);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        const uint8_t *src = _src - H265_EPEL_EXTRA_BEFORE * _src_stride;
        h265_mc_filter_h_neon(tmp_array, MAX_PB, src, _src_stride,
                              height + H265_EPEL_EXTRA, width, hf, 4,
                              bit_depth - 8, bit_depth > 8);
        h265_mc_filter_v_neon(dst, MAX_PB,
                              (const uint8_t *)(tmp_array +
                                  H265_EPEL_EXTRA_BEFORE * MAX_PB),
                              MAX_PB * (ptrdiff_t)sizeof(int16_t),
                              height, width, vf, 4, 6, 1, 1);
        return;
    }
#endif

    if (bit_depth <= 8) {
        const uint8_t *src = _src - H265_EPEL_EXTRA_BEFORE * _src_stride;
        for (y = 0; y < height + H265_EPEL_EXTRA; y++) {
            for (x = 0; x < width; x++)
                tmp[x] = (int16_t)(CHROMA_FILTER(src, 1) >> (bit_depth - 8));
            src += _src_stride;
            tmp += MAX_PB;
        }
        tmp = tmp_array + H265_EPEL_EXTRA_BEFORE * MAX_PB;
        filter = vf;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(CHROMA_FILTER(tmp, MAX_PB) >> 6);
            tmp += MAX_PB;
            dst += MAX_PB;
        }
    } else {
        ptrdiff_t stride = _src_stride >> 1;
        const uint16_t *src = (const uint16_t *)_src - H265_EPEL_EXTRA_BEFORE * stride;
        for (y = 0; y < height + H265_EPEL_EXTRA; y++) {
            for (x = 0; x < width; x++)
                tmp[x] = (int16_t)(CHROMA_FILTER(src, 1) >> (bit_depth - 8));
            src += stride;
            tmp += MAX_PB;
        }
        tmp = tmp_array + H265_EPEL_EXTRA_BEFORE * MAX_PB;
        filter = vf;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++)
                dst[x] = (int16_t)(CHROMA_FILTER(tmp, MAX_PB) >> 6);
            tmp += MAX_PB;
            dst += MAX_PB;
        }
    }
}

/* ------------------------------------------------------------------ */
/* uni-prediction                                                      */
/* ------------------------------------------------------------------ */
H265_INLINE int clip_pixel(int v, int bit_depth)
{
    int max = (1 << bit_depth) - 1;
    return v < 0 ? 0 : v > max ? max : v;
}

static void uni_write(uint8_t *dst8, ptrdiff_t dststride, const int16_t *src2,
                      int height, int width, int shift, int add, int bit_depth,
                      int wx, int ox, int denom)
{
    int x, y;

#ifdef H265_X86
    if (h265_have_avx2) {
        h265_mc_uni_write_avx2(dst8, dststride, src2, MAX_PB, width, height,
                               shift, add, bit_depth, wx, ox);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_mc_uni_write_neon(dst8, dststride, src2, MAX_PB, width, height,
                               shift, add, bit_depth, wx, ox);
        return;
    }
#endif

    if (bit_depth <= 8) {
        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            for (x = 0; x < width; x++) {
                int v = src2[x];
                if (wx)
                    v = ((v * wx + add) >> shift) + ox;
                else
                    v = (v + add) >> shift;
                dst[x] = (uint8_t)clip_pixel(v, bit_depth);
            }
            dst8 += dststride;
            src2 += MAX_PB;
        }
    } else {
        ptrdiff_t stride = dststride >> 1;
        for (y = 0; y < height; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            for (x = 0; x < width; x++) {
                int v = src2[x];
                if (wx)
                    v = ((v * wx + add) >> shift) + ox;
                else
                    v = (v + add) >> shift;
                dst[x] = (uint16_t)clip_pixel(v, bit_depth);
            }
            dst8 += stride * 2;
            src2 += MAX_PB;
        }
    }
    (void)denom;
}

static void copy_pixels_dsp(uint8_t *dst, ptrdiff_t ds, const uint8_t *src,
                            ptrdiff_t ss, int w, int h, int pixel_shift)
{
#ifdef H265_X86
    if (h265_have_avx2) {
        h265_copy_pixels_avx2(dst, ds, src, ss, w, h, pixel_shift);
        return;
    }
    if (h265_have_sse2) {
        h265_copy_pixels_sse2(dst, ds, src, ss, w, h, pixel_shift);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_copy_pixels_neon(dst, ds, src, ss, w, h, pixel_shift);
        return;
    }
#endif
    {
        int y;
        if (!pixel_shift) {
            for (y = 0; y < h; y++) {
                memcpy(dst, src, w);
                dst += ds;
                src += ss;
            }
        } else {
            for (y = 0; y < h; y++) {
                memcpy(dst, src, w * 2);
                dst += ds;
                src += ss;
            }
        }
    }
}

static void avg_pixels_dsp(uint8_t *dst, ptrdiff_t ds, const uint8_t *s1,
                           ptrdiff_t ss1, const uint8_t *s2, ptrdiff_t ss2,
                           int w, int h, int pixel_shift)
{
#ifdef H265_X86
    if (h265_have_avx2) {
        h265_avg_pixels_avx2(dst, ds, s1, ss1, s2, ss2, w, h, pixel_shift);
        return;
    }
    if (h265_have_sse2) {
        h265_avg_pixels_sse2(dst, ds, s1, ss1, s2, ss2, w, h, pixel_shift);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_avg_pixels_neon(dst, ds, s1, ss1, s2, ss2, w, h, pixel_shift);
        return;
    }
#endif
    {
        int x, y;
        if (!pixel_shift) {
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++)
                    dst[x] = (uint8_t)((s1[x] + s2[x] + 1) >> 1);
                dst += ds;
                s1 += ss1;
                s2 += ss2;
            }
        } else {
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++) {
                    uint16_t v = (uint16_t)((((const uint16_t *)s1)[x] +
                                             ((const uint16_t *)s2)[x] + 1) >> 1);
                    ((uint16_t *)dst)[x] = v;
                }
                dst += ds;
                s1 += ss1;
                s2 += ss2;
            }
        }
    }
}

void h265_mc_luma_uni(H265Slice *sh, uint8_t *dst, ptrdiff_t dststride,
                      const H265Frame *ref, const H265Mv *mv, int x_off,
                      int y_off, int block_w, int block_h, int luma_weight,
                      int luma_offset)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    const uint8_t *src = ref->data[0];
    ptrdiff_t srcstride = ref->linesize[0];
    int pic_width = sps->width;
    int pic_height = sps->height;
    int mx = mv->x & 3;
    int my = mv->y & 3;
    int weight_flag = (sh->slice_type == H265_SLICE_P && pps->weighted_pred_flag) ||
                      (sh->slice_type == H265_SLICE_B && pps->weighted_bipred_flag);
    int16_t *tmp = sh->tmp;
    int16_t *dst_tmp = tmp;

    x_off += mv->x >> 2;
    y_off += mv->y >> 2;
    src += y_off * srcstride + (x_off << sps->pixel_shift);

    if (!weight_flag && !mx && !my && x_off >= 0 && y_off >= 0 &&
        x_off + block_w <= pic_width && y_off + block_h <= pic_height) {
        copy_pixels_dsp(dst, dststride, src, srcstride, block_w, block_h,
                        sps->pixel_shift);
        return;
    }

    if (x_off < H265_QPEL_EXTRA_BEFORE || y_off < H265_QPEL_EXTRA_AFTER ||
        x_off >= pic_width - block_w - H265_QPEL_EXTRA_AFTER ||
        y_off >= pic_height - block_h - H265_QPEL_EXTRA_AFTER ||
        ref->data[0] == sh->frame->data[0]) {
        ptrdiff_t edge_emu_stride = H265_EDGE_EMU_BUFFER_STRIDE << sps->pixel_shift;
        int offset = H265_QPEL_EXTRA_BEFORE * srcstride +
                     (H265_QPEL_EXTRA_BEFORE << sps->pixel_shift);
        int buf_offset = H265_QPEL_EXTRA_BEFORE * edge_emu_stride +
                         (H265_QPEL_EXTRA_BEFORE << sps->pixel_shift);
        h265_emulated_edge_mc(sh->edge_emu_buffer, src - offset,
                              edge_emu_stride, srcstride,
                              block_w + H265_QPEL_EXTRA,
                              block_h + H265_QPEL_EXTRA,
                              x_off - H265_QPEL_EXTRA_BEFORE,
                              y_off - H265_QPEL_EXTRA_BEFORE,
                              pic_width, pic_height, sps->pixel_shift);
        src = sh->edge_emu_buffer + buf_offset;
        srcstride = edge_emu_stride;
    }

    if (!weight_flag && !mx && !my) {
        copy_pixels_dsp(dst, dststride, src, srcstride, block_w, block_h,
                        sps->pixel_shift);
        return;
    }

#ifdef H265_X86
    if (weight_flag && !mx && !my && h265_have_avx2) {
        if (luma_weight)
            h265_mc_weighted_int_avx2(dst, dststride, src, srcstride,
                                      block_w, block_h, luma_weight,
                                      luma_offset << (sps->bit_depth - 8),
                                      sh->luma_log2_weight_denom,
                                      sps->bit_depth, sps->pixel_shift);
        else
            copy_pixels_dsp(dst, dststride, src, srcstride, block_w, block_h,
                            sps->pixel_shift);
        return;
    }
#endif
#ifdef H265_NEON
    if (weight_flag && !mx && !my && h265_have_neon) {
        if (luma_weight)
            h265_mc_weighted_int_neon(dst, dststride, src, srcstride,
                                      block_w, block_h, luma_weight,
                                      luma_offset << (sps->bit_depth - 8),
                                      sh->luma_log2_weight_denom,
                                      sps->bit_depth, sps->pixel_shift);
        else
            copy_pixels_dsp(dst, dststride, src, srcstride, block_w, block_h,
                            sps->pixel_shift);
        return;
    }
#endif

    if (!my) {
        if (!mx) {
            put_pixels(dst_tmp, src, srcstride, block_h, block_w, sps->bit_depth);
        } else {
            put_luma_h(dst_tmp, src, srcstride, block_h, block_w,
                       h265_qpel_filters[mx], sps->bit_depth);
        }
    } else if (!mx) {
        put_luma_v(dst_tmp, src, srcstride, block_h, block_w,
                   h265_qpel_filters[my], sps->bit_depth);
    } else {
        put_luma_hv(dst_tmp, src, srcstride, block_h, block_w,
                    h265_qpel_filters[mx], h265_qpel_filters[my],
                    sps->bit_depth);
    }

    if (!weight_flag) {
        int shift = 14 - sps->bit_depth;
        int add = 1 << (shift - 1);
        uni_write(dst, dststride, dst_tmp, block_h, block_w, shift, add,
                  sps->bit_depth, 0, 0, 0);
    } else {
        int shift = sh->luma_log2_weight_denom + 14 - sps->bit_depth;
        int add = 1 << (shift - 1);
        int ox = luma_offset << (sps->bit_depth - 8);
        uni_write(dst, dststride, dst_tmp, block_h, block_w, shift, add,
                  sps->bit_depth, luma_weight, ox, sh->luma_log2_weight_denom);
    }
}

/* ------------------------------------------------------------------ */
/* bi-prediction                                                       */
/* ------------------------------------------------------------------ */
/* Both inputs are 14-bit intermediates obtained from the two reference
 * blocks.  The combination is the same arithmetic FFmpeg performs inside its
 * second-reference bi kernels. */
static void combine_bi(uint8_t *dst8, ptrdiff_t dststride,
                       const int16_t *first, const int16_t *second,
                       int height, int width, int bit_depth,
                       int weighted, int denom, int wx0, int wx1, int ox0,
                       int ox1)
{
    int x, y;
    int shift = 15 - bit_depth;
    int offset = 1 << (shift - 1);
    int log2Wd = denom + shift - 1;

#ifdef H265_X86
    if (h265_have_avx2) {
        if (!weighted) {
            h265_mc_combine_bi_avx2(dst8, dststride, first, second, MAX_PB,
                                    width, height, shift, offset, bit_depth,
                                    0, 0, 0);
        } else {
            int ox0s = ox0 * (1 << (bit_depth - 8));
            int ox1s = ox1 * (1 << (bit_depth - 8));
            int add = (ox0s + ox1s + 1) << log2Wd;
            h265_mc_combine_bi_avx2(dst8, dststride, first, second, MAX_PB,
                                    width, height, log2Wd + 1, add, bit_depth,
                                    1, wx0, wx1);
        }
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        if (!weighted) {
            h265_mc_combine_bi_neon(dst8, dststride, first, second, MAX_PB,
                                    width, height, shift, offset, bit_depth,
                                    0, 0, 0);
        } else {
            int ox0s = ox0 * (1 << (bit_depth - 8));
            int ox1s = ox1 * (1 << (bit_depth - 8));
            int add = (ox0s + ox1s + 1) << log2Wd;
            h265_mc_combine_bi_neon(dst8, dststride, first, second, MAX_PB,
                                    width, height, log2Wd + 1, add, bit_depth,
                                    1, wx0, wx1);
        }
        return;
    }
#endif

    if (!weighted) {
        if (bit_depth <= 8) {
            for (y = 0; y < height; y++) {
                uint8_t *dst = dst8;
                for (x = 0; x < width; x++)
                    dst[x] = (uint8_t)clip_pixel((first[x] + second[x] + offset) >> shift,
                                                 bit_depth);
                dst8 += dststride;
                first += MAX_PB;
                second += MAX_PB;
            }
        } else {
            ptrdiff_t st = dststride >> 1;
            for (y = 0; y < height; y++) {
                uint16_t *dst = (uint16_t *)dst8;
                for (x = 0; x < width; x++)
                    dst[x] = (uint16_t)clip_pixel((first[x] + second[x] + offset) >> shift,
                                                  bit_depth);
                dst8 += st * 2;
                first += MAX_PB;
                second += MAX_PB;
            }
        }
    } else {
        ox0 = ox0 * (1 << (bit_depth - 8));
        ox1 = ox1 * (1 << (bit_depth - 8));
        if (bit_depth <= 8) {
            for (y = 0; y < height; y++) {
                uint8_t *dst = dst8;
                for (x = 0; x < width; x++)
                    dst[x] = (uint8_t)clip_pixel((second[x] * wx1 + first[x] * wx0 +
                                                   ((ox0 + ox1 + 1) << log2Wd)) >>
                                                  (log2Wd + 1), bit_depth);
                dst8 += dststride;
                first += MAX_PB;
                second += MAX_PB;
            }
        } else {
            ptrdiff_t st = dststride >> 1;
            for (y = 0; y < height; y++) {
                uint16_t *dst = (uint16_t *)dst8;
                for (x = 0; x < width; x++)
                    dst[x] = (uint16_t)clip_pixel((second[x] * wx1 + first[x] * wx0 +
                                                   ((ox0 + ox1 + 1) << log2Wd)) >>
                                                  (log2Wd + 1), bit_depth);
                dst8 += st * 2;
                first += MAX_PB;
                second += MAX_PB;
            }
        }
    }
}

void h265_mc_luma_bi(H265Slice *sh, uint8_t *dst, ptrdiff_t dststride,
                     const H265Frame *ref0, const H265Mv *mv0, int x_off,
                     int y_off, int block_w, int block_h,
                     const H265Frame *ref1, const H265Mv *mv1,
                     H265MvField *current_mv)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    ptrdiff_t src0stride = ref0->linesize[0];
    ptrdiff_t src1stride = ref1->linesize[0];
    int pic_width = sps->width;
    int pic_height = sps->height;
    int mx0 = mv0->x & 3, my0 = mv0->y & 3;
    int mx1 = mv1->x & 3, my1 = mv1->y & 3;
    int weight_flag = (sh->slice_type == H265_SLICE_P && pps->weighted_pred_flag) ||
                      (sh->slice_type == H265_SLICE_B && pps->weighted_bipred_flag);
    int x_off0 = x_off + (mv0->x >> 2);
    int y_off0 = y_off + (mv0->y >> 2);
    int x_off1 = x_off + (mv1->x >> 2);
    int y_off1 = y_off + (mv1->y >> 2);
    const uint8_t *src0 = ref0->data[0] + y_off0 * src0stride +
                          (x_off0 << sps->pixel_shift);
    const uint8_t *src1;
    int16_t *tmp = sh->tmp;

    if (!weight_flag && !mx0 && !my0 && !mx1 && !my1 &&
        x_off0 >= 0 && y_off0 >= 0 && x_off1 >= 0 && y_off1 >= 0 &&
        x_off0 + block_w <= pic_width && y_off0 + block_h <= pic_height &&
        x_off1 + block_w <= pic_width && y_off1 + block_h <= pic_height) {
        const uint8_t *s1 = ref1->data[0] + y_off1 * src1stride +
                            (x_off1 << sps->pixel_shift);
        avg_pixels_dsp(dst, dststride, src0, src0stride, s1, src1stride,
                       block_w, block_h, sps->pixel_shift);
        return;
    }

    if (x_off0 < H265_QPEL_EXTRA_BEFORE || y_off0 < H265_QPEL_EXTRA_AFTER ||
        x_off0 >= pic_width - block_w - H265_QPEL_EXTRA_AFTER ||
        y_off0 >= pic_height - block_h - H265_QPEL_EXTRA_AFTER) {
        ptrdiff_t edge_emu_stride = H265_EDGE_EMU_BUFFER_STRIDE << sps->pixel_shift;
        int offset = H265_QPEL_EXTRA_BEFORE * src0stride +
                     (H265_QPEL_EXTRA_BEFORE << sps->pixel_shift);
        int buf_offset = H265_QPEL_EXTRA_BEFORE * edge_emu_stride +
                         (H265_QPEL_EXTRA_BEFORE << sps->pixel_shift);
        h265_emulated_edge_mc(sh->edge_emu_buffer, src0 - offset,
                              edge_emu_stride, src0stride,
                              block_w + H265_QPEL_EXTRA,
                              block_h + H265_QPEL_EXTRA,
                              x_off0 - H265_QPEL_EXTRA_BEFORE,
                              y_off0 - H265_QPEL_EXTRA_BEFORE,
                              pic_width, pic_height, sps->pixel_shift);
        src0 = sh->edge_emu_buffer + buf_offset;
        src0stride = edge_emu_stride;
    }

    src1 = ref1->data[0] + y_off1 * src1stride + (x_off1 << sps->pixel_shift);
    if (x_off1 < H265_QPEL_EXTRA_BEFORE || y_off1 < H265_QPEL_EXTRA_AFTER ||
        x_off1 >= pic_width - block_w - H265_QPEL_EXTRA_AFTER ||
        y_off1 >= pic_height - block_h - H265_QPEL_EXTRA_AFTER) {
        ptrdiff_t edge_emu_stride = H265_EDGE_EMU_BUFFER_STRIDE << sps->pixel_shift;
        int offset = H265_QPEL_EXTRA_BEFORE * src1stride +
                     (H265_QPEL_EXTRA_BEFORE << sps->pixel_shift);
        int buf_offset = H265_QPEL_EXTRA_BEFORE * edge_emu_stride +
                         (H265_QPEL_EXTRA_BEFORE << sps->pixel_shift);
        h265_emulated_edge_mc(sh->edge_emu_buffer2, src1 - offset,
                              edge_emu_stride, src1stride,
                              block_w + H265_QPEL_EXTRA,
                              block_h + H265_QPEL_EXTRA,
                              x_off1 - H265_QPEL_EXTRA_BEFORE,
                              y_off1 - H265_QPEL_EXTRA_BEFORE,
                              pic_width, pic_height, sps->pixel_shift);
        src1 = sh->edge_emu_buffer2 + buf_offset;
        src1stride = edge_emu_stride;
    }

    /* first reference */
    if (!my0) {
        if (!mx0)
            put_pixels(tmp, src0, src0stride, block_h, block_w, sps->bit_depth);
        else
            put_luma_h(tmp, src0, src0stride, block_h, block_w,
                       h265_qpel_filters[mx0], sps->bit_depth);
    } else if (!mx0) {
        put_luma_v(tmp, src0, src0stride, block_h, block_w,
                   h265_qpel_filters[my0], sps->bit_depth);
    } else {
        put_luma_hv(tmp, src0, src0stride, block_h, block_w,
                    h265_qpel_filters[mx0], h265_qpel_filters[my0],
                    sps->bit_depth);
    }

    /* second reference into a local temp and combine */
    {
        int16_t tmp2_array[MAX_PB * MAX_PB];
        int16_t *tmp2 = tmp2_array;
        if (!my1) {
            if (!mx1)
                put_pixels(tmp2, src1, src1stride, block_h, block_w, sps->bit_depth);
            else
                put_luma_h(tmp2, src1, src1stride, block_h, block_w,
                           h265_qpel_filters[mx1], sps->bit_depth);
        } else if (!mx1) {
            put_luma_v(tmp2, src1, src1stride, block_h, block_w,
                       h265_qpel_filters[my1], sps->bit_depth);
        } else {
            put_luma_hv(tmp2, src1, src1stride, block_h, block_w,
                        h265_qpel_filters[mx1], h265_qpel_filters[my1],
                        sps->bit_depth);
        }
        if (!weight_flag) {
            combine_bi(dst, dststride, tmp, tmp2, block_h, block_w, sps->bit_depth,
                     0, 0, 0, 0, 0, 0);
        } else {
            combine_bi(dst, dststride, tmp, tmp2, block_h, block_w, sps->bit_depth,
                     1, sh->luma_log2_weight_denom,
                     sh->luma_weight_l0[current_mv->ref_idx[0]],
                     sh->luma_weight_l1[current_mv->ref_idx[1]],
                     sh->luma_offset_l0[current_mv->ref_idx[0]],
                     sh->luma_offset_l1[current_mv->ref_idx[1]]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* chroma                                                              */
/* ------------------------------------------------------------------ */
static void chroma_uni_write(uint8_t *dst8, ptrdiff_t dststride,
                             const int16_t *src2, int height, int width,
                             int shift, int add, int bit_depth, int wx, int ox)
{
    int x, y;

#ifdef H265_X86
    if (h265_have_avx2) {
        h265_mc_uni_write_avx2(dst8, dststride, src2, MAX_PB, width, height,
                               shift, add, bit_depth, wx, ox);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_mc_uni_write_neon(dst8, dststride, src2, MAX_PB, width, height,
                               shift, add, bit_depth, wx, ox);
        return;
    }
#endif

    if (bit_depth <= 8) {
        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            for (x = 0; x < width; x++) {
                int v = src2[x];
                if (wx)
                    v = ((v * wx + add) >> shift) + ox;
                else
                    v = (v + add) >> shift;
                dst[x] = (uint8_t)clip_pixel(v, bit_depth);
            }
            dst8 += dststride;
            src2 += MAX_PB;
        }
    } else {
        ptrdiff_t st = dststride >> 1;
        for (y = 0; y < height; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            for (x = 0; x < width; x++) {
                int v = src2[x];
                if (wx)
                    v = ((v * wx + add) >> shift) + ox;
                else
                    v = (v + add) >> shift;
                dst[x] = (uint16_t)clip_pixel(v, bit_depth);
            }
            dst8 += st * 2;
            src2 += MAX_PB;
        }
    }
}

void h265_mc_chroma_uni(H265Slice *sh, uint8_t *dst0, ptrdiff_t dststride,
                        const uint8_t *src0, ptrdiff_t srcstride, int reflist,
                        int x_off, int y_off, int block_w, int block_h,
                        const H265MvField *current_mv, int chroma_weight,
                        int chroma_offset)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    int pic_width = sps->width >> sps->hshift[1];
    int pic_height = sps->height >> sps->vshift[1];
    const H265Mv *mv = &current_mv->mv[reflist];
    int weight_flag = (sh->slice_type == H265_SLICE_P && pps->weighted_pred_flag) ||
                      (sh->slice_type == H265_SLICE_B && pps->weighted_bipred_flag);
    int hshift = sps->hshift[1];
    int vshift = sps->vshift[1];
    intptr_t mx = mv->x & ((1 << (2 + hshift)) - 1);
    intptr_t my = mv->y & ((1 << (2 + vshift)) - 1);
    intptr_t _mx = mx << (1 - hshift);
    intptr_t _my = my << (1 - vshift);
    int emu = src0 == sh->frame->data[1] || src0 == sh->frame->data[2];
    int16_t *tmp = sh->tmp;

    x_off += mv->x >> (2 + hshift);
    y_off += mv->y >> (2 + vshift);
    src0 += y_off * srcstride + (x_off << sps->pixel_shift);

    if (!weight_flag && !_mx && !_my && !emu && x_off >= 0 && y_off >= 0 &&
        x_off + block_w <= pic_width && y_off + block_h <= pic_height) {
        copy_pixels_dsp(dst0, dststride, src0, srcstride, block_w, block_h,
                        sps->pixel_shift);
        return;
    }

    if (x_off < H265_EPEL_EXTRA_BEFORE || y_off < H265_EPEL_EXTRA_AFTER ||
        x_off >= pic_width - block_w - H265_EPEL_EXTRA_AFTER ||
        y_off >= pic_height - block_h - H265_EPEL_EXTRA_AFTER || emu) {
        int edge_emu_stride = H265_EDGE_EMU_BUFFER_STRIDE << sps->pixel_shift;
        int offset0 = H265_EPEL_EXTRA_BEFORE * (srcstride + (1 << sps->pixel_shift));
        int buf_offset0 = H265_EPEL_EXTRA_BEFORE *
                          (edge_emu_stride + (1 << sps->pixel_shift));
        h265_emulated_edge_mc(sh->edge_emu_buffer, src0 - offset0,
                              edge_emu_stride, srcstride,
                              block_w + H265_EPEL_EXTRA, block_h + H265_EPEL_EXTRA,
                              x_off - H265_EPEL_EXTRA_BEFORE,
                              y_off - H265_EPEL_EXTRA_BEFORE,
                              pic_width, pic_height, sps->pixel_shift);
        src0 = sh->edge_emu_buffer + buf_offset0;
        srcstride = edge_emu_stride;
    }

    if (!weight_flag && !_mx && !_my) {
        copy_pixels_dsp(dst0, dststride, src0, srcstride, block_w, block_h,
                        sps->pixel_shift);
        return;
    }

#ifdef H265_X86
    if (weight_flag && !_mx && !_my && h265_have_avx2) {
        if (chroma_weight)
            h265_mc_weighted_int_avx2(dst0, dststride, src0, srcstride,
                                      block_w, block_h, chroma_weight,
                                      chroma_offset << (sps->bit_depth - 8),
                                      sh->chroma_log2_weight_denom,
                                      sps->bit_depth, sps->pixel_shift);
        else
            copy_pixels_dsp(dst0, dststride, src0, srcstride, block_w, block_h,
                            sps->pixel_shift);
        return;
    }
#endif
#ifdef H265_NEON
    if (weight_flag && !_mx && !_my && h265_have_neon) {
        if (chroma_weight)
            h265_mc_weighted_int_neon(dst0, dststride, src0, srcstride,
                                      block_w, block_h, chroma_weight,
                                      chroma_offset << (sps->bit_depth - 8),
                                      sh->chroma_log2_weight_denom,
                                      sps->bit_depth, sps->pixel_shift);
        else
            copy_pixels_dsp(dst0, dststride, src0, srcstride, block_w, block_h,
                            sps->pixel_shift);
        return;
    }
#endif

    if (!_my) {
        if (!_mx)
            put_pixels(tmp, src0, srcstride, block_h, block_w, sps->bit_depth);
        else
            put_chroma_h(tmp, src0, srcstride, block_h, block_w,
                         h265_epel_filters[_mx], sps->bit_depth);
    } else if (!_mx) {
        put_chroma_v(tmp, src0, srcstride, block_h, block_w,
                     h265_epel_filters[_my], sps->bit_depth);
    } else {
        put_chroma_hv(tmp, src0, srcstride, block_h, block_w,
                      h265_epel_filters[_mx], h265_epel_filters[_my],
                      sps->bit_depth);
    }

    if (!weight_flag) {
        int shift = 14 - sps->bit_depth;
        int add = 1 << (shift - 1);
        chroma_uni_write(dst0, dststride, tmp, block_h, block_w, shift, add,
                         sps->bit_depth, 0, 0);
    } else {
        int shift = sh->chroma_log2_weight_denom + 14 - sps->bit_depth;
        int add = 1 << (shift - 1);
        int ox = chroma_offset << (sps->bit_depth - 8);
        chroma_uni_write(dst0, dststride, tmp, block_h, block_w, shift, add,
                         sps->bit_depth, chroma_weight, ox);
    }
}

void h265_mc_chroma_bi(H265Slice *sh, uint8_t *dst0, ptrdiff_t dststride,
                       const H265Frame *ref0, const H265Frame *ref1,
                       int x_off, int y_off, int block_w, int block_h,
                       const H265MvField *current_mv, int cidx)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    const uint8_t *src1 = ref0->data[cidx + 1];
    const uint8_t *src2 = ref1->data[cidx + 1];
    ptrdiff_t src1stride = ref0->linesize[cidx + 1];
    ptrdiff_t src2stride = ref1->linesize[cidx + 1];
    int weight_flag = (sh->slice_type == H265_SLICE_P && pps->weighted_pred_flag) ||
                      (sh->slice_type == H265_SLICE_B && pps->weighted_bipred_flag);
    int pic_width = sps->width >> sps->hshift[1];
    int pic_height = sps->height >> sps->vshift[1];
    const H265Mv *mv0 = &current_mv->mv[0];
    const H265Mv *mv1 = &current_mv->mv[1];
    int hshift = sps->hshift[1];
    int vshift = sps->vshift[1];
    intptr_t mx0 = mv0->x & ((1 << (2 + hshift)) - 1);
    intptr_t my0 = mv0->y & ((1 << (2 + vshift)) - 1);
    intptr_t mx1 = mv1->x & ((1 << (2 + hshift)) - 1);
    intptr_t my1 = mv1->y & ((1 << (2 + vshift)) - 1);
    intptr_t _mx0 = mx0 << (1 - hshift);
    intptr_t _my0 = my0 << (1 - vshift);
    intptr_t _mx1 = mx1 << (1 - hshift);
    intptr_t _my1 = my1 << (1 - vshift);
    int x_off0 = x_off + (mv0->x >> (2 + hshift));
    int y_off0 = y_off + (mv0->y >> (2 + vshift));
    int x_off1 = x_off + (mv1->x >> (2 + hshift));
    int y_off1 = y_off + (mv1->y >> (2 + vshift));
    int16_t *tmp = sh->tmp;
    int16_t tmp2_array[MAX_PB * MAX_PB];
    int16_t *tmp2;

    src1 += y_off0 * src1stride + (x_off0 << sps->pixel_shift);
    src2 += y_off1 * src2stride + (x_off1 << sps->pixel_shift);

    if (!weight_flag && !_mx0 && !_my0 && !_mx1 && !_my1 &&
        x_off0 >= 0 && y_off0 >= 0 && x_off1 >= 0 && y_off1 >= 0 &&
        x_off0 + block_w <= pic_width && y_off0 + block_h <= pic_height &&
        x_off1 + block_w <= pic_width && y_off1 + block_h <= pic_height) {
        avg_pixels_dsp(dst0, dststride, src1, src1stride, src2, src2stride,
                       block_w, block_h, sps->pixel_shift);
        return;
    }

    if (x_off0 < H265_EPEL_EXTRA_BEFORE || y_off0 < H265_EPEL_EXTRA_AFTER ||
        x_off0 >= pic_width - block_w - H265_EPEL_EXTRA_AFTER ||
        y_off0 >= pic_height - block_h - H265_EPEL_EXTRA_AFTER) {
        int edge_emu_stride = H265_EDGE_EMU_BUFFER_STRIDE << sps->pixel_shift;
        int offset1 = H265_EPEL_EXTRA_BEFORE * (src1stride + (1 << sps->pixel_shift));
        int buf_offset1 = H265_EPEL_EXTRA_BEFORE *
                          (edge_emu_stride + (1 << sps->pixel_shift));
        h265_emulated_edge_mc(sh->edge_emu_buffer, src1 - offset1,
                              edge_emu_stride, src1stride,
                              block_w + H265_EPEL_EXTRA, block_h + H265_EPEL_EXTRA,
                              x_off0 - H265_EPEL_EXTRA_BEFORE,
                              y_off0 - H265_EPEL_EXTRA_BEFORE,
                              pic_width, pic_height, sps->pixel_shift);
        src1 = sh->edge_emu_buffer + buf_offset1;
        src1stride = edge_emu_stride;
    }
    if (x_off1 < H265_EPEL_EXTRA_BEFORE || y_off1 < H265_EPEL_EXTRA_AFTER ||
        x_off1 >= pic_width - block_w - H265_EPEL_EXTRA_AFTER ||
        y_off1 >= pic_height - block_h - H265_EPEL_EXTRA_AFTER) {
        int edge_emu_stride = H265_EDGE_EMU_BUFFER_STRIDE << sps->pixel_shift;
        int offset1 = H265_EPEL_EXTRA_BEFORE * (src2stride + (1 << sps->pixel_shift));
        int buf_offset1 = H265_EPEL_EXTRA_BEFORE *
                          (edge_emu_stride + (1 << sps->pixel_shift));
        h265_emulated_edge_mc(sh->edge_emu_buffer2, src2 - offset1,
                              edge_emu_stride, src2stride,
                              block_w + H265_EPEL_EXTRA, block_h + H265_EPEL_EXTRA,
                              x_off1 - H265_EPEL_EXTRA_BEFORE,
                              y_off1 - H265_EPEL_EXTRA_BEFORE,
                              pic_width, pic_height, sps->pixel_shift);
        src2 = sh->edge_emu_buffer2 + buf_offset1;
        src2stride = edge_emu_stride;
    }

    if (!_my0) {
        if (!_mx0)
            put_pixels(tmp, src1, src1stride, block_h, block_w, sps->bit_depth);
        else
            put_chroma_h(tmp, src1, src1stride, block_h, block_w,
                         h265_epel_filters[_mx0], sps->bit_depth);
    } else if (!_mx0) {
        put_chroma_v(tmp, src1, src1stride, block_h, block_w,
                     h265_epel_filters[_my0], sps->bit_depth);
    } else {
        put_chroma_hv(tmp, src1, src1stride, block_h, block_w,
                      h265_epel_filters[_mx0], h265_epel_filters[_my0],
                      sps->bit_depth);
    }

    tmp2 = tmp2_array;
    if (!_my1) {
        if (!_mx1)
            put_pixels(tmp2, src2, src2stride, block_h, block_w, sps->bit_depth);
        else
            put_chroma_h(tmp2, src2, src2stride, block_h, block_w,
                         h265_epel_filters[_mx1], sps->bit_depth);
    } else if (!_mx1) {
        put_chroma_v(tmp2, src2, src2stride, block_h, block_w,
                     h265_epel_filters[_my1], sps->bit_depth);
    } else {
        put_chroma_hv(tmp2, src2, src2stride, block_h, block_w,
                      h265_epel_filters[_mx1], h265_epel_filters[_my1],
                      sps->bit_depth);
    }

    if (!weight_flag) {
        combine_bi(dst0, dststride, tmp, tmp2, block_h, block_w, sps->bit_depth,
                 0, 0, 0, 0, 0, 0);
    } else {
        combine_bi(dst0, dststride, tmp, tmp2, block_h, block_w, sps->bit_depth,
                 1, sh->chroma_log2_weight_denom,
                 sh->chroma_weight_l0[current_mv->ref_idx[0]][cidx],
                 sh->chroma_weight_l1[current_mv->ref_idx[1]][cidx],
                 sh->chroma_offset_l0[current_mv->ref_idx[0]][cidx],
                 sh->chroma_offset_l1[current_mv->ref_idx[1]][cidx]);
    }
}
