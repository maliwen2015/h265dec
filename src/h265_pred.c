/*
 * HEVC intra prediction (planar, DC, 33 angular modes) with reference
 * sample smoothing.
 *
 * Derived from FFmpeg's hevc/pred_template.c (LGPL-2.1+).
 */
#include "h265_internal.h"

#define POS(x, y) src[(x) + stride * (y)]

H265_INLINE void ref_filter_3tap(int16_t *filtered_left,
                                        int16_t *filtered_top,
                                        const int16_t *left,
                                        const int16_t *top,
                                        int size)
{
    int i;
    int n = 2 * size;

    filtered_left[n - 1] = left[n - 1];
    filtered_top[n - 1] = top[n - 1];
    for (i = n - 2; i >= 0; i--)
        filtered_left[i] = (left[i + 1] + 2 * left[i] + left[i - 1] + 2) >> 2;
    filtered_top[-1] =
    filtered_left[-1] = (left[0] + 2 * left[-1] + top[0] + 2) >> 2;
    for (i = n - 2; i >= 0; i--)
        filtered_top[i] = (top[i + 1] + 2 * top[i] + top[i - 1] + 2) >> 2;
}

static void ref_filter_strong(int16_t *filtered_top, int16_t *left,
                              const int16_t *top)
{
    int i;

    filtered_top[-1] = top[-1];
    filtered_top[63] = top[63];
    for (i = 0; i < 63; i++)
        filtered_top[i] = ((64 - (i + 1)) * top[-1] +
                           (i + 1) * top[63] + 32) >> 6;
    for (i = 0; i < 63; i++)
        left[i] = ((64 - (i + 1)) * left[-1] + (i + 1) * left[63] + 32) >> 6;
}

H265_INLINE void pred_planar(uint8_t *_src, const int16_t *top,
                                    const int16_t *left, ptrdiff_t stride,
                                    int log2_size, int bit_depth)
{
    int x, y;
    int size = 1 << log2_size;

    if (bit_depth == 8) {
        uint8_t *src = _src;
        for (y = 0; y < size; y++)
            for (x = 0; x < size; x++)
                POS(x, y) = (uint8_t)(((size - 1 - x) * left[y] + (x + 1) * top[size] +
                                       (size - 1 - y) * top[x] + (y + 1) * left[size] + size) >>
                                      (log2_size + 1));
    } else {
        for (y = 0; y < size; y++) {
            uint16_t *src = (uint16_t *)_src + y * (stride >> 1);
            for (x = 0; x < size; x++)
                src[x] = (uint16_t)(((size - 1 - x) * left[y] + (x + 1) * top[size] +
                                     (size - 1 - y) * top[x] + (y + 1) * left[size] + size) >>
                                    (log2_size + 1));
        }
    }
}

static void pred_dc(uint8_t *_src, const int16_t *top, const int16_t *left,
                    ptrdiff_t stride, int log2_size, int c_idx, int bit_depth)
{
    int i, j, x, y;
    int size = (1 << log2_size);
    int dc = size;

    for (i = 0; i < size; i++)
        dc += left[i] + top[i];
    dc >>= log2_size + 1;

    if (bit_depth == 8) {
        uint8_t *src = _src;
        for (i = 0; i < size; i++)
            for (j = 0; j < size; j++)
                POS(j, i) = (uint8_t)dc;
        if (c_idx == 0 && size < 32) {
            POS(0, 0) = (uint8_t)((left[0] + 2 * dc + top[0] + 2) >> 2);
            for (x = 1; x < size; x++)
                POS(x, 0) = (uint8_t)((top[x] + 3 * dc + 2) >> 2);
            for (y = 1; y < size; y++)
                POS(0, y) = (uint8_t)((left[y] + 3 * dc + 2) >> 2);
        }
    } else {
        for (i = 0; i < size; i++) {
            uint16_t *src = (uint16_t *)_src + i * (stride >> 1);
            for (j = 0; j < size; j++)
                src[j] = (uint16_t)dc;
        }
        if (c_idx == 0 && size < 32) {
            uint16_t *src = (uint16_t *)_src;
            ptrdiff_t st = stride >> 1;
            src[0] = (uint16_t)((left[0] + 2 * dc + top[0] + 2) >> 2);
            for (x = 1; x < size; x++)
                src[x] = (uint16_t)((top[x] + 3 * dc + 2) >> 2);
            for (y = 1; y < size; y++)
                src[y * st] = (uint16_t)((left[y] + 3 * dc + 2) >> 2);
        }
    }
}

static const int intra_pred_angle[] = {
     32,  26,  21,  17, 13,  9,  5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32,
    -26, -21, -17, -13, -9, -5, -2, 0, 2,  5,  9, 13,  17,  21,  26,  32
};
static const int inv_angle[] = {
    -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482,
    -630, -910, -1638, -4096
};

static void pred_angular(uint8_t *_src, const int16_t *top, const int16_t *left,
                         ptrdiff_t stride, int c_idx, int mode, int size,
                         int bit_depth)
{
    int x, y;
    int angle = intra_pred_angle[mode - 2];
    int16_t ref_array[3 * H265_MAX_TB_SIZE + 4];
    int16_t *ref_tmp = ref_array + size;
    const int16_t *ref;
    int last = (size * angle) >> 5;
    int maxv = (1 << bit_depth) - 1;

    if (bit_depth == 8) {
        uint8_t *src = _src;
        if (mode >= 18) {
            ref = top - 1;
            if (angle < 0 && last < -1) {
                for (x = 0; x <= size; x++)
                    ref_tmp[x] = top[x - 1];
                for (x = last; x <= -1; x++)
                    ref_tmp[x] = left[-1 + ((x * inv_angle[mode - 11] + 128) >> 8)];
                ref = ref_tmp;
            }
            for (y = 0; y < size; y++) {
                int idx = ((y + 1) * angle) >> 5;
                int fact = ((y + 1) * angle) & 31;
                if (fact) {
                    for (x = 0; x < size; x++)
                        POS(x, y) = (uint8_t)(((32 - fact) * ref[x + idx + 1] +
                                               fact * ref[x + idx + 2] + 16) >> 5);
                } else {
                    for (x = 0; x < size; x++)
                        POS(x, y) = (uint8_t)ref[x + idx + 1];
                }
            }
            if (mode == 26 && c_idx == 0 && size < 32) {
                for (y = 0; y < size; y++) {
                    int v = top[0] + ((left[y] - left[-1]) >> 1);
                    POS(0, y) = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
        } else {
            ref = left - 1;
            if (angle < 0 && last < -1) {
                for (x = 0; x <= size; x++)
                    ref_tmp[x] = left[x - 1];
                for (x = last; x <= -1; x++)
                    ref_tmp[x] = top[-1 + ((x * inv_angle[mode - 11] + 128) >> 8)];
                ref = ref_tmp;
            }
            for (x = 0; x < size; x++) {
                int idx = ((x + 1) * angle) >> 5;
                int fact = ((x + 1) * angle) & 31;
                if (fact) {
                    for (y = 0; y < size; y++)
                        POS(x, y) = (uint8_t)(((32 - fact) * ref[y + idx + 1] +
                                               fact * ref[y + idx + 2] + 16) >> 5);
                } else {
                    for (y = 0; y < size; y++)
                        POS(x, y) = (uint8_t)ref[y + idx + 1];
                }
            }
            if (mode == 10 && c_idx == 0 && size < 32) {
                for (x = 0; x < size; x++) {
                    int v = left[0] + ((top[x] - top[-1]) >> 1);
                    POS(x, 0) = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
        }
    } else {
        ptrdiff_t st = stride >> 1;
        if (mode >= 18) {
            ref = top - 1;
            if (angle < 0 && last < -1) {
                for (x = 0; x <= size; x++)
                    ref_tmp[x] = top[x - 1];
                for (x = last; x <= -1; x++)
                    ref_tmp[x] = left[-1 + ((x * inv_angle[mode - 11] + 128) >> 8)];
                ref = ref_tmp;
            }
            for (y = 0; y < size; y++) {
                uint16_t *src = (uint16_t *)_src + y * st;
                int idx = ((y + 1) * angle) >> 5;
                int fact = ((y + 1) * angle) & 31;
                if (fact) {
                    for (x = 0; x < size; x++)
                        src[x] = (uint16_t)(((32 - fact) * ref[x + idx + 1] +
                                             fact * ref[x + idx + 2] + 16) >> 5);
                } else {
                    for (x = 0; x < size; x++)
                        src[x] = (uint16_t)ref[x + idx + 1];
                }
            }
            if (mode == 26 && c_idx == 0 && size < 32) {
                for (y = 0; y < size; y++) {
                    uint16_t *src = (uint16_t *)_src + y * st;
                    int v = top[0] + ((left[y] - left[-1]) >> 1);
                    src[0] = (uint16_t)(v < 0 ? 0 : v > maxv ? maxv : v);
                }
            }
        } else {
            ref = left - 1;
            if (angle < 0 && last < -1) {
                for (x = 0; x <= size; x++)
                    ref_tmp[x] = left[x - 1];
                for (x = last; x <= -1; x++)
                    ref_tmp[x] = top[-1 + ((x * inv_angle[mode - 11] + 128) >> 8)];
                ref = ref_tmp;
            }
            for (x = 0; x < size; x++) {
                int idx = ((x + 1) * angle) >> 5;
                int fact = ((x + 1) * angle) & 31;
                if (fact) {
                    for (y = 0; y < size; y++) {
                        uint16_t *src = (uint16_t *)_src + y * st;
                        src[x] = (uint16_t)(((32 - fact) * ref[y + idx + 1] +
                                             fact * ref[y + idx + 2] + 16) >> 5);
                    }
                } else {
                    for (y = 0; y < size; y++) {
                        uint16_t *src = (uint16_t *)_src + y * st;
                        src[x] = (uint16_t)ref[y + idx + 1];
                    }
                }
            }
            if (mode == 10 && c_idx == 0 && size < 32) {
                uint16_t *src = (uint16_t *)_src;
                for (x = 0; x < size; x++) {
                    int v = left[0] + ((top[x] - top[-1]) >> 1);
                    src[x] = (uint16_t)(v < 0 ? 0 : v > maxv ? maxv : v);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* reference sample preparation                                        */
/* ------------------------------------------------------------------ */
void h265_intra_pred(H265Slice *sh, int x0, int y0, int log2_size, int c_idx)
{
    H265Frame *f = sh->frame;
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    int i;
    int hshift = sps->hshift[c_idx];
    int vshift = sps->vshift[c_idx];
    int size = 1 << log2_size;
    int size_in_luma_h = size << hshift;
    int size_in_luma_v = size << vshift;
    int size_in_tbs_h = size_in_luma_h >> sps->log2_min_tb_size;
    int size_in_tbs_v = size_in_luma_v >> sps->log2_min_tb_size;
    int x = x0 >> hshift;
    int y = y0 >> vshift;
    int x_tb = (x0 >> sps->log2_min_tb_size) & sps->tb_mask;
    int y_tb = (y0 >> sps->log2_min_tb_size) & sps->tb_mask;
    int spin = c_idx && !size_in_tbs_v && ((2 * y0) & (1 << sps->log2_min_tb_size));
    int min_pu_width = sps->min_pu_width;
    int min_cb_width = sps->min_cb_width;
    int cur_tb_addr;
    ptrdiff_t stride = f->linesize[c_idx];
    uint8_t *src = f->data[c_idx] + (y * stride) + (x << sps->pixel_shift);
    int mode = c_idx ? sh->tu.intra_pred_mode_c : sh->tu.intra_pred_mode;
    int16_t left_array[2 * H265_MAX_TB_SIZE + 1 + 16];
    int16_t filtered_left_array[2 * H265_MAX_TB_SIZE + 1];
    int16_t top_array[2 * H265_MAX_TB_SIZE + 1 + 16];
    int16_t filtered_top_array[2 * H265_MAX_TB_SIZE + 1];
    int16_t *left = left_array + 1;
    int16_t *top = top_array + 1;
    int16_t *filtered_left = filtered_left_array + 1;
    int16_t *filtered_top = filtered_top_array + 1;
    int maxv = (1 << sps->bit_depth) - 1;
    int mid = 1 << (sps->bit_depth - 1);
    int cand_bottom_left, cand_left, cand_up_left, cand_up, cand_up_right;
    int bottom_left_size, top_right_size;

#define MVF(x, y) (f->tab_mvf[(x) + (y) * min_pu_width])
#define PU(x) ((x) >> sps->log2_min_pu_size)
#define MVF_PU(x, y) \
    MVF(PU(x0 + ((x) * (1 << hshift))), PU(y0 + ((y) * (1 << vshift))))
#define IS_INTRA(x, y) (MVF_PU(x, y).pred_flag == H265_PF_INTRA)
#define MIN_TB_ADDR_ZS(x, y) \
    pps->min_tb_addr_zs[(y) * (sps->tb_mask + 2) + (x)]
#define EXTEND(ptr, val, len)                 \
    do {                                      \
        int _i;                               \
        for (_i = 0; _i < (len); _i++)        \
            (ptr)[_i] = (int16_t)(val);       \
    } while (0)

    cur_tb_addr = MIN_TB_ADDR_ZS(x_tb, y_tb);
    cand_bottom_left = sh->na.cand_bottom_left &&
                       cur_tb_addr > MIN_TB_ADDR_ZS(x_tb - 1, (y_tb + size_in_tbs_v + spin) & sps->tb_mask);
    cand_left = sh->na.cand_left;
    cand_up_left = sh->na.cand_up_left;
    cand_up = sh->na.cand_up;
    cand_up_right = sh->na.cand_up_right && !spin &&
                    cur_tb_addr > MIN_TB_ADDR_ZS((x_tb + size_in_tbs_h) & sps->tb_mask, y_tb - 1);

    bottom_left_size = (H265_MIN(y0 + 2 * size_in_luma_v, sps->height) -
                        (y0 + size_in_luma_v)) >> vshift;
    top_right_size = (H265_MIN(x0 + 2 * size_in_luma_h, sps->width) -
                      (x0 + size_in_luma_h)) >> hshift;

    if (pps->constrained_intra_pred_flag == 1) {
        int size_in_luma_pu_v = PU(size_in_luma_v);
        int size_in_luma_pu_h = PU(size_in_luma_h);
        int on_pu_edge_x = !(x0 & ((1 << sps->log2_min_pu_size) - 1));
        int on_pu_edge_y = !(y0 & ((1 << sps->log2_min_pu_size) - 1));
        if (!size_in_luma_pu_h)
            size_in_luma_pu_h++;
        if (cand_bottom_left == 1 && on_pu_edge_x) {
            int x_left_pu = PU(x0 - 1);
            int y_bottom_pu = PU(y0 + size_in_luma_v);
            int max = H265_MIN(size_in_luma_pu_v, sps->min_pu_height - y_bottom_pu);
            cand_bottom_left = 0;
            for (i = 0; i < max; i += 2)
                cand_bottom_left |= (MVF(x_left_pu, y_bottom_pu + i).pred_flag == H265_PF_INTRA);
        }
        if (cand_left == 1 && on_pu_edge_x) {
            int x_left_pu = PU(x0 - 1);
            int y_left_pu = PU(y0);
            int max = H265_MIN(size_in_luma_pu_v, sps->min_pu_height - y_left_pu);
            cand_left = 0;
            for (i = 0; i < max; i += 2)
                cand_left |= (MVF(x_left_pu, y_left_pu + i).pred_flag == H265_PF_INTRA);
        }
        if (cand_up_left == 1) {
            int x_left_pu = PU(x0 - 1);
            int y_top_pu = PU(y0 - 1);
            cand_up_left = MVF(x_left_pu, y_top_pu).pred_flag == H265_PF_INTRA;
        }
        if (cand_up == 1 && on_pu_edge_y) {
            int x_top_pu = PU(x0);
            int y_top_pu = PU(y0 - 1);
            int max = H265_MIN(size_in_luma_pu_h, sps->min_pu_width - x_top_pu);
            cand_up = 0;
            for (i = 0; i < max; i += 2)
                cand_up |= (MVF(x_top_pu + i, y_top_pu).pred_flag == H265_PF_INTRA);
        }
        if (cand_up_right == 1 && on_pu_edge_y) {
            int y_top_pu = PU(y0 - 1);
            int x_right_pu = PU(x0 + size_in_luma_h);
            int max = H265_MIN(size_in_luma_pu_h, sps->min_pu_width - x_right_pu);
            cand_up_right = 0;
            for (i = 0; i < max; i += 2)
                cand_up_right |= (MVF(x_right_pu + i, y_top_pu).pred_flag == H265_PF_INTRA);
        }
        memset(left, 128, 2 * H265_MAX_TB_SIZE * sizeof(int16_t));
        memset(top, 128, 2 * H265_MAX_TB_SIZE * sizeof(int16_t));
        top[-1] = 128;
    }

    if (cand_up_left) {
        left[-1] = sps->pixel_shift ?
            ((uint16_t *)src)[-1 - (stride >> 1)] : src[-1 - stride];
        top[-1] = left[-1];
    }
    if (cand_up) {
        for (i = 0; i < size; i++)
            top[i] = sps->pixel_shift ?
                     ((const uint16_t *)(src - stride))[i] : src[i - stride];
    }
    if (cand_up_right) {
        for (i = 0; i < size; i++)
            top[size + i] = sps->pixel_shift ?
                ((const uint16_t *)(src - stride))[size + i] : src[size + i - stride];
        {
            int val = sps->pixel_shift ?
                ((const uint16_t *)(src - stride))[size + top_right_size - 1] :
                src[size + top_right_size - 1 - stride];
            for (i = size + top_right_size; i < 2 * size; i++)
                top[i] = (int16_t)val;
        }
    }
    if (cand_left) {
        for (i = 0; i < size; i++)
            left[i] = sps->pixel_shift ?
                      ((const uint16_t *)(src + i * stride))[-1] : src[i * stride - 1];
    }
    if (cand_bottom_left) {
        for (i = size; i < size + bottom_left_size; i++)
            left[i] = sps->pixel_shift ?
                      ((const uint16_t *)(src + i * stride))[-1] : src[i * stride - 1];
        {
            int val = sps->pixel_shift ?
                ((const uint16_t *)(src + (size + bottom_left_size - 1) * stride))[-1] :
                src[(size + bottom_left_size - 1) * stride - 1];
            for (i = size + bottom_left_size; i < 2 * size; i++)
                left[i] = (int16_t)val;
        }
    }

    if (pps->constrained_intra_pred_flag == 1) {
        if (cand_bottom_left || cand_left || cand_up_left || cand_up || cand_up_right) {
            int size_max_x = x0 + ((2 * size) << hshift) < sps->width ?
                             2 * size : (sps->width - x0) >> hshift;
            int size_max_y = y0 + ((2 * size) << vshift) < sps->height ?
                             2 * size : (sps->height - y0) >> vshift;
            int j = size + (cand_bottom_left ? bottom_left_size : 0) - 1;
            if (!cand_up_right)
                size_max_x = x0 + ((size) << hshift) < sps->width ?
                             size : (sps->width - x0) >> hshift;
            if (!cand_bottom_left)
                size_max_y = y0 + ((size) << vshift) < sps->height ?
                             size : (sps->height - y0) >> vshift;
            if (cand_bottom_left || cand_left || cand_up_left) {
                while (j > -1 && !IS_INTRA(-1, j))
                    j--;
                if (!IS_INTRA(-1, j)) {
                    j = 0;
                    while (j < size_max_x && !IS_INTRA(j, -1))
                        j++;
                    for (i = j; i >= 0; i--)
                        top[i - 1] = top[i];
                    left[-1] = top[-1];
                }
            } else {
                j = 0;
                while (j < size_max_x && !IS_INTRA(j, -1))
                    j++;
                if (j > 0) {
                    for (i = j; i > 0; i--)
                        top[i - 1] = top[i];
                    top[-1] = top[0];
                }
                left[-1] = top[-1];
            }
            left[-1] = top[-1];
            if (cand_bottom_left || cand_left) {
                int16_t a = left[-1];
                for (i = 0; i < size_max_y; i++)
                    if (!IS_INTRA(-1, i))
                        left[i] = a;
                    else
                        a = left[i];
            }
            if (!cand_left)
                EXTEND(left, left[-1], size);
            if (!cand_bottom_left)
                EXTEND(left + size, left[size - 1], size);
            if (x0 != 0 && y0 != 0) {
                int16_t a = left[size_max_y - 1];
                for (i = size_max_y - 1; i >= 0; i--)
                    if (!IS_INTRA(-1, i - 1))
                        left[i - 1] = a;
                    else
                        a = left[i - 1];
                if (!IS_INTRA(-1, -1))
                    left[-1] = left[0];
            } else if (x0 == 0) {
                EXTEND(left, 0, size_max_y);
            } else {
                int16_t a = left[size_max_y - 1];
                for (i = size_max_y - 1; i >= 0; i--)
                    if (!IS_INTRA(-1, i - 1))
                        left[i - 1] = a;
                    else
                        a = left[i - 1];
            }
            top[-1] = left[-1];
            if (y0 != 0) {
                int16_t a = left[-1];
                for (i = 0; i < size_max_x; i++)
                    if (!IS_INTRA(i, -1))
                        top[i] = a;
                    else
                        a = top[i];
            }
        }
    }

    if (!cand_bottom_left) {
        if (cand_left) {
            EXTEND(left + size, left[size - 1], size);
        } else if (cand_up_left) {
            EXTEND(left, left[-1], 2 * size);
            cand_left = 1;
        } else if (cand_up) {
            left[-1] = top[0];
            EXTEND(left, left[-1], 2 * size);
            cand_up_left = 1;
            cand_left = 1;
        } else if (cand_up_right) {
            EXTEND(top, top[size], size);
            left[-1] = top[size];
            EXTEND(left, left[-1], 2 * size);
            cand_up = 1;
            cand_up_left = 1;
            cand_left = 1;
        } else {
            left[-1] = mid;
            EXTEND(top, left[-1], 2 * size);
            EXTEND(left, left[-1], 2 * size);
        }
    }
    if (!cand_left)
        EXTEND(left, left[size], size);
    if (!cand_up_left)
        left[-1] = left[0];
    if (!cand_up)
        EXTEND(top, left[-1], size);
    if (!cand_up_right)
        EXTEND(top + size, top[size - 1], size);
    top[-1] = left[-1];

    if (!sps->intra_smoothing_disabled && (c_idx == 0 || sps->chroma_format_idc == 3)) {
        if (mode != H265_INTRA_DC && size != 4) {
            int intra_hor_ver_dist_thresh[] = { 7, 1, 0 };
            int min_dist_vert_hor = H265_MIN(H265_ABS((int)(mode - 26U)),
                                             H265_ABS((int)(mode - 10U)));
            if (min_dist_vert_hor > intra_hor_ver_dist_thresh[log2_size - 3]) {
                int threshold = 1 << (sps->bit_depth - 5);
                if (sps->strong_intra_smoothing_enabled && c_idx == 0 &&
                    log2_size == 5 &&
                    H265_ABS(top[-1] + top[63] - 2 * top[31]) < threshold &&
                    H265_ABS(left[-1] + left[63] - 2 * left[31]) < threshold) {
                    ref_filter_strong(filtered_top, left, top);
                    top = filtered_top;
                } else {
                    ref_filter_3tap(filtered_left, filtered_top, left, top, size);
                    left = filtered_left;
                    top = filtered_top;
                }
            }
        }
    }

    switch (mode) {
    case H265_INTRA_PLANAR:
        pred_planar(src, top, left, stride, log2_size, sps->bit_depth);
        break;
    case H265_INTRA_DC:
        pred_dc(src, top, left, stride, log2_size, c_idx, sps->bit_depth);
        break;
    default:
        pred_angular(src, top, left, stride, c_idx, mode, size, sps->bit_depth);
        break;
    }
    (void)min_cb_width;
    (void)maxv;
#undef MVF
#undef PU
#undef MVF_PU
#undef IS_INTRA
#undef MIN_TB_ADDR_ZS
#undef EXTEND
}
