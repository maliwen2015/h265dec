/*
 * HEVC sample adaptive offset filter.
 *
 * Derived from FFmpeg's hevc/filter.c and h26x/h2656_sao_template.c
 * (LGPL-2.1+).
 */
#include "h265_internal.h"

#define SAO_STRIDE (2 * H265_MAX_PB_SIZE + 64)

static void copy_CTB(uint8_t *dst, const uint8_t *src, int width_bytes,
                     int height, ptrdiff_t stride_dst, ptrdiff_t stride_src)
{
    int y;
    for (y = 0; y < height; y++) {
        memcpy(dst, src, width_bytes);
        dst += stride_dst;
        src += stride_src;
    }
}

static void copy_pixel(uint8_t *dst, const uint8_t *src, int pixel_shift)
{
    if (pixel_shift)
        *(uint16_t *)dst = *(uint16_t *)src;
    else
        *dst = *src;
}

static void copy_vert(uint8_t *dst, const uint8_t *src, int pixel_shift,
                      int height, ptrdiff_t stride_dst, ptrdiff_t stride_src)
{
    int i;
    if (!pixel_shift) {
        for (i = 0; i < height; i++) {
            *dst = *src;
            dst += stride_dst;
            src += stride_src;
        }
    } else {
        for (i = 0; i < height; i++) {
            *(uint16_t *)dst = *(uint16_t *)src;
            dst += stride_dst;
            src += stride_src;
        }
    }
}

static void copy_CTB_to_hv(const H265Frame *f, const H265SPS *sps,
                           const uint8_t *src, ptrdiff_t stride_src,
                           int x, int y, int width, int height, int c_idx,
                           int x_ctb, int y_ctb)
{
    int sh = sps->pixel_shift;
    int w = sps->width >> sps->hshift[c_idx];
    int h = sps->height >> sps->vshift[c_idx];

    memcpy(f->sao_pixel_buffer_h[c_idx] + (((2 * y_ctb) * w + x) << sh),
           src, width << sh);
    memcpy(f->sao_pixel_buffer_h[c_idx] + (((2 * y_ctb + 1) * w + x) << sh),
           src + stride_src * (height - 1), width << sh);

    copy_vert(f->sao_pixel_buffer_v[c_idx] + (((2 * x_ctb) * h + y) << sh),
              src, sh, height, 1 << sh, stride_src);
    copy_vert(f->sao_pixel_buffer_v[c_idx] + (((2 * x_ctb + 1) * h + y) << sh),
              src + ((width - 1) << sh), sh, height, 1 << sh, stride_src);
}

static void restore_tqb_pixels2(const H265Frame *f, const H265PPS *pps,
                                const H265SPS *sps, uint8_t *src1,
                                const uint8_t *dst1, ptrdiff_t stride_src,
                                ptrdiff_t stride_dst, int x0, int y0,
                                int width, int height, int c_idx)
{
    if (pps->transquant_bypass_enable_flag ||
        (sps->pcm_loop_filter_disabled && sps->pcm_enabled)) {
        int x, y, n;
        int min_pu_size = 1 << sps->log2_min_pu_size;
        int hshift = sps->hshift[c_idx];
        int vshift = sps->vshift[c_idx];
        int x_min = x0 >> sps->log2_min_pu_size;
        int y_min = y0 >> sps->log2_min_pu_size;
        int x_max = (x0 + width) >> sps->log2_min_pu_size;
        int y_max = (y0 + height) >> sps->log2_min_pu_size;
        int len = (min_pu_size >> hshift) << sps->pixel_shift;

        for (y = y_min; y < y_max; y++) {
            for (x = x_min; x < x_max; x++) {
                if (f->is_pcm[y * sps->min_pu_width + x]) {
                    uint8_t *src = src1 +
                        (((y << sps->log2_min_pu_size) - y0) >> vshift) * stride_src +
                        ((((x << sps->log2_min_pu_size) - x0) >> hshift) << sps->pixel_shift);
                    const uint8_t *dst = dst1 +
                        (((y << sps->log2_min_pu_size) - y0) >> vshift) * stride_dst +
                        ((((x << sps->log2_min_pu_size) - x0) >> hshift) << sps->pixel_shift);
                    for (n = 0; n < (min_pu_size >> vshift); n++) {
                        memcpy(src, dst, len);
                        src += stride_src;
                        dst += stride_dst;
                    }
                }
            }
        }
    }
}

#define CMP(a, b) (((a) > (b)) - ((a) < (b)))

static void sao_band_filter(uint8_t *dst8, const uint8_t *src8,
                            ptrdiff_t stride_dst, ptrdiff_t stride_src,
                            const int16_t *sao_offset_val, int sao_left_class,
                            int width, int height, int bit_depth)
{
    int offset_table[32] = { 0 };
    int k, y, x;
    int shift = bit_depth - 5;

    for (k = 0; k < 4; k++)
        offset_table[(k + sao_left_class) & 31] = sao_offset_val[k + 1];

#ifdef H265_X86
    if (h265_have_avx2) {
        h265_sao_band_avx2(dst8, stride_dst, src8, stride_src, offset_table,
                           width, height, bit_depth > 8, bit_depth);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        h265_sao_band_neon(dst8, stride_dst, src8, stride_src, offset_table,
                           width, height, bit_depth > 8, bit_depth);
        return;
    }
#endif

    if (bit_depth <= 8) {
        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            const uint8_t *src = src8;
            for (x = 0; x < width; x++) {
                int v = src[x] + offset_table[(src[x] >> shift) & 31];
                dst[x] = (uint8_t)H265_CLIP(v, 0, 255);
            }
            dst8 += stride_dst;
            src8 += stride_src;
        }
    } else {
        int maxv = (1 << bit_depth) - 1;
        for (y = 0; y < height; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            const uint16_t *src = (const uint16_t *)src8;
            ptrdiff_t st = stride_src >> 1;
            for (x = 0; x < width; x++) {
                int v = src[x] + offset_table[(src[x] >> shift) & 31];
                dst[x] = (uint16_t)H265_CLIP(v, 0, maxv);
            }
            dst8 += stride_dst;
            src8 += st * 2;
        }
    }
}

static void sao_edge_filter(uint8_t *dst8, const uint8_t *src8,
                            ptrdiff_t stride_dst, const int16_t *sao_offset_val,
                            int eo, int width, int height, int pixel_shift,
                            int bit_depth)
{
    static const uint8_t edge_idx[] = { 1, 2, 0, 3, 4 };
    static const int8_t pos[4][2][2] = {
        { { -1, 0 }, { 1, 0 } },
        { { 0, -1 }, { 0, 1 } },
        { { -1, -1 }, { 1, 1 } },
        { { 1, -1 }, { -1, 1 } },
    };
    int a_stride, b_stride;
    int x, y;
    ptrdiff_t stride_src = (SAO_STRIDE >> pixel_shift) * (1 << pixel_shift);
    int maxv = (1 << bit_depth) - 1;
    int bps = pixel_shift ? 2 : 1;

#ifdef H265_X86
    if (h265_have_avx2) {
        if (pixel_shift) {
            a_stride = pos[eo][0][0] * 2 + pos[eo][0][1] * (int)stride_src;
            b_stride = pos[eo][1][0] * 2 + pos[eo][1][1] * (int)stride_src;
        } else {
            a_stride = pos[eo][0][0] + pos[eo][0][1] * (int)stride_src;
            b_stride = pos[eo][1][0] + pos[eo][1][1] * (int)stride_src;
        }
        h265_sao_edge_avx2(dst8, stride_dst, src8, stride_src, a_stride,
                           b_stride, sao_offset_val, width, height,
                           pixel_shift, bit_depth);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon) {
        if (pixel_shift) {
            a_stride = pos[eo][0][0] * 2 + pos[eo][0][1] * (int)stride_src;
            b_stride = pos[eo][1][0] * 2 + pos[eo][1][1] * (int)stride_src;
        } else {
            a_stride = pos[eo][0][0] + pos[eo][0][1] * (int)stride_src;
            b_stride = pos[eo][1][0] + pos[eo][1][1] * (int)stride_src;
        }
        h265_sao_edge_neon(dst8, stride_dst, src8, stride_src, a_stride,
                           b_stride, sao_offset_val, width, height,
                           pixel_shift, bit_depth);
        return;
    }
#endif

    if (pixel_shift) {
        a_stride = pos[eo][0][0] * 2 + pos[eo][0][1] * (int)stride_src;
        b_stride = pos[eo][1][0] * 2 + pos[eo][1][1] * (int)stride_src;
        for (y = 0; y < height; y++) {
            uint16_t *dst = (uint16_t *)dst8;
            const uint16_t *src = (const uint16_t *)src8;
            for (x = 0; x < width; x++) {
                int diff0 = CMP(src[x], *(const uint16_t *)((const uint8_t *)src + x * 2 + a_stride));
                int diff1 = CMP(src[x], *(const uint16_t *)((const uint8_t *)src + x * 2 + b_stride));
                int offset_val = edge_idx[2 + diff0 + diff1];
                int v = src[x] + sao_offset_val[offset_val];
                dst[x] = (uint16_t)H265_CLIP(v, 0, maxv);
            }
            dst8 += stride_dst;
            src8 += stride_src;
        }
    } else {
        a_stride = pos[eo][0][0] + pos[eo][0][1] * (int)stride_src;
        b_stride = pos[eo][1][0] + pos[eo][1][1] * (int)stride_src;
        for (y = 0; y < height; y++) {
            uint8_t *dst = dst8;
            const uint8_t *src = src8;
            for (x = 0; x < width; x++) {
                int diff0 = CMP(src[x], src[x + a_stride]);
                int diff1 = CMP(src[x], src[x + b_stride]);
                int offset_val = edge_idx[2 + diff0 + diff1];
                int v = src[x] + sao_offset_val[offset_val];
                dst[x] = (uint8_t)H265_CLIP(v, 0, maxv);
            }
            dst8 += stride_dst;
            src8 += stride_src;
        }
    }
    (void)bps;
}

static void sao_edge_restore_0(uint8_t *dst8, const uint8_t *src8,
                               ptrdiff_t stride_dst, ptrdiff_t stride_src,
                               const H265SAOParams *sao, const int *borders,
                               int _width, int _height, int c_idx,
                               const uint8_t *vert_edge, const uint8_t *horiz_edge,
                               const uint8_t *diag_edge, int pixel_shift, int bit_depth)
{
    int x, y;
    const int16_t *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];
    int init_x = 0, width = _width, height = _height;
    int maxv = (1 << bit_depth) - 1;

    (void)vert_edge;
    (void)horiz_edge;
    (void)diag_edge;

#define DGET(off) (pixel_shift ? ((const uint16_t *)(src8 + (off)))[0] : src8[(off)])
#define DSET(off, v) do { \
    if (pixel_shift) ((uint16_t *)(dst8 + (off)))[0] = (uint16_t)(v); \
    else dst8[(off)] = (uint8_t)(v); } while (0)

    if (sao_eo_class != H265_SAO_EO_VERT) {
        if (borders[0]) {
            int offset_val = sao_offset_val[0];
            for (y = 0; y < height; y++)
                DSET(y * stride_dst, H265_CLIP(DGET(y * stride_src) + offset_val, 0, maxv));
            init_x = 1;
        }
        if (borders[2]) {
            int offset_val = sao_offset_val[0];
            int offset = width - 1;
            for (x = 0; x < height; x++)
                DSET(x * stride_dst + offset * (pixel_shift ? 2 : 1),
                     H265_CLIP(DGET(x * stride_src + offset * (pixel_shift ? 2 : 1)) + offset_val, 0, maxv));
            width--;
        }
    }
    if (sao_eo_class != H265_SAO_EO_HORIZ) {
        if (borders[1]) {
            int offset_val = sao_offset_val[0];
            for (x = init_x; x < width; x++)
                DSET(x * (pixel_shift ? 2 : 1),
                     H265_CLIP(DGET(x * (pixel_shift ? 2 : 1)) + offset_val, 0, maxv));
        }
        if (borders[3]) {
            int offset_val = sao_offset_val[0];
            ptrdiff_t y_stride_dst = stride_dst * (height - 1);
            ptrdiff_t y_stride_src = stride_src * (height - 1);
            for (x = init_x; x < width; x++)
                DSET(x * (pixel_shift ? 2 : 1) + y_stride_dst,
                     H265_CLIP(DGET(x * (pixel_shift ? 2 : 1) + y_stride_src) + offset_val, 0, maxv));
            height--;
        }
    }
#undef DGET
#undef DSET
}

static void sao_edge_restore_1(uint8_t *dst8, const uint8_t *src8,
                               ptrdiff_t stride_dst, ptrdiff_t stride_src,
                               const H265SAOParams *sao, const int *borders,
                               int _width, int _height, int c_idx,
                               const uint8_t *vert_edge, const uint8_t *horiz_edge,
                               const uint8_t *diag_edge, int pixel_shift, int bit_depth)
{
    int x, y;
    const int16_t *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];
    int init_x = 0, init_y = 0, width = _width, height = _height;
    int maxv = (1 << bit_depth) - 1;

#define DGET(off) (pixel_shift ? ((const uint16_t *)(src8 + (off)))[0] : src8[(off)])
#define DSET(off, v) do { \
    if (pixel_shift) ((uint16_t *)(dst8 + (off)))[0] = (uint16_t)(v); \
    else dst8[(off)] = (uint8_t)(v); } while (0)

    if (sao_eo_class != H265_SAO_EO_VERT) {
        if (borders[0]) {
            int offset_val = sao_offset_val[0];
            for (y = 0; y < height; y++)
                DSET(y * stride_dst, H265_CLIP(DGET(y * stride_src) + offset_val, 0, maxv));
            init_x = 1;
        }
        if (borders[2]) {
            int offset_val = sao_offset_val[0];
            int offset = width - 1;
            for (x = 0; x < height; x++)
                DSET(x * stride_dst + offset * (pixel_shift ? 2 : 1),
                     H265_CLIP(DGET(x * stride_src + offset * (pixel_shift ? 2 : 1)) + offset_val, 0, maxv));
            width--;
        }
    }
    if (sao_eo_class != H265_SAO_EO_HORIZ) {
        if (borders[1]) {
            int offset_val = sao_offset_val[0];
            for (x = init_x; x < width; x++)
                DSET(x * (pixel_shift ? 2 : 1),
                     H265_CLIP(DGET(x * (pixel_shift ? 2 : 1)) + offset_val, 0, maxv));
            init_y = 1;
        }
        if (borders[3]) {
            int offset_val = sao_offset_val[0];
            ptrdiff_t y_stride_dst = stride_dst * (height - 1);
            ptrdiff_t y_stride_src = stride_src * (height - 1);
            for (x = init_x; x < width; x++)
                DSET(x * (pixel_shift ? 2 : 1) + y_stride_dst,
                     H265_CLIP(DGET(x * (pixel_shift ? 2 : 1) + y_stride_src) + offset_val, 0, maxv));
            height--;
        }
    }

    {
        int save_upper_left = !diag_edge[0] && sao_eo_class == H265_SAO_EO_135D && !borders[0] && !borders[1];
        int save_upper_right = !diag_edge[1] && sao_eo_class == H265_SAO_EO_45D && !borders[1] && !borders[2];
        int save_lower_right = !diag_edge[2] && sao_eo_class == H265_SAO_EO_135D && !borders[2] && !borders[3];
        int save_lower_left = !diag_edge[3] && sao_eo_class == H265_SAO_EO_45D && !borders[0] && !borders[3];

        if (vert_edge[0] && sao_eo_class != H265_SAO_EO_VERT) {
            for (y = init_y + save_upper_left; y < height - save_lower_left; y++)
                DSET(y * stride_dst, DGET(y * stride_src));
        }
        if (vert_edge[1] && sao_eo_class != H265_SAO_EO_VERT) {
            for (y = init_y + save_upper_right; y < height - save_lower_right; y++)
                DSET(y * stride_dst + (width - 1) * (pixel_shift ? 2 : 1),
                     DGET(y * stride_src + (width - 1) * (pixel_shift ? 2 : 1)));
        }
        if (horiz_edge[0] && sao_eo_class != H265_SAO_EO_HORIZ) {
            for (x = init_x + save_upper_left; x < width - save_upper_right; x++)
                DSET(x * (pixel_shift ? 2 : 1), DGET(x * (pixel_shift ? 2 : 1)));
        }
        if (horiz_edge[1] && sao_eo_class != H265_SAO_EO_HORIZ) {
            for (x = init_x + save_lower_left; x < width - save_lower_right; x++)
                DSET((height - 1) * stride_dst + x * (pixel_shift ? 2 : 1),
                     DGET((height - 1) * stride_src + x * (pixel_shift ? 2 : 1)));
        }
        if (diag_edge[0] && sao_eo_class == H265_SAO_EO_135D)
            DSET(0, DGET(0));
        if (diag_edge[1] && sao_eo_class == H265_SAO_EO_45D)
            DSET((width - 1) * (pixel_shift ? 2 : 1), DGET((width - 1) * (pixel_shift ? 2 : 1)));
        if (diag_edge[2] && sao_eo_class == H265_SAO_EO_135D)
            DSET(stride_dst * (height - 1) + (width - 1) * (pixel_shift ? 2 : 1),
                 DGET(stride_src * (height - 1) + (width - 1) * (pixel_shift ? 2 : 1)));
        if (diag_edge[3] && sao_eo_class == H265_SAO_EO_45D)
            DSET(stride_dst * (height - 1), DGET(stride_src * (height - 1)));
    }
#undef DGET
#undef DSET
}

/* ------------------------------------------------------------------ */
/* per-CTB SAO                                                         */
/* ------------------------------------------------------------------ */
static void sao_filter_CTB(H265Slice *sh, int x, int y)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    int c_idx;
    int edges[4];
    int x_ctb = x >> sps->log2_ctb_size;
    int y_ctb = y >> sps->log2_ctb_size;
    int ctb_addr_rs = y_ctb * sps->ctb_width + x_ctb;
    int ctb_addr_ts = pps->ctb_addr_rs_to_ts[ctb_addr_rs];
    H265SAOParams *sao = &f->sao[ctb_addr_rs];
    uint8_t vert_edge[] = { 0, 0 };
    uint8_t horiz_edge[] = { 0, 0 };
    uint8_t diag_edge[] = { 0, 0, 0, 0 };
    uint8_t lfase = f->filter_slice_edges[ctb_addr_rs];
    uint8_t no_tile_filter = pps->tiles_enabled_flag &&
                             !pps->loop_filter_across_tiles_enabled_flag;
    uint8_t restore = no_tile_filter || !lfase;
    uint8_t left_tile_edge = 0, right_tile_edge = 0;
    uint8_t up_tile_edge = 0, bottom_tile_edge = 0;

#define CTB(tab, xx, yy) ((tab)[(yy) * sps->ctb_width + (xx)])

    edges[0] = x_ctb == 0;
    edges[1] = y_ctb == 0;
    edges[2] = x_ctb == sps->ctb_width - 1;
    edges[3] = y_ctb == sps->ctb_height - 1;

    if (restore) {
        if (!edges[0]) {
            left_tile_edge = no_tile_filter &&
                pps->tile_id[ctb_addr_ts] != pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs - 1]];
            vert_edge[0] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                            CTB(f->tab_slice_address, x_ctb - 1, y_ctb)) || left_tile_edge;
        }
        if (!edges[2]) {
            right_tile_edge = no_tile_filter &&
                pps->tile_id[ctb_addr_ts] != pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs + 1]];
            vert_edge[1] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                            CTB(f->tab_slice_address, x_ctb + 1, y_ctb)) || right_tile_edge;
        }
        if (!edges[1]) {
            up_tile_edge = no_tile_filter &&
                pps->tile_id[ctb_addr_ts] != pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs - sps->ctb_width]];
            horiz_edge[0] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                             CTB(f->tab_slice_address, x_ctb, y_ctb - 1)) || up_tile_edge;
        }
        if (!edges[3]) {
            bottom_tile_edge = no_tile_filter &&
                pps->tile_id[ctb_addr_ts] != pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs + sps->ctb_width]];
            horiz_edge[1] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                             CTB(f->tab_slice_address, x_ctb, y_ctb + 1)) || bottom_tile_edge;
        }
        if (!edges[0] && !edges[1])
            diag_edge[0] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                            CTB(f->tab_slice_address, x_ctb - 1, y_ctb - 1)) ||
                           left_tile_edge || up_tile_edge;
        if (!edges[1] && !edges[2])
            diag_edge[1] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                            CTB(f->tab_slice_address, x_ctb + 1, y_ctb - 1)) ||
                           right_tile_edge || up_tile_edge;
        if (!edges[2] && !edges[3])
            diag_edge[2] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                            CTB(f->tab_slice_address, x_ctb + 1, y_ctb + 1)) ||
                           right_tile_edge || bottom_tile_edge;
        if (!edges[0] && !edges[3])
            diag_edge[3] = (!lfase && CTB(f->tab_slice_address, x_ctb, y_ctb) !=
                            CTB(f->tab_slice_address, x_ctb - 1, y_ctb + 1)) ||
                           left_tile_edge || bottom_tile_edge;
    }

    for (c_idx = 0; c_idx < (sps->chroma_format_idc ? 3 : 1); c_idx++) {
        int x0 = x >> sps->hshift[c_idx];
        int y0 = y >> sps->vshift[c_idx];
        ptrdiff_t stride_src = f->linesize[c_idx];
        int ctb_size_h = (1 << sps->log2_ctb_size) >> sps->hshift[c_idx];
        int ctb_size_v = (1 << sps->log2_ctb_size) >> sps->vshift[c_idx];
        int width = H265_MIN(ctb_size_h, (sps->width >> sps->hshift[c_idx]) - x0);
        int height = H265_MIN(ctb_size_v, (sps->height >> sps->vshift[c_idx]) - y0);
        uint8_t *src = &f->data[c_idx][y0 * stride_src + (x0 << sps->pixel_shift)];
        ptrdiff_t stride_dst;
        uint8_t *dst;
        int psh = sps->pixel_shift;

        switch (sao->type_idx[c_idx]) {
        case H265_SAO_BAND:
            copy_CTB_to_hv(f, sps, src, stride_src, x0, y0, width, height, c_idx,
                           x_ctb, y_ctb);
            if (pps->transquant_bypass_enable_flag ||
                (sps->pcm_loop_filter_disabled && sps->pcm_enabled)) {
                dst = sh->edge_emu_buffer;
                stride_dst = 2 * H265_MAX_PB_SIZE;
                copy_CTB(dst, src, width << psh, height, stride_dst, stride_src);
                sao_band_filter(src, dst, stride_src, stride_dst,
                                sao->offset_val[c_idx], sao->band_position[c_idx],
                                width, height, sps->bit_depth);
                restore_tqb_pixels2(f, pps, sps, src, dst, stride_src, stride_dst,
                                    x, y, width, height, c_idx);
            } else {
                sao_band_filter(src, src, stride_src, stride_src,
                                sao->offset_val[c_idx], sao->band_position[c_idx],
                                width, height, sps->bit_depth);
            }
            sao->type_idx[c_idx] = H265_SAO_APPLIED;
            break;
        case H265_SAO_EDGE: {
            int w = sps->width >> sps->hshift[c_idx];
            int h = sps->height >> sps->vshift[c_idx];
            int left_edge = edges[0];
            int top_edge = edges[1];
            int right_edge = edges[2];
            int bottom_edge = edges[3];
            int left_pixels, right_pixels;

            stride_dst = SAO_STRIDE;
            dst = sh->edge_emu_buffer + stride_dst + 64;

            if (!top_edge) {
                int left = 1 - left_edge;
                int right = 1 - right_edge;
                const uint8_t *src1[2];
                uint8_t *dst1;
                int src_idx, pos;

                dst1 = dst - stride_dst - (left << psh);
                src1[0] = src - stride_src - (left << psh);
                src1[1] = f->sao_pixel_buffer_h[c_idx] +
                          (((2 * y_ctb - 1) * w + x0 - left) << psh);
                pos = 0;
                if (left) {
                    src_idx = CTB(f->sao, x_ctb - 1, y_ctb - 1).type_idx[c_idx] == H265_SAO_APPLIED;
                    copy_pixel(dst1, src1[src_idx], psh);
                    pos += (1 << psh);
                }
                src_idx = CTB(f->sao, x_ctb, y_ctb - 1).type_idx[c_idx] == H265_SAO_APPLIED;
                memcpy(dst1 + pos, src1[src_idx] + pos, width << psh);
                if (right) {
                    pos += width << psh;
                    src_idx = CTB(f->sao, x_ctb + 1, y_ctb - 1).type_idx[c_idx] == H265_SAO_APPLIED;
                    copy_pixel(dst1 + pos, src1[src_idx] + pos, psh);
                }
            }
            if (!bottom_edge) {
                int left = 1 - left_edge;
                int right = 1 - right_edge;
                const uint8_t *src1[2];
                uint8_t *dst1;
                int src_idx, pos;

                dst1 = dst + height * stride_dst - (left << psh);
                src1[0] = src + height * stride_src - (left << psh);
                src1[1] = f->sao_pixel_buffer_h[c_idx] +
                          (((2 * y_ctb + 2) * w + x0 - left) << psh);
                pos = 0;
                if (left) {
                    src_idx = CTB(f->sao, x_ctb - 1, y_ctb + 1).type_idx[c_idx] == H265_SAO_APPLIED;
                    copy_pixel(dst1, src1[src_idx], psh);
                    pos += (1 << psh);
                }
                src_idx = CTB(f->sao, x_ctb, y_ctb + 1).type_idx[c_idx] == H265_SAO_APPLIED;
                memcpy(dst1 + pos, src1[src_idx] + pos, width << psh);
                if (right) {
                    pos += width << psh;
                    src_idx = CTB(f->sao, x_ctb + 1, y_ctb + 1).type_idx[c_idx] == H265_SAO_APPLIED;
                    copy_pixel(dst1 + pos, src1[src_idx] + pos, psh);
                }
            }
            left_pixels = 0;
            if (!left_edge) {
                if (CTB(f->sao, x_ctb - 1, y_ctb).type_idx[c_idx] == H265_SAO_APPLIED) {
                    copy_vert(dst - (1 << psh),
                              f->sao_pixel_buffer_v[c_idx] +
                              (((2 * x_ctb - 1) * h + y0) << psh),
                              psh, height, stride_dst, 1 << psh);
                } else {
                    left_pixels = 1;
                }
            }
            right_pixels = 0;
            if (!right_edge) {
                if (CTB(f->sao, x_ctb + 1, y_ctb).type_idx[c_idx] == H265_SAO_APPLIED) {
                    copy_vert(dst + (width << psh),
                              f->sao_pixel_buffer_v[c_idx] +
                              (((2 * x_ctb + 2) * h + y0) << psh),
                              psh, height, stride_dst, 1 << psh);
                } else {
                    right_pixels = 1;
                }
            }

            copy_CTB(dst - (left_pixels << psh), src - (left_pixels << psh),
                     (width + left_pixels + right_pixels) << psh, height,
                     stride_dst, stride_src);
            copy_CTB_to_hv(f, sps, src, stride_src, x0, y0, width, height, c_idx,
                           x_ctb, y_ctb);
            sao_edge_filter(src, dst, stride_src, sao->offset_val[c_idx],
                            sao->eo_class[c_idx], width, height, psh, sps->bit_depth);
            if (restore)
                sao_edge_restore_1(src, dst, stride_src, stride_dst, sao, edges,
                                   width, height, c_idx, vert_edge, horiz_edge,
                                   diag_edge, psh, sps->bit_depth);
            else
                sao_edge_restore_0(src, dst, stride_src, stride_dst, sao, edges,
                                   width, height, c_idx, vert_edge, horiz_edge,
                                   diag_edge, psh, sps->bit_depth);
            restore_tqb_pixels2(f, pps, sps, src, dst, stride_src, stride_dst,
                                x, y, width, height, c_idx);
            sao->type_idx[c_idx] = H265_SAO_APPLIED;
            break;
        }
        default:
            break;
        }
    }
#undef CTB
}

void h265_sao_frame(H265Slice *sh)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    int ctb_count, ctb_size, ts;

    if (sh->h->param.disable_sao || !sps->sao_enabled)
        return;
    ctb_count = sps->ctb_width * sps->ctb_height;
    ctb_size = 1 << sps->log2_ctb_size;
    for (ts = 0; ts < ctb_count; ts++) {
        int rs = pps->ctb_addr_ts_to_rs[ts];
        int x0 = (rs % sps->ctb_width) * ctb_size;
        int y0 = (rs / sps->ctb_width) * ctb_size;
        sao_filter_CTB(sh, x0, y0);
    }
}

void h265_sao_ctb_row(H265Slice *sh, int y0)
{
    H265SPS *sps = sh->sps;
    int ctb_size = 1 << sps->log2_ctb_size;
    int x;
    if (sh->h->param.disable_sao || !sps->sao_enabled)
        return;
    for (x = 0; x < sps->ctb_width; x++)
        sao_filter_CTB(sh, x * ctb_size, y0);
}
