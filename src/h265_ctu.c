/*
 * HEVC coding tree unit decoding (quadtree, prediction units, transform
 * units, PCM, residuals) and the slice data loop.
 *
 * Derived from ITU-T Rec. H.265 and FFmpeg's hevc/hevcdec.c (LGPL-2.1+).
 */
#include "h265_internal.h"

#define SAMPLE_CTB(tab, x, y, w) ((tab)[(y) * (w) + (x)])
#define CTB(tab, x, y, w) ((tab)[(y) * (w) + (x)])

/* ------------------------------------------------------------------ */
/* SAO parameters                                                      */
/* ------------------------------------------------------------------ */
#define SET_SAO(elem, value)                                     \
    do {                                                         \
        if (!sao_merge_up_flag && !sao_merge_left_flag)          \
            sao->elem = (value);                                 \
        else if (sao_merge_left_flag)                            \
            sao->elem = CTB(f->sao, rx - 1, ry, sps->ctb_width).elem; \
        else                                                     \
            sao->elem = CTB(f->sao, rx, ry - 1, sps->ctb_width).elem; \
    } while (0)

static void h265_sao_param(H265Slice *sh, int rx, int ry)
{
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int sao_merge_left_flag = 0;
    int sao_merge_up_flag = 0;
    H265SAOParams *sao = &CTB(f->sao, rx, ry, sps->ctb_width);
    int c_idx, i;

    if (sh->slice_sample_adaptive_offset_flag[0] ||
        sh->slice_sample_adaptive_offset_flag[1]) {
        if (rx > 0 && sh->ctb_left_flag)
            sao_merge_left_flag = h265_sao_merge_flag_decode(sh);
        if (ry > 0 && !sao_merge_left_flag && sh->ctb_up_flag)
            sao_merge_up_flag = h265_sao_merge_flag_decode(sh);
    }

    for (c_idx = 0; c_idx < (sps->chroma_format_idc ? 3 : 1); c_idx++) {
        int log2_sao_offset_scale = c_idx == 0 ? sh->pps->log2_sao_offset_scale_luma :
                                                 sh->pps->log2_sao_offset_scale_chroma;

        if (!sh->slice_sample_adaptive_offset_flag[c_idx]) {
            sao->type_idx[c_idx] = H265_SAO_NOT_APPLIED;
            continue;
        }
        if (c_idx == 2) {
            sao->type_idx[2] = sao->type_idx[1];
            sao->eo_class[2] = sao->eo_class[1];
        } else {
            SET_SAO(type_idx[c_idx], h265_sao_type_idx_decode(sh));
        }
        if (sao->type_idx[c_idx] == H265_SAO_NOT_APPLIED)
            continue;
        for (i = 0; i < 4; i++)
            SET_SAO(offset_abs[c_idx][i], h265_sao_offset_abs_decode(sh, sps->bit_depth));
        if (sao->type_idx[c_idx] == H265_SAO_BAND) {
            for (i = 0; i < 4; i++) {
                if (sao->offset_abs[c_idx][i])
                    SET_SAO(offset_sign[c_idx][i], h265_sao_offset_sign_decode(sh));
                else
                    sao->offset_sign[c_idx][i] = 0;
            }
            SET_SAO(band_position[c_idx], h265_sao_band_position_decode(sh));
        } else if (c_idx != 2) {
            SET_SAO(eo_class[c_idx], h265_sao_eo_class_decode(sh));
        }
        sao->offset_val[c_idx][0] = 0;
        for (i = 0; i < 4; i++) {
            sao->offset_val[c_idx][i + 1] = (int16_t)sao->offset_abs[c_idx][i];
            if (sao->type_idx[c_idx] == H265_SAO_EDGE) {
                if (i > 1)
                    sao->offset_val[c_idx][i + 1] = -sao->offset_val[c_idx][i + 1];
            } else if (sao->offset_sign[c_idx][i]) {
                sao->offset_val[c_idx][i + 1] = -sao->offset_val[c_idx][i + 1];
            }
            sao->offset_val[c_idx][i + 1] *= 1 << log2_sao_offset_scale;
        }
    }
}
#undef SET_SAO

static int h265_cross_component_pred(H265Slice *sh, int idx)
{
    int log2_res_scale_abs_plus1 = h265_log2_res_scale_abs(sh, idx);
    if (log2_res_scale_abs_plus1 != 0) {
        int sign = h265_res_scale_sign_flag(sh, idx);
        sh->tu.res_scale_val = (1 << (log2_res_scale_abs_plus1 - 1)) * (1 - 2 * sign);
    } else {
        sh->tu.res_scale_val = 0;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* transform unit                                                      */
/* ------------------------------------------------------------------ */
static void h265_transform_unit(H265Slice *sh, int x0, int y0,
                               int xBase, int yBase, int cb_xBase, int cb_yBase,
                               int log2_cb_size, int log2_trafo_size,
                               int blk_idx, int cbf_luma, int *cbf_cb, int *cbf_cr)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    const int log2_trafo_size_c = log2_trafo_size - sps->hshift[1];
    int i;

    if (sh->cu.pred_mode == H265_MODE_INTRA) {
        int trafo_size = 1 << log2_trafo_size;
        h265_set_neighbour_available(sh, x0, y0, trafo_size, trafo_size,
                                     sps->log2_ctb_size);
        h265_intra_pred(sh, x0, y0, log2_trafo_size, 0);
    }

    if (cbf_luma || cbf_cb[0] || cbf_cr[0] ||
        (sps->chroma_format_idc == 2 && (cbf_cb[1] || cbf_cr[1]))) {
        int scan_idx = H265_SCAN_DIAG;
        int scan_idx_c = H265_SCAN_DIAG;
        int cbf_chroma = cbf_cb[0] || cbf_cr[0] ||
                         (sps->chroma_format_idc == 2 && (cbf_cb[1] || cbf_cr[1]));

        if (pps->cu_qp_delta_enabled_flag && !sh->tu.is_cu_qp_delta_coded) {
            sh->tu.cu_qp_delta = h265_cu_qp_delta_abs(sh);
            if (sh->tu.cu_qp_delta != 0)
                if (h265_cu_qp_delta_sign_flag(sh) == 1)
                    sh->tu.cu_qp_delta = -sh->tu.cu_qp_delta;
            sh->tu.is_cu_qp_delta_coded = 1;
            if (sh->tu.cu_qp_delta < -(26 + sps->qp_bd_offset / 2) ||
                sh->tu.cu_qp_delta > (25 + sps->qp_bd_offset / 2))
                sh->error = 1;
            h265_set_qPy(sh, cb_xBase, cb_yBase, log2_cb_size);
        }
        if (sh->cu_chroma_qp_offset_enabled_flag && cbf_chroma &&
            !sh->cu.cu_transquant_bypass_flag &&
            !sh->tu.is_cu_chroma_qp_offset_coded) {
            int flag = h265_cu_chroma_qp_offset_flag(sh);
            if (flag) {
                int idx = 0;
                if (pps->chroma_qp_offset_list_len_minus1 > 0)
                    idx = h265_cu_chroma_qp_offset_idx(sh,
                            pps->chroma_qp_offset_list_len_minus1);
                sh->tu.cu_qp_offset_cb = pps->cb_qp_offset_list[idx];
                sh->tu.cu_qp_offset_cr = pps->cr_qp_offset_list[idx];
            } else {
                sh->tu.cu_qp_offset_cb = 0;
                sh->tu.cu_qp_offset_cr = 0;
            }
            sh->tu.is_cu_chroma_qp_offset_coded = 1;
        }

        if (sh->cu.pred_mode == H265_MODE_INTRA && log2_trafo_size < 4) {
            if (sh->tu.intra_pred_mode >= 6 && sh->tu.intra_pred_mode <= 14)
                scan_idx = H265_SCAN_VERT;
            else if (sh->tu.intra_pred_mode >= 22 && sh->tu.intra_pred_mode <= 30)
                scan_idx = H265_SCAN_HORIZ;
            if (sh->tu.intra_pred_mode_c >= 6 && sh->tu.intra_pred_mode_c <= 14)
                scan_idx_c = H265_SCAN_VERT;
            else if (sh->tu.intra_pred_mode_c >= 22 && sh->tu.intra_pred_mode_c <= 30)
                scan_idx_c = H265_SCAN_HORIZ;
        }

        sh->tu.cross_pf = 0;
        if (cbf_luma)
            h265_hls_residual_coding(sh, x0, y0, log2_trafo_size, scan_idx, 0);

        if (sps->chroma_format_idc &&
            (log2_trafo_size > 2 || sps->chroma_format_idc == 3)) {
            int trafo_size_h = 1 << (log2_trafo_size_c + sps->hshift[1]);
            int trafo_size_v = 1 << (log2_trafo_size_c + sps->vshift[1]);
            sh->tu.cross_pf = (pps->cross_component_prediction_enabled_flag &&
                               cbf_luma &&
                               (sh->cu.pred_mode == H265_MODE_INTER ||
                                sh->tu.chroma_mode_c == 4));
            if (sh->tu.cross_pf)
                h265_cross_component_pred(sh, 0);
            for (i = 0; i < (sps->chroma_format_idc == 2 ? 2 : 1); i++) {
                if (sh->cu.pred_mode == H265_MODE_INTRA) {
                    h265_set_neighbour_available(sh, x0, y0 + (i << log2_trafo_size_c),
                                                 trafo_size_h, trafo_size_v,
                                                 sps->log2_ctb_size);
                    h265_intra_pred(sh, x0, y0 + (i << log2_trafo_size_c),
                                    log2_trafo_size_c, 1);
                }
                if (cbf_cb[i]) {
                    h265_hls_residual_coding(sh, x0, y0 + (i << log2_trafo_size_c),
                                             log2_trafo_size_c, scan_idx_c, 1);
                } else if (sh->tu.cross_pf) {
                    ptrdiff_t stride = f->linesize[1];
                    int hshift = sps->hshift[1];
                    int vshift = sps->vshift[1];
                    const int16_t *coeffs_y = (int16_t *)sh->edge_emu_buffer;
                    int16_t *coeffs = (int16_t *)sh->edge_emu_buffer2;
                    int size = 1 << log2_trafo_size_c;
                    uint8_t *dst = &f->data[1][(y0 >> vshift) * stride +
                                               ((x0 >> hshift) << sps->pixel_shift)];
                    int k;
                    for (k = 0; k < size * size; k++)
                        coeffs[k] = (int16_t)((sh->tu.res_scale_val * coeffs_y[k]) >> 3);
                    h265_add_residual(dst, coeffs, stride, size, sps->pixel_shift,
                                      sps->bit_depth);
                }
            }
            if (sh->tu.cross_pf)
                h265_cross_component_pred(sh, 1);
            for (i = 0; i < (sps->chroma_format_idc == 2 ? 2 : 1); i++) {
                if (sh->cu.pred_mode == H265_MODE_INTRA) {
                    h265_set_neighbour_available(sh, x0, y0 + (i << log2_trafo_size_c),
                                                 trafo_size_h, trafo_size_v,
                                                 sps->log2_ctb_size);
                    h265_intra_pred(sh, x0, y0 + (i << log2_trafo_size_c),
                                    log2_trafo_size_c, 2);
                }
                if (cbf_cr[i]) {
                    h265_hls_residual_coding(sh, x0, y0 + (i << log2_trafo_size_c),
                                             log2_trafo_size_c, scan_idx_c, 2);
                } else if (sh->tu.cross_pf) {
                    ptrdiff_t stride = f->linesize[2];
                    int hshift = sps->hshift[2];
                    int vshift = sps->vshift[2];
                    const int16_t *coeffs_y = (int16_t *)sh->edge_emu_buffer;
                    int16_t *coeffs = (int16_t *)sh->edge_emu_buffer2;
                    int size = 1 << log2_trafo_size_c;
                    uint8_t *dst = &f->data[2][(y0 >> vshift) * stride +
                                               ((x0 >> hshift) << sps->pixel_shift)];
                    int k;
                    for (k = 0; k < size * size; k++)
                        coeffs[k] = (int16_t)((sh->tu.res_scale_val * coeffs_y[k]) >> 3);
                    h265_add_residual(dst, coeffs, stride, size, sps->pixel_shift,
                                      sps->bit_depth);
                }
            }
        } else if (sps->chroma_format_idc && blk_idx == 3) {
            int trafo_size_h = 1 << (log2_trafo_size + 1);
            int trafo_size_v = 1 << (log2_trafo_size + sps->vshift[1]);
            for (i = 0; i < (sps->chroma_format_idc == 2 ? 2 : 1); i++) {
                if (sh->cu.pred_mode == H265_MODE_INTRA) {
                    h265_set_neighbour_available(sh, xBase, yBase + (i << log2_trafo_size),
                                                 trafo_size_h, trafo_size_v,
                                                 sps->log2_ctb_size);
                    h265_intra_pred(sh, xBase, yBase + (i << log2_trafo_size),
                                    log2_trafo_size, 1);
                }
                if (cbf_cb[i])
                    h265_hls_residual_coding(sh, xBase, yBase + (i << log2_trafo_size),
                                             log2_trafo_size, scan_idx_c, 1);
            }
            for (i = 0; i < (sps->chroma_format_idc == 2 ? 2 : 1); i++) {
                if (sh->cu.pred_mode == H265_MODE_INTRA) {
                    h265_set_neighbour_available(sh, xBase, yBase + (i << log2_trafo_size),
                                                 trafo_size_h, trafo_size_v,
                                                 sps->log2_ctb_size);
                    h265_intra_pred(sh, xBase, yBase + (i << log2_trafo_size),
                                    log2_trafo_size, 2);
                }
                if (cbf_cr[i])
                    h265_hls_residual_coding(sh, xBase, yBase + (i << log2_trafo_size),
                                             log2_trafo_size, scan_idx_c, 2);
            }
        }
    } else if (sps->chroma_format_idc && sh->cu.pred_mode == H265_MODE_INTRA) {
        if (log2_trafo_size > 2 || sps->chroma_format_idc == 3) {
            int trafo_size_h = 1 << (log2_trafo_size_c + sps->hshift[1]);
            int trafo_size_v = 1 << (log2_trafo_size_c + sps->vshift[1]);
            h265_set_neighbour_available(sh, x0, y0, trafo_size_h, trafo_size_v,
                                         sps->log2_ctb_size);
            h265_intra_pred(sh, x0, y0, log2_trafo_size_c, 1);
            h265_intra_pred(sh, x0, y0, log2_trafo_size_c, 2);
            if (sps->chroma_format_idc == 2) {
                h265_set_neighbour_available(sh, x0, y0 + (1 << log2_trafo_size_c),
                                             trafo_size_h, trafo_size_v,
                                             sps->log2_ctb_size);
                h265_intra_pred(sh, x0, y0 + (1 << log2_trafo_size_c),
                                log2_trafo_size_c, 1);
                h265_intra_pred(sh, x0, y0 + (1 << log2_trafo_size_c),
                                log2_trafo_size_c, 2);
            }
        } else if (blk_idx == 3) {
            int trafo_size_h = 1 << (log2_trafo_size + 1);
            int trafo_size_v = 1 << (log2_trafo_size + sps->vshift[1]);
            h265_set_neighbour_available(sh, xBase, yBase, trafo_size_h,
                                         trafo_size_v, sps->log2_ctb_size);
            h265_intra_pred(sh, xBase, yBase, log2_trafo_size, 1);
            h265_intra_pred(sh, xBase, yBase, log2_trafo_size, 2);
            if (sps->chroma_format_idc == 2) {
                h265_set_neighbour_available(sh, xBase, yBase + (1 << log2_trafo_size),
                                             trafo_size_h, trafo_size_v,
                                             sps->log2_ctb_size);
                h265_intra_pred(sh, xBase, yBase + (1 << log2_trafo_size),
                                log2_trafo_size, 1);
                h265_intra_pred(sh, xBase, yBase + (1 << log2_trafo_size),
                                log2_trafo_size, 2);
            }
        }
    }
}

static void set_deblocking_bypass(H265Frame *f, H265SPS *sps, int x0, int y0,
                                  int log2_cb_size)
{
    int cb_size = 1 << log2_cb_size;
    int log2_min_pu_size = sps->log2_min_pu_size;
    int min_pu_width = sps->min_pu_width;
    int x_end = H265_MIN(x0 + cb_size, sps->width);
    int y_end = H265_MIN(y0 + cb_size, sps->height);
    int i, j;

    for (j = (y0 >> log2_min_pu_size); j < (y_end >> log2_min_pu_size); j++)
        for (i = (x0 >> log2_min_pu_size); i < (x_end >> log2_min_pu_size); i++)
            f->is_pcm[i + j * min_pu_width] = 2;
}

static void h265_transform_tree(H265Slice *sh, int x0, int y0,
                               int xBase, int yBase, int cb_xBase, int cb_yBase,
                               int log2_cb_size, int log2_trafo_size,
                               int trafo_depth, int blk_idx,
                               const int *base_cbf_cb, const int *base_cbf_cr)
{
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int split_transform_flag;
    int cbf_cb[2], cbf_cr[2];

    cbf_cb[0] = base_cbf_cb[0];
    cbf_cb[1] = base_cbf_cb[1];
    cbf_cr[0] = base_cbf_cr[0];
    cbf_cr[1] = base_cbf_cr[1];

    if (sh->cu.intra_split_flag) {
        if (trafo_depth == 1) {
            sh->tu.intra_pred_mode = sh->pu.intra_pred_mode[blk_idx];
            if (sps->chroma_format_idc == 3) {
                sh->tu.intra_pred_mode_c = sh->pu.intra_pred_mode_c[blk_idx];
                sh->tu.chroma_mode_c = sh->pu.chroma_mode_c[blk_idx];
            } else {
                sh->tu.intra_pred_mode_c = sh->pu.intra_pred_mode_c[0];
                sh->tu.chroma_mode_c = sh->pu.chroma_mode_c[0];
            }
        }
    } else {
        sh->tu.intra_pred_mode = sh->pu.intra_pred_mode[0];
        sh->tu.intra_pred_mode_c = sh->pu.intra_pred_mode_c[0];
        sh->tu.chroma_mode_c = sh->pu.chroma_mode_c[0];
    }

    if (log2_trafo_size <= sps->log2_max_trafo_size &&
        log2_trafo_size > sps->log2_min_tb_size &&
        trafo_depth < sh->cu.max_trafo_depth &&
        !(sh->cu.intra_split_flag && trafo_depth == 0)) {
        split_transform_flag = h265_split_transform_flag_decode(sh, log2_trafo_size);
    } else {
        int inter_split = sps->max_transform_hierarchy_depth_inter == 0 &&
                          sh->cu.pred_mode == H265_MODE_INTER &&
                          sh->cu.part_mode != H265_PART_2Nx2N &&
                          trafo_depth == 0;
        split_transform_flag = log2_trafo_size > sps->log2_max_trafo_size ||
                               (sh->cu.intra_split_flag && trafo_depth == 0) ||
                               inter_split;
    }

    if (sps->chroma_format_idc &&
        (log2_trafo_size > 2 || sps->chroma_format_idc == 3)) {
        if (trafo_depth == 0 || cbf_cb[0]) {
            cbf_cb[0] = h265_cbf_cb_cr_decode(sh, trafo_depth);
            if (sps->chroma_format_idc == 2 &&
                (!split_transform_flag || log2_trafo_size == 3))
                cbf_cb[1] = h265_cbf_cb_cr_decode(sh, trafo_depth);
        }
        if (trafo_depth == 0 || cbf_cr[0]) {
            cbf_cr[0] = h265_cbf_cb_cr_decode(sh, trafo_depth);
            if (sps->chroma_format_idc == 2 &&
                (!split_transform_flag || log2_trafo_size == 3))
                cbf_cr[1] = h265_cbf_cb_cr_decode(sh, trafo_depth);
        }
    }

    if (split_transform_flag) {
        const int trafo_size_split = 1 << (log2_trafo_size - 1);
        const int x1 = x0 + trafo_size_split;
        const int y1 = y0 + trafo_size_split;
        h265_transform_tree(sh, x0, y0, x0, y0, cb_xBase, cb_yBase,
                            log2_cb_size, log2_trafo_size - 1, trafo_depth + 1,
                            0, cbf_cb, cbf_cr);
        h265_transform_tree(sh, x1, y0, x0, y0, cb_xBase, cb_yBase,
                            log2_cb_size, log2_trafo_size - 1, trafo_depth + 1,
                            1, cbf_cb, cbf_cr);
        h265_transform_tree(sh, x0, y1, x0, y0, cb_xBase, cb_yBase,
                            log2_cb_size, log2_trafo_size - 1, trafo_depth + 1,
                            2, cbf_cb, cbf_cr);
        h265_transform_tree(sh, x1, y1, x0, y0, cb_xBase, cb_yBase,
                            log2_cb_size, log2_trafo_size - 1, trafo_depth + 1,
                            3, cbf_cb, cbf_cr);
    } else {
        int min_tu_size = 1 << sps->log2_min_tb_size;
        int log2_min_tu_size = sps->log2_min_tb_size;
        int min_tu_width = sps->min_tb_width;
        int cbf_luma = 1;

        if (sh->cu.pred_mode == H265_MODE_INTRA || trafo_depth != 0 ||
            cbf_cb[0] || cbf_cr[0] ||
            (sps->chroma_format_idc == 2 && (cbf_cb[1] || cbf_cr[1])))
            cbf_luma = h265_cbf_luma_decode(sh, trafo_depth);

        h265_transform_unit(sh, x0, y0, xBase, yBase, cb_xBase, cb_yBase,
                            log2_cb_size, log2_trafo_size, blk_idx, cbf_luma,
                            cbf_cb, cbf_cr);
        if (cbf_luma) {
            int i, j;
            for (i = 0; i < (1 << log2_trafo_size); i += min_tu_size)
                for (j = 0; j < (1 << log2_trafo_size); j += min_tu_size) {
                    int x_tu = (x0 + j) >> log2_min_tu_size;
                    int y_tu = (y0 + i) >> log2_min_tu_size;
                    f->cbf_luma[y_tu * min_tu_width + x_tu] = 1;
                }
        }
        if (!sh->disable_deblocking_filter_flag) {
            h265_deblocking_boundary_strengths(sh, x0, y0, log2_trafo_size);
            if (sh->pps->transquant_bypass_enable_flag &&
                sh->cu.cu_transquant_bypass_flag)
                set_deblocking_bypass(f, sps, x0, y0, log2_trafo_size);
        }
    }
}

static void h265_pcm_sample(H265Slice *sh, int x0, int y0, int log2_cb_size)
{
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    H265BitReader gb;
    int cb_size = 1 << log2_cb_size;
    ptrdiff_t stride0 = f->linesize[0];
    uint8_t *dst0 = &f->data[0][y0 * stride0 + (x0 << sps->pixel_shift)];
    int length = cb_size * cb_size * sps->pcm.bit_depth +
                 (sps->chroma_format_idc != 0 ?
                  (((cb_size >> sps->hshift[1]) * (cb_size >> sps->vshift[1])) +
                   ((cb_size >> sps->hshift[2]) * (cb_size >> sps->vshift[2]))) *
                  sps->pcm.bit_depth_chroma : 0);
    const uint8_t *pcm = h265_cabac_skip_bytes(sh, (length + 7) >> 3);

    if (!sh->disable_deblocking_filter_flag)
        h265_deblocking_boundary_strengths(sh, x0, y0, log2_cb_size);
    if (!pcm)
        return;

    h265_br_init(&gb, pcm, length);
    h265_put_pcm(dst0, stride0, cb_size, cb_size, &gb, sps->pcm.bit_depth,
                 sps->pixel_shift, sps->bit_depth);
    if (sps->chroma_format_idc) {
        ptrdiff_t stride1 = f->linesize[1];
        ptrdiff_t stride2 = f->linesize[2];
        uint8_t *dst1 = &f->data[1][(y0 >> sps->vshift[1]) * stride1 +
                                    ((x0 >> sps->hshift[1]) << sps->pixel_shift)];
        uint8_t *dst2 = &f->data[2][(y0 >> sps->vshift[2]) * stride2 +
                                    ((x0 >> sps->hshift[2]) << sps->pixel_shift)];
        h265_put_pcm(dst1, stride1, cb_size >> sps->hshift[1],
                     cb_size >> sps->vshift[1], &gb, sps->pcm.bit_depth_chroma,
                     sps->pixel_shift, sps->bit_depth);
        h265_put_pcm(dst2, stride2, cb_size >> sps->hshift[2],
                     cb_size >> sps->vshift[2], &gb, sps->pcm.bit_depth_chroma,
                     sps->pixel_shift, sps->bit_depth);
    }
}

/* ------------------------------------------------------------------ */
/* prediction                                                          */
/* ------------------------------------------------------------------ */
static void h265_prediction_unit(H265Slice *sh, int x0, int y0, int nPbW,
                                 int nPbH, int log2_cb_size, int partIdx, int idx)
{
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int merge_idx = 0;
    H265MvField current_mv;
    int min_pu_width = sps->min_pu_width;
    H265MvField *tab_mvf = f->tab_mvf;
    const int *linesize = f->linesize;
    uint8_t *dst0 = f->data[0] + y0 * linesize[0] + (x0 << sps->pixel_shift);
    uint8_t *dst1 = f->data[1] ? f->data[1] + (y0 >> sps->vshift[1]) * linesize[1] +
                                 ((x0 >> sps->hshift[1]) << sps->pixel_shift) : NULL;
    uint8_t *dst2 = f->data[2] ? f->data[2] + (y0 >> sps->vshift[2]) * linesize[2] +
                                 ((x0 >> sps->hshift[2]) << sps->pixel_shift) : NULL;
    int log2_min_cb_size = sps->log2_min_cb_size;
    int min_cb_width = sps->min_cb_width;
    int x_cb = x0 >> log2_min_cb_size;
    int y_cb = y0 >> log2_min_cb_size;
    int x_pu, y_pu;
    int i, j;
    int skip_flag = SAMPLE_CTB(f->skip_flag, x_cb, y_cb, min_cb_width);
    H265Frame *ref0 = NULL, *ref1 = NULL;

    memset(&current_mv, 0, sizeof(current_mv));

    if (!skip_flag)
        sh->pu.merge_flag = (uint8_t)h265_merge_flag_decode(sh);

    if (skip_flag || sh->pu.merge_flag) {
        if (sh->max_num_merge_cand > 1)
            merge_idx = h265_merge_idx_decode(sh);
        else
            merge_idx = 0;
        h265_luma_mv_merge_mode(sh, x0, y0, nPbW, nPbH, log2_cb_size, partIdx,
                                merge_idx, &current_mv);
    } else {
        H265MvField mv;
        int inter_pred_idc = H265_PRED_L0;
        int mvp_flag;

        memset(&mv, 0, sizeof(mv));
        h265_set_neighbour_available(sh, x0, y0, nPbW, nPbH, sps->log2_ctb_size);
        if (sh->slice_type == H265_SLICE_B)
            inter_pred_idc = h265_inter_pred_idc_decode(sh, nPbW, nPbH);

        if (inter_pred_idc != H265_PRED_L1) {
            if (sh->nb_refs[H265_L0])
                mv.ref_idx[0] = (int8_t)h265_ref_idx_lx_decode(sh, sh->nb_refs[H265_L0]);
            mv.pred_flag = H265_PF_L0;
            h265_hls_mvd_coding(sh);
            mvp_flag = h265_mvp_lx_flag_decode(sh);
            h265_luma_mv_mvp_mode(sh, x0, y0, nPbW, nPbH, log2_cb_size, partIdx,
                                  merge_idx, &mv, mvp_flag, 0);
            mv.mv[0].x = (int16_t)(mv.mv[0].x + sh->pu.mvd.x);
            mv.mv[0].y = (int16_t)(mv.mv[0].y + sh->pu.mvd.y);
        }
        if (inter_pred_idc != H265_PRED_L0) {
            if (sh->nb_refs[H265_L1])
                mv.ref_idx[1] = (int8_t)h265_ref_idx_lx_decode(sh, sh->nb_refs[H265_L1]);
            if (sh->mvd_l1_zero_flag == 1 && inter_pred_idc == H265_PRED_BI) {
                sh->pu.mvd.x = 0;
                sh->pu.mvd.y = 0;
            } else {
                h265_hls_mvd_coding(sh);
            }
            mv.pred_flag = (int8_t)(mv.pred_flag + H265_PF_L1);
            mvp_flag = h265_mvp_lx_flag_decode(sh);
            h265_luma_mv_mvp_mode(sh, x0, y0, nPbW, nPbH, log2_cb_size, partIdx,
                                  merge_idx, &mv, mvp_flag, 1);
            mv.mv[1].x = (int16_t)(mv.mv[1].x + sh->pu.mvd.x);
            mv.mv[1].y = (int16_t)(mv.mv[1].y + sh->pu.mvd.y);
        }
        current_mv = mv;
    }

    x_pu = x0 >> sps->log2_min_pu_size;
    y_pu = y0 >> sps->log2_min_pu_size;
    for (j = 0; j < nPbH >> sps->log2_min_pu_size; j++)
        for (i = 0; i < nPbW >> sps->log2_min_pu_size; i++)
            tab_mvf[(y_pu + j) * min_pu_width + x_pu + i] = current_mv;

    if (current_mv.pred_flag & H265_PF_L0) {
        ref0 = sh->ref_frame[0][current_mv.ref_idx[0]];
        if (!ref0)
            return;
        {
            int need = (current_mv.mv[0].y >> 2) + y0 + nPbH + 9;
            if (need < 1) need = 1;
            h265_frame_wait_rows(ref0, need);
        }
    }
    if (current_mv.pred_flag & H265_PF_L1) {
        ref1 = sh->ref_frame[1][current_mv.ref_idx[1]];
        if (!ref1)
            return;
        {
            int need = (current_mv.mv[1].y >> 2) + y0 + nPbH + 9;
            if (need < 1) need = 1;
            h265_frame_wait_rows(ref1, need);
        }
    }

    if (current_mv.pred_flag == H265_PF_L0) {
        int x0_c = x0 >> sps->hshift[1];
        int y0_c = y0 >> sps->vshift[1];
        int nPbW_c = nPbW >> sps->hshift[1];
        int nPbH_c = nPbH >> sps->vshift[1];

        h265_mc_luma_uni(sh, dst0, linesize[0], ref0, &current_mv.mv[0], x0, y0,
                         nPbW, nPbH,
                         sh->luma_weight_l0[current_mv.ref_idx[0]],
                         sh->luma_offset_l0[current_mv.ref_idx[0]]);
        if (sps->chroma_format_idc) {
            h265_mc_chroma_uni(sh, dst1, linesize[1], ref0->data[1],
                               ref0->linesize[1], 0, x0_c, y0_c, nPbW_c, nPbH_c,
                               &current_mv,
                               sh->chroma_weight_l0[current_mv.ref_idx[0]][0],
                               sh->chroma_offset_l0[current_mv.ref_idx[0]][0]);
            h265_mc_chroma_uni(sh, dst2, linesize[2], ref0->data[2],
                               ref0->linesize[2], 0, x0_c, y0_c, nPbW_c, nPbH_c,
                               &current_mv,
                               sh->chroma_weight_l0[current_mv.ref_idx[0]][1],
                               sh->chroma_offset_l0[current_mv.ref_idx[0]][1]);
        }
    } else if (current_mv.pred_flag == H265_PF_L1) {
        int x0_c = x0 >> sps->hshift[1];
        int y0_c = y0 >> sps->vshift[1];
        int nPbW_c = nPbW >> sps->hshift[1];
        int nPbH_c = nPbH >> sps->vshift[1];

        h265_mc_luma_uni(sh, dst0, linesize[0], ref1, &current_mv.mv[1], x0, y0,
                         nPbW, nPbH,
                         sh->luma_weight_l1[current_mv.ref_idx[1]],
                         sh->luma_offset_l1[current_mv.ref_idx[1]]);
        if (sps->chroma_format_idc) {
            h265_mc_chroma_uni(sh, dst1, linesize[1], ref1->data[1],
                               ref1->linesize[1], 1, x0_c, y0_c, nPbW_c, nPbH_c,
                               &current_mv,
                               sh->chroma_weight_l1[current_mv.ref_idx[1]][0],
                               sh->chroma_offset_l1[current_mv.ref_idx[1]][0]);
            h265_mc_chroma_uni(sh, dst2, linesize[2], ref1->data[2],
                               ref1->linesize[2], 1, x0_c, y0_c, nPbW_c, nPbH_c,
                               &current_mv,
                               sh->chroma_weight_l1[current_mv.ref_idx[1]][1],
                               sh->chroma_offset_l1[current_mv.ref_idx[1]][1]);
        }
    } else if (current_mv.pred_flag == H265_PF_BI) {
        int x0_c = x0 >> sps->hshift[1];
        int y0_c = y0 >> sps->vshift[1];
        int nPbW_c = nPbW >> sps->hshift[1];
        int nPbH_c = nPbH >> sps->vshift[1];

        h265_mc_luma_bi(sh, dst0, linesize[0], ref0, &current_mv.mv[0], x0, y0,
                        nPbW, nPbH, ref1, &current_mv.mv[1], &current_mv);
        if (sps->chroma_format_idc) {
            h265_mc_chroma_bi(sh, dst1, linesize[1], ref0, ref1, x0_c, y0_c,
                              nPbW_c, nPbH_c, &current_mv, 0);
            h265_mc_chroma_bi(sh, dst2, linesize[2], ref0, ref1, x0_c, y0_c,
                              nPbW_c, nPbH_c, &current_mv, 1);
        }
    }
    (void)idx;
}

static int luma_intra_pred_mode(H265Slice *sh, int x0, int y0, int pu_size,
                                int prev_intra_luma_pred_flag)
{
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int x_pu = x0 >> sps->log2_min_pu_size;
    int y_pu = y0 >> sps->log2_min_pu_size;
    int min_pu_width = sps->min_pu_width;
    int size_in_pus = pu_size >> sps->log2_min_pu_size;
    int x0b = x0 & ((1 << sps->log2_ctb_size) - 1);
    int y0b = y0 & ((1 << sps->log2_ctb_size) - 1);
    int cand_up = (sh->ctb_up_flag || y0b) ?
                  f->tab_ipm[(y_pu - 1) * min_pu_width + x_pu] : H265_INTRA_DC;
    int cand_left = (sh->ctb_left_flag || x0b) ?
                    f->tab_ipm[y_pu * min_pu_width + x_pu - 1] : H265_INTRA_DC;
    int y_ctb = (y0 >> sps->log2_ctb_size) << sps->log2_ctb_size;
    H265MvField *tab_mvf = f->tab_mvf;
    int intra_pred_mode;
    int candidate[3];
    int i, j;

    if ((y0 - 1) < y_ctb)
        cand_up = H265_INTRA_DC;

    if (cand_left == cand_up) {
        if (cand_left < 2) {
            candidate[0] = H265_INTRA_PLANAR;
            candidate[1] = H265_INTRA_DC;
            candidate[2] = H265_INTRA_ANGULAR_26;
        } else {
            candidate[0] = cand_left;
            candidate[1] = 2 + ((cand_left - 2 - 1 + 32) & 31);
            candidate[2] = 2 + ((cand_left - 2 + 1) & 31);
        }
    } else {
        candidate[0] = cand_left;
        candidate[1] = cand_up;
        if (candidate[0] != H265_INTRA_PLANAR && candidate[1] != H265_INTRA_PLANAR)
            candidate[2] = H265_INTRA_PLANAR;
        else if (candidate[0] != H265_INTRA_DC && candidate[1] != H265_INTRA_DC)
            candidate[2] = H265_INTRA_DC;
        else
            candidate[2] = H265_INTRA_ANGULAR_26;
    }

    if (prev_intra_luma_pred_flag) {
        intra_pred_mode = candidate[sh->pu.mpm_idx];
    } else {
        if (candidate[0] > candidate[1]) {
            int t = candidate[0]; candidate[0] = candidate[1]; candidate[1] = t;
        }
        if (candidate[0] > candidate[2]) {
            int t = candidate[0]; candidate[0] = candidate[2]; candidate[2] = t;
        }
        if (candidate[1] > candidate[2]) {
            int t = candidate[1]; candidate[1] = candidate[2]; candidate[2] = t;
        }
        intra_pred_mode = sh->pu.rem_intra_luma_pred_mode;
        for (i = 0; i < 3; i++)
            if (intra_pred_mode >= candidate[i])
                intra_pred_mode++;
    }

    if (!size_in_pus)
        size_in_pus = 1;
    for (i = 0; i < size_in_pus; i++) {
        memset(&f->tab_ipm[(y_pu + i) * min_pu_width + x_pu], intra_pred_mode,
               size_in_pus);
        for (j = 0; j < size_in_pus; j++)
            tab_mvf[(y_pu + j) * min_pu_width + x_pu + i].pred_flag = H265_PF_INTRA;
    }
    return intra_pred_mode;
}

static void set_ct_depth(H265SPS *sps, uint8_t *tab_ct_depth, int x0, int y0,
                         int log2_cb_size, int ct_depth)
{
    int length = (1 << log2_cb_size) >> sps->log2_min_cb_size;
    int x_cb = x0 >> sps->log2_min_cb_size;
    int y_cb = y0 >> sps->log2_min_cb_size;
    int y;
    for (y = 0; y < length; y++)
        memset(&tab_ct_depth[(y_cb + y) * sps->min_cb_width + x_cb],
               ct_depth, length);
}

static const uint8_t tab_mode_idx[] = {
     0,  1,  2,  2,  2,  2,  3,  5,  7,  8, 10, 12, 13, 15, 17, 18, 19, 20,
    21, 22, 23, 23, 24, 24, 25, 25, 26, 27, 27, 28, 28, 29, 29, 30, 31
};

static void intra_prediction_unit(H265Slice *sh, int x0, int y0, int log2_cb_size)
{
    H265SPS *sps = sh->sps;
    static const uint8_t intra_chroma_table[4] = { 0, 26, 10, 1 };
    uint8_t prev_intra_luma_pred_flag[4];
    int split = sh->cu.part_mode == H265_PART_NxN;
    int pb_size = (1 << log2_cb_size) >> split;
    int side = split + 1;
    int chroma_mode;
    int i, j;

    for (i = 0; i < side; i++)
        for (j = 0; j < side; j++)
            prev_intra_luma_pred_flag[2 * i + j] =
                (uint8_t)h265_prev_intra_luma_pred_flag_decode(sh);

    for (i = 0; i < side; i++) {
        for (j = 0; j < side; j++) {
            if (prev_intra_luma_pred_flag[2 * i + j])
                sh->pu.mpm_idx = h265_mpm_idx_decode(sh);
            else
                sh->pu.rem_intra_luma_pred_mode = h265_rem_intra_luma_pred_mode_decode(sh);
            sh->pu.intra_pred_mode[2 * i + j] =
                (uint8_t)luma_intra_pred_mode(sh, x0 + pb_size * j,
                                              y0 + pb_size * i, pb_size,
                                              prev_intra_luma_pred_flag[2 * i + j]);
        }
    }

    if (sps->chroma_format_idc == 3) {
        for (i = 0; i < side; i++) {
            for (j = 0; j < side; j++) {
                sh->pu.chroma_mode_c[2 * i + j] = chroma_mode =
                    (uint8_t)h265_intra_chroma_pred_mode_decode(sh);
                if (chroma_mode != 4) {
                    if (sh->pu.intra_pred_mode[2 * i + j] == intra_chroma_table[chroma_mode])
                        sh->pu.intra_pred_mode_c[2 * i + j] = 34;
                    else
                        sh->pu.intra_pred_mode_c[2 * i + j] = intra_chroma_table[chroma_mode];
                } else {
                    sh->pu.intra_pred_mode_c[2 * i + j] = sh->pu.intra_pred_mode[2 * i + j];
                }
            }
        }
    } else if (sps->chroma_format_idc == 2) {
        int mode_idx;
        sh->pu.chroma_mode_c[0] = chroma_mode =
            (uint8_t)h265_intra_chroma_pred_mode_decode(sh);
        if (chroma_mode != 4) {
            if (sh->pu.intra_pred_mode[0] == intra_chroma_table[chroma_mode])
                mode_idx = 34;
            else
                mode_idx = intra_chroma_table[chroma_mode];
        } else {
            mode_idx = sh->pu.intra_pred_mode[0];
        }
        sh->pu.intra_pred_mode_c[0] = tab_mode_idx[mode_idx];
    } else if (sps->chroma_format_idc != 0) {
        chroma_mode = h265_intra_chroma_pred_mode_decode(sh);
        if (chroma_mode != 4) {
            if (sh->pu.intra_pred_mode[0] == intra_chroma_table[chroma_mode])
                sh->pu.intra_pred_mode_c[0] = 34;
            else
                sh->pu.intra_pred_mode_c[0] = intra_chroma_table[chroma_mode];
        } else {
            sh->pu.intra_pred_mode_c[0] = sh->pu.intra_pred_mode[0];
        }
    }
}

static void intra_prediction_unit_default_value(H265Slice *sh, int x0, int y0,
                                                int log2_cb_size)
{
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int pb_size = 1 << log2_cb_size;
    int size_in_pus = pb_size >> sps->log2_min_pu_size;
    int min_pu_width = sps->min_pu_width;
    H265MvField *tab_mvf = f->tab_mvf;
    int x_pu = x0 >> sps->log2_min_pu_size;
    int y_pu = y0 >> sps->log2_min_pu_size;
    int j, k;

    if (size_in_pus == 0)
        size_in_pus = 1;
    for (j = 0; j < size_in_pus; j++)
        memset(&f->tab_ipm[(y_pu + j) * min_pu_width + x_pu], H265_INTRA_DC,
               size_in_pus);
    if (sh->cu.pred_mode == H265_MODE_INTRA) {
        for (j = 0; j < size_in_pus; j++)
            for (k = 0; k < size_in_pus; k++)
                tab_mvf[(y_pu + j) * min_pu_width + x_pu + k].pred_flag =
                    H265_PF_INTRA;
    }
}

/* ------------------------------------------------------------------ */
/* coding unit                                                         */
/* ------------------------------------------------------------------ */
static void h265_coding_unit(H265Slice *sh, int x0, int y0, int log2_cb_size)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    int cb_size = 1 << log2_cb_size;
    int log2_min_cb_size = sps->log2_min_cb_size;
    int length = cb_size >> log2_min_cb_size;
    int min_cb_width = sps->min_cb_width;
    int x_cb = x0 >> log2_min_cb_size;
    int y_cb = y0 >> log2_min_cb_size;
    int idx = log2_cb_size - 2;
    int qp_block_mask = (1 << (sps->log2_ctb_size - pps->diff_cu_qp_delta_depth)) - 1;
    int x, y;

    sh->cu.x = x0;
    sh->cu.y = y0;
    sh->cu.pred_mode = H265_MODE_INTRA;
    sh->cu.part_mode = H265_PART_2Nx2N;
    sh->cu.intra_split_flag = 0;

    SAMPLE_CTB(f->skip_flag, x_cb, y_cb, min_cb_width) = 0;
    for (x = 0; x < 4; x++)
        sh->pu.intra_pred_mode[x] = 1;
    if (pps->transquant_bypass_enable_flag) {
        sh->cu.cu_transquant_bypass_flag = (uint8_t)h265_cu_transquant_bypass_flag_decode(sh);
        if (sh->cu.cu_transquant_bypass_flag)
            set_deblocking_bypass(f, sps, x0, y0, log2_cb_size);
    } else {
        sh->cu.cu_transquant_bypass_flag = 0;
    }

    if (sh->slice_type != H265_SLICE_I) {
        const int x0b = x0 & ((1 << sps->log2_ctb_size) - 1);
        const int y0b = y0 & ((1 << sps->log2_ctb_size) - 1);
        uint8_t skip_flag = (uint8_t)h265_skip_flag_decode(sh, f->skip_flag,
                                                           x0b, y0b, x_cb, y_cb,
                                                           min_cb_width);
        x = y_cb * min_cb_width + x_cb;
        for (y = 0; y < length; y++) {
            memset(&f->skip_flag[x], skip_flag, length);
            x += min_cb_width;
        }
        sh->cu.pred_mode = skip_flag ? H265_MODE_SKIP : H265_MODE_INTER;
    } else {
        x = y_cb * min_cb_width + x_cb;
        for (y = 0; y < length; y++) {
            memset(&f->skip_flag[x], 0, length);
            x += min_cb_width;
        }
    }

    if (SAMPLE_CTB(f->skip_flag, x_cb, y_cb, min_cb_width)) {
        h265_prediction_unit(sh, x0, y0, cb_size, cb_size, log2_cb_size, 0, idx);
        intra_prediction_unit_default_value(sh, x0, y0, log2_cb_size);
        if (!sh->disable_deblocking_filter_flag)
            h265_deblocking_boundary_strengths(sh, x0, y0, log2_cb_size);
    } else {
        int pcm_flag = 0;

        if (sh->slice_type != H265_SLICE_I)
            sh->cu.pred_mode = h265_pred_mode_decode(sh);
        if (sh->cu.pred_mode != H265_MODE_INTRA ||
            log2_cb_size == sps->log2_min_cb_size) {
            sh->cu.part_mode = h265_part_mode_decode(sh, sps, log2_cb_size);
            sh->cu.intra_split_flag = sh->cu.part_mode == H265_PART_NxN &&
                                      sh->cu.pred_mode == H265_MODE_INTRA;
        }

        if (sh->cu.pred_mode == H265_MODE_INTRA) {
            if (sh->cu.part_mode == H265_PART_2Nx2N && sps->pcm_enabled &&
                log2_cb_size >= sps->pcm.log2_min_pcm_cb_size &&
                log2_cb_size <= sps->pcm.log2_max_pcm_cb_size)
                pcm_flag = h265_pcm_flag_decode(sh);
            if (pcm_flag) {
                intra_prediction_unit_default_value(sh, x0, y0, log2_cb_size);
                h265_pcm_sample(sh, x0, y0, log2_cb_size);
                if (sps->pcm_loop_filter_disabled)
                    set_deblocking_bypass(f, sps, x0, y0, log2_cb_size);
                for (x = y_cb * min_cb_width + x_cb, y = 0; y < length; y++, x += min_cb_width)
                    memset(&f->qp_y_tab[x], sh->qp_y, length);
                set_ct_depth(sps, f->tab_ct_depth, x0, y0, log2_cb_size, sh->ct_depth);
                return;
            } else {
                intra_prediction_unit(sh, x0, y0, log2_cb_size);
            }
        } else {
            intra_prediction_unit_default_value(sh, x0, y0, log2_cb_size);
            switch (sh->cu.part_mode) {
            case H265_PART_2Nx2N:
                h265_prediction_unit(sh, x0, y0, cb_size, cb_size, log2_cb_size, 0, idx);
                break;
            case H265_PART_2NxN:
                h265_prediction_unit(sh, x0, y0, cb_size, cb_size / 2, log2_cb_size, 0, idx);
                h265_prediction_unit(sh, x0, y0 + cb_size / 2, cb_size, cb_size / 2, log2_cb_size, 1, idx);
                break;
            case H265_PART_Nx2N:
                h265_prediction_unit(sh, x0, y0, cb_size / 2, cb_size, log2_cb_size, 0, idx - 1);
                h265_prediction_unit(sh, x0 + cb_size / 2, y0, cb_size / 2, cb_size, log2_cb_size, 1, idx - 1);
                break;
            case H265_PART_2NxnU:
                h265_prediction_unit(sh, x0, y0, cb_size, cb_size / 4, log2_cb_size, 0, idx);
                h265_prediction_unit(sh, x0, y0 + cb_size / 4, cb_size, cb_size * 3 / 4, log2_cb_size, 1, idx);
                break;
            case H265_PART_2NxnD:
                h265_prediction_unit(sh, x0, y0, cb_size, cb_size * 3 / 4, log2_cb_size, 0, idx);
                h265_prediction_unit(sh, x0, y0 + cb_size * 3 / 4, cb_size, cb_size / 4, log2_cb_size, 1, idx);
                break;
            case H265_PART_nLx2N:
                h265_prediction_unit(sh, x0, y0, cb_size / 4, cb_size, log2_cb_size, 0, idx - 2);
                h265_prediction_unit(sh, x0 + cb_size / 4, y0, cb_size * 3 / 4, cb_size, log2_cb_size, 1, idx - 2);
                break;
            case H265_PART_nRx2N:
                h265_prediction_unit(sh, x0, y0, cb_size * 3 / 4, cb_size, log2_cb_size, 0, idx - 2);
                h265_prediction_unit(sh, x0 + cb_size * 3 / 4, y0, cb_size / 4, cb_size, log2_cb_size, 1, idx - 2);
                break;
            case H265_PART_NxN:
                h265_prediction_unit(sh, x0, y0, cb_size / 2, cb_size / 2, log2_cb_size, 0, idx - 1);
                h265_prediction_unit(sh, x0 + cb_size / 2, y0, cb_size / 2, cb_size / 2, log2_cb_size, 1, idx - 1);
                h265_prediction_unit(sh, x0, y0 + cb_size / 2, cb_size / 2, cb_size / 2, log2_cb_size, 2, idx - 1);
                h265_prediction_unit(sh, x0 + cb_size / 2, y0 + cb_size / 2, cb_size / 2, cb_size / 2, log2_cb_size, 3, idx - 1);
                break;
            }
        }

        if (!pcm_flag) {
            int rqt_root_cbf = 1;
            if (sh->cu.pred_mode != H265_MODE_INTRA &&
                !(sh->cu.part_mode == H265_PART_2Nx2N && sh->pu.merge_flag))
                rqt_root_cbf = h265_no_residual_syntax_flag_decode(sh);
            if (rqt_root_cbf) {
                const int cbf[2] = { 0 };
                sh->cu.max_trafo_depth = sh->cu.pred_mode == H265_MODE_INTRA ?
                    sps->max_transform_hierarchy_depth_intra + sh->cu.intra_split_flag :
                    sps->max_transform_hierarchy_depth_inter;
                h265_transform_tree(sh, x0, y0, x0, y0, x0, y0, log2_cb_size,
                                    log2_cb_size, 0, 0, cbf, cbf);
            } else if (!sh->disable_deblocking_filter_flag) {
                h265_deblocking_boundary_strengths(sh, x0, y0, log2_cb_size);
            }
        }
    }

    if (pps->cu_qp_delta_enabled_flag && sh->tu.is_cu_qp_delta_coded == 0)
        h265_set_qPy(sh, x0, y0, log2_cb_size);

    x = y_cb * min_cb_width + x_cb;
    for (y = 0; y < length; y++) {
        memset(&f->qp_y_tab[x], sh->qp_y, length);
        x += min_cb_width;
    }

    if (((x0 + (1 << log2_cb_size)) & qp_block_mask) == 0 &&
        ((y0 + (1 << log2_cb_size)) & qp_block_mask) == 0)
        sh->qPy_pred = sh->qp_y;

    set_ct_depth(sps, f->tab_ct_depth, x0, y0, log2_cb_size, sh->ct_depth);
}

static int h265_coding_quadtree(H265Slice *sh, int x0, int y0,
                                int log2_cb_size, int cb_depth)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    const int cb_size = 1 << log2_cb_size;
    int split_cu;

    sh->ct_depth = cb_depth;

    if (x0 + cb_size <= sps->width && y0 + cb_size <= sps->height &&
        log2_cb_size > sps->log2_min_cb_size) {
        split_cu = h265_split_coding_unit_flag_decode(sh, f->tab_ct_depth,
                                                      sps, cb_depth, x0, y0);
    } else {
        split_cu = (log2_cb_size > sps->log2_min_cb_size);
    }

    if (pps->cu_qp_delta_enabled_flag &&
        log2_cb_size >= sps->log2_ctb_size - pps->diff_cu_qp_delta_depth) {
        sh->tu.is_cu_qp_delta_coded = 0;
        sh->tu.cu_qp_delta = 0;
    }
    if (sh->cu_chroma_qp_offset_enabled_flag &&
        log2_cb_size >= sps->log2_ctb_size - pps->diff_cu_chroma_qp_offset_depth)
        sh->tu.is_cu_chroma_qp_offset_coded = 0;

    if (split_cu) {
        int qp_block_mask = (1 << (sps->log2_ctb_size - pps->diff_cu_qp_delta_depth)) - 1;
        const int cb_size_split = cb_size >> 1;
        const int x1 = x0 + cb_size_split;
        const int y1 = y0 + cb_size_split;
        int more_data = 0;

        more_data = h265_coding_quadtree(sh, x0, y0, log2_cb_size - 1, cb_depth + 1);
        if (more_data && x1 < sps->width)
            more_data = h265_coding_quadtree(sh, x1, y0, log2_cb_size - 1, cb_depth + 1);
        if (more_data && y1 < sps->height)
            more_data = h265_coding_quadtree(sh, x0, y1, log2_cb_size - 1, cb_depth + 1);
        if (more_data && x1 < sps->width && y1 < sps->height)
            more_data = h265_coding_quadtree(sh, x1, y1, log2_cb_size - 1, cb_depth + 1);

        if (((x0 + (1 << log2_cb_size)) & qp_block_mask) == 0 &&
            ((y0 + (1 << log2_cb_size)) & qp_block_mask) == 0)
            sh->qPy_pred = sh->qp_y;

        if (more_data)
            return ((x1 + cb_size_split) < sps->width ||
                    (y1 + cb_size_split) < sps->height);
        return 0;
    } else {
        h265_coding_unit(sh, x0, y0, log2_cb_size);
        if ((!((x0 + cb_size) % (1 << sps->log2_ctb_size)) ||
             (x0 + cb_size >= sps->width)) &&
            (!((y0 + cb_size) % (1 << sps->log2_ctb_size)) ||
             (y0 + cb_size >= sps->height))) {
            int end_of_slice_flag = h265_end_of_slice_flag_decode(sh);
            return !end_of_slice_flag;
        }
        return 1;
    }
}

static void h265_decode_neighbour(H265Slice *sh, int x_ctb, int y_ctb,
                                  int ctb_addr_ts)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    int ctb_size = 1 << sps->log2_ctb_size;
    int ctb_addr_rs = pps->ctb_addr_ts_to_rs[ctb_addr_ts];
    int ctb_addr_in_slice = ctb_addr_rs - (int)sh->slice_addr;

    f->tab_slice_address[ctb_addr_rs] = (int32_t)sh->slice_addr;

    if (pps->entropy_coding_sync_enabled_flag) {
        if (x_ctb == 0 && (y_ctb & (ctb_size - 1)) == 0)
            sh->first_qp_group = 1;
        sh->end_of_tiles_x = sps->width;
    } else if (pps->tiles_enabled_flag) {
        if (ctb_addr_ts && pps->tile_id[ctb_addr_ts] != pps->tile_id[ctb_addr_ts - 1]) {
            int idxX = pps->col_idxX[x_ctb >> sps->log2_ctb_size];
            sh->end_of_tiles_x = x_ctb + (pps->column_width[idxX] << sps->log2_ctb_size);
            sh->first_qp_group = 1;
        }
    } else {
        sh->end_of_tiles_x = sps->width;
    }
    sh->end_of_tiles_y = H265_MIN(y_ctb + ctb_size, sps->height);

    sh->boundary_flags = 0;
    if (pps->tiles_enabled_flag) {
        if (x_ctb > 0 && pps->tile_id[ctb_addr_ts] !=
            pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs - 1]])
            sh->boundary_flags |= H265_BOUNDARY_LEFT_TILE;
        if (x_ctb > 0 && f->tab_slice_address[ctb_addr_rs] !=
            f->tab_slice_address[ctb_addr_rs - 1])
            sh->boundary_flags |= H265_BOUNDARY_LEFT_SLICE;
        if (y_ctb > 0 && pps->tile_id[ctb_addr_ts] !=
            pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs - sps->ctb_width]])
            sh->boundary_flags |= H265_BOUNDARY_UPPER_TILE;
        if (y_ctb > 0 && f->tab_slice_address[ctb_addr_rs] !=
            f->tab_slice_address[ctb_addr_rs - sps->ctb_width])
            sh->boundary_flags |= H265_BOUNDARY_UPPER_SLICE;
    } else {
        if (ctb_addr_in_slice <= 0)
            sh->boundary_flags |= H265_BOUNDARY_LEFT_SLICE;
        if (ctb_addr_in_slice < sps->ctb_width)
            sh->boundary_flags |= H265_BOUNDARY_UPPER_SLICE;
    }

    sh->ctb_left_flag = (x_ctb > 0) && (ctb_addr_in_slice > 0) &&
                        !(sh->boundary_flags & H265_BOUNDARY_LEFT_TILE);
    sh->ctb_up_flag = (y_ctb > 0) && (ctb_addr_in_slice >= sps->ctb_width) &&
                      !(sh->boundary_flags & H265_BOUNDARY_UPPER_TILE);
    sh->ctb_up_right_flag = (y_ctb > 0) &&
                            (ctb_addr_in_slice + 1 >= sps->ctb_width) &&
                            (pps->tile_id[ctb_addr_ts] ==
                             pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs + 1 - sps->ctb_width]]);
    sh->ctb_up_left_flag = (x_ctb > 0) && (y_ctb > 0) &&
                           (ctb_addr_in_slice - 1 >= sps->ctb_width) &&
                           (pps->tile_id[ctb_addr_ts] ==
                            pps->tile_id[pps->ctb_addr_rs_to_ts[ctb_addr_rs - 1 - sps->ctb_width]]);
}

/* ------------------------------------------------------------------ */
/* slice data                                                          */
/* ------------------------------------------------------------------ */
static int cabac_init_ctb(H265Slice *sh, int ctb_addr_ts, int start_ts,
                          const uint8_t *data, int size)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;

    if (ctb_addr_ts == start_ts) {
        if (h265_cabac_init(sh, data + sh->data_offset,
                            size - (int)sh->data_offset) < 0)
            return -1;
        if (sh->dependent_slice_segment_flag == 0 ||
            (pps->tiles_enabled_flag &&
             pps->tile_id[ctb_addr_ts] != pps->tile_id[ctb_addr_ts - 1]))
            h265_cabac_init_states(sh);
        if (!sh->first_slice_in_pic_flag && pps->entropy_coding_sync_enabled_flag) {
            if (ctb_addr_ts % sps->ctb_width == 0) {
                if (sps->ctb_width == 1)
                    h265_cabac_init_states(sh);
                else if (sh->dependent_slice_segment_flag == 1)
                    memcpy(sh->cabac_state, sh->cabac_saved, H265_CONTEXTS);
            }
        }
    } else {
        if (pps->tiles_enabled_flag &&
            pps->tile_id[ctb_addr_ts] != pps->tile_id[ctb_addr_ts - 1]) {
            if (h265_cabac_reinit(sh) < 0)
                return -1;
            h265_cabac_init_states(sh);
        }
        if (pps->entropy_coding_sync_enabled_flag &&
            ctb_addr_ts % sps->ctb_width == 0) {
            h265_cabac_terminate(sh);
            if (h265_cabac_reinit(sh) < 0)
                return -1;
            if (sps->ctb_width == 1)
                h265_cabac_init_states(sh);
            else
                memcpy(sh->cabac_state, sh->cabac_saved, H265_CONTEXTS);
        }
    }
    return 0;
}

int h265_decode_wpp_row(H265Slice *sh, H265WPPContext *ctx, int row)
{
    H265PPS *pps = sh->pps;
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int ctb_width = sps->ctb_width;
    int start_rs = row * ctb_width;
    int row_ctus = H265_MIN(ctb_width, sps->ctb_size - start_rs);
    int ctb_addr_ts = pps->ctb_addr_rs_to_ts[start_rs];
    int count = 0;
    int more_data = 1;

    if (!f)
        return -1;

    sh->first_qp_group = 1;
    if (!pps->cu_qp_delta_enabled_flag)
        sh->qp_y = sh->slice_qp;
    sh->tu.cu_qp_offset_cb = 0;
    sh->tu.cu_qp_offset_cr = 0;

    if (row > 0) {
        h265_mutex_lock(&ctx->mutex);
        while (ctx->progress[row - 1] < H265_MIN(3, row_ctus) && !ctx->error)
            h265_cond_wait(&ctx->cond, &ctx->mutex);
        if (ctx->error) {
            h265_mutex_unlock(&ctx->mutex);
            return -1;
        }
        memcpy(sh->cabac_state, ctx->slices[row - 1]->cabac_saved,
               H265_CONTEXTS);
        if (sps->persistent_rice_adaptation_enabled)
            memcpy(sh->stat_coeff, ctx->slices[row - 1]->stat_coeff_saved,
                   H265_STAT_COEFFS);
        h265_mutex_unlock(&ctx->mutex);
    }

    if (h265_cabac_init(sh, ctx->rbsp + ctx->row_offset[row],
                        ctx->row_size[row]) < 0)
        goto error;
    if (row == 0)
        h265_cabac_init_states(sh);

    while (count < row_ctus && ctb_addr_ts < sps->ctb_size) {
        int ctb_addr_rs = pps->ctb_addr_ts_to_rs[ctb_addr_ts];
        int x_ctb = (ctb_addr_rs % ctb_width) << sps->log2_ctb_size;
        int y_ctb = (ctb_addr_rs / ctb_width) << sps->log2_ctb_size;

        if (row > 0 && count > 0) {
            int need = H265_MIN(count + 3, row_ctus);
            h265_mutex_lock(&ctx->mutex);
            while (ctx->progress[row - 1] < need && !ctx->error)
                h265_cond_wait(&ctx->cond, &ctx->mutex);
            if (ctx->error) {
                h265_mutex_unlock(&ctx->mutex);
                goto error;
            }
            h265_mutex_unlock(&ctx->mutex);
        }

        h265_decode_neighbour(sh, x_ctb, y_ctb, ctb_addr_ts);
        h265_sao_param(sh, x_ctb >> sps->log2_ctb_size,
                       y_ctb >> sps->log2_ctb_size);
        f->deblock[ctb_addr_rs].beta_offset = sh->beta_offset;
        f->deblock[ctb_addr_rs].tc_offset = sh->tc_offset;
        f->deblock[ctb_addr_rs].disabled = sh->disable_deblocking_filter_flag;
        f->filter_slice_edges[ctb_addr_rs] =
            sh->slice_loop_filter_across_slices_enabled_flag;

        more_data = h265_coding_quadtree(sh, x_ctb, y_ctb,
                                         sps->log2_ctb_size, 0);
        if (sh->error) {
            f->tab_slice_address[ctb_addr_rs] = -1;
            goto error;
        }
        ctb_addr_ts++;
        count++;
        h265_save_states(sh, ctb_addr_ts);

        h265_mutex_lock(&ctx->mutex);
        ctx->progress[row] = count;
        h265_cond_broadcast(&ctx->cond);
        h265_mutex_unlock(&ctx->mutex);

        if (!more_data && count < row_ctus) {
            f->tab_slice_address[ctb_addr_rs] = -1;
            goto error;
        }
    }

    if (count != row_ctus) {
        goto error;
    }
    return 0;

error:
    h265_mutex_lock(&ctx->mutex);
    ctx->error = 1;
    h265_cond_broadcast(&ctx->cond);
    h265_mutex_unlock(&ctx->mutex);
    sh->error = 1;
    return -1;
}

int h265_decode_slice(H265Slice *sh)
{
    H265Decoder *h = sh->h;
    H265PPS *pps = sh->pps;
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int more_data = 1;
    int ctb_addr_ts;
    int start_ts;
    int x_ctb = 0, y_ctb = 0;

    if (!sh->frame)
        sh->frame = h->cur_frame;
    f = sh->frame;
    if (!f)
        return -1;

    if (!sh->rpl_valid) {
        if (h->threaded)
            return -1;
        if (!sh->dependent_slice_segment_flag && sh->slice_type != H265_SLICE_I) {
            if (h265_slice_rpl(h, sh) < 0)
                return -1;
        }
        sh->rpl_valid = 1;
    }

    sh->first_qp_group = !sh->dependent_slice_segment_flag;
    if (!pps->cu_qp_delta_enabled_flag)
        sh->qp_y = sh->slice_qp;
    sh->tu.cu_qp_offset_cb = 0;
    sh->tu.cu_qp_offset_cr = 0;

    start_ts = pps->ctb_addr_rs_to_ts[sh->slice_ctb_addr_rs];
    ctb_addr_ts = start_ts;

    while (more_data && ctb_addr_ts < sps->ctb_size) {
        int ctb_addr_rs = pps->ctb_addr_ts_to_rs[ctb_addr_ts];
        x_ctb = (ctb_addr_rs % sps->ctb_width) << sps->log2_ctb_size;
        y_ctb = (ctb_addr_rs / sps->ctb_width) << sps->log2_ctb_size;

        h265_decode_neighbour(sh, x_ctb, y_ctb, ctb_addr_ts);

        if (cabac_init_ctb(sh, ctb_addr_ts, start_ts, sh->rbsp,
                           sh->rbsp_size) < 0) {
            f->tab_slice_address[ctb_addr_rs] = -1;
            return -1;
        }

        h265_sao_param(sh, x_ctb >> sps->log2_ctb_size,
                       y_ctb >> sps->log2_ctb_size);

        f->deblock[ctb_addr_rs].beta_offset = sh->beta_offset;
        f->deblock[ctb_addr_rs].tc_offset = sh->tc_offset;
        f->deblock[ctb_addr_rs].disabled = sh->disable_deblocking_filter_flag;
        f->filter_slice_edges[ctb_addr_rs] =
            sh->slice_loop_filter_across_slices_enabled_flag;

        more_data = h265_coding_quadtree(sh, x_ctb, y_ctb, sps->log2_ctb_size, 0);
        if (sh->error) {
            f->tab_slice_address[ctb_addr_rs] = -1;
            return -1;
        }

        ctb_addr_ts++;
        h265_save_states(sh, ctb_addr_ts);

        if (sh->row_progress) {
            int row_done = 0;
            int last = ctb_addr_ts >= sps->ctb_size;
            if (last) {
                row_done = 1;
            } else {
                int next_rs = pps->ctb_addr_ts_to_rs[ctb_addr_ts];
                int next_y = (next_rs / sps->ctb_width) << sps->log2_ctb_size;
                row_done = next_y != y_ctb;
            }
            if (row_done)
                h265_filter_row(sh, y_ctb, last);
        }
    }

    return 0;
}
