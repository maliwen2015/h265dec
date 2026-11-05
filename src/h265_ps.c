/*
 * HEVC VPS / SPS / PPS parsing.
 *
 * Derived from ITU-T Rec. H.265 and FFmpeg's HEVC decoder (LGPL-2.1+).
 */
#include "h265_internal.h"

extern const uint8_t h265_default_scaling_list_intra[64];
extern const uint8_t h265_default_scaling_list_inter[64];

/* ------------------------------------------------------------------ */
/* reference counting                                                  */
/* ------------------------------------------------------------------ */
H265VPS *h265_vps_ref(H265VPS *v)
{
    if (v)
        v->refcount++;
    return v;
}

H265SPS *h265_sps_ref(H265SPS *s)
{
    if (s)
        s->refcount++;
    return s;
}

H265PPS *h265_pps_ref(H265PPS *p)
{
    if (p)
        p->refcount++;
    return p;
}

void h265_vps_unref(H265VPS **pv)
{
    H265VPS *v = *pv;
    if (!v)
        return;
    if (--v->refcount <= 0)
        free(v);
    *pv = NULL;
}

void h265_sps_unref(H265SPS **ps)
{
    H265SPS *s = *ps;
    if (!s)
        return;
    if (--s->refcount <= 0)
        free(s);
    *ps = NULL;
}

void h265_pps_unref(H265PPS **pp)
{
    H265PPS *p = *pp;
    if (!p)
        return;
    if (--p->refcount <= 0) {
        free(p->column_width);
        free(p->row_height);
        free(p->col_bd);
        free(p->row_bd);
        free(p->col_idxX);
        free(p->ctb_addr_rs_to_ts);
        free(p->ctb_addr_ts_to_rs);
        free(p->tile_id);
        free(p->tile_pos_rs);
        free(p->min_tb_addr_zs_tab);
        h265_sps_unref(&p->sps);
        free(p);
    }
    *pp = NULL;
}

void h265_vps_free(H265VPS *v)
{
    if (v)
        free(v);
}

void h265_sps_free(H265SPS *s)
{
    if (s)
        free(s);
}

void h265_pps_free(H265PPS *p)
{
    h265_pps_unref(&p);
}

/* ------------------------------------------------------------------ */
/* bitstream helpers                                                   */
/* ------------------------------------------------------------------ */
static void parse_ptl_full(H265BitReader *br, H265PTL *ptl, int profile_present,
                           int max_sub_layers, uint8_t *profile_flags,
                           uint8_t *level_flags)
{
    int i;

    if (profile_present) {
        ptl->profile_space = h265_br_get_bits(br, 2);
        ptl->tier_flag     = h265_br_get_bits1(br);
        ptl->profile_idc   = h265_br_get_bits(br, 5);
        h265_br_skip(br, 32);
        h265_br_skip(br, 4);
        h265_br_skip(br, 44);
    } else {
        memset(ptl, 0, sizeof(*ptl));
    }
    ptl->level_idc = h265_br_get_bits(br, 8);

    for (i = 0; i < max_sub_layers - 1; i++) {
        profile_flags[i] = h265_br_get_bits1(br);
        level_flags[i]   = h265_br_get_bits1(br);
    }
    if (max_sub_layers - 1 > 0)
        h265_br_skip(br, 2 * (8 - (max_sub_layers - 1)));
    for (i = 0; i < max_sub_layers - 1; i++) {
        if (profile_flags[i])
            h265_br_skip(br, 88);
        if (level_flags[i])
            h265_br_skip(br, 8);
    }
}

static void set_default_scaling_list(H265ScalingList *sl)
{
    int m;

    for (m = 0; m < 6; m++) {
        memset(sl->sl[0][m], 16, 16);
        sl->sl_dc[0][m] = 16;
        sl->sl_dc[1][m] = 16;
    }
    memcpy(sl->sl[1][0], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[1][1], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[1][2], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[1][3], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[1][4], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[1][5], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[2][0], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[2][1], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[2][2], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[2][3], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[2][4], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[2][5], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[3][0], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[3][1], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[3][2], h265_default_scaling_list_intra, 64);
    memcpy(sl->sl[3][3], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[3][4], h265_default_scaling_list_inter, 64);
    memcpy(sl->sl[3][5], h265_default_scaling_list_inter, 64);
}

static int scaling_list_data(H265BitReader *br, H265ScalingList *sl,
                             int chroma_format_idc)
{
    uint8_t dc_coef[2][6];
    int size_id, matrix_id, i;

    memset(dc_coef, 0, sizeof(dc_coef));

    for (size_id = 0; size_id < 4; size_id++) {
        for (matrix_id = 0; matrix_id < 6; matrix_id += (size_id == 3 ? 3 : 1)) {
            int pred_mode = h265_br_get_bits1(br);
            if (!pred_mode) {
                unsigned delta = h265_br_get_ue(br);
                if (delta) {
                    delta *= (size_id == 3) ? 3 : 1;
                    if (matrix_id < (int)delta)
                        return -1;
                    memcpy(sl->sl[size_id][matrix_id],
                           sl->sl[size_id][matrix_id - delta],
                           size_id > 0 ? 64 : 16);
                    if (size_id > 1)
                        sl->sl_dc[size_id - 2][matrix_id] =
                            sl->sl_dc[size_id - 2][matrix_id - delta];
                }
            } else {
                int next_coef = 8;
                int coef_num = H265_MIN(64, 1 << (4 + (size_id << 1)));
                if (size_id > 1) {
                    int dc = h265_br_get_se(br);
                    if (dc < -7 || dc > 247)
                        return -1;
                    dc_coef[size_id - 2][matrix_id] = dc + 8;
                    next_coef = dc_coef[size_id - 2][matrix_id];
                    sl->sl_dc[size_id - 2][matrix_id] = next_coef;
                }
                for (i = 0; i < coef_num; i++) {
                    int pos;
                    int delta;
                    if (size_id == 0)
                        pos = 4 * h265_diag_scan4x4_y[i] + h265_diag_scan4x4_x[i];
                    else
                        pos = 8 * h265_diag_scan8x8_y[i] + h265_diag_scan8x8_x[i];
                    delta = h265_br_get_se(br);
                    next_coef = (next_coef + 256 + delta) % 256;
                    sl->sl[size_id][matrix_id][pos] = next_coef;
                }
            }
        }
    }

    if (chroma_format_idc == 3) {
        for (i = 0; i < 64; i++) {
            sl->sl[3][1][i] = sl->sl[2][1][i];
            sl->sl[3][2][i] = sl->sl[2][2][i];
            sl->sl[3][4][i] = sl->sl[2][4][i];
            sl->sl[3][5][i] = sl->sl[2][5][i];
        }
        sl->sl_dc[1][1] = sl->sl_dc[0][1];
        sl->sl_dc[1][2] = sl->sl_dc[0][2];
        sl->sl_dc[1][4] = sl->sl_dc[0][4];
        sl->sl_dc[1][5] = sl->sl_dc[0][5];
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* short-term RPS                                                      */
/* ------------------------------------------------------------------ */
int h265_parse_short_term_rps(H265BitReader *br, H265ShortTermRPS *rps,
                              const H265SPS *sps, int is_slice_header)
{
    int delta_poc;
    int k0 = 0;
    int k = 0;
    int i;

    rps->used = 0;
    rps->rps_predict = 0;

    if (rps != sps->st_rps && sps->nb_st_rps)
        rps->rps_predict = h265_br_get_bits1(br);

    if (rps->rps_predict) {
        const H265ShortTermRPS *rps_ridx;
        uint8_t used[32] = { 0 };
        int delta_rps;

        if (is_slice_header) {
            rps->delta_idx = h265_br_get_ue(br) + 1;
            if (rps->delta_idx > sps->nb_st_rps)
                return -1;
            rps_ridx = &sps->st_rps[sps->nb_st_rps - rps->delta_idx];
            rps->rps_idx_num_delta_pocs = rps_ridx->num_delta_pocs;
        } else {
            rps_ridx = &sps->st_rps[rps - sps->st_rps - 1];
        }

        rps->delta_rps_sign = h265_br_get_bits1(br);
        rps->abs_delta_rps  = h265_br_get_ue(br) + 1;
        if (rps->abs_delta_rps > 32768)
            return -1;
        delta_rps = (1 - (rps->delta_rps_sign << 1)) * rps->abs_delta_rps;
        for (i = 0; i <= rps_ridx->num_delta_pocs; i++) {
            used[k] = h265_br_get_bits1(br);

            rps->use_delta = 0;
            if (!used[k])
                rps->use_delta = h265_br_get_bits1(br);

            if (used[k] || rps->use_delta) {
                if (i < rps_ridx->num_delta_pocs)
                    delta_poc = delta_rps + rps_ridx->delta_poc[i];
                else
                    delta_poc = delta_rps;
                rps->delta_poc[k] = delta_poc;
                if (delta_poc < 0)
                    k0++;
                k++;
            }
        }
        if (k >= 32)
            return -1;

        rps->num_delta_pocs    = k;
        rps->num_negative_pics = k0;
        if (rps->num_delta_pocs != 0) {
            int u, tmp;
            for (i = 1; i < rps->num_delta_pocs; i++) {
                delta_poc = rps->delta_poc[i];
                u = used[i];
                for (k = i - 1; k >= 0; k--) {
                    tmp = rps->delta_poc[k];
                    if (delta_poc < tmp) {
                        rps->delta_poc[k + 1] = tmp;
                        used[k + 1] = used[k];
                        rps->delta_poc[k] = delta_poc;
                        used[k] = u;
                    }
                }
            }
        }
        if ((rps->num_negative_pics >> 1) != 0) {
            int u;
            k = rps->num_negative_pics - 1;
            for (i = 0; i < rps->num_negative_pics >> 1; i++) {
                delta_poc = rps->delta_poc[i];
                u = used[i];
                rps->delta_poc[i] = rps->delta_poc[k];
                used[i] = used[k];
                rps->delta_poc[k] = delta_poc;
                used[k] = u;
                k--;
            }
        }
        for (i = 0; i < 32; i++)
            rps->used |= (uint32_t)used[i] << i;
    } else {
        unsigned nb_positive_pics;

        rps->num_negative_pics = h265_br_get_ue(br);
        nb_positive_pics       = h265_br_get_ue(br);

        if (rps->num_negative_pics >= H265_MAX_REFS ||
            nb_positive_pics >= H265_MAX_REFS)
            return -1;

        rps->num_delta_pocs = rps->num_negative_pics + nb_positive_pics;
        if (rps->num_delta_pocs) {
            int prev = 0;
            for (i = 0; i < rps->num_negative_pics; i++) {
                delta_poc = h265_br_get_ue(br) + 1;
                if (delta_poc < 1 || delta_poc > 32768)
                    return -1;
                prev -= delta_poc;
                rps->delta_poc[i] = prev;
                rps->used |= h265_br_get_bits1(br) * (1 << i);
            }
            prev = 0;
            for (i = 0; i < (int)nb_positive_pics; i++) {
                delta_poc = h265_br_get_ue(br) + 1;
                if (delta_poc < 1 || delta_poc > 32768)
                    return -1;
                prev += delta_poc;
                rps->delta_poc[rps->num_negative_pics + i] = prev;
                rps->used |= h265_br_get_bits1(br) <<
                             (rps->num_negative_pics + i);
            }
        }
    }
    return 0;
}

int h265_compute_poc(unsigned log2_max_poc_lsb, int poc_tid0, int poc_lsb,
                     int nal_unit_type)
{
    int max_poc_lsb = 1 << log2_max_poc_lsb;
    int prev_poc_lsb = poc_tid0 % max_poc_lsb;
    int prev_poc_msb = poc_tid0 - prev_poc_lsb;
    int poc_msb;

    if (poc_lsb < prev_poc_lsb && prev_poc_lsb - poc_lsb >= max_poc_lsb / 2)
        poc_msb = prev_poc_msb + max_poc_lsb;
    else if (poc_lsb > prev_poc_lsb && poc_lsb - prev_poc_lsb > max_poc_lsb / 2)
        poc_msb = prev_poc_msb - max_poc_lsb;
    else
        poc_msb = prev_poc_msb;

    if (nal_unit_type == H265_NAL_BLA_W_LP ||
        nal_unit_type == H265_NAL_BLA_W_RADL ||
        nal_unit_type == H265_NAL_BLA_N_LP)
        poc_msb = 0;

    return poc_msb + poc_lsb;
}

/* ------------------------------------------------------------------ */
/* VUI / HRD                                                           */
/* ------------------------------------------------------------------ */
static void skip_hrd(H265BitReader *br, int max_sub_layers)
{
    int nal_hrd = h265_br_get_bits1(br);
    int vcl_hrd = h265_br_get_bits1(br);
    int sub_pic = 0, i, j;
    int cpb_cnt[H265_MAX_SUB_LAYERS] = { 0 };
    int low_delay[H265_MAX_SUB_LAYERS] = { 0 };

    if (nal_hrd || vcl_hrd) {
        sub_pic = h265_br_get_bits1(br);
        if (sub_pic) {
            h265_br_skip(br, 8 + 5 + 1 + 5);
        }
        h265_br_skip(br, 4 + 4);
        if (sub_pic)
            h265_br_skip(br, 4);
        h265_br_skip(br, 5 + 5 + 5);
    }

    for (i = 0; i < max_sub_layers; i++) {
        int fixed_general = h265_br_get_bits1(br);
        int fixed_within = fixed_general;
        if (!fixed_general)
            fixed_within = h265_br_get_bits1(br);
        if (fixed_within) {
            h265_br_get_ue(br);        /* elemental_duration_in_tc_minus1 */
            low_delay[i] = 0;
        } else {
            low_delay[i] = h265_br_get_bits1(br);
        }
        if (!low_delay[i]) {
            cpb_cnt[i] = h265_br_get_ue(br);
            if (cpb_cnt[i] > 31)
                cpb_cnt[i] = 31;
        }
    }

    for (i = 0; i < max_sub_layers; i++) {
        for (j = 0; j <= cpb_cnt[i]; j++) {
            if (nal_hrd) {
                h265_br_get_ue(br);
                h265_br_get_ue(br);
                if (sub_pic) {
                    h265_br_get_ue(br);
                    h265_br_get_ue(br);
                }
                h265_br_get_bits1(br);
            }
            if (vcl_hrd) {
                h265_br_get_ue(br);
                h265_br_get_ue(br);
                if (sub_pic) {
                    h265_br_get_ue(br);
                    h265_br_get_ue(br);
                }
                h265_br_get_bits1(br);
            }
        }
    }
}

static void parse_vui(H265BitReader *br, H265SPS *sps, int max_sub_layers)
{
    int present;

    present = h265_br_get_bits1(br);
    if (present) {
        int aspect = h265_br_get_bits(br, 8);
        if (aspect == 255) {
            sps->sar_num = h265_br_get_bits(br, 16);
            sps->sar_den = h265_br_get_bits(br, 16);
        } else {
            static const int sar_tab[17][2] = {
                {0,0},{1,1},{12,11},{10,11},{16,11},{40,33},{24,11},{20,11},
                {32,11},{80,33},{18,11},{15,11},{64,33},{160,99},{4,3},{3,2},{2,1}
            };
            if (aspect < 17) {
                sps->sar_num = sar_tab[aspect][0];
                sps->sar_den = sar_tab[aspect][1];
            }
        }
    }

    if (h265_br_get_bits1(br))        /* overscan_info_present_flag */
        h265_br_get_bits1(br);

    if (h265_br_get_bits1(br)) {      /* video_signal_type_present_flag */
        h265_br_skip(br, 3);          /* video_format */
        sps->video_full_range = h265_br_get_bits1(br);
        if (h265_br_get_bits1(br)) {  /* colour_description_present_flag */
            sps->colour_primaries       = h265_br_get_bits(br, 8);
            sps->transfer_characteristics = h265_br_get_bits(br, 8);
            sps->matrix_coeffs          = h265_br_get_bits(br, 8);
        }
    }

    if (h265_br_get_bits1(br)) {      /* chroma_loc_info_present_flag */
        sps->chroma_loc_type_top    = h265_br_get_ue(br);
        sps->chroma_loc_type_bottom = h265_br_get_ue(br);
    }

    sps->neutral_chroma_indication = h265_br_get_bits1(br);
    sps->field_seq_flag = h265_br_get_bits1(br);
    sps->frame_field_info_present = h265_br_get_bits1(br);

    if (h265_br_get_bits1(br)) {      /* default_display_window_flag */
        h265_br_get_ue(br);
        h265_br_get_ue(br);
        h265_br_get_ue(br);
        h265_br_get_ue(br);
    }

    if (h265_br_get_bits1(br)) {      /* vui_timing_info_present_flag */
        sps->vui_num_units_in_tick = h265_br_get_bits(br, 32);
        sps->vui_time_scale         = h265_br_get_bits(br, 32);
        if (h265_br_get_bits1(br))    /* poc_proportional_to_timing_flag */
            h265_br_get_ue(br);
        if (h265_br_get_bits1(br))    /* vui_hrd_parameters_present_flag */
            skip_hrd(br, max_sub_layers);
    }

    if (h265_br_get_bits1(br)) {      /* bitstream_restriction_flag */
        h265_br_get_bits1(br);
        h265_br_get_bits1(br);
        h265_br_get_bits1(br);
        h265_br_get_ue(br);
        h265_br_get_ue(br);
        h265_br_get_ue(br);
        h265_br_get_ue(br);
        h265_br_get_ue(br);
    }

    if (sps->vui_num_units_in_tick && sps->vui_time_scale) {
        sps->fps_num = (int)sps->vui_time_scale;
        sps->fps_den = (int)sps->vui_num_units_in_tick;
    }
}

/* ------------------------------------------------------------------ */
/* VPS                                                                 */
/* ------------------------------------------------------------------ */
int h265_parse_vps(H265Decoder *h, const uint8_t *rbsp, int size)
{
    H265BitReader br;
    H265VPS *vps;
    unsigned id;
    int max_sub_layers, i, start;

    h265_br_init(&br, rbsp, size * 8);
    h265_br_skip(&br, 16);            /* nal header */
    id = h265_br_get_bits(&br, 4);
    if (id >= H265_MAX_VPS_COUNT)
        return -1;

    vps = (H265VPS *)calloc(1, sizeof(*vps));
    if (!vps)
        return -1;

    vps->vps_id = id;
    h265_br_skip(&br, 2);             /* base_layer_internal/external */
    vps->vps_max_layers = h265_br_get_bits(&br, 6) + 1;
    max_sub_layers = h265_br_get_bits(&br, 3) + 1;
    vps->vps_max_sub_layers = max_sub_layers;
    vps->vps_temporal_id_nesting_flag = h265_br_get_bits1(&br);
    h265_br_skip(&br, 16);

    if (max_sub_layers > H265_MAX_SUB_LAYERS) {
        free(vps);
        return -1;
    }

    {
        uint8_t pf[8] = {0}, lf[8] = {0};
        parse_ptl_full(&br, &vps->ptl, 1, max_sub_layers, pf, lf);
    }

    vps->vps_sub_layer_ordering_info_present_flag = h265_br_get_bits1(&br);
    start = vps->vps_sub_layer_ordering_info_present_flag ? 0 : max_sub_layers - 1;
    for (i = start; i < max_sub_layers; i++) {
        vps->vps_max_dec_pic_buffering[i] = h265_br_get_ue(&br) + 1;
        vps->vps_num_reorder_pics[i] = h265_br_get_ue(&br);
        vps->vps_max_latency_increase[i] = h265_br_get_ue(&br);
    }
    for (i = 0; i < start; i++) {
        vps->vps_max_dec_pic_buffering[i] = vps->vps_max_dec_pic_buffering[start];
        vps->vps_num_reorder_pics[i] = vps->vps_num_reorder_pics[start];
        vps->vps_max_latency_increase[i] = vps->vps_max_latency_increase[start];
    }

    {
        int max_layer_id = h265_br_get_bits(&br, 6);
        int num_layer_sets = h265_br_get_ue(&br) + 1;
        if (num_layer_sets > 1)
            h265_br_skip(&br, (num_layer_sets - 1) * (max_layer_id + 1));
        if (h265_br_get_bits1(&br)) {           /* timing info */
            h265_br_skip(&br, 32 + 32);
            if (h265_br_get_bits1(&br))
                h265_br_get_ue(&br);
            {
                int num_hrd = h265_br_get_ue(&br);
                int hi;
                for (hi = 0; hi < num_hrd; hi++) {
                    h265_br_get_ue(&br);
                    if (hi)
                        h265_br_get_bits1(&br);
                    skip_hrd(&br, max_sub_layers);
                }
            }
        }
    }

    vps->layer_id_in_nuh[0] = 0;
    vps->layer_id_in_nuh[1] = 1;
    if (vps->vps_max_layers > 1 && h265_br_get_bits1(&br)) {
        /* Minimal MV-HEVC VPS extension parse: enough to identify an
         * auxiliary (alpha/depth) layer. */
        H265PTL ptl;
        uint8_t pf[8] = {0}, lf[8] = {0};
        int splitting, mask, num_types = 0, dim_len[16], n = 0, i2;
        int view_id_len;

        parse_ptl_full(&br, &ptl, 0, max_sub_layers, pf, lf);
        splitting = h265_br_get_bits1(&br);
        mask = (int)h265_br_get_bits(&br, 16);
        for (i2 = 0; i2 < 16; i2++)
            if (mask & (1 << i2))
                num_types++;
        if (num_types > 0 && num_types <= 16) {
            for (i2 = 0; i2 < num_types - splitting; i2++) {
                dim_len[i2] = h265_br_get_bits(&br, 3) + 1;
                n += dim_len[i2];
            }
            if (splitting && num_types > 0)
                dim_len[num_types - 1] = 5 - n;
            if (h265_br_get_bits1(&br))
                vps->layer_id_in_nuh[1] = h265_br_get_bits(&br, 6);
            if (!splitting) {
                for (i2 = 0; i2 < num_types; i2++)
                    h265_br_skip(&br, dim_len[i2]);
            }
            view_id_len = h265_br_get_bits(&br, 4);
            if (view_id_len > 0) {
                n = (mask & (1 << 14)) ? 2 : 1;   /* MULTIVIEW */
                for (i2 = 0; i2 < n; i2++)
                    vps->view_id[i2] = h265_br_get_bits(&br, view_id_len);
            }
            vps->multilayer = 1;
            vps->multilayer_auxiliary = !!(mask & (1 << 12)) ||
                (view_id_len > 0 && vps->view_id[0] == vps->view_id[1]);
        }
    }

    if (br.error) {
        free(vps);
        return -1;
    }

    vps->valid = 1;
    vps->refcount = 1;
    h265_vps_unref(&h->ps.vps_list[id]);
    h->ps.vps_list[id] = vps;
    return 0;
}

/* ------------------------------------------------------------------ */
/* SPS                                                                 */
/* ------------------------------------------------------------------ */
static int parse_window(H265BitReader *br, H265Window *win, int chroma_format,
                        int width, int height)
{
    static const int sub_w[4] = { 1, 2, 2, 1 };
    static const int sub_h[4] = { 1, 2, 1, 1 };
    int64_t left, right, top, bottom;

    left   = (int64_t)h265_br_get_ue(br) * sub_w[chroma_format];
    right  = (int64_t)h265_br_get_ue(br) * sub_w[chroma_format];
    top    = (int64_t)h265_br_get_ue(br) * sub_h[chroma_format];
    bottom = (int64_t)h265_br_get_ue(br) * sub_h[chroma_format];

    if (left < 0 || right < 0 || top < 0 || bottom < 0 ||
        width <= left + right || height <= top + bottom) {
        memset(win, 0, sizeof(*win));
        return -1;
    }
    win->left_offset = (unsigned)left;
    win->right_offset = (unsigned)right;
    win->top_offset = (unsigned)top;
    win->bottom_offset = (unsigned)bottom;
    return 0;
}

int h265_parse_sps(H265Decoder *h, const uint8_t *rbsp, int size)
{
    H265BitReader br;
    H265SPS *sps;
    H265VPS *vps = NULL;
    unsigned sps_id, vps_id;
    int max_sub_layers, i, start, ret = 0;
    int bit_depth_chroma;

    h265_br_init(&br, rbsp, size * 8);
    h265_br_skip(&br, 16);

    vps_id = h265_br_get_bits(&br, 4);
    if (vps_id >= H265_MAX_VPS_COUNT)
        return -1;
    max_sub_layers = h265_br_get_bits(&br, 3) + 1;
    if (max_sub_layers > H265_MAX_SUB_LAYERS)
        return -1;
    h265_br_skip(&br, 1);             /* temporal_id_nesting_flag */

    sps = (H265SPS *)calloc(1, sizeof(*sps));
    if (!sps)
        return -1;
    sps->vps_id = vps_id;
    sps->max_sub_layers = max_sub_layers;
    sps->colour_primaries = 2;          /* unspecified */
    sps->transfer_characteristics = 2;
    sps->matrix_coeffs = 2;
    sps->chroma_loc_type_top = -1;
    sps->chroma_loc_type_bottom = -1;

    {
        uint8_t pf[8] = {0}, lf[8] = {0};
        parse_ptl_full(&br, &sps->ptl, 1, max_sub_layers, pf, lf);
    }

    if (vps_id < H265_MAX_VPS_COUNT)
        vps = h->ps.vps_list[vps_id];

    sps_id = h265_br_get_ue(&br);
    if (sps_id >= H265_MAX_SPS_COUNT) {
        free(sps);
        return -1;
    }

    sps->chroma_format_idc = h265_br_get_ue(&br);
    if (sps->chroma_format_idc > 3) {
        free(sps);
        return -1;
    }
    if (sps->chroma_format_idc == 3)
        sps->separate_colour_plane = h265_br_get_bits1(&br);
    if (sps->separate_colour_plane) {
        /* Each picture carries one colour plane and is decoded with the
         * monochrome syntax, as FFmpeg does.  colour_plane_id tells the caller
         * which plane the returned gray picture contains. */
        sps->chroma_format_idc = 0;
    }
    sps->width  = h265_br_get_ue(&br);
    sps->height = h265_br_get_ue(&br);
    if (sps->width <= 0 || sps->height <= 0 || sps->width > 16384 ||
        sps->height > 16384) {
        free(sps);
        return -1;
    }

    sps->conformance_window = h265_br_get_bits1(&br);
    if (sps->conformance_window)
        parse_window(&br, &sps->conf_win, sps->chroma_format_idc,
                     sps->width, sps->height);

    sps->bit_depth = h265_br_get_ue(&br) + 8;
    if (sps->bit_depth > 12) {
        /* HEVC profiles defined by FFmpeg's native decoder cover 8/9/10/12-bit. */
        free(sps);
        return -1;
    }
    bit_depth_chroma = h265_br_get_ue(&br) + 8;
    if (bit_depth_chroma > 16 ||
        (sps->chroma_format_idc && bit_depth_chroma != sps->bit_depth)) {
        free(sps);
        return -1;
    }
    sps->bit_depth_chroma = bit_depth_chroma;
    sps->pixel_shift = sps->bit_depth > 8;

    sps->output_window = sps->conf_win;

    sps->log2_max_poc_lsb = h265_br_get_ue(&br) + 4;
    if (sps->log2_max_poc_lsb > 16) {
        free(sps);
        return -1;
    }

    sps->temporal_layer[0].max_dec_pic_buffering = 1;
    sps->temporal_layer[0].num_reorder_pics = 0;
    sps->temporal_layer[0].max_latency_increase = 0;

    start = h265_br_get_bits1(&br) ? 0 : max_sub_layers - 1;
    for (i = start; i < max_sub_layers; i++) {
        sps->temporal_layer[i].max_dec_pic_buffering = h265_br_get_ue(&br) + 1;
        sps->temporal_layer[i].num_reorder_pics      = h265_br_get_ue(&br);
        sps->temporal_layer[i].max_latency_increase  = h265_br_get_ue(&br);
        if (sps->temporal_layer[i].max_dec_pic_buffering > H265_MAX_DPB_SIZE)
            ret = -1;
        if (sps->temporal_layer[i].num_reorder_pics >=
            sps->temporal_layer[i].max_dec_pic_buffering)
            sps->temporal_layer[i].max_dec_pic_buffering =
                sps->temporal_layer[i].num_reorder_pics + 1;
    }
    for (i = 0; i < start; i++)
        sps->temporal_layer[i] = sps->temporal_layer[start];

    sps->log2_min_cb_size = h265_br_get_ue(&br) + 3;
    sps->log2_diff_max_min_coding_block_size = h265_br_get_ue(&br);
    sps->log2_min_tb_size = h265_br_get_ue(&br) + 2;
    sps->log2_diff_max_min_transform_block_size = h265_br_get_ue(&br);
    sps->log2_max_trafo_size = sps->log2_diff_max_min_transform_block_size +
                               sps->log2_min_tb_size;

    if (sps->log2_min_cb_size < 3 || sps->log2_min_cb_size > 30 ||
        sps->log2_diff_max_min_coding_block_size > 30 ||
        sps->log2_min_tb_size >= sps->log2_min_cb_size ||
        sps->log2_min_tb_size < 2 ||
        sps->log2_diff_max_min_transform_block_size > 30) {
        free(sps);
        return -1;
    }

    sps->max_transform_hierarchy_depth_inter = h265_br_get_ue(&br);
    sps->max_transform_hierarchy_depth_intra = h265_br_get_ue(&br);

    sps->scaling_list_enabled = h265_br_get_bits1(&br);
    if (sps->scaling_list_enabled) {
        set_default_scaling_list(&sps->scaling_list);
        if (h265_br_get_bits1(&br)) {
            if (scaling_list_data(&br, &sps->scaling_list,
                                  sps->chroma_format_idc) < 0) {
                free(sps);
                return -1;
            }
        }
    }

    sps->amp_enabled = h265_br_get_bits1(&br);
    sps->sao_enabled = h265_br_get_bits1(&br);

    sps->pcm_enabled = h265_br_get_bits1(&br);
    if (sps->pcm_enabled) {
        sps->pcm.bit_depth = h265_br_get_bits(&br, 4) + 1;
        sps->pcm.bit_depth_chroma = h265_br_get_bits(&br, 4) + 1;
        sps->pcm.log2_min_pcm_cb_size = h265_br_get_ue(&br) + 3;
        sps->pcm.log2_max_pcm_cb_size = sps->pcm.log2_min_pcm_cb_size +
                                        h265_br_get_ue(&br);
        if (H265_MAX(sps->pcm.bit_depth, sps->pcm.bit_depth_chroma) >
            sps->bit_depth) {
            free(sps);
            return -1;
        }
        sps->pcm_loop_filter_disabled = h265_br_get_bits1(&br);
    }

    sps->nb_st_rps = h265_br_get_ue(&br);
    if (sps->nb_st_rps > H265_MAX_ST_RPS) {
        free(sps);
        return -1;
    }
    for (i = 0; i < sps->nb_st_rps; i++) {
        if (h265_parse_short_term_rps(&br, &sps->st_rps[i], sps, 0) < 0) {
            free(sps);
            return -1;
        }
    }

    sps->long_term_ref_pics_present = h265_br_get_bits1(&br);
    if (sps->long_term_ref_pics_present) {
        sps->num_long_term_ref_pics_sps = h265_br_get_ue(&br);
        if (sps->num_long_term_ref_pics_sps > H265_MAX_LONG_TERM_REF_PICS) {
            free(sps);
            return -1;
        }
        sps->used_by_curr_pic_lt = 0;
        for (i = 0; i < sps->num_long_term_ref_pics_sps; i++) {
            sps->lt_ref_pic_poc_lsb_sps[i] =
                h265_br_get_bits(&br, sps->log2_max_poc_lsb);
            sps->used_by_curr_pic_lt |= h265_br_get_bits1(&br) << i;
        }
    }

    sps->temporal_mvp_enabled = h265_br_get_bits1(&br);
    sps->strong_intra_smoothing_enabled = h265_br_get_bits1(&br);

    sps->vui_present = h265_br_get_bits1(&br);
    if (sps->vui_present)
        parse_vui(&br, sps, max_sub_layers);

    if (h265_br_get_bits1(&br)) {     /* extension_present_flag */
        int range_extension = h265_br_get_bits1(&br);
        int multilayer = h265_br_get_bits1(&br);
        int sps3d = h265_br_get_bits1(&br);
        int scc = h265_br_get_bits1(&br);
        h265_br_skip(&br, 4);
        if (range_extension) {
            sps->range_extension = 1;
            sps->transform_skip_rotation_enabled = h265_br_get_bits1(&br);
            sps->transform_skip_context_enabled  = h265_br_get_bits1(&br);
            sps->implicit_rdpcm_enabled          = h265_br_get_bits1(&br);
            sps->explicit_rdpcm_enabled          = h265_br_get_bits1(&br);
            h265_br_get_bits1(&br);   /* extended_precision_processing */
            sps->intra_smoothing_disabled        = h265_br_get_bits1(&br);
            h265_br_get_bits1(&br);   /* high_precision_offsets_enabled */
            sps->persistent_rice_adaptation_enabled = h265_br_get_bits1(&br);
            sps->cabac_bypass_alignment_enabled  = h265_br_get_bits1(&br);
        }
        if (multilayer)
            h265_br_get_bits1(&br);
        if (sps3d) {
            for (i = 0; i < 2; i++) {
                h265_br_get_bits1(&br);
                h265_br_get_bits1(&br);
                if (i == 0) {
                    h265_br_get_ue(&br);
                    h265_br_skip(&br, 4);
                } else {
                    h265_br_get_ue(&br);
                    h265_br_skip(&br, 5);
                }
            }
        }
        if (scc) {
            sps->motion_vector_resolution_control_idc = 0;
            h265_br_get_bits1(&br);   /* curr_pic_ref_enabled */
            if (h265_br_get_bits1(&br)) {  /* palette_mode_enabled */
                h265_br_get_ue(&br);
                h265_br_get_ue(&br);
                if (h265_br_get_bits1(&br)) {
                    int n = h265_br_get_ue(&br) + 1;
                    int comp, k;
                    for (comp = 0; comp < (sps->chroma_format_idc ? 3 : 1); comp++)
                        for (k = 0; k < n; k++)
                            h265_br_get_bits(&br, 8);
                }
            }
            sps->motion_vector_resolution_control_idc = h265_br_get_bits(&br, 2);
            h265_br_get_bits1(&br);
        }
    }

    sps->log2_ctb_size = sps->log2_min_cb_size +
                         sps->log2_diff_max_min_coding_block_size;
    sps->log2_min_pu_size = sps->log2_min_cb_size - 1;
    if (sps->log2_ctb_size > 6 || sps->log2_ctb_size < 4 ||
        sps->log2_max_trafo_size > H265_MIN(sps->log2_ctb_size, 5) ||
        sps->max_transform_hierarchy_depth_inter >
            sps->log2_ctb_size - sps->log2_min_tb_size ||
        sps->max_transform_hierarchy_depth_intra >
            sps->log2_ctb_size - sps->log2_min_tb_size) {
        free(sps);
        return -1;
    }

    sps->ctb_width  = (sps->width + (1 << sps->log2_ctb_size) - 1) >>
                      sps->log2_ctb_size;
    sps->ctb_height = (sps->height + (1 << sps->log2_ctb_size) - 1) >>
                      sps->log2_ctb_size;
    sps->ctb_size = sps->ctb_width * sps->ctb_height;
    sps->min_cb_width  = sps->width >> sps->log2_min_cb_size;
    sps->min_cb_height = sps->height >> sps->log2_min_cb_size;
    sps->min_tb_width  = sps->width >> sps->log2_min_tb_size;
    sps->min_tb_height = sps->height >> sps->log2_min_tb_size;
    sps->min_pu_width  = sps->width >> sps->log2_min_pu_size;
    sps->min_pu_height = sps->height >> sps->log2_min_pu_size;
    sps->tb_mask = (1 << (sps->log2_ctb_size - sps->log2_min_tb_size)) - 1;
    sps->qp_bd_offset = 6 * (sps->bit_depth - 8);

    sps->hshift[0] = 0;
    sps->vshift[0] = 0;
    if (sps->chroma_format_idc == 1) {
        sps->hshift[1] = sps->hshift[2] = 1;
        sps->vshift[1] = sps->vshift[2] = 1;
    } else if (sps->chroma_format_idc == 2) {
        sps->hshift[1] = sps->hshift[2] = 1;
        sps->vshift[1] = sps->vshift[2] = 0;
    } else {
        sps->hshift[1] = sps->hshift[2] = 0;
        sps->vshift[1] = sps->vshift[2] = 0;
    }

    if (br.error || ret < 0) {
        free(sps);
        return -1;
    }

    sps->valid = 1;
    sps->refcount = 1;
    h265_sps_unref(&h->ps.sps_list[sps_id]);
    h->ps.sps_list[sps_id] = sps;
    /* Make the newest SPS available to prefix SEI parsed before the first
     * slice header without disturbing the SPS of an in-flight picture. */
    h->sei_sps = sps;
    (void)vps;
    return 0;
}

/* ------------------------------------------------------------------ */
/* PPS                                                                 */
/* ------------------------------------------------------------------ */
static int setup_pps(H265PPS *pps, const H265SPS *sps)
{
    int log2_diff;
    int pic_area_in_ctbs;
    int i, j, x, y, ctb_addr_rs, tile_id;
    int num_tile_columns = pps->num_tile_columns;
    int num_tile_rows = pps->num_tile_rows;

    pps->col_bd = (unsigned *)malloc((num_tile_columns + 1) * sizeof(unsigned));
    pps->row_bd = (unsigned *)malloc((num_tile_rows + 1) * sizeof(unsigned));
    pps->col_idxX = (int *)malloc(sps->ctb_width * sizeof(int));
    if (!pps->col_bd || !pps->row_bd || !pps->col_idxX)
        return -1;

    if (pps->uniform_spacing_flag) {
        if (!pps->column_width) {
            pps->column_width = (unsigned *)malloc(num_tile_columns * sizeof(unsigned));
            pps->row_height   = (unsigned *)malloc(num_tile_rows * sizeof(unsigned));
        }
        if (!pps->column_width || !pps->row_height)
            return -1;
        for (i = 0; i < num_tile_columns; i++)
            pps->column_width[i] =
                ((i + 1) * sps->ctb_width) / num_tile_columns -
                (i * sps->ctb_width) / num_tile_columns;
        for (i = 0; i < num_tile_rows; i++)
            pps->row_height[i] =
                ((i + 1) * sps->ctb_height) / num_tile_rows -
                (i * sps->ctb_height) / num_tile_rows;
    }

    pps->col_bd[0] = 0;
    for (i = 0; i < num_tile_columns; i++)
        pps->col_bd[i + 1] = pps->col_bd[i] + pps->column_width[i];
    pps->row_bd[0] = 0;
    for (i = 0; i < num_tile_rows; i++)
        pps->row_bd[i + 1] = pps->row_bd[i] + pps->row_height[i];

    for (i = 0, j = 0; i < sps->ctb_width; i++) {
        if (i > (int)pps->col_bd[j])
            j++;
        pps->col_idxX[i] = j;
    }

    pic_area_in_ctbs = sps->ctb_width * sps->ctb_height;
    pps->ctb_addr_rs_to_ts = (int *)malloc(pic_area_in_ctbs * sizeof(int));
    pps->ctb_addr_ts_to_rs = (int *)malloc(pic_area_in_ctbs * sizeof(int));
    pps->tile_id = (int *)malloc(pic_area_in_ctbs * sizeof(int));
    pps->min_tb_addr_zs_tab =
        (int *)malloc((sps->tb_mask + 2) * (sps->tb_mask + 2) * sizeof(int));
    if (!pps->ctb_addr_rs_to_ts || !pps->ctb_addr_ts_to_rs || !pps->tile_id ||
        !pps->min_tb_addr_zs_tab)
        return -1;

    for (ctb_addr_rs = 0; ctb_addr_rs < pic_area_in_ctbs; ctb_addr_rs++) {
        int tb_x = ctb_addr_rs % sps->ctb_width;
        int tb_y = ctb_addr_rs / sps->ctb_width;
        int tile_x = 0, tile_y = 0, val = 0;

        for (i = 0; i < num_tile_columns; i++) {
            if (tb_x < (int)pps->col_bd[i + 1]) {
                tile_x = i;
                break;
            }
        }
        for (i = 0; i < num_tile_rows; i++) {
            if (tb_y < (int)pps->row_bd[i + 1]) {
                tile_y = i;
                break;
            }
        }
        for (i = 0; i < tile_x; i++)
            val += pps->row_height[tile_y] * pps->column_width[i];
        for (i = 0; i < tile_y; i++)
            val += sps->ctb_width * pps->row_height[i];
        val += (tb_y - pps->row_bd[tile_y]) * pps->column_width[tile_x] +
               tb_x - pps->col_bd[tile_x];

        pps->ctb_addr_rs_to_ts[ctb_addr_rs] = val;
        pps->ctb_addr_ts_to_rs[val] = ctb_addr_rs;
    }

    for (j = 0, tile_id = 0; j < num_tile_rows; j++)
        for (i = 0; i < num_tile_columns; i++, tile_id++)
            for (y = pps->row_bd[j]; y < (int)pps->row_bd[j + 1]; y++)
                for (x = pps->col_bd[i]; x < (int)pps->col_bd[i + 1]; x++)
                    pps->tile_id[pps->ctb_addr_rs_to_ts[y * sps->ctb_width + x]] =
                        tile_id;

    pps->tile_pos_rs = (int *)malloc(tile_id * sizeof(int));
    if (!pps->tile_pos_rs)
        return -1;
    for (j = 0; j < num_tile_rows; j++)
        for (i = 0; i < num_tile_columns; i++)
            pps->tile_pos_rs[j * num_tile_columns + i] =
                pps->row_bd[j] * sps->ctb_width + pps->col_bd[i];

    log2_diff = sps->log2_ctb_size - sps->log2_min_tb_size;
    pps->min_tb_addr_zs = &pps->min_tb_addr_zs_tab[1 * (sps->tb_mask + 2) + 1];
    for (y = 0; y < sps->tb_mask + 2; y++) {
        pps->min_tb_addr_zs_tab[y * (sps->tb_mask + 2)] = -1;
        pps->min_tb_addr_zs_tab[y] = -1;
    }
    for (y = 0; y < sps->tb_mask + 1; y++) {
        for (x = 0; x < sps->tb_mask + 1; x++) {
            int tb_x = x >> log2_diff;
            int tb_y = y >> log2_diff;
            int rs = sps->ctb_width * tb_y + tb_x;
            int val = pps->ctb_addr_rs_to_ts[rs] << (log2_diff * 2);
            for (i = 0; i < log2_diff; i++) {
                int m = 1 << i;
                val += (m & x ? m * m : 0) + (m & y ? 2 * m * m : 0);
            }
            pps->min_tb_addr_zs[y * (sps->tb_mask + 2) + x] = val;
        }
    }
    return 0;
}

int h265_parse_pps(H265Decoder *h, const uint8_t *rbsp, int size)
{
    H265BitReader br;
    H265PPS *pps;
    H265SPS *sps = NULL;
    unsigned pps_id, sps_id;
    int i;

    h265_br_init(&br, rbsp, size * 8);
    h265_br_skip(&br, 16);

    pps_id = h265_br_get_ue(&br);
    if (pps_id >= H265_MAX_PPS_COUNT)
        return -1;

    pps = (H265PPS *)calloc(1, sizeof(*pps));
    if (!pps)
        return -1;

    pps->pps_id = pps_id;
    pps->loop_filter_across_tiles_enabled_flag = 1;
    pps->num_tile_columns = 1;
    pps->num_tile_rows = 1;
    pps->uniform_spacing_flag = 1;
    pps->log2_max_transform_skip_block_size = 2;
    pps->log2_parallel_merge_level = 2;

    sps_id = h265_br_get_ue(&br);
    if (sps_id >= H265_MAX_SPS_COUNT || !h->ps.sps_list[sps_id]) {
        free(pps);
        return -1;
    }
    sps = h->ps.sps_list[sps_id];
    pps->sps_id = sps_id;
    pps->sps = h265_sps_ref(sps);

    pps->dependent_slice_segments_enabled_flag = h265_br_get_bits1(&br);
    pps->output_flag_present_flag = h265_br_get_bits1(&br);
    pps->num_extra_slice_header_bits = h265_br_get_bits(&br, 3);
    pps->sign_data_hiding_flag = h265_br_get_bits1(&br);
    pps->cabac_init_present_flag = h265_br_get_bits1(&br);
    pps->num_ref_idx_l0_default_active = h265_br_get_ue(&br) + 1;
    pps->num_ref_idx_l1_default_active = h265_br_get_ue(&br) + 1;
    if (pps->num_ref_idx_l0_default_active >= H265_MAX_REFS ||
        pps->num_ref_idx_l1_default_active >= H265_MAX_REFS) {
        goto fail;
    }
    pps->pic_init_qp_minus26 = h265_br_get_se(&br);
    pps->constrained_intra_pred_flag = h265_br_get_bits1(&br);
    pps->transform_skip_enabled_flag = h265_br_get_bits1(&br);
    pps->cu_qp_delta_enabled_flag = h265_br_get_bits1(&br);
    if (pps->cu_qp_delta_enabled_flag)
        pps->diff_cu_qp_delta_depth = h265_br_get_ue(&br);
    if (pps->diff_cu_qp_delta_depth < 0 ||
        pps->diff_cu_qp_delta_depth > sps->log2_diff_max_min_coding_block_size)
        goto fail;
    pps->cb_qp_offset = h265_br_get_se(&br);
    pps->cr_qp_offset = h265_br_get_se(&br);
    if (pps->cb_qp_offset < -12 || pps->cb_qp_offset > 12 ||
        pps->cr_qp_offset < -12 || pps->cr_qp_offset > 12)
        goto fail;
    pps->pic_slice_level_chroma_qp_offsets_present_flag = h265_br_get_bits1(&br);
    pps->weighted_pred_flag = h265_br_get_bits1(&br);
    pps->weighted_bipred_flag = h265_br_get_bits1(&br);
    pps->transquant_bypass_enable_flag = h265_br_get_bits1(&br);
    pps->tiles_enabled_flag = h265_br_get_bits1(&br);
    pps->entropy_coding_sync_enabled_flag = h265_br_get_bits1(&br);

    if (pps->tiles_enabled_flag) {
        int cols = h265_br_get_ue(&br) + 1;
        int rows = h265_br_get_ue(&br) + 1;
        if (cols < 1 || cols > sps->ctb_width || rows < 1 ||
            rows > sps->ctb_height)
            goto fail;
        pps->num_tile_columns = cols;
        pps->num_tile_rows = rows;
        pps->column_width = (unsigned *)malloc(cols * sizeof(unsigned));
        pps->row_height = (unsigned *)malloc(rows * sizeof(unsigned));
        if (!pps->column_width || !pps->row_height)
            goto fail;
        pps->uniform_spacing_flag = h265_br_get_bits1(&br);
        if (!pps->uniform_spacing_flag) {
            uint64_t sum = 0;
            for (i = 0; i < cols - 1; i++) {
                pps->column_width[i] = h265_br_get_ue(&br) + 1;
                sum += pps->column_width[i];
            }
            if (sum >= (uint64_t)sps->ctb_width)
                goto fail;
            pps->column_width[cols - 1] = sps->ctb_width - (unsigned)sum;
            sum = 0;
            for (i = 0; i < rows - 1; i++) {
                pps->row_height[i] = h265_br_get_ue(&br) + 1;
                sum += pps->row_height[i];
            }
            if (sum >= (uint64_t)sps->ctb_height)
                goto fail;
            pps->row_height[rows - 1] = sps->ctb_height - (unsigned)sum;
        }
        pps->loop_filter_across_tiles_enabled_flag = h265_br_get_bits1(&br);
    }

    pps->seq_loop_filter_across_slices_enabled_flag = h265_br_get_bits1(&br);
    pps->deblocking_filter_control_present_flag = h265_br_get_bits1(&br);
    if (pps->deblocking_filter_control_present_flag) {
        pps->deblocking_filter_override_enabled_flag = h265_br_get_bits1(&br);
        pps->disable_dbf = h265_br_get_bits1(&br);
        if (!pps->disable_dbf) {
            int beta = h265_br_get_se(&br);
            int tc = h265_br_get_se(&br);
            if (beta < -6 || beta > 6 || tc < -6 || tc > 6)
                goto fail;
            pps->beta_offset = 2 * beta;
            pps->tc_offset = 2 * tc;
        }
    }

    pps->scaling_list_data_present_flag = h265_br_get_bits1(&br);
    if (pps->scaling_list_data_present_flag) {
        set_default_scaling_list(&pps->scaling_list);
        if (scaling_list_data(&br, &pps->scaling_list, sps->chroma_format_idc) < 0)
            goto fail;
    }

    pps->lists_modification_present_flag = h265_br_get_bits1(&br);
    {
        unsigned level = h265_br_get_ue(&br);
        if (level > (unsigned)sps->log2_ctb_size)
            goto fail;
        pps->log2_parallel_merge_level = level + 2;
    }
    pps->slice_header_extension_present_flag = h265_br_get_bits1(&br);

    pps->pps_extension_present_flag = h265_br_get_bits1(&br);
    if (pps->pps_extension_present_flag) {
        int range_ext = h265_br_get_bits1(&br);
        int multilayer = h265_br_get_bits1(&br);
        int pps3d = h265_br_get_bits1(&br);
        int scc = h265_br_get_bits1(&br);
        h265_br_skip(&br, 4);
        (void)pps3d;
        (void)multilayer;
        if (range_ext && sps->ptl.profile_idc >= 4) {
            pps->pps_range_extensions_flag = 1;
            if (pps->transform_skip_enabled_flag)
                pps->log2_max_transform_skip_block_size = h265_br_get_ue(&br) + 2;
            pps->cross_component_prediction_enabled_flag = h265_br_get_bits1(&br);
            pps->chroma_qp_offset_list_enabled_flag = h265_br_get_bits1(&br);
            if (pps->chroma_qp_offset_list_enabled_flag) {
                int k;
                pps->diff_cu_chroma_qp_offset_depth = h265_br_get_ue(&br);
                pps->chroma_qp_offset_list_len_minus1 = h265_br_get_ue(&br);
                if (pps->chroma_qp_offset_list_len_minus1 > 5)
                    goto fail;
                for (k = 0; k <= pps->chroma_qp_offset_list_len_minus1; k++) {
                    pps->cb_qp_offset_list[k] = (int8_t)h265_br_get_se(&br);
                    pps->cr_qp_offset_list[k] = (int8_t)h265_br_get_se(&br);
                }
            }
            pps->log2_sao_offset_scale_luma = h265_br_get_ue(&br);
            pps->log2_sao_offset_scale_chroma = h265_br_get_ue(&br);
            if (pps->log2_sao_offset_scale_luma >
                    H265_MAX(sps->bit_depth - 10, 0) ||
                pps->log2_sao_offset_scale_chroma >
                    H265_MAX(sps->bit_depth_chroma - 10, 0))
                goto fail;
        }
        if (scc) {
            pps->pps_scc_extension_flag = 1;
            pps->pps_curr_pic_ref_enabled_flag = h265_br_get_bits1(&br);
            if (h265_br_get_bits1(&br)) {      /* residual_adaptive_colour_transform */
                pps->pps_slice_act_qp_offsets_present_flag = h265_br_get_bits1(&br);
                pps->pps_act_y_qp_offset = (int8_t)(h265_br_get_se(&br) - 5);
                pps->pps_act_cb_qp_offset = (int8_t)(h265_br_get_se(&br) - 5);
                pps->pps_act_cr_qp_offset = (int8_t)(h265_br_get_se(&br) - 3);
            }
            if (h265_br_get_bits1(&br)) {      /* palette predictor initializers */
                int n = h265_br_get_ue(&br);
                if (n > 0) {
                    int mono = h265_br_get_bits1(&br);
                    int bd = h265_br_get_ue(&br) + 8;
                    int comp, k, ncomp = mono ? 1 : 3;
                    if (!mono)
                        h265_br_get_ue(&br);
                    for (comp = 0; comp < ncomp; comp++)
                        for (k = 0; k < n; k++)
                            h265_br_get_bits(&br, bd);
                }
            }
        }
    }

    if (setup_pps(pps, sps) < 0)
        goto fail;
    if (br.error)
        goto fail;

    pps->valid = 1;
    pps->refcount = 1;
    h265_pps_unref(&h->ps.pps_list[pps_id]);
    h->ps.pps_list[pps_id] = pps;
    return 0;

fail:
    h265_pps_unref(&pps);
    return -1;
}

static int sei_clip_int(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static void sei_parse_pic_timing(H265Decoder *h, const uint8_t *data, int size)
{
    H265SPS *sps = h->sei_sps ? h->sei_sps : h->sps;
    H265BitReader br;
    if (!sps || !sps->frame_field_info_present || size < 1)
        return;
    h265_br_init(&br, data, size * 8);
    h->sei.pic_struct = h265_br_get_bits(&br, 4);
}

static void sei_parse_recovery(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    h265_br_init(&br, data, size * 8);
    h->sei.recovery_present = 1;
    h->sei.recovery_poc_cnt = h265_br_get_se(&br);
    h->sei.recovery_exact_match = h265_br_get_bits1(&br);
    h->sei.recovery_broken_link = h265_br_get_bits1(&br);
}

static void sei_parse_timecode(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    int i, n;

    h265_br_init(&br, data, size * 8);
    n = h265_br_get_bits(&br, 2);
    h->sei.timecode_present = 1;
    h->sei.timecode_num_clock_ts = n;
    for (i = 0; i < n && i < 3; i++) {
        int flag = h265_br_get_bits1(&br);
        h->sei.timecode_clock_timestamp_flag[i] = flag;
        if (!flag)
            continue;
        h->sei.timecode_units_field_based[i] = h265_br_get_bits1(&br);
        h->sei.timecode_counting_type[i] = h265_br_get_bits(&br, 5);
        h->sei.timecode_full_timestamp[i] = h265_br_get_bits1(&br);
        h->sei.timecode_discontinuity[i] = h265_br_get_bits1(&br);
        h->sei.timecode_cnt_dropped[i] = h265_br_get_bits1(&br);
        h->sei.timecode_n_frames[i] = h265_br_get_bits(&br, 9);
        if (h->sei.timecode_full_timestamp[i]) {
            h->sei.timecode_seconds[i] = sei_clip_int(h265_br_get_bits(&br, 6), 0, 59);
            h->sei.timecode_minutes[i] = sei_clip_int(h265_br_get_bits(&br, 6), 0, 59);
            h->sei.timecode_hours[i] = sei_clip_int(h265_br_get_bits(&br, 5), 0, 23);
        } else {
            int seconds_flag = h265_br_get_bits1(&br);
            if (seconds_flag) {
                h->sei.timecode_seconds[i] = sei_clip_int(h265_br_get_bits(&br, 6), 0, 59);
                if (h265_br_get_bits1(&br)) {
                    h->sei.timecode_minutes[i] = sei_clip_int(h265_br_get_bits(&br, 6), 0, 59);
                    if (h265_br_get_bits1(&br))
                        h->sei.timecode_hours[i] = sei_clip_int(h265_br_get_bits(&br, 5), 0, 23);
                }
            }
        }
        {
            int len = h265_br_get_bits(&br, 5);
            if (len > 0)
                h265_br_skip(&br, len);
        }
    }
}

static void sei_parse_mastering(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    int i;
    if (size < 24)
        return;
    h265_br_init(&br, data, size * 8);
    for (i = 0; i < 3; i++) {
        h->sei.display_primaries[i][0] = h265_br_get_bits(&br, 16);
        h->sei.display_primaries[i][1] = h265_br_get_bits(&br, 16);
    }
    h->sei.white_point[0] = h265_br_get_bits(&br, 16);
    h->sei.white_point[1] = h265_br_get_bits(&br, 16);
    h->sei.max_luminance = h265_br_get_bits(&br, 32);
    h->sei.min_luminance = h265_br_get_bits(&br, 32);
    h->sei.mastering_present = 1;
}

static void sei_parse_content_light(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    if (size < 4)
        return;
    h265_br_init(&br, data, size * 8);
    h->sei.max_content_light_level = h265_br_get_bits(&br, 16);
    h->sei.max_pic_average_light_level = h265_br_get_bits(&br, 16);
    h->sei.content_light_present = 1;
}

static void sei_parse_alternative_transfer(H265Decoder *h, const uint8_t *data,
                                           int size)
{
    H265BitReader br;
    if (size < 1)
        return;
    h265_br_init(&br, data, size * 8);
    h->sei.preferred_transfer_characteristics = h265_br_get_bits(&br, 8);
    h->sei.alternative_transfer_present = 1;
}

static void sei_parse_orientation(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    int cancel;
    h265_br_init(&br, data, size * 8);
    cancel = h265_br_get_bits1(&br);
    h->sei.display_orientation_present = !cancel;
    if (!cancel) {
        h->sei.display_orientation_hflip = h265_br_get_bits1(&br);
        h->sei.display_orientation_vflip = h265_br_get_bits1(&br);
        h->sei.display_orientation_rotation = h265_br_get_bits(&br, 16);
    }
}

static void sei_parse_ambient(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    if (size < 8)
        return;
    h265_br_init(&br, data, size * 8);
    h->sei.ambient_illuminance = h265_br_get_bits(&br, 32);
    h->sei.ambient_light_x = h265_br_get_bits(&br, 16);
    h->sei.ambient_light_y = h265_br_get_bits(&br, 16);
    h->sei.ambient_present = h->sei.ambient_illuminance != 0;
}

static void sei_parse_frame_packing(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    h265_br_init(&br, data, size * 8);
    h265_br_get_ue(&br);                    /* arrangement_id */
    if (h265_br_get_bits1(&br)) {           /* arrangement_cancel_flag */
        h->sei.frame_packing_present = 0;
        return;
    }
    h->sei.frame_packing_present = 1;
    h->sei.frame_packing_type = h265_br_get_bits(&br, 7);
    h265_br_get_bits1(&br);                 /* quincunx_sampling_flag */
    h->sei.frame_packing_content = h265_br_get_bits(&br, 6);
    h265_br_skip(&br, 3);                   /* spatial/frame0/field_views */
    h->sei.frame_packing_current_frame_is_frame0 = h265_br_get_bits1(&br);
}

static void sei_parse_film_grain(H265Decoder *h, const uint8_t *data, int size)
{
    H265BitReader br;
    H265FilmGrainParams *p = &h->sei.film_grain;
    int cancel, c, i, j;

    h265_br_init(&br, data, size * 8);
    cancel = h265_br_get_bits1(&br);
    if (cancel) {
        p->present = 0;
        return;
    }
    memset(p, 0, sizeof(*p));
    p->model_id = h265_br_get_bits(&br, 2);
    if (h265_br_get_bits1(&br)) {          /* separate_colour_description */
        h265_br_skip(&br, 3 + 3 + 1 + 8 + 8 + 8);
    }
    p->blending_mode_id = h265_br_get_bits(&br, 2);
    p->log2_scale_factor = h265_br_get_bits(&br, 4);
    for (c = 0; c < 3; c++)
        p->component_model_present[c] = h265_br_get_bits1(&br);
    for (c = 0; c < 3; c++) {
        if (!p->component_model_present[c])
            continue;
        p->num_intensity_intervals[c] = h265_br_get_bits(&br, 8) + 1;
        p->num_model_values[c] = h265_br_get_bits(&br, 3) + 1;
        if (p->num_model_values[c] > 6)
            return;
        for (i = 0; i < p->num_intensity_intervals[c]; i++) {
            p->intensity_interval_lower_bound[c][i] = h265_br_get_bits(&br, 8);
            p->intensity_interval_upper_bound[c][i] = h265_br_get_bits(&br, 8);
            for (j = 0; j < p->num_model_values[c]; j++)
                p->comp_model_value[c][i][j] = h265_br_get_se(&br);
        }
    }
    /* persistence/repetition flag */
    h265_br_get_bits1(&br);
    p->present = 1;
}

static void sei_store_blob(H265Decoder *h, int type, const uint8_t *data,
                           int size)
{
    uint8_t **pdst;
    int *psize;
    if (h->cur_frame) {
        if (type == 4) {
            pdst = &h->cur_frame->t35_data;
            psize = &h->cur_frame->t35_size;
        } else {
            pdst = &h->cur_frame->unreg_data;
            psize = &h->cur_frame->unreg_size;
        }
        h265_set_side_blob(pdst, psize, data, size);
    } else if (type == 4) {
        h265_set_side_blob(&h->pending_t35, &h->pending_t35_size, data, size);
    } else {
        h265_set_side_blob(&h->pending_unreg, &h->pending_unreg_size, data,
                           size);
    }
}

static void sei_parse_payload(H265Decoder *h, int type, const uint8_t *data,
                              int size, int suffix)
{
    switch (type) {
    case 1:   sei_parse_pic_timing(h, data, size); break;
    case 4:   sei_store_blob(h, 4, data, size); break;
    case 5:   sei_store_blob(h, 5, data, size); break;
    case 6:   sei_parse_recovery(h, data, size); break;
    case 19:  sei_parse_film_grain(h, data, size); break;
    case 45:  sei_parse_frame_packing(h, data, size); break;
    case 47:  sei_parse_orientation(h, data, size); break;
    case 136: sei_parse_timecode(h, data, size); break;
    case 137: sei_parse_mastering(h, data, size); break;
    case 144: sei_parse_content_light(h, data, size); break;
    case 147: sei_parse_alternative_transfer(h, data, size); break;
    case 148: sei_parse_ambient(h, data, size); break;
    default: break;
    }
    /* Prefix SEI describes the next picture, suffix SEI the current one. */
    if (suffix && h->cur_frame)
        h->cur_frame->sei = h->sei;
}

void h265_sei_reset_au(H265Decoder *h)
{
    h265_free_side_blob(&h->pending_t35, &h->pending_t35_size);
    h265_free_side_blob(&h->pending_unreg, &h->pending_unreg_size);
    h265_free_side_blob(&h->pending_dovi, &h->pending_dovi_size);
    h->sei.pic_struct = -1;
    h->sei.recovery_present = 0;
    h->sei.recovery_poc_cnt = 0;
    h->sei.recovery_exact_match = 0;
    h->sei.recovery_broken_link = 0;
    h->sei.timecode_present = 0;
    h->sei.timecode_num_clock_ts = 0;
    h->sei.frame_packing_present = 0;
}

int h265_parse_sei(H265Decoder *h, const uint8_t *rbsp, int size, int nal_type)
{
    int pos = 2;                    /* NAL header */
    (void)nal_type;

    while (pos + 2 <= size) {
        int payload_type = 0;
        int payload_size = 0;
        int byte;
        do {
            byte = rbsp[pos++];
            payload_type += byte;
        } while (byte == 0xff && pos < size);
        do {
            byte = rbsp[pos++];
            payload_size += byte;
        } while (byte == 0xff && pos < size);
        if (payload_size < 0 || payload_size > size - pos)
            break;
        sei_parse_payload(h, payload_type, rbsp + pos, payload_size,
                          nal_type == H265_NAL_SUFFIX_SEI);
        pos += payload_size;
    }
    return 0;
}
