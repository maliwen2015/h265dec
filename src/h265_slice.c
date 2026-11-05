/*
 * HEVC slice header parsing, reference picture set construction and DPB
 * output ordering.
 *
 * Derived from ITU-T Rec. H.265 and FFmpeg's hevc/hevcdec.c and refs.c
 * (LGPL-2.1+).
 */
#include "h265_internal.h"


static int ceil_log2(unsigned v)
{
    int n = 0;
    while ((1u << n) < v)
        n++;
    return n;
}

/* ------------------------------------------------------------------ */
/* weighted prediction table                                           */
/* ------------------------------------------------------------------ */
static int pred_weight_table(H265Slice *sh, H265SPS *sps, H265BitReader *br)
{
    int i, j;
    unsigned luma_log2_weight_denom;
    unsigned luma_weight_flags, chroma_weight_flags;

    luma_log2_weight_denom = h265_br_get_ue(br);
    if (luma_log2_weight_denom > 7)
        return -1;
    sh->luma_log2_weight_denom = (uint8_t)luma_log2_weight_denom;
    if (sps->chroma_format_idc != 0) {
        int chroma = (int)luma_log2_weight_denom + h265_br_get_se(br);
        if (chroma < 0 || chroma > 7)
            return -1;
        sh->chroma_log2_weight_denom = (int16_t)chroma;
    }

    luma_weight_flags = h265_br_get_bits(br, sh->nb_refs[H265_L0]);
    chroma_weight_flags = sps->chroma_format_idc ?
                          h265_br_get_bits(br, sh->nb_refs[H265_L0]) : 0;
    for (i = 0; i < (int)sh->nb_refs[H265_L0]; i++) {
        unsigned flag_bit = 1u << (sh->nb_refs[H265_L0] - 1 - i);
        if (luma_weight_flags & flag_bit) {
            int delta = h265_br_get_se(br);
            if ((int8_t)delta != delta)
                return -1;
            sh->luma_weight_l0[i] = (int16_t)((1 << sh->luma_log2_weight_denom) + delta);
            sh->luma_offset_l0[i] = (int16_t)h265_br_get_se(br);
        } else {
            sh->luma_weight_l0[i] = (int16_t)(1 << sh->luma_log2_weight_denom);
            sh->luma_offset_l0[i] = 0;
        }
        if (chroma_weight_flags & flag_bit) {
            for (j = 0; j < 2; j++) {
                int delta_w = h265_br_get_se(br);
                int delta_o = h265_br_get_se(br);
                if ((int8_t)delta_w != delta_w || delta_o < -(1 << 17) ||
                    delta_o > (1 << 17))
                    return -1;
                sh->chroma_weight_l0[i][j] = (int16_t)((1 << sh->chroma_log2_weight_denom) + delta_w);
                sh->chroma_offset_l0[i][j] = (int16_t)H265_CLIP(
                    delta_o - ((128 * sh->chroma_weight_l0[i][j]) >>
                               sh->chroma_log2_weight_denom) + 128, -128, 127);
            }
        } else {
            sh->chroma_weight_l0[i][0] = sh->chroma_weight_l0[i][1] =
                (int16_t)(1 << sh->chroma_log2_weight_denom);
            sh->chroma_offset_l0[i][0] = sh->chroma_offset_l0[i][1] = 0;
        }
    }

    if (sh->slice_type == H265_SLICE_B) {
        luma_weight_flags = h265_br_get_bits(br, sh->nb_refs[H265_L1]);
        chroma_weight_flags = sps->chroma_format_idc ?
                              h265_br_get_bits(br, sh->nb_refs[H265_L1]) : 0;
        for (i = 0; i < (int)sh->nb_refs[H265_L1]; i++) {
            unsigned flag_bit = 1u << (sh->nb_refs[H265_L1] - 1 - i);
            if (luma_weight_flags & flag_bit) {
                int delta = h265_br_get_se(br);
                if ((int8_t)delta != delta)
                    return -1;
                sh->luma_weight_l1[i] = (int16_t)((1 << sh->luma_log2_weight_denom) + delta);
                sh->luma_offset_l1[i] = (int16_t)h265_br_get_se(br);
            } else {
                sh->luma_weight_l1[i] = (int16_t)(1 << sh->luma_log2_weight_denom);
                sh->luma_offset_l1[i] = 0;
            }
            if (chroma_weight_flags & flag_bit) {
                for (j = 0; j < 2; j++) {
                    int delta_w = h265_br_get_se(br);
                    int delta_o = h265_br_get_se(br);
                    if ((int8_t)delta_w != delta_w || delta_o < -(1 << 17) ||
                        delta_o > (1 << 17))
                        return -1;
                    sh->chroma_weight_l1[i][j] = (int16_t)((1 << sh->chroma_log2_weight_denom) + delta_w);
                    sh->chroma_offset_l1[i][j] = (int16_t)H265_CLIP(
                        delta_o - ((128 * sh->chroma_weight_l1[i][j]) >>
                                   sh->chroma_log2_weight_denom) + 128, -128, 127);
                }
            } else {
                sh->chroma_weight_l1[i][0] = sh->chroma_weight_l1[i][1] =
                    (int16_t)(1 << sh->chroma_log2_weight_denom);
                sh->chroma_offset_l1[i][0] = sh->chroma_offset_l1[i][1] = 0;
            }
        }
    }
    return 0;
}

static int decode_lt_rps(H265SPS *sps, H265LongTermRPS *rps, H265BitReader *br,
                         int cur_poc, int poc_lsb)
{
    int max_poc_lsb = 1 << sps->log2_max_poc_lsb;
    int prev_delta_msb = 0;
    unsigned nb_sps = 0, nb_sh;
    int i;

    rps->nb_refs = 0;
    if (!sps->long_term_ref_pics_present)
        return 0;

    if (sps->num_long_term_ref_pics_sps > 0)
        nb_sps = h265_br_get_ue(br);
    nb_sh = h265_br_get_ue(br);
    if (nb_sps > (unsigned)sps->num_long_term_ref_pics_sps ||
        nb_sh + nb_sps > 32)
        return -1;

    rps->nb_refs = (uint8_t)(nb_sh + nb_sps);
    for (i = 0; i < rps->nb_refs; i++) {
        if (i < (int)nb_sps) {
            uint8_t lt_idx_sps = 0;
            if (sps->num_long_term_ref_pics_sps > 1)
                lt_idx_sps = h265_br_get_bits(br, ceil_log2(sps->num_long_term_ref_pics_sps));
            if (lt_idx_sps >= sps->num_long_term_ref_pics_sps)
                return -1;
            rps->poc[i] = sps->lt_ref_pic_poc_lsb_sps[lt_idx_sps];
            rps->used[i] = !!(sps->used_by_curr_pic_lt & (1u << lt_idx_sps));
        } else {
            rps->poc[i] = h265_br_get_bits(br, sps->log2_max_poc_lsb);
            rps->used[i] = h265_br_get_bits1(br);
        }
        rps->poc_msb_present[i] = h265_br_get_bits1(br);
        if (rps->poc_msb_present[i]) {
            int64_t delta = h265_br_get_ue(br);
            int64_t poc;
            if (i && i != (int)nb_sps)
                delta += prev_delta_msb;
            poc = rps->poc[i] + cur_poc - delta * max_poc_lsb - poc_lsb;
            if (poc != (int32_t)poc)
                return -1;
            rps->poc[i] = (int)poc;
            prev_delta_msb = (int)delta;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* slice header                                                        */
/* ------------------------------------------------------------------ */
int h265_parse_slice_header(H265Decoder *h, H265Slice *sh, const uint8_t *rbsp,
                            int size)
{
    H265BitReader br;
    H265PPS *pps;
    H265SPS *sps;
    unsigned pps_id;
    int i, ret = 0;

    sh->h = h;
    sh->rbsp = (uint8_t *)rbsp;
    sh->rbsp_size = size;

    h265_br_init(&br, rbsp, size * 8);
    h265_br_skip(&br, 16);

    sh->first_slice_in_pic_flag = h265_br_get_bits1(&br);
    sh->no_output_of_prior_pics_flag = 0;
    if (H265_IS_IRAP(h->nal_unit_type))
        sh->no_output_of_prior_pics_flag = h265_br_get_bits1(&br);

    pps_id = h265_br_get_ue(&br);
    if (pps_id >= H265_MAX_PPS_COUNT || !h->ps.pps_list[pps_id])
        return -1;
    sh->pps_id = pps_id;
    pps = h->ps.pps_list[pps_id];
    sps = pps->sps;
    sh->pps = pps;
    sh->sps = sps;
    sh->slice_idx = h->nb_slices;

    if (h->nal_unit_type == H265_NAL_CRA_NUT && h->last_eos == 1)
        sh->no_output_of_prior_pics_flag = 1;

    sh->dependent_slice_segment_flag = 0;
    if (!sh->first_slice_in_pic_flag) {
        int slice_address_length;
        if (pps->dependent_slice_segments_enabled_flag)
            sh->dependent_slice_segment_flag = h265_br_get_bits1(&br);
        if (sh->dependent_slice_segment_flag && !h->cur_frame)
            return -1;
        slice_address_length = ceil_log2(sps->ctb_width * sps->ctb_height);
        sh->slice_segment_addr = h265_br_get_bits(&br, slice_address_length);
        if (sh->slice_segment_addr >= (unsigned)(sps->ctb_width * sps->ctb_height))
            return -1;
        if (!sh->dependent_slice_segment_flag)
            sh->slice_addr = sh->slice_segment_addr;
    } else {
        sh->slice_segment_addr = sh->slice_addr = 0;
    }

    if (!sh->dependent_slice_segment_flag) {
        for (i = 0; i < pps->num_extra_slice_header_bits; i++)
            h265_br_skip(&br, 1);
        sh->slice_type = h265_br_get_ue(&br);
        if (sh->slice_type != H265_SLICE_I && sh->slice_type != H265_SLICE_P &&
            sh->slice_type != H265_SLICE_B)
            return -1;
        if (H265_IS_IRAP(h->nal_unit_type) && sh->slice_type != H265_SLICE_I)
            return -1;

        sh->pic_output_flag = 1;
        if (pps->output_flag_present_flag)
            sh->pic_output_flag = h265_br_get_bits1(&br);
        if (sps->separate_colour_plane)
            sh->colour_plane_id = h265_br_get_bits(&br, 2);

        if (!H265_IS_IDR(h->nal_unit_type)) {
            int poc;
            sh->pic_order_cnt_lsb = h265_br_get_bits(&br, sps->log2_max_poc_lsb);
            poc = h265_compute_poc(sps->log2_max_poc_lsb, h->poc_tid0,
                                   sh->pic_order_cnt_lsb, h->nal_unit_type);
            sh->poc = poc;
        } else {
            sh->poc = 0;
            sh->pic_order_cnt_lsb = 0;
        }

        if (!H265_IS_IDR(h->nal_unit_type)) {
            sh->short_term_ref_pic_set_sps_flag = h265_br_get_bits1(&br);
            if (!sh->short_term_ref_pic_set_sps_flag) {
                if (h265_parse_short_term_rps(&br, &sh->slice_rps, sps, 1) < 0)
                    return -1;
                sh->short_term_rps = &sh->slice_rps;
            } else {
                int numbits, rps_idx;
                if (!sps->nb_st_rps)
                    return -1;
                numbits = ceil_log2(sps->nb_st_rps);
                rps_idx = numbits > 0 ? h265_br_get_bits(&br, numbits) : 0;
                if (rps_idx >= sps->nb_st_rps)
                    return -1;
                sh->short_term_rps = &sps->st_rps[rps_idx];
            }
            if (decode_lt_rps(sps, &sh->long_term_rps, &br, sh->poc,
                              sh->pic_order_cnt_lsb) < 0)
                return -1;
            if (sps->temporal_mvp_enabled)
                sh->slice_temporal_mvp_enabled_flag = h265_br_get_bits1(&br);
            else
                sh->slice_temporal_mvp_enabled_flag = 0;
        } else {
            sh->short_term_ref_pic_set_sps_flag = 0;
            sh->short_term_rps = NULL;
            sh->long_term_rps.nb_refs = 0;
            sh->slice_temporal_mvp_enabled_flag = 0;
        }

        if (sps->sao_enabled) {
            sh->slice_sample_adaptive_offset_flag[0] = h265_br_get_bits1(&br);
            if (sps->chroma_format_idc) {
                sh->slice_sample_adaptive_offset_flag[1] =
                sh->slice_sample_adaptive_offset_flag[2] = h265_br_get_bits1(&br);
            }
        }

        sh->nb_refs[H265_L0] = sh->nb_refs[H265_L1] = 0;
        if (sh->slice_type == H265_SLICE_P || sh->slice_type == H265_SLICE_B) {
            int nb_refs;
            sh->nb_refs[H265_L0] = pps->num_ref_idx_l0_default_active;
            if (sh->slice_type == H265_SLICE_B)
                sh->nb_refs[H265_L1] = pps->num_ref_idx_l1_default_active;
            if (h265_br_get_bits1(&br)) {
                sh->nb_refs[H265_L0] = h265_br_get_ue(&br) + 1;
                if (sh->slice_type == H265_SLICE_B)
                    sh->nb_refs[H265_L1] = h265_br_get_ue(&br) + 1;
            }
            if (sh->nb_refs[H265_L0] >= H265_MAX_REFS ||
                sh->nb_refs[H265_L1] >= H265_MAX_REFS)
                return -1;
            sh->rpl_modification_flag[0] = 0;
            sh->rpl_modification_flag[1] = 0;
            nb_refs = h265_frame_nb_refs(sh);
            if (!nb_refs)
                return -1;
            if (pps->lists_modification_present_flag && nb_refs > 1) {
                sh->rpl_modification_flag[0] = h265_br_get_bits1(&br);
                if (sh->rpl_modification_flag[0])
                    for (i = 0; i < (int)sh->nb_refs[H265_L0]; i++)
                        sh->list_entry_lx[0][i] = h265_br_get_bits(&br, ceil_log2(nb_refs));
                if (sh->slice_type == H265_SLICE_B) {
                    sh->rpl_modification_flag[1] = h265_br_get_bits1(&br);
                    if (sh->rpl_modification_flag[1])
                        for (i = 0; i < (int)sh->nb_refs[H265_L1]; i++)
                            sh->list_entry_lx[1][i] = h265_br_get_bits(&br, ceil_log2(nb_refs));
                }
            }
            if (sh->slice_type == H265_SLICE_B)
                sh->mvd_l1_zero_flag = h265_br_get_bits1(&br);
            sh->cabac_init_flag = pps->cabac_init_present_flag ?
                                  h265_br_get_bits1(&br) : 0;
            sh->collocated_ref_idx = 0;
            if (sh->slice_temporal_mvp_enabled_flag) {
                sh->collocated_list = H265_L0;
                if (sh->slice_type == H265_SLICE_B)
                    sh->collocated_list = !h265_br_get_bits1(&br);
                if (sh->nb_refs[sh->collocated_list] > 1) {
                    sh->collocated_ref_idx = h265_br_get_ue(&br);
                    if (sh->collocated_ref_idx >= sh->nb_refs[sh->collocated_list])
                        return -1;
                }
            }
            if ((pps->weighted_pred_flag && sh->slice_type == H265_SLICE_P) ||
                (pps->weighted_bipred_flag && sh->slice_type == H265_SLICE_B)) {
                if (pred_weight_table(sh, sps, &br) < 0)
                    return -1;
            }
            sh->max_num_merge_cand = (uint8_t)(5 - h265_br_get_ue(&br));
            if (sh->max_num_merge_cand < 1 || sh->max_num_merge_cand > 5)
                return -1;
            if (sps->motion_vector_resolution_control_idc == 2)
                sh->use_integer_mv_flag = h265_br_get_bits1(&br);
            else
                sh->use_integer_mv_flag = (uint8_t)sps->motion_vector_resolution_control_idc;
        }

        sh->slice_qp_delta = h265_br_get_se(&br);
        if (pps->pic_slice_level_chroma_qp_offsets_present_flag) {
            sh->slice_cb_qp_offset = h265_br_get_se(&br);
            sh->slice_cr_qp_offset = h265_br_get_se(&br);
        }
        if (pps->pps_slice_act_qp_offsets_present_flag) {
            sh->slice_act_y_qp_offset = h265_br_get_se(&br);
            sh->slice_act_cb_qp_offset = h265_br_get_se(&br);
            sh->slice_act_cr_qp_offset = h265_br_get_se(&br);
        }
        sh->cu_chroma_qp_offset_enabled_flag =
            pps->chroma_qp_offset_list_enabled_flag ? h265_br_get_bits1(&br) : 0;

        if (pps->deblocking_filter_control_present_flag) {
            int deblocking_filter_override_flag = 0;
            if (pps->deblocking_filter_override_enabled_flag)
                deblocking_filter_override_flag = h265_br_get_bits1(&br);
            if (deblocking_filter_override_flag) {
                sh->disable_deblocking_filter_flag = h265_br_get_bits1(&br);
                if (!sh->disable_deblocking_filter_flag) {
                    int beta = h265_br_get_se(&br);
                    int tc = h265_br_get_se(&br);
                    if (beta < -6 || beta > 6 || tc < -6 || tc > 6)
                        return -1;
                    sh->beta_offset = beta * 2;
                    sh->tc_offset = tc * 2;
                }
            } else {
                sh->disable_deblocking_filter_flag = (uint8_t)pps->disable_dbf;
                sh->beta_offset = pps->beta_offset;
                sh->tc_offset = pps->tc_offset;
            }
        } else {
            sh->disable_deblocking_filter_flag = 0;
            sh->beta_offset = 0;
            sh->tc_offset = 0;
        }

        if (pps->seq_loop_filter_across_slices_enabled_flag &&
            (sh->slice_sample_adaptive_offset_flag[0] ||
             sh->slice_sample_adaptive_offset_flag[1] ||
             !sh->disable_deblocking_filter_flag))
            sh->slice_loop_filter_across_slices_enabled_flag = h265_br_get_bits1(&br);
        else
            sh->slice_loop_filter_across_slices_enabled_flag =
                (uint8_t)pps->seq_loop_filter_across_slices_enabled_flag;
    }

    sh->num_entry_point_offsets = 0;
    if (pps->tiles_enabled_flag || pps->entropy_coding_sync_enabled_flag) {
        unsigned n = h265_br_get_ue(&br);
        if (n > (unsigned)(h265_br_bits_left(&br)) || n > 65535)
            return -1;
        sh->num_entry_point_offsets = (int)n;
        if (n > 0) {
            int offset_len = h265_br_get_ue(&br) + 1;
            if (offset_len < 1 || offset_len > 32)
                return -1;
            if (sh->num_entry_point_alloc < (int)n) {
                free(sh->entry_point_offset);
                free(sh->offset);
                free(sh->size);
                sh->entry_point_offset = (unsigned *)malloc(n * sizeof(unsigned));
                sh->offset = (int *)malloc((n + 1) * sizeof(int));
                sh->size = (int *)malloc((n + 1) * sizeof(int));
                sh->num_entry_point_alloc = (int)n;
                if (!sh->entry_point_offset || !sh->offset || !sh->size)
                    return -1;
            }
            for (i = 0; i < (int)n; i++)
                sh->entry_point_offset[i] = h265_br_get_bits(&br, offset_len) + 1;
        }
    }

    if (pps->slice_header_extension_present_flag) {
        unsigned length = h265_br_get_ue(&br);
        if ((int64_t)length * 8 > h265_br_bits_left(&br))
            return -1;
        for (i = 0; i < (int)length; i++)
            h265_br_skip(&br, 8);
    }

    ret = h265_br_get_bits1(&br);
    if (!ret && h265_br_bits_left(&br) >= 0)
        return -1;
    h265_br_align(&br);
    sh->data_offset = br.pos >> 3;

    if (br.error || h265_br_bits_left(&br) < 0)
        return -1;

    sh->slice_qp = (int8_t)(26 + pps->pic_init_qp_minus26 + sh->slice_qp_delta);
    if (sh->slice_qp > 51 || sh->slice_qp < -sps->qp_bd_offset)
        return -1;
    sh->slice_ctb_addr_rs = (int)sh->slice_segment_addr;

    if (sh->dependent_slice_segment_flag) {
        /* inherit the independent slice's address and type-related fields */
        int ctb_addr_ts = pps->ctb_addr_rs_to_ts[sh->slice_segment_addr];
        if (!ctb_addr_ts)
            return -1;
        sh->slice_addr = (unsigned)h->cur_sh.slice_addr;
    }

    return 0;
}

static int count_epb_in_range(const H265Slice *sh, int offset, int raw_len)
{
    int start = offset + raw_len;
    int count = 0;
    int i;

    for (i = 0; i < sh->nb_epb; i++) {
        int p = sh->epb_pos[i];
        if (p >= offset && p < start) {
            start--;
            count++;
        }
    }
    return count;
}

int h265_slice_compute_entry_offsets(H265Slice *sh)
{
    int n = sh->num_entry_point_offsets;
    int i, offset, cmpt;

    if (n <= 0 || !sh->entry_point_offset || !sh->offset || !sh->size)
        return -1;

    offset = (int)sh->data_offset;
    cmpt = count_epb_in_range(sh, offset, (int)sh->entry_point_offset[0]);
    for (i = 1; i < n; i++) {
        offset += (int)sh->entry_point_offset[i - 1] - cmpt;
        cmpt = count_epb_in_range(sh, offset,
                                  (int)sh->entry_point_offset[i]);
        sh->size[i] = (int)sh->entry_point_offset[i] - cmpt;
        sh->offset[i] = offset;
    }
    offset += (int)sh->entry_point_offset[n - 1] - cmpt;
    if (offset <= (int)sh->data_offset || offset > sh->rbsp_size)
        return -1;
    sh->size[n] = sh->rbsp_size - offset;
    sh->offset[n] = offset;
    sh->offset[0] = (int)sh->data_offset;
    sh->size[0] = sh->offset[1] - sh->offset[0];

    for (i = 0; i <= n; i++)
        if (sh->size[i] <= 0)
            return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* DPB                                                                 */
/* ------------------------------------------------------------------ */
int h265_frame_nb_refs(H265Slice *sh)
{
    int ret = 0, i;
    const H265ShortTermRPS *rps = sh->short_term_rps;
    const H265LongTermRPS *long_rps = &sh->long_term_rps;

    if (rps) {
        for (i = 0; i < rps->num_negative_pics; i++)
            ret += !!(rps->used & (1u << i));
        for (; i < rps->num_delta_pocs; i++)
            ret += !!(rps->used & (1u << i));
    }
    for (i = 0; i < long_rps->nb_refs; i++)
        ret += !!long_rps->used[i];
    return ret;
}

static void dpb_remove(H265Decoder *h, int idx)
{
    h265_frame_unref(h->DPB[idx]);
    memmove(&h->DPB[idx], &h->DPB[idx + 1],
            (h->nb_dpb - idx - 1) * sizeof(H265Frame *));
    h->nb_dpb--;
}

static H265Frame *find_ref_idx(H265Decoder *h, int poc, int use_msb,
                               int colour_plane_id)
{
    int mask = use_msb ? ~0 : (1 << h->sps->log2_max_poc_lsb) - 1;
    int i;
    for (i = 0; i < h->nb_dpb; i++) {
        H265Frame *ref = h->DPB[i];
        if (ref) {
            if ((ref->poc & mask) == poc && (use_msb || ref->poc != h->poc) &&
                (!h->sps->separate_colour_plane ||
                 ref->colour_plane_id == colour_plane_id))
                return ref;
        }
    }
    return NULL;
}

static H265Frame *generate_missing_ref(H265Decoder *h, int poc,
                                       int colour_plane_id)
{
    H265Frame *f = h265_frame_alloc(h);
    int i, y;
    if (!f)
        return NULL;
    if (h265_frame_alloc_aux(f, h->sps) < 0) {
        h265_frame_unref(f);
        return NULL;
    }
    h265_frame_reset_meta(f);
    f->pps = h265_pps_ref(h->pps);
    f->colour_plane_id = colour_plane_id;
    for (i = 0; i < f->num_planes; i++) {
        int w = h->sps->width >> h->sps->hshift[i];
        int hh = h->sps->height >> h->sps->vshift[i];
        if (!h->sps->pixel_shift) {
            memset(f->data[i], 1 << (h->sps->bit_depth - 1),
                   (size_t)f->linesize[i] * hh);
        } else {
            for (y = 0; y < hh; y++) {
                uint16_t *row = (uint16_t *)(f->data[i] + y * f->linesize[i]);
                int x;
                for (x = 0; x < w; x++)
                    row[x] = (uint16_t)(1 << (h->sps->bit_depth - 1));
            }
        }
    }
    f->poc = poc;
    f->flags = H265_FRAME_FLAG_UNAVAILABLE;
    f->decode_done = 1;
    f->in_use = 1;
    if (h->nb_dpb >= H265_MAX_DPB_SIZE) {
        h265_frame_unref(f);
        return NULL;
    }
    h->DPB[h->nb_dpb++] = f;
    return f;
}

static int add_candidate_ref(H265Decoder *h, H265RefList *rps, int list_idx,
                             int poc, int ref_flag, int use_msb)
{
    H265RefList *list = &rps[list_idx];
    int plane = h->cur_frame ? h->cur_frame->colour_plane_id : -1;
    H265Frame *ref = find_ref_idx(h, poc, use_msb, plane);

    if (ref == h->cur_frame || list->nb_refs >= H265_MAX_REFS)
        return -1;
    if (!ref) {
        ref = generate_missing_ref(h, poc, plane);
        if (!ref)
            return -1;
    }
    list->list[list->nb_refs] = ref->poc;
    list->isLongTerm[list->nb_refs] = (ref_flag == H265_FRAME_FLAG_LONG_REF);
    list->nb_refs++;
    ref->flags &= ~(H265_FRAME_FLAG_SHORT_REF | H265_FRAME_FLAG_LONG_REF);
    ref->flags |= ref_flag;
    return 0;
}

static int build_frame_rps(H265Decoder *h, H265Slice *sh)
{
    const H265ShortTermRPS *short_rps = sh->short_term_rps;
    const H265LongTermRPS *long_rps = &sh->long_term_rps;
    int i, ret = 0;

    for (i = 0; i < h->nb_dpb; ) {
        H265Frame *df = h->DPB[i];
        if (df != h->cur_frame && (df->flags & H265_FRAME_FLAG_UNAVAILABLE))
            dpb_remove(h, i);
        else
            i++;
    }
    for (i = 0; i < h->nb_dpb; i++) {
        if (h->DPB[i] != h->cur_frame)
            h->DPB[i]->flags &= ~(H265_FRAME_FLAG_SHORT_REF | H265_FRAME_FLAG_LONG_REF);
    }
    for (i = 0; i < H265_NB_RPS_TYPE; i++)
        h->rps[i].nb_refs = 0;

    if (short_rps) {
        for (i = 0; i < short_rps->num_delta_pocs; i++) {
            int poc = sh->poc + short_rps->delta_poc[i];
            int list;
            if (!(short_rps->used & (1u << i)))
                list = H265_ST_FOLL;
            else if (i < short_rps->num_negative_pics)
                list = H265_ST_CURR_BEF;
            else
                list = H265_ST_CURR_AFT;
            ret = add_candidate_ref(h, h->rps, list, poc,
                                    H265_FRAME_FLAG_SHORT_REF, 1);
            if (ret < 0)
                goto fail;
        }
    }
    for (i = 0; i < long_rps->nb_refs; i++) {
        int poc = long_rps->poc[i];
        int list = long_rps->used[i] ? H265_LT_CURR : H265_LT_FOLL;
        ret = add_candidate_ref(h, h->rps, list, poc,
                                H265_FRAME_FLAG_LONG_REF,
                                long_rps->poc_msb_present[i]);
        if (ret < 0)
            goto fail;
    }
fail:
    for (i = 0; i < h->nb_dpb; ) {
        H265Frame *f = h->DPB[i];
        if (f != h->cur_frame && !(f->flags & (H265_FRAME_FLAG_SHORT_REF |
                                               H265_FRAME_FLAG_LONG_REF |
                                               H265_FRAME_FLAG_OUTPUT)))
            dpb_remove(h, i);
        else
            i++;
    }
    return ret;
}

static void fill_rpl_tab(H265Decoder *h, H265Slice *sh);

int h265_slice_rpl(H265Decoder *h, H265Slice *sh)
{
    int nb_list = sh->slice_type == H265_SLICE_B ? 2 : 1;
    int list_idx, i, j;

    for (list_idx = 0; list_idx < nb_list; list_idx++) {
        H265RefList rpl_tmp;
        H265RefList *rpl = &sh->ref_list[list_idx];
        int cand_lists[3];
        int n_cand = 3;

        memset(&rpl_tmp, 0, sizeof(rpl_tmp));
        cand_lists[0] = list_idx ? H265_ST_CURR_AFT : H265_ST_CURR_BEF;
        cand_lists[1] = list_idx ? H265_ST_CURR_BEF : H265_ST_CURR_AFT;
        cand_lists[2] = H265_LT_CURR;

        while (rpl_tmp.nb_refs < (int)sh->nb_refs[list_idx]) {
            int added = 0;
            for (i = 0; i < n_cand; i++) {
                H265RefList *rps = &h->rps[cand_lists[i]];
                for (j = 0; j < rps->nb_refs && rpl_tmp.nb_refs < H265_MAX_REFS; j++) {
                    rpl_tmp.list[rpl_tmp.nb_refs] = rps->list[j];
                    rpl_tmp.isLongTerm[rpl_tmp.nb_refs] =
                        (cand_lists[i] == H265_LT_CURR) ? 1 : rps->isLongTerm[j];
                    rpl_tmp.nb_refs++;
                    added++;
                }
            }
            if (!added)
                break;
        }

        if (sh->rpl_modification_flag[list_idx]) {
            for (i = 0; i < (int)sh->nb_refs[list_idx]; i++) {
                int idx = (int)sh->list_entry_lx[list_idx][i];
                if (idx >= rpl_tmp.nb_refs)
                    return -1;
                rpl->list[i] = rpl_tmp.list[idx];
                rpl->isLongTerm[i] = rpl_tmp.isLongTerm[idx];
                rpl->nb_refs++;
            }
        } else {
            memcpy(rpl, &rpl_tmp, sizeof(rpl_tmp));
            rpl->nb_refs = H265_MIN(rpl->nb_refs, (int)sh->nb_refs[list_idx]);
        }

        for (i = 0; i < rpl->nb_refs; i++) {
            H265Frame *ref = find_ref_idx(h, rpl->list[i], 1,
                                          sh->colour_plane_id);
            if (!ref)
                return -1;
            sh->ref_frame[list_idx][i] = ref;
        }
    }

    if (sh->slice_temporal_mvp_enabled_flag &&
        sh->collocated_list < 2) {
        int idx = (int)sh->collocated_ref_idx;
        if (idx < sh->ref_list[sh->collocated_list].nb_refs)
            sh->collocated_ref = sh->ref_frame[sh->collocated_list][idx];
        else
            sh->collocated_ref = NULL;
    } else {
        sh->collocated_ref = NULL;
    }
    h->collocated_ref = sh->collocated_ref;
    fill_rpl_tab(h, sh);
    return 0;
}

/* Copy a slice's reference lists into the per-CTB table for deblocking and
 * collocated-picture use. */
static void fill_rpl_tab(H265Decoder *h, H265Slice *sh)
{
    H265Frame *f = sh->frame;
    H265PPS *pps = sh->pps;
    H265SPS *sps = sh->sps;
    int ctb_count = sps->ctb_width * sps->ctb_height;
    int start = pps->ctb_addr_rs_to_ts[sh->slice_ctb_addr_rs];
    int i;

    if (!f)
        return;
    for (i = start; i < ctb_count; i++) {
        f->rpl_tab[i].refPicList[0] = sh->ref_list[0];
        f->rpl_tab[i].refPicList[1] = sh->ref_list[1];
    }
}

/* ------------------------------------------------------------------ */
/* output                                                              */
/* ------------------------------------------------------------------ */
void h265_output_frames(H265Decoder *h, int max_output, int max_dpb, int discard)
{
    for (;;) {
        int nb_output = 0;
        int min_poc = INT_MAX, min_idx = -1;
        int i;

        for (i = 0; i < h->nb_dpb; i++) {
            H265Frame *f = h->DPB[i];
            if (f->flags & H265_FRAME_FLAG_OUTPUT) {
                nb_output++;
                if (f->poc < min_poc) {
                    min_poc = f->poc;
                    min_idx = i;
                }
            }
        }
        if (nb_output > max_output || (nb_output && h->nb_dpb > max_dpb)) {
            H265Frame *f = h->DPB[min_idx];
            if (!f->decode_done)
                break;
            f->flags &= ~H265_FRAME_FLAG_OUTPUT;
            if (!discard && !h->param.skip_non_ref &&
                f->flags & (H265_FRAME_FLAG_SHORT_REF | H265_FRAME_FLAG_LONG_REF)) {
                if (h->out_count < (int)H265_ARRAY_ELEMS(h->out_q)) {
                    h265_frame_addref(f);
                    h->out_q[h->out_count++] = f;
                }
            } else if (!discard && !h->param.skip_non_ref) {
                if (h->out_count < (int)H265_ARRAY_ELEMS(h->out_q)) {
                    h265_frame_addref(f);
                    h->out_q[h->out_count++] = f;
                }
            }
            if (!(f->flags & (H265_FRAME_FLAG_SHORT_REF | H265_FRAME_FLAG_LONG_REF)))
                dpb_remove(h, min_idx);
        } else {
            break;
        }
    }
}

/* ------------------------------------------------------------------ */
/* frame start                                                         */
/* ------------------------------------------------------------------ */
int h265_frame_start(H265Decoder *h, H265Slice *sh)
{
    H265PPS *pps = sh->pps;
    H265SPS *sps = pps->sps;
    H265Frame *f;
    int new_sequence = H265_IS_IDR(h->nal_unit_type) ||
                       H265_IS_BLA(h->nal_unit_type) || h->last_eos;
    int i;

    h->sps = sps;
    h->pps = pps;
    h->poc = sh->poc;

    if (H265_IS_IDR(h->nal_unit_type)) {
        for (i = 0; i < h->nb_dpb; i++)
            h->DPB[i]->flags &= ~(H265_FRAME_FLAG_SHORT_REF |
                                  H265_FRAME_FLAG_LONG_REF);
    }

    if (H265_IS_IRAP(h->nal_unit_type))
        h->no_rasl_output_flag = H265_IS_IDR(h->nal_unit_type) ||
                                 H265_IS_BLA(h->nal_unit_type) ||
                                 (h->nal_unit_type == H265_NAL_CRA_NUT && h->last_eos);

    if (h->temporal_id == 0 &&
        h->nal_unit_type != H265_NAL_TRAIL_N &&
        h->nal_unit_type != H265_NAL_TSA_N &&
        h->nal_unit_type != H265_NAL_STSA_N &&
        h->nal_unit_type != H265_NAL_RADL_N &&
        h->nal_unit_type != H265_NAL_RADL_R &&
        h->nal_unit_type != H265_NAL_RASL_N &&
        h->nal_unit_type != H265_NAL_RASL_R)
        h->poc_tid0 = h->poc;

    if (new_sequence) {
        h265_output_frames(h, 0, 0, sh->no_output_of_prior_pics_flag);
    }

    f = h265_frame_alloc(h);
    if (!f)
        return -1;
    if (h265_frame_alloc_aux(f, sps) < 0) {
        h265_frame_unref(f);
        return -1;
    }
    f->pps = h265_pps_ref(pps);
    h265_frame_reset_meta(f);
    f->sei = h->sei;
    f->t35_data = h->pending_t35;
    f->t35_size = h->pending_t35_size;
    f->unreg_data = h->pending_unreg;
    f->unreg_size = h->pending_unreg_size;
    f->dovi_rpu = h->pending_dovi;
    f->dovi_rpu_size = h->pending_dovi_size;
    h->pending_t35 = NULL;
    h->pending_t35_size = 0;
    h->pending_unreg = NULL;
    h->pending_unreg_size = 0;
    h->pending_dovi = NULL;
    h->pending_dovi_size = 0;
    f->pts = h->packet_pts;
    f->dts = h->packet_dts;
    f->poc = sh->poc;
    f->temporal_id = h->temporal_id;
    f->nal_unit_type = h->nal_unit_type;
    f->is_irap = H265_IS_IRAP(h->nal_unit_type);
    f->colour_plane_id = sps->separate_colour_plane ?
                         sh->colour_plane_id : -1;
    f->layer_id = sh->layer_id;
    f->slice_type = sh->slice_type;
    if (sh->pic_output_flag)
        f->flags = H265_FRAME_FLAG_OUTPUT | H265_FRAME_FLAG_SHORT_REF;
    else
        f->flags = H265_FRAME_FLAG_SHORT_REF;
    h->cur_frame = f;
    sh->frame = f;
    h->cur_sh = *sh;
    h->cur_sh.frame = f;

    if (h->nb_dpb >= H265_MAX_DPB_SIZE) {
        /* drop the oldest non-referenced frame */
        for (i = 0; i < h->nb_dpb; i++) {
            if (!(h->DPB[i]->flags & (H265_FRAME_FLAG_SHORT_REF |
                                      H265_FRAME_FLAG_LONG_REF |
                                      H265_FRAME_FLAG_OUTPUT))) {
                dpb_remove(h, i);
                break;
            }
        }
        if (h->nb_dpb >= H265_MAX_DPB_SIZE) {
            h265_frame_unref(f);
            h->cur_frame = NULL;
            return -1;
        }
    }
    h->DPB[h->nb_dpb++] = f;

    if (build_frame_rps(h, sh) < 0)
        return -1;

    if (h265_slice_rpl(h, sh) < 0)
        return -1;
    sh->rpl_valid = 1;
    h->cur_sh.rpl_valid = 1;

    h265_output_frames(h,
                       sps->temporal_layer[sps->max_sub_layers - 1].num_reorder_pics,
                       sps->temporal_layer[sps->max_sub_layers - 1].max_dec_pic_buffering,
                       0);
    return 0;
}

void h265_frame_filter(H265Decoder *h, H265Slice *sh)
{
    H265Frame *f = sh->frame;
    if (!f)
        return;
    if (!sh->row_progress) {
        if (!h->param.disable_lf)
            h265_deblock_frame(sh);
        if (!h->param.disable_sao)
            h265_sao_frame(sh);
        h265_frame_report_rows(f, f->height);
    }
    h265_frame_mark_done(f);
}

void h265_filter_row(H265Slice *sh, int y0, int last)
{
    H265SPS *sps = sh->sps;
    H265Frame *f = sh->frame;
    int ctb_size = 1 << sps->log2_ctb_size;

    /* Filtering lags decoding by one CTB row: intra prediction of row y must
     * see the unfiltered samples of row y-1. */
    if (y0 >= ctb_size)
        h265_deblock_ctb_row(sh, y0 - ctb_size);
    if (y0 >= 2 * ctb_size)
        h265_sao_ctb_row(sh, y0 - 2 * ctb_size);
    if (y0 >= ctb_size)
        h265_frame_report_rows(f, y0 - ctb_size);
    if (last) {
        h265_deblock_ctb_row(sh, y0);
        if (y0 >= ctb_size)
            h265_sao_ctb_row(sh, y0 - ctb_size);
        h265_frame_report_rows(f, y0);
        h265_sao_ctb_row(sh, y0);
        h265_frame_report_rows(f, f->height);
    }
}

void h265_frame_finish(H265Decoder *h, H265Slice *sh)
{
    H265Frame *f = sh->frame;
    H265SPS *sps;
    if (!f)
        return;
    h265_frame_filter(h, sh);
    sps = sh->sps ? sh->sps : h->sps;
    h265_output_frames(h,
                       sps->temporal_layer[sps->max_sub_layers - 1].num_reorder_pics,
                       sps->temporal_layer[sps->max_sub_layers - 1].max_dec_pic_buffering,
                       0);
}
