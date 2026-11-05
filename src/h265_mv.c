/*
 * HEVC merge and AMVP candidate derivation.
 *
 * Derived from FFmpeg's hevc/mvs.c (LGPL-2.1+).
 */
#include "h265_internal.h"

static const uint8_t l0_l1_cand_idx[12][2] = {
    { 0, 1 }, { 1, 0 }, { 0, 2 }, { 2, 0 }, { 1, 2 }, { 2, 1 },
    { 0, 3 }, { 3, 0 }, { 1, 3 }, { 3, 1 }, { 2, 3 }, { 3, 2 },
};

void h265_set_neighbour_available(H265Slice *sh, int x0, int y0, int nPbW,
                                  int nPbH, int log2_ctb_size)
{
    int x0b = x0 & ((1 << log2_ctb_size) - 1);
    int y0b = y0 & ((1 << log2_ctb_size) - 1);

    sh->na.cand_up       = (sh->ctb_up_flag || y0b);
    sh->na.cand_left     = (sh->ctb_left_flag || x0b);
    sh->na.cand_up_left  = (x0b || y0b) ?
                           sh->na.cand_left && sh->na.cand_up :
                           sh->ctb_up_left_flag;
    sh->na.cand_up_right_sap =
        (x0b + nPbW == 1 << log2_ctb_size) ?
            sh->ctb_up_right_flag && !y0b : sh->na.cand_up;
    sh->na.cand_up_right =
        sh->na.cand_up_right_sap && (x0 + nPbW) < sh->end_of_tiles_x;
    sh->na.cand_bottom_left =
        ((y0 + nPbH) >= sh->end_of_tiles_y) ? 0 : sh->na.cand_left;
}

H265_INLINE int z_scan_block_avail(H265Slice *sh, int xCurr, int yCurr,
                                          int xN, int yN)
{
    H265PPS *pps = sh->pps;
    H265SPS *sps = sh->sps;
    int xCurr_ctb = xCurr >> sps->log2_ctb_size;
    int yCurr_ctb = yCurr >> sps->log2_ctb_size;
    int xN_ctb = xN >> sps->log2_ctb_size;
    int yN_ctb = yN >> sps->log2_ctb_size;

#define MIN_TB_ADDR_ZS(x, y) pps->min_tb_addr_zs[(y) * (sps->tb_mask + 2) + (x)]
    if (yN_ctb < yCurr_ctb || xN_ctb < xCurr_ctb)
        return 1;
    else {
        int Curr = MIN_TB_ADDR_ZS((xCurr >> sps->log2_min_tb_size) & sps->tb_mask,
                                  (yCurr >> sps->log2_min_tb_size) & sps->tb_mask);
        int N = MIN_TB_ADDR_ZS((xN >> sps->log2_min_tb_size) & sps->tb_mask,
                               (yN >> sps->log2_min_tb_size) & sps->tb_mask);
        return N <= Curr;
    }
#undef MIN_TB_ADDR_ZS
}

H265_INLINE int is_diff_mer(H265PPS *pps, int xN, int yN, int xP, int yP)
{
    uint8_t plevel = pps->log2_parallel_merge_level;
    return xN >> plevel == xP >> plevel && yN >> plevel == yP >> plevel;
}

H265_INLINE int compare_mv_ref_idx(H265MvField A, H265MvField B)
{
    int a_pf = A.pred_flag;
    int b_pf = B.pred_flag;

    if (a_pf == b_pf) {
        if (a_pf == H265_PF_BI) {
            return A.ref_idx[0] == B.ref_idx[0] &&
                   A.mv[0].x == B.mv[0].x && A.mv[0].y == B.mv[0].y &&
                   A.ref_idx[1] == B.ref_idx[1] &&
                   A.mv[1].x == B.mv[1].x && A.mv[1].y == B.mv[1].y;
        } else if (a_pf == H265_PF_L0) {
            return A.ref_idx[0] == B.ref_idx[0] &&
                   A.mv[0].x == B.mv[0].x && A.mv[0].y == B.mv[0].y;
        } else if (a_pf == H265_PF_L1) {
            return A.ref_idx[1] == B.ref_idx[1] &&
                   A.mv[1].x == B.mv[1].x && A.mv[1].y == B.mv[1].y;
        }
    }
    return 0;
}

H265_INLINE int16_t clip_int16v(int v)
{
    return v < -32768 ? -32768 : v > 32767 ? 32767 : (int16_t)v;
}

H265_INLINE void mv_scale(H265Mv *dst, const H265Mv *src, int td, int tb)
{
    int tx, scale_factor;

    td = H265_CLIP(td, -128, 127);
    tb = H265_CLIP(tb, -128, 127);
    tx = (0x4000 + H265_ABS(td / 2)) / td;
    scale_factor = H265_CLIP((tb * tx + 32) >> 6, -(1 << 12), (1 << 12) - 1);
    dst->x = clip_int16v((scale_factor * src->x + 127 +
                          (scale_factor * src->x < 0)) >> 8);
    dst->y = clip_int16v((scale_factor * src->y + 127 +
                          (scale_factor * src->y < 0)) >> 8);
}

static int check_mvset(H265Mv *mvLXCol, const H265Mv *mvCol, int colPic,
                       int poc, const H265RefList *refPicList, int X,
                       int refIdxLx, const H265RefList *refPicList_col,
                       int listCol, int refidxCol)
{
    int cur_lt = refPicList[X].isLongTerm[refIdxLx];
    int col_lt = refPicList_col[listCol].isLongTerm[refidxCol];
    int col_poc_diff, cur_poc_diff;

    if (cur_lt != col_lt) {
        mvLXCol->x = 0;
        mvLXCol->y = 0;
        return 0;
    }
    col_poc_diff = colPic - refPicList_col[listCol].list[refidxCol];
    cur_poc_diff = poc - refPicList[X].list[refIdxLx];

    if (cur_lt || col_poc_diff == cur_poc_diff || !col_poc_diff) {
        mvLXCol->x = mvCol->x;
        mvLXCol->y = mvCol->y;
    } else {
        mv_scale(mvLXCol, mvCol, col_poc_diff, cur_poc_diff);
    }
    return 1;
}

#define CHECK_MVSET(l)                                                   \
    check_mvset(mvLXCol, temp_col.mv + l, colPic, sh->poc,               \
                refPicList, X, refIdxLx, refPicList_col, H265_L##l,       \
                temp_col.ref_idx[l])

static int derive_temporal_colocated_mvs(H265Slice *sh, H265MvField temp_col,
                                         int refIdxLx, H265Mv *mvLXCol, int X,
                                         int colPic, const H265RefList *refPicList_col)
{
    const H265RefList *refPicList = sh->ref_list;

    if (temp_col.pred_flag == H265_PF_INTRA)
        return 0;

    if (!(temp_col.pred_flag & H265_PF_L0))
        return CHECK_MVSET(1);
    else if (temp_col.pred_flag == H265_PF_L0)
        return CHECK_MVSET(0);
    else if (temp_col.pred_flag == H265_PF_BI) {
        int check_diffpicount = 0;
        int i, j;
        for (j = 0; j < 2; j++) {
            for (i = 0; i < refPicList[j].nb_refs; i++) {
                if (refPicList[j].list[i] > sh->poc) {
                    check_diffpicount++;
                    break;
                }
            }
        }
        if (!check_diffpicount) {
            if (X == 0)
                return CHECK_MVSET(0);
            else
                return CHECK_MVSET(1);
        } else {
            if (sh->collocated_list == H265_L1)
                return CHECK_MVSET(0);
            else
                return CHECK_MVSET(1);
        }
    }
    return 0;
}

static const H265RefList *get_ref_list(const H265Frame *frame, int x0, int y0,
                                       int log2_ctb_size, int ctb_width,
                                       const H265PPS *pps)
{
    int x_cb = x0 >> log2_ctb_size;
    int y_cb = y0 >> log2_ctb_size;
    int ctb_addr_ts = pps->ctb_addr_rs_to_ts[y_cb * ctb_width + x_cb];
    return &frame->rpl_tab[ctb_addr_ts].refPicList[0];
}

#define DERIVE_TEMPORAL_COLOCATED_MVS                                   \
    derive_temporal_colocated_mvs(sh, temp_col, refIdxLx, mvLXCol, X,   \
                                  colPic, ff_ref_list)

static int temporal_luma_motion_vector(H265Slice *sh, int x0, int y0,
                                       int nPbW, int nPbH, int refIdxLx,
                                       H265Mv *mvLXCol, int X)
{
    H265SPS *sps = sh->sps;
    const H265Frame *ref = sh->collocated_ref;
    const H265MvField *tab_mvf;
    H265MvField temp_col;
    int x, y, x_pu, y_pu;
    int min_pu_width = sps->min_pu_width;
    int availableFlagLXCol = 0;
    int colPic;
    const H265RefList *ff_ref_list;

    if (!ref) {
        memset(mvLXCol, 0, sizeof(*mvLXCol));
        return 0;
    }
    tab_mvf = ref->tab_mvf;
    colPic = ref->poc;

    x = x0 + nPbW;
    y = y0 + nPbH;
    if (tab_mvf &&
        (y0 >> sps->log2_ctb_size) == (y >> sps->log2_ctb_size) &&
        y < sps->height && x < sps->width) {
        x &= ~15;
        y &= ~15;
        h265_frame_wait_rows((H265Frame *)ref, y + 1);
        x_pu = x >> sps->log2_min_pu_size;
        y_pu = y >> sps->log2_min_pu_size;
        temp_col = tab_mvf[y_pu * min_pu_width + x_pu];
        ff_ref_list = get_ref_list(ref, x, y, sps->log2_ctb_size,
                                   sps->ctb_width, ref->pps);
        availableFlagLXCol = DERIVE_TEMPORAL_COLOCATED_MVS;
    }
    if (tab_mvf && !availableFlagLXCol) {
        x = x0 + (nPbW >> 1);
        y = y0 + (nPbH >> 1);
        x &= ~15;
        y &= ~15;
        h265_frame_wait_rows((H265Frame *)ref, y + 1);
        x_pu = x >> sps->log2_min_pu_size;
        y_pu = y >> sps->log2_min_pu_size;
        temp_col = tab_mvf[y_pu * min_pu_width + x_pu];
        ff_ref_list = get_ref_list(ref, x, y, sps->log2_ctb_size,
                                   sps->ctb_width, ref->pps);
        availableFlagLXCol = DERIVE_TEMPORAL_COLOCATED_MVS;
    }
    return availableFlagLXCol;
}

static void derive_spatial_merge_candidates(H265Slice *sh, int x0, int y0,
                                            int nPbW, int nPbH, int log2_cb_size,
                                            int singleMCLFlag, int part_idx,
                                            int merge_idx, H265MvField mergecandlist[])
{
    H265PPS *pps = sh->pps;
    H265SPS *sps = sh->sps;
    const H265RefList *refPicList = sh->ref_list;
    const H265MvField *tab_mvf = sh->frame->tab_mvf;
    const int min_pu_width = sps->min_pu_width;
    const int cand_bottom_left = sh->na.cand_bottom_left;
    const int cand_left = sh->na.cand_left;
    const int cand_up_left = sh->na.cand_up_left;
    const int cand_up = sh->na.cand_up;
    const int cand_up_right = sh->na.cand_up_right_sap;
    const int xA1 = x0 - 1, yA1 = y0 + nPbH - 1;
    const int xB1 = x0 + nPbW - 1, yB1 = y0 - 1;
    const int xB0 = x0 + nPbW, yB0 = y0 - 1;
    const int xA0 = x0 - 1, yA0 = y0 + nPbH;
    const int xB2 = x0 - 1, yB2 = y0 - 1;
    const int nb_refs = (sh->slice_type == H265_SLICE_P) ?
                        (int)sh->nb_refs[0] :
                        H265_MIN((int)sh->nb_refs[0], (int)sh->nb_refs[1]);
    int zero_idx = 0;
    int nb_merge_cand = 0;
    int nb_orig_merge_cand = 0;
    int is_available_a0, is_available_a1, is_available_b0, is_available_b1, is_available_b2;

#define TAB_MVF(x, y) (tab_mvf[(y) * min_pu_width + (x)])
#define TAB_MVF_PU(v) TAB_MVF(((x##v) >> sps->log2_min_pu_size), \
                              ((y##v) >> sps->log2_min_pu_size))
#define AVAILABLE(cand, v) (cand && !(TAB_MVF_PU(v).pred_flag == H265_PF_INTRA))
#define PRED_BLOCK_AVAILABLE(v) z_scan_block_avail(sh, x0, y0, x##v, y##v)
#define COMPARE_MV_REFIDX(a, b) compare_mv_ref_idx(TAB_MVF_PU(a), TAB_MVF_PU(b))

    if (((!singleMCLFlag && part_idx == 1 &&
          (sh->cu.part_mode == H265_PART_Nx2N ||
           sh->cu.part_mode == H265_PART_nLx2N ||
           sh->cu.part_mode == H265_PART_nRx2N)) ||
         is_diff_mer(pps, xA1, yA1, x0, y0))) {
        is_available_a1 = 0;
    } else {
        is_available_a1 = AVAILABLE(cand_left, A1);
        if (is_available_a1) {
            mergecandlist[nb_merge_cand] = TAB_MVF_PU(A1);
            if (merge_idx == 0)
                return;
            nb_merge_cand++;
        }
    }

    if (((!singleMCLFlag && part_idx == 1 &&
          (sh->cu.part_mode == H265_PART_2NxN ||
           sh->cu.part_mode == H265_PART_2NxnU ||
           sh->cu.part_mode == H265_PART_2NxnD)) ||
         is_diff_mer(pps, xB1, yB1, x0, y0))) {
        is_available_b1 = 0;
    } else {
        is_available_b1 = AVAILABLE(cand_up, B1);
        if (is_available_b1 && !(is_available_a1 && COMPARE_MV_REFIDX(B1, A1))) {
            mergecandlist[nb_merge_cand] = TAB_MVF_PU(B1);
            if (merge_idx == nb_merge_cand)
                return;
            nb_merge_cand++;
        }
    }

    is_available_b0 = AVAILABLE(cand_up_right, B0) &&
                      xB0 < sps->width && PRED_BLOCK_AVAILABLE(B0) &&
                      !is_diff_mer(pps, xB0, yB0, x0, y0);
    if (is_available_b0 && !(is_available_b1 && COMPARE_MV_REFIDX(B0, B1))) {
        mergecandlist[nb_merge_cand] = TAB_MVF_PU(B0);
        if (merge_idx == nb_merge_cand)
            return;
        nb_merge_cand++;
    }

    is_available_a0 = AVAILABLE(cand_bottom_left, A0) &&
                      yA0 < sps->height && PRED_BLOCK_AVAILABLE(A0) &&
                      !is_diff_mer(pps, xA0, yA0, x0, y0);
    if (is_available_a0 && !(is_available_a1 && COMPARE_MV_REFIDX(A0, A1))) {
        mergecandlist[nb_merge_cand] = TAB_MVF_PU(A0);
        if (merge_idx == nb_merge_cand)
            return;
        nb_merge_cand++;
    }

    is_available_b2 = AVAILABLE(cand_up_left, B2) &&
                      !is_diff_mer(pps, xB2, yB2, x0, y0);
    if (is_available_b2 &&
        !(is_available_a1 && COMPARE_MV_REFIDX(B2, A1)) &&
        !(is_available_b1 && COMPARE_MV_REFIDX(B2, B1)) &&
        nb_merge_cand != 4) {
        mergecandlist[nb_merge_cand] = TAB_MVF_PU(B2);
        if (merge_idx == nb_merge_cand)
            return;
        nb_merge_cand++;
    }

    if (sh->slice_temporal_mvp_enabled_flag &&
        nb_merge_cand < sh->max_num_merge_cand) {
        H265Mv mv_l0_col = { 0 }, mv_l1_col = { 0 };
        int available_l0 = temporal_luma_motion_vector(sh, x0, y0, nPbW, nPbH,
                                                       0, &mv_l0_col, 0);
        int available_l1 = (sh->slice_type == H265_SLICE_B) ?
                           temporal_luma_motion_vector(sh, x0, y0, nPbW, nPbH,
                                                       0, &mv_l1_col, 1) : 0;
        if (available_l0 || available_l1) {
            mergecandlist[nb_merge_cand].pred_flag = available_l0 + (available_l1 << 1);
            mergecandlist[nb_merge_cand].ref_idx[0] = 0;
            mergecandlist[nb_merge_cand].ref_idx[1] = 0;
            mergecandlist[nb_merge_cand].mv[0] = mv_l0_col;
            mergecandlist[nb_merge_cand].mv[1] = mv_l1_col;
            if (merge_idx == nb_merge_cand)
                return;
            nb_merge_cand++;
        }
    }

    nb_orig_merge_cand = nb_merge_cand;

    if (sh->slice_type == H265_SLICE_B && nb_orig_merge_cand > 1 &&
        nb_orig_merge_cand < sh->max_num_merge_cand) {
        int comb_idx;
        for (comb_idx = 0; nb_merge_cand < sh->max_num_merge_cand &&
                           comb_idx < nb_orig_merge_cand * (nb_orig_merge_cand - 1);
             comb_idx++) {
            int l0_cand_idx = l0_l1_cand_idx[comb_idx][0];
            int l1_cand_idx = l0_l1_cand_idx[comb_idx][1];
            H265MvField l0_cand = mergecandlist[l0_cand_idx];
            H265MvField l1_cand = mergecandlist[l1_cand_idx];

            if ((l0_cand.pred_flag & H265_PF_L0) &&
                (l1_cand.pred_flag & H265_PF_L1) &&
                (refPicList[0].list[l0_cand.ref_idx[0]] !=
                     refPicList[1].list[l1_cand.ref_idx[1]] ||
                 l0_cand.mv[0].x != l1_cand.mv[1].x ||
                 l0_cand.mv[0].y != l1_cand.mv[1].y)) {
                mergecandlist[nb_merge_cand].ref_idx[0] = l0_cand.ref_idx[0];
                mergecandlist[nb_merge_cand].ref_idx[1] = l1_cand.ref_idx[1];
                mergecandlist[nb_merge_cand].pred_flag = H265_PF_BI;
                mergecandlist[nb_merge_cand].mv[0] = l0_cand.mv[0];
                mergecandlist[nb_merge_cand].mv[1] = l1_cand.mv[1];
                if (merge_idx == nb_merge_cand)
                    return;
                nb_merge_cand++;
            }
        }
    }

    while (nb_merge_cand < sh->max_num_merge_cand) {
        mergecandlist[nb_merge_cand].pred_flag =
            H265_PF_L0 + ((sh->slice_type == H265_SLICE_B) << 1);
        mergecandlist[nb_merge_cand].mv[0].x = 0;
        mergecandlist[nb_merge_cand].mv[0].y = 0;
        mergecandlist[nb_merge_cand].mv[1].x = 0;
        mergecandlist[nb_merge_cand].mv[1].y = 0;
        mergecandlist[nb_merge_cand].ref_idx[0] = zero_idx < nb_refs ? zero_idx : 0;
        mergecandlist[nb_merge_cand].ref_idx[1] = zero_idx < nb_refs ? zero_idx : 0;
        if (merge_idx == nb_merge_cand)
            return;
        nb_merge_cand++;
        zero_idx++;
    }
#undef TAB_MVF
#undef TAB_MVF_PU
#undef AVAILABLE
#undef PRED_BLOCK_AVAILABLE
#undef COMPARE_MV_REFIDX
}

void h265_luma_mv_merge_mode(H265Slice *sh, int x0, int y0, int nPbW, int nPbH,
                             int log2_cb_size, int part_idx, int merge_idx,
                             H265MvField *mv)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    int singleMCLFlag = 0;
    int nCS = 1 << log2_cb_size;
    H265MvField mergecand_list[H265_MRG_MAX_NUM_CANDS];

    if (pps->log2_parallel_merge_level > 2 && nCS == 8) {
        singleMCLFlag = 1;
        x0 = sh->cu.x;
        y0 = sh->cu.y;
        nPbW = nCS;
        nPbH = nCS;
        part_idx = 0;
    }
    h265_set_neighbour_available(sh, x0, y0, nPbW, nPbH, sps->log2_ctb_size);
    derive_spatial_merge_candidates(sh, x0, y0, nPbW, nPbH, log2_cb_size,
                                    singleMCLFlag, part_idx, merge_idx,
                                    mergecand_list);

    if (mergecand_list[merge_idx].pred_flag == H265_PF_BI &&
        (nPbW + nPbH) == 12)
        mergecand_list[merge_idx].pred_flag = H265_PF_L0;

    *mv = mergecand_list[merge_idx];
}

H265_INLINE void dist_scale(H265Slice *sh, H265Mv *mv, int x, int y,
                                   int elist, int ref_idx_curr, int ref_idx)
{
    const H265RefList *refPicList = sh->ref_list;
    const H265MvField *tab_mvf = sh->frame->tab_mvf;
    int ref_pic_elist = refPicList[elist].list[tab_mvf[y * sh->sps->min_pu_width + x].ref_idx[elist]];
    int ref_pic_curr = refPicList[ref_idx_curr].list[ref_idx];

    if (ref_pic_elist != ref_pic_curr) {
        int poc_diff = sh->poc - ref_pic_elist;
        if (!poc_diff)
            poc_diff = 1;
        mv_scale(mv, mv, poc_diff, sh->poc - ref_pic_curr);
    }
}

static int mv_mp_mode_mx(H265Slice *sh, int x, int y, int pred_flag_index,
                         H265Mv *mv, int ref_idx_curr, int ref_idx)
{
    const H265MvField *tab_mvf = sh->frame->tab_mvf;
    const H265RefList *refPicList = sh->ref_list;
    int min_pu_width = sh->sps->min_pu_width;

    if ((tab_mvf[y * min_pu_width + x].pred_flag & (1 << pred_flag_index)) &&
        refPicList[pred_flag_index].list[tab_mvf[y * min_pu_width + x].ref_idx[pred_flag_index]] ==
            refPicList[ref_idx_curr].list[ref_idx]) {
        *mv = tab_mvf[y * min_pu_width + x].mv[pred_flag_index];
        return 1;
    }
    return 0;
}

static int mv_mp_mode_mx_lt(H265Slice *sh, int x, int y, int pred_flag_index,
                            H265Mv *mv, int ref_idx_curr, int ref_idx)
{
    const H265MvField *tab_mvf = sh->frame->tab_mvf;
    const H265RefList *refPicList = sh->ref_list;
    int min_pu_width = sh->sps->min_pu_width;

    if (tab_mvf[y * min_pu_width + x].pred_flag & (1 << pred_flag_index)) {
        int currIsLongTerm = refPicList[ref_idx_curr].isLongTerm[ref_idx];
        int colIsLongTerm = refPicList[pred_flag_index].isLongTerm[
            tab_mvf[y * min_pu_width + x].ref_idx[pred_flag_index]];
        if (colIsLongTerm == currIsLongTerm) {
            *mv = tab_mvf[y * min_pu_width + x].mv[pred_flag_index];
            if (!currIsLongTerm)
                dist_scale(sh, mv, x, y, pred_flag_index, ref_idx_curr, ref_idx);
            return 1;
        }
    }
    return 0;
}

#define MP_MX(v, pred, mx)                                      \
    mv_mp_mode_mx(sh,                                           \
                  (x##v) >> sh->sps->log2_min_pu_size,          \
                  (y##v) >> sh->sps->log2_min_pu_size,          \
                  pred, &mx, ref_idx_curr, ref_idx)
#define MP_MX_LT(v, pred, mx)                                   \
    mv_mp_mode_mx_lt(sh,                                        \
                     (x##v) >> sh->sps->log2_min_pu_size,      \
                     (y##v) >> sh->sps->log2_min_pu_size,      \
                     pred, &mx, ref_idx_curr, ref_idx)

void h265_luma_mv_mvp_mode(H265Slice *sh, int x0, int y0, int nPbW, int nPbH,
                           int log2_cb_size, int part_idx, int merge_idx,
                           H265MvField *mv, int mvp_lx_flag, int LX)
{
    H265SPS *sps = sh->sps;
    int isScaledFlag_L0 = 0;
    int availableFlagLXA0 = 1;
    int availableFlagLXB0 = 1;
    int numMVPCandLX = 0;
    int min_pu_width = sps->min_pu_width;
    int xA0, yA0, is_available_a0;
    int xA1, yA1, is_available_a1;
    int xB0, yB0, is_available_b0;
    int xB1, yB1, is_available_b1;
    int xB2, yB2, is_available_b2;
    H265Mv mvpcand_list[2] = { { 0, 0 }, { 0, 0 } };
    H265Mv mxA, mxB;
    int ref_idx_curr;
    int ref_idx = 0;
    int pred_flag_index_l0;
    int pred_flag_index_l1;
    const int cand_bottom_left = sh->na.cand_bottom_left;
    const int cand_left = sh->na.cand_left;
    const int cand_up_left = sh->na.cand_up_left;
    const int cand_up = sh->na.cand_up;
    const int cand_up_right = sh->na.cand_up_right_sap;
    const H265MvField *tab_mvf = sh->frame->tab_mvf;

#define TAB_MVF(x, y) (tab_mvf[(y) * min_pu_width + (x)])
#define TAB_MVF_PU(v) TAB_MVF(((x##v) >> sps->log2_min_pu_size), \
                              ((y##v) >> sps->log2_min_pu_size))
#define AVAILABLE(cand, v) (cand && !(TAB_MVF_PU(v).pred_flag == H265_PF_INTRA))
#define PRED_BLOCK_AVAILABLE(v) z_scan_block_avail(sh, x0, y0, x##v, y##v)

    ref_idx_curr = LX;
    ref_idx = mv->ref_idx[LX];
    pred_flag_index_l0 = LX;
    pred_flag_index_l1 = !LX;
    (void)tab_mvf;

    xA0 = x0 - 1;
    yA0 = y0 + nPbH;
    is_available_a0 = AVAILABLE(cand_bottom_left, A0) &&
                      yA0 < sps->height && PRED_BLOCK_AVAILABLE(A0);

    xA1 = x0 - 1;
    yA1 = y0 + nPbH - 1;
    is_available_a1 = AVAILABLE(cand_left, A1);
    if (is_available_a0 || is_available_a1)
        isScaledFlag_L0 = 1;

    if (is_available_a0) {
        if (MP_MX(A0, pred_flag_index_l0, mxA))
            goto b_candidates;
        if (MP_MX(A0, pred_flag_index_l1, mxA))
            goto b_candidates;
    }
    if (is_available_a1) {
        if (MP_MX(A1, pred_flag_index_l0, mxA))
            goto b_candidates;
        if (MP_MX(A1, pred_flag_index_l1, mxA))
            goto b_candidates;
    }
    if (is_available_a0) {
        if (MP_MX_LT(A0, pred_flag_index_l0, mxA))
            goto b_candidates;
        if (MP_MX_LT(A0, pred_flag_index_l1, mxA))
            goto b_candidates;
    }
    if (is_available_a1) {
        if (MP_MX_LT(A1, pred_flag_index_l0, mxA))
            goto b_candidates;
        if (MP_MX_LT(A1, pred_flag_index_l1, mxA))
            goto b_candidates;
    }
    availableFlagLXA0 = 0;

b_candidates:
    xB0 = x0 + nPbW;
    yB0 = y0 - 1;
    is_available_b0 = AVAILABLE(cand_up_right, B0) &&
                      xB0 < sps->width && PRED_BLOCK_AVAILABLE(B0);
    xB1 = x0 + nPbW - 1;
    yB1 = y0 - 1;
    is_available_b1 = AVAILABLE(cand_up, B1);
    xB2 = x0 - 1;
    yB2 = y0 - 1;
    is_available_b2 = AVAILABLE(cand_up_left, B2);

    if (is_available_b0) {
        if (MP_MX(B0, pred_flag_index_l0, mxB))
            goto scalef;
        if (MP_MX(B0, pred_flag_index_l1, mxB))
            goto scalef;
    }
    if (is_available_b1) {
        if (MP_MX(B1, pred_flag_index_l0, mxB))
            goto scalef;
        if (MP_MX(B1, pred_flag_index_l1, mxB))
            goto scalef;
    }
    if (is_available_b2) {
        if (MP_MX(B2, pred_flag_index_l0, mxB))
            goto scalef;
        if (MP_MX(B2, pred_flag_index_l1, mxB))
            goto scalef;
    }
    availableFlagLXB0 = 0;

scalef:
    if (!isScaledFlag_L0) {
        if (availableFlagLXB0) {
            availableFlagLXA0 = 1;
            mxA = mxB;
        }
        availableFlagLXB0 = 0;
        if (is_available_b0) {
            availableFlagLXB0 = MP_MX_LT(B0, pred_flag_index_l0, mxB);
            if (!availableFlagLXB0)
                availableFlagLXB0 = MP_MX_LT(B0, pred_flag_index_l1, mxB);
        }
        if (is_available_b1 && !availableFlagLXB0) {
            availableFlagLXB0 = MP_MX_LT(B1, pred_flag_index_l0, mxB);
            if (!availableFlagLXB0)
                availableFlagLXB0 = MP_MX_LT(B1, pred_flag_index_l1, mxB);
        }
        if (is_available_b2 && !availableFlagLXB0) {
            availableFlagLXB0 = MP_MX_LT(B2, pred_flag_index_l0, mxB);
            if (!availableFlagLXB0)
                availableFlagLXB0 = MP_MX_LT(B2, pred_flag_index_l1, mxB);
        }
    }

    if (availableFlagLXA0)
        mvpcand_list[numMVPCandLX++] = mxA;
    if (availableFlagLXB0 && (!availableFlagLXA0 ||
                              mxA.x != mxB.x || mxA.y != mxB.y))
        mvpcand_list[numMVPCandLX++] = mxB;

    if (numMVPCandLX < 2 && sh->slice_temporal_mvp_enabled_flag &&
        mvp_lx_flag == numMVPCandLX) {
        H265Mv mv_col;
        int available_col = temporal_luma_motion_vector(sh, x0, y0, nPbW, nPbH,
                                                        ref_idx, &mv_col, LX);
        if (available_col)
            mvpcand_list[numMVPCandLX++] = mv_col;
    }

    mv->mv[LX] = mvpcand_list[mvp_lx_flag];
#undef TAB_MVF
#undef TAB_MVF_PU
#undef AVAILABLE
#undef PRED_BLOCK_AVAILABLE
}
