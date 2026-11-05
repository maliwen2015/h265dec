/*
 * HEVC CABAC engine and syntax element decoding.
 *
 * Engine ported from FFmpeg's cabac_functions.h / cabac.c and the syntax
 * decoder from hevc/cabac.c (LGPL-2.1+).
 */
#include "h265_internal.h"

#define H265_CABAC_NORM_SHIFT  (h265_cabac_tables + H265_NORM_SHIFT_OFFSET)
#define H265_CABAC_LPS_RANGE   (h265_cabac_tables + H265_LPS_RANGE_OFFSET)
#define H265_CABAC_MLPS_STATE  (h265_cabac_tables + H265_MLPS_STATE_OFFSET)

#define CABAC_MAX_BIN 31

/* Context offsets from the HEVC CABAC_ELEMS table (FFmpeg hevc/cabac.c). */
enum {
    SAO_MERGE_FLAG_OFFSET = 0,
    SAO_TYPE_IDX_OFFSET = 1,
    SPLIT_CODING_UNIT_FLAG_OFFSET = 2,
    CU_TRANSQUANT_BYPASS_FLAG_OFFSET = 5,
    SKIP_FLAG_OFFSET = 6,
    CU_QP_DELTA_OFFSET = 9,
    PRED_MODE_FLAG_OFFSET = 12,
    PART_MODE_OFFSET = 13,
    PREV_INTRA_LUMA_PRED_FLAG_OFFSET = 17,
    INTRA_CHROMA_PRED_MODE_OFFSET = 18,
    MERGE_FLAG_OFFSET = 20,
    MERGE_IDX_OFFSET = 21,
    INTER_PRED_IDC_OFFSET = 22,
    REF_IDX_L0_OFFSET = 27,
    ABS_MVD_GREATER0_FLAG_OFFSET = 31,
    ABS_MVD_GREATER1_FLAG_OFFSET = 33,
    MVP_LX_FLAG_OFFSET = 35,
    NO_RESIDUAL_DATA_FLAG_OFFSET = 36,
    SPLIT_TRANSFORM_FLAG_OFFSET = 37,
    CBF_LUMA_OFFSET = 40,
    CBF_CB_CR_OFFSET = 42,
    TRANSFORM_SKIP_FLAG_OFFSET = 47,
    EXPLICIT_RDPCM_FLAG_OFFSET = 49,
    EXPLICIT_RDPCM_DIR_FLAG_OFFSET = 51,
    LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET = 53,
    LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET = 71,
    SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET = 89,
    SIGNIFICANT_COEFF_FLAG_OFFSET = 93,
    COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET = 137,
    COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET = 161,
    LOG2_RES_SCALE_ABS_OFFSET = 167,
    RES_SCALE_SIGN_FLAG_OFFSET = 175,
    CU_CHROMA_QP_OFFSET_FLAG_OFFSET = 177,
    CU_CHROMA_QP_OFFSET_IDX_OFFSET = 178,
};

H265_INLINE int cabac_byte(const H265CABAC *restrict c, int off)
{
    const uint8_t *p = c->bytestream + off;
    return p < c->bytestream_end ? *p : 0;
}

H265_INLINE void cabac_refill(H265CABAC *restrict c)
{
    const uint8_t *p = c->bytestream;
    if (p + 1 < c->bytestream_end)
        c->low += (p[0] << 9) + (p[1] << 1);
    else
        c->low += (cabac_byte(c, 0) << 9) + (cabac_byte(c, 1) << 1);
    c->low -= H265_CABAC_MASK;
    c->bytestream += H265_CABAC_BITS / 8;
}

H265_INLINE void cabac_refill2(H265CABAC *restrict c)
{
    int i = __builtin_ctz((unsigned)c->low) - H265_CABAC_BITS;
    unsigned x = -H265_CABAC_MASK;
    {
        const uint8_t *p = c->bytestream;
        if (p + 1 < c->bytestream_end)
            x += (p[0] << 9) + (p[1] << 1);
        else
            x += (cabac_byte(c, 0) << 9) + (cabac_byte(c, 1) << 1);
    }
    c->low += x << i;
    c->bytestream += H265_CABAC_BITS / 8;
}

H265_INLINE void cabac_renorm_once(H265CABAC *restrict c)
{
    int shift = (uint32_t)(c->range - 0x100) >> 31;
    c->range <<= shift;
    c->low <<= shift;
    if (!(uint16_t)c->low)
        cabac_refill(c);
}

H265_INLINE int get_cabac(H265CABAC *restrict c, uint8_t *restrict const state)
{
    int s = *state;
    int RangeLPS = H265_CABAC_LPS_RANGE[(unsigned)(2 * (c->range & 0xC0) + s)];
    int bit, lps_mask;

    c->range -= RangeLPS;
    lps_mask = ((c->range << (H265_CABAC_BITS + 1)) - c->low) >> 31;

    c->low -= (c->range << (H265_CABAC_BITS + 1)) & lps_mask;
    c->range += (RangeLPS - c->range) & lps_mask;

    s ^= lps_mask;
    *state = (H265_CABAC_MLPS_STATE + 128)[s];
    bit = s & 1;

    lps_mask = H265_CABAC_NORM_SHIFT[(unsigned)c->range];
    c->range <<= lps_mask;
    c->low <<= lps_mask;
    if (!(uint16_t)c->low)
        cabac_refill2(c);
    return bit;
}

#define get_cabac_noinline get_cabac

H265_INLINE int get_cabac_bypass(H265CABAC *restrict c)
{
    int range;
    c->low += c->low;

    if (!(uint16_t)c->low)
        cabac_refill(c);

    range = c->range << (H265_CABAC_BITS + 1);
    if (c->low < range)
        return 0;
    c->low -= range;
    return 1;
}

H265_INLINE int get_cabac_bypass_sign(H265CABAC *restrict c, int val)
{
    int range, mask;
    c->low += c->low;

    if (!(uint16_t)c->low)
        cabac_refill(c);

    range = c->range << (H265_CABAC_BITS + 1);
    c->low -= range;
    mask = c->low >> 31;
    range &= mask;
    c->low += range;
    return (val ^ mask) - mask;
}

H265_INLINE int get_cabac_terminate(H265CABAC *restrict c)
{
    c->range -= 2;
    if (c->low < c->range << (H265_CABAC_BITS + 1)) {
        cabac_renorm_once(c);
        return 0;
    }
    return 1;
}

static int cabac_reinit(H265Slice *sh)
{
    H265CABAC *c = &sh->cabac;
    const uint8_t *ptr = c->bytestream;
    int size;

    if (c->low & 0x1)
        ptr--;
    if (c->low & 0x1FF)
        ptr--;
    if (ptr < c->bytestream_start)
        ptr = c->bytestream_start;
    size = (int)(c->bytestream_end - ptr);
    if (size < 0)
        return -1;
    return h265_cabac_init(sh, ptr, size);
}

int h265_cabac_init(H265Slice *sh, const uint8_t *data, int size)
{
    H265CABAC *c = &sh->cabac;
    const uint8_t *p = data;

    if (size < 1)
        return -1;

    c->bytestream_start = c->bytestream = data;
    c->bytestream_end = data + size;

    c->low = (cabac_byte(c, 0) << 18) + (cabac_byte(c, 1) << 10);
    p += 2;
    if (((uintptr_t)p & 1) == 0) {
        c->low += 1 << 9;
    } else {
        c->low += (p < c->bytestream_end ? *p : 0) * 4 + 2;
        p++;
    }
    c->bytestream = p;
    c->range = 0x1FE;
    if ((c->range << (H265_CABAC_BITS + 1)) < c->low)
        return -1;
    return 0;
}

/* init_values[3][199], from FFmpeg's hevc/cabac.c */
static const uint8_t init_values[3][H265_CONTEXTS] = {
    {
      153, 200, 139, 141, 157, 154, 154, 154, 154, 154, 154, 154, 154, 184, 154, 154,
      154, 184, 63, 139, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
      154, 154, 154, 154, 154, 153, 138, 138, 111, 141, 94, 138, 182, 154, 154, 139,
      139, 139, 139, 139, 139, 110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111,
      143, 127, 111, 79, 108, 123, 63, 110, 110, 124, 125, 140, 153, 125, 127, 140,
      109, 111, 143, 127, 111, 79, 108, 123, 63, 91, 171, 134, 141, 111, 111, 125,
      110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179,
      153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136,
      153, 136, 139, 111, 136, 139, 111, 141, 111, 140, 92, 137, 138, 140, 152, 138,
      139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122,
      197, 138, 153, 136, 167, 152, 152, 154, 154, 154, 154, 154, 154, 154, 154, 154,
      154, 154, 154
    },
    {
      153, 185, 107, 139, 126, 154, 197, 185, 201, 154, 154, 154, 149, 154, 139, 154,
      154, 154, 152, 139, 110, 122, 95, 79, 63, 31, 31, 153, 153, 153, 153, 140,
      198, 140, 198, 168, 79, 124, 138, 94, 153, 111, 149, 107, 167, 154, 154, 139,
      139, 139, 139, 139, 139, 125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110,
      111, 111, 95, 94, 108, 123, 108, 125, 110, 94, 110, 95, 79, 125, 111, 110,
      78, 110, 111, 111, 95, 94, 108, 123, 108, 121, 140, 61, 154, 155, 154, 139,
      153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136,
      153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121,
      167, 151, 183, 140, 151, 183, 140, 140, 140, 154, 196, 196, 167, 154, 152, 167,
      182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137,
      182, 107, 167, 91, 122, 107, 167, 154, 154, 154, 154, 154, 154, 154, 154, 154,
      154, 154, 154
    },
    {
      153, 160, 107, 139, 126, 154, 197, 185, 201, 154, 154, 154, 134, 154, 139, 154,
      154, 183, 152, 139, 154, 137, 95, 79, 63, 31, 31, 153, 153, 153, 153, 169,
      198, 169, 198, 168, 79, 224, 167, 122, 153, 111, 149, 92, 167, 154, 154, 139,
      139, 139, 139, 139, 139, 125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125,
      126, 111, 111, 79, 108, 123, 93, 125, 110, 124, 110, 95, 94, 125, 111, 111,
      79, 125, 126, 111, 111, 79, 108, 123, 93, 121, 140, 61, 154, 170, 154, 139,
      153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136,
      153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121,
      167, 151, 183, 140, 151, 183, 140, 140, 140, 154, 196, 167, 167, 154, 152, 167,
      182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208, 166, 167, 154, 152, 167,
      182, 107, 167, 91, 107, 107, 167, 154, 154, 154, 154, 154, 154, 154, 154, 154,
      154, 154, 154
    },
};

int h265_cabac_init_states(H265Slice *sh)
{
    int init_type = 2 - sh->slice_type;
    int i;

    if (sh->cabac_init_flag && sh->slice_type != H265_SLICE_I)
        init_type ^= 3;

    for (i = 0; i < H265_CONTEXTS; i++) {
        int init_value = init_values[init_type][i];
        int m = (init_value >> 4) * 5 - 45;
        int n = ((init_value & 15) << 3) - 16;
        int qp = H265_CLIP(sh->slice_qp, 0, 51);
        int pre = 2 * (((m * qp) >> 4) + n) - 127;

        pre ^= pre >> 31;
        if (pre > 124)
            pre = 124 + (pre & 1);
        sh->cabac_state[i] = pre;
    }
    for (i = 0; i < H265_STAT_COEFFS; i++)
        sh->stat_coeff[i] = 0;
    return 0;
}

void h265_save_states(H265Slice *sh, int ctb_addr_ts)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;

    if (pps->entropy_coding_sync_enabled_flag &&
        (ctb_addr_ts % sps->ctb_width == 2 ||
         (sps->ctb_width == 2 && ctb_addr_ts % sps->ctb_width == 0))) {
        memcpy(sh->cabac_saved, sh->cabac_state, H265_CONTEXTS);
        if (sps->persistent_rice_adaptation_enabled)
            memcpy(sh->stat_coeff_saved, sh->stat_coeff, H265_STAT_COEFFS);
    }
}

int h265_cabac_reinit(H265Slice *sh)
{
    return cabac_reinit(sh);
}

/* ------------------------------------------------------------------ */
/* syntax elements                                                     */
/* ------------------------------------------------------------------ */
#define GET_CABAC(ctx) get_cabac(&sh->cabac, &sh->cabac_state[ctx])
#define SAMPLE_CTB(tab, x, y) ((tab)[(y) * min_cb_width + (x)])

int h265_sao_merge_flag_decode(H265Slice *sh)
{
    return GET_CABAC(SAO_MERGE_FLAG_OFFSET);
}

int h265_sao_type_idx_decode(H265Slice *sh)
{
    if (!GET_CABAC(SAO_TYPE_IDX_OFFSET))
        return 0;
    if (!get_cabac_bypass(&sh->cabac))
        return H265_SAO_BAND;
    return H265_SAO_EDGE;
}

int h265_sao_band_position_decode(H265Slice *sh)
{
    int i;
    int value = get_cabac_bypass(&sh->cabac);
    for (i = 0; i < 4; i++)
        value = (value << 1) | get_cabac_bypass(&sh->cabac);
    return value;
}

int h265_sao_offset_abs_decode(H265Slice *sh, int bit_depth)
{
    int i = 0;
    int length = (1 << (H265_MIN(bit_depth, 10) - 5)) - 1;
    while (i < length && get_cabac_bypass(&sh->cabac))
        i++;
    return i;
}

int h265_sao_offset_sign_decode(H265Slice *sh)
{
    return get_cabac_bypass(&sh->cabac);
}

int h265_sao_eo_class_decode(H265Slice *sh)
{
    int ret = get_cabac_bypass(&sh->cabac) << 1;
    ret |= get_cabac_bypass(&sh->cabac);
    return ret;
}

int h265_end_of_slice_flag_decode(H265Slice *sh)
{
    return get_cabac_terminate(&sh->cabac);
}

int h265_cu_transquant_bypass_flag_decode(H265Slice *sh)
{
    return GET_CABAC(CU_TRANSQUANT_BYPASS_FLAG_OFFSET);
}

int h265_skip_flag_decode(H265Slice *sh, uint8_t *skip_flag, int x0, int y0,
                          int x_cb, int y_cb, int min_cb_width)
{
    int inc = 0;
    H265SPS *sps = sh->sps;
    if (sh->ctb_left_flag || x0)
        inc = !!SAMPLE_CTB(skip_flag, x_cb - 1, y_cb);
    if (sh->ctb_up_flag || y0)
        inc += !!SAMPLE_CTB(skip_flag, x_cb, y_cb - 1);
    (void)sps;
    return GET_CABAC(SKIP_FLAG_OFFSET + inc);
}

int h265_pred_mode_decode(H265Slice *sh)
{
    return GET_CABAC(PRED_MODE_FLAG_OFFSET);
}

int h265_split_coding_unit_flag_decode(H265Slice *sh, uint8_t *tab_ct_depth,
                                       const H265SPS *sps, int ct_depth,
                                       int x0, int y0)
{
    int inc = 0, depth_left = 0, depth_top = 0;
    int x0b = x0 & ((1 << sps->log2_ctb_size) - 1);
    int y0b = y0 & ((1 << sps->log2_ctb_size) - 1);
    int x_cb = x0 >> sps->log2_min_cb_size;
    int y_cb = y0 >> sps->log2_min_cb_size;

    if (sh->ctb_left_flag || x0b)
        depth_left = tab_ct_depth[(y_cb) * sps->min_cb_width + x_cb - 1];
    if (sh->ctb_up_flag || y0b)
        depth_top = tab_ct_depth[(y_cb - 1) * sps->min_cb_width + x_cb];

    inc += (depth_left > ct_depth);
    inc += (depth_top > ct_depth);
    return GET_CABAC(SPLIT_CODING_UNIT_FLAG_OFFSET + inc);
}

int h265_part_mode_decode(H265Slice *sh, const H265SPS *sps, int log2_cb_size)
{
    if (GET_CABAC(PART_MODE_OFFSET))
        return H265_PART_2Nx2N;
    if (log2_cb_size == sps->log2_min_cb_size) {
        if (sh->cu.pred_mode == H265_MODE_INTRA)
            return H265_PART_NxN;
        if (GET_CABAC(PART_MODE_OFFSET + 1))
            return H265_PART_2NxN;
        if (log2_cb_size == 3)
            return H265_PART_Nx2N;
        if (GET_CABAC(PART_MODE_OFFSET + 2))
            return H265_PART_Nx2N;
        return H265_PART_NxN;
    }

    if (!sps->amp_enabled) {
        if (GET_CABAC(PART_MODE_OFFSET + 1))
            return H265_PART_2NxN;
        return H265_PART_Nx2N;
    }

    if (GET_CABAC(PART_MODE_OFFSET + 1)) {
        if (GET_CABAC(PART_MODE_OFFSET + 3))
            return H265_PART_2NxN;
        if (get_cabac_bypass(&sh->cabac))
            return H265_PART_2NxnD;
        return H265_PART_2NxnU;
    }
    if (GET_CABAC(PART_MODE_OFFSET + 3))
        return H265_PART_Nx2N;
    if (get_cabac_bypass(&sh->cabac))
        return H265_PART_nRx2N;
    return H265_PART_nLx2N;
}

int h265_pcm_flag_decode(H265Slice *sh)
{
    return get_cabac_terminate(&sh->cabac);
}

int h265_prev_intra_luma_pred_flag_decode(H265Slice *sh)
{
    return GET_CABAC(PREV_INTRA_LUMA_PRED_FLAG_OFFSET);
}

int h265_mpm_idx_decode(H265Slice *sh)
{
    int i = 0;
    while (i < 2 && get_cabac_bypass(&sh->cabac))
        i++;
    return i;
}

int h265_rem_intra_luma_pred_mode_decode(H265Slice *sh)
{
    int i;
    int value = get_cabac_bypass(&sh->cabac);
    for (i = 0; i < 4; i++)
        value = (value << 1) | get_cabac_bypass(&sh->cabac);
    return value;
}

int h265_intra_chroma_pred_mode_decode(H265Slice *sh)
{
    int ret;
    if (!GET_CABAC(INTRA_CHROMA_PRED_MODE_OFFSET))
        return 4;
    ret = get_cabac_bypass(&sh->cabac) << 1;
    ret |= get_cabac_bypass(&sh->cabac);
    return ret;
}

int h265_merge_idx_decode(H265Slice *sh)
{
    int i = GET_CABAC(MERGE_IDX_OFFSET);
    if (i != 0) {
        while (i < sh->max_num_merge_cand - 1 && get_cabac_bypass(&sh->cabac))
            i++;
    }
    return i;
}

int h265_merge_flag_decode(H265Slice *sh)
{
    return GET_CABAC(MERGE_FLAG_OFFSET);
}

int h265_inter_pred_idc_decode(H265Slice *sh, int nPbW, int nPbH)
{
    if (nPbW + nPbH == 12)
        return GET_CABAC(INTER_PRED_IDC_OFFSET + 4);
    if (GET_CABAC(INTER_PRED_IDC_OFFSET + sh->ct_depth))
        return H265_PRED_BI;
    return GET_CABAC(INTER_PRED_IDC_OFFSET + 4);
}

int h265_ref_idx_lx_decode(H265Slice *sh, int num_ref_idx_lx)
{
    int i = 0;
    int max = num_ref_idx_lx - 1;
    int max_ctx = H265_MIN(max, 2);
    while (i < max_ctx && GET_CABAC(REF_IDX_L0_OFFSET + i))
        i++;
    if (i == 2) {
        while (i < max && get_cabac_bypass(&sh->cabac))
            i++;
    }
    return i;
}

int h265_mvp_lx_flag_decode(H265Slice *sh)
{
    return GET_CABAC(MVP_LX_FLAG_OFFSET);
}

int h265_no_residual_syntax_flag_decode(H265Slice *sh)
{
    return GET_CABAC(NO_RESIDUAL_DATA_FLAG_OFFSET);
}

H265_INLINE int abs_mvd_greater0_flag_decode(H265Slice *sh)
{
    return GET_CABAC(ABS_MVD_GREATER0_FLAG_OFFSET);
}

H265_INLINE int abs_mvd_greater1_flag_decode(H265Slice *sh)
{
    return GET_CABAC(ABS_MVD_GREATER1_FLAG_OFFSET + 1);
}

H265_INLINE int mvd_decode(H265Slice *sh)
{
    int ret = 2;
    int k = 1;
    while (k < CABAC_MAX_BIN && get_cabac_bypass(&sh->cabac)) {
        ret += 1U << k;
        k++;
    }
    if (k == CABAC_MAX_BIN)
        return 0;
    while (k--)
        ret += get_cabac_bypass(&sh->cabac) << k;
    return get_cabac_bypass_sign(&sh->cabac, -ret);
}

H265_INLINE int mvd_sign_flag_decode(H265Slice *sh)
{
    return get_cabac_bypass_sign(&sh->cabac, -1);
}

int h265_split_transform_flag_decode(H265Slice *sh, int log2_trafo_size)
{
    return GET_CABAC(SPLIT_TRANSFORM_FLAG_OFFSET + 5 - log2_trafo_size);
}

int h265_cbf_cb_cr_decode(H265Slice *sh, int trafo_depth)
{
    return GET_CABAC(CBF_CB_CR_OFFSET + trafo_depth);
}

int h265_cbf_luma_decode(H265Slice *sh, int trafo_depth)
{
    return GET_CABAC(CBF_LUMA_OFFSET + !trafo_depth);
}

H265_INLINE int transform_skip_flag_decode(H265Slice *sh, int c_idx)
{
    return GET_CABAC(TRANSFORM_SKIP_FLAG_OFFSET + !!c_idx);
}

H265_INLINE int explicit_rdpcm_flag_decode(H265Slice *sh, int c_idx)
{
    return GET_CABAC(EXPLICIT_RDPCM_FLAG_OFFSET + !!c_idx);
}

H265_INLINE int explicit_rdpcm_dir_flag_decode(H265Slice *sh, int c_idx)
{
    return GET_CABAC(EXPLICIT_RDPCM_DIR_FLAG_OFFSET + !!c_idx);
}

int h265_log2_res_scale_abs(H265Slice *sh, int idx)
{
    int i = 0;
    while (i < 4 && GET_CABAC(LOG2_RES_SCALE_ABS_OFFSET + 4 * idx + i))
        i++;
    return i;
}

int h265_res_scale_sign_flag(H265Slice *sh, int idx)
{
    return GET_CABAC(RES_SCALE_SIGN_FLAG_OFFSET + idx);
}

int h265_cu_qp_delta_abs(H265Slice *sh)
{
    int prefix_val = 0;
    int suffix_val = 0;
    int inc = 0;

    while (prefix_val < 5 && GET_CABAC(CU_QP_DELTA_OFFSET + inc)) {
        prefix_val++;
        inc = 1;
    }
    if (prefix_val >= 5) {
        int k = 0;
        while (k < 7 && get_cabac_bypass(&sh->cabac)) {
            suffix_val += 1 << k;
            k++;
        }
        if (k == 7)
            return -1;
        while (k--)
            suffix_val += get_cabac_bypass(&sh->cabac) << k;
    }
    return prefix_val + suffix_val;
}

int h265_cu_qp_delta_sign_flag(H265Slice *sh)
{
    return get_cabac_bypass(&sh->cabac);
}

int h265_cu_chroma_qp_offset_flag(H265Slice *sh)
{
    return GET_CABAC(CU_CHROMA_QP_OFFSET_FLAG_OFFSET);
}

int h265_cu_chroma_qp_offset_idx(H265Slice *sh, int chroma_qp_offset_list_len_minus1)
{
    int c_max = H265_MAX(5, chroma_qp_offset_list_len_minus1);
    int i = 0;
    while (i < c_max && GET_CABAC(CU_CHROMA_QP_OFFSET_IDX_OFFSET))
        i++;
    return i;
}

/* ------------------------------------------------------------------ */
/* transform coefficient decoding                                      */
/* ------------------------------------------------------------------ */
static const uint8_t scan_1x1[1] = { 0 };
static const uint8_t horiz_scan2x2_x[4] = { 0, 1, 0, 1 };
static const uint8_t horiz_scan2x2_y[4] = { 0, 0, 1, 1 };
static const uint8_t horiz_scan4x4_x[16] = {
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
};
static const uint8_t horiz_scan4x4_y[16] = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
};
static const uint8_t horiz_scan8x8_inv[8][8] = {
    {  0,  1,  2,  3, 16, 17, 18, 19, },
    {  4,  5,  6,  7, 20, 21, 22, 23, },
    {  8,  9, 10, 11, 24, 25, 26, 27, },
    { 12, 13, 14, 15, 28, 29, 30, 31, },
    { 32, 33, 34, 35, 48, 49, 50, 51, },
    { 36, 37, 38, 39, 52, 53, 54, 55, },
    { 40, 41, 42, 43, 56, 57, 58, 59, },
    { 44, 45, 46, 47, 60, 61, 62, 63, },
};
static const uint8_t diag_scan2x2_x[4] = { 0, 0, 1, 1 };
static const uint8_t diag_scan2x2_y[4] = { 0, 1, 0, 1 };
static const uint8_t diag_scan2x2_inv[2][2] = {
    { 0, 2, },
    { 1, 3, },
};
static const uint8_t diag_scan4x4_inv[4][4] = {
    { 0,  2,  5,  9, },
    { 1,  4,  8, 12, },
    { 3,  7, 11, 14, },
    { 6, 10, 13, 15, },
};
static const uint8_t diag_scan8x8_inv[8][8] = {
    {  0,  2,  5,  9, 14, 20, 27, 35, },
    {  1,  4,  8, 13, 19, 26, 34, 42, },
    {  3,  7, 12, 18, 25, 33, 41, 48, },
    {  6, 11, 17, 24, 32, 40, 47, 53, },
    { 10, 16, 23, 31, 39, 46, 52, 57, },
    { 15, 22, 30, 38, 45, 51, 56, 60, },
    { 21, 29, 37, 44, 50, 55, 59, 62, },
    { 28, 36, 43, 49, 54, 58, 61, 63, },
};

H265_INLINE void last_significant_coeff_xy_prefix_decode(H265Slice *sh,
                                            int c_idx, int log2_size,
                                            int *last_scx_prefix,
                                            int *last_scy_prefix)
{
    int i = 0;
    int max = (log2_size << 1) - 1;
    int ctx_offset, ctx_shift;

    if (!c_idx) {
        ctx_offset = 3 * (log2_size - 2) + ((log2_size - 1) >> 2);
        ctx_shift = (log2_size + 1) >> 2;
    } else {
        ctx_offset = 15;
        ctx_shift = log2_size - 2;
    }
    while (i < max &&
           GET_CABAC(LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET + (i >> ctx_shift) + ctx_offset))
        i++;
    *last_scx_prefix = i;

    i = 0;
    while (i < max &&
           GET_CABAC(LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET + (i >> ctx_shift) + ctx_offset))
        i++;
    *last_scy_prefix = i;
}

H265_INLINE int last_significant_coeff_suffix_decode(H265Slice *sh,
                                                            int prefix)
{
    int i;
    int length = (prefix >> 1) - 1;
    int value = get_cabac_bypass(&sh->cabac);
    for (i = 1; i < length; i++)
        value = (value << 1) | get_cabac_bypass(&sh->cabac);
    return value;
}

H265_INLINE int significant_coeff_group_flag_decode(H265Slice *sh,
                                                           int c_idx, int ctx_cg)
{
    int inc = H265_MIN(ctx_cg, 1) + (c_idx > 0 ? 2 : 0);
    return GET_CABAC(SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET + inc);
}

H265_INLINE int significant_coeff_flag_decode(H265Slice *sh, int x_c,
                                                     int y_c, int offset,
                                                     const uint8_t *ctx_idx_map)
{
    int inc = ctx_idx_map[(y_c << 2) + x_c] + offset;
    return GET_CABAC(SIGNIFICANT_COEFF_FLAG_OFFSET + inc);
}

H265_INLINE int significant_coeff_flag_decode_0(H265Slice *sh, int c_idx,
                                                       int offset)
{
    return GET_CABAC(SIGNIFICANT_COEFF_FLAG_OFFSET + offset);
}

H265_INLINE int coeff_abs_level_greater1_flag_decode(H265Slice *sh,
                                                            int c_idx, int inc)
{
    if (c_idx > 0)
        inc += 16;
    return GET_CABAC(COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET + inc);
}

H265_INLINE int coeff_abs_level_greater2_flag_decode(H265Slice *sh,
                                                            int c_idx, int inc)
{
    if (c_idx > 0)
        inc += 4;
    return GET_CABAC(COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET + inc);
}

H265_INLINE int coeff_abs_level_remaining_decode(H265Slice *sh,
                                                        int rc_rice_param)
{
    int prefix = 0;
    int suffix = 0;
    int remaining;
    int i;

    while (prefix < CABAC_MAX_BIN && get_cabac_bypass(&sh->cabac))
        prefix++;

    if (prefix < 3) {
        for (i = 0; i < rc_rice_param; i++)
            suffix = (suffix << 1) | get_cabac_bypass(&sh->cabac);
        remaining = (prefix << rc_rice_param) + suffix;
    } else {
        int prefix_minus3 = prefix - 3;
        if (prefix == CABAC_MAX_BIN || prefix_minus3 + rc_rice_param > 16 + 6)
            return 0;
        for (i = 0; i < prefix_minus3 + rc_rice_param; i++)
            suffix = (suffix << 1) | get_cabac_bypass(&sh->cabac);
        remaining = (((1 << prefix_minus3) + 3 - 1) << rc_rice_param) + suffix;
    }
    return remaining;
}

H265_INLINE int coeff_sign_flag_decode(H265Slice *sh, uint8_t nb)
{
    int i;
    int ret = 0;
    for (i = 0; i < nb; i++)
        ret = (ret << 1) | get_cabac_bypass(&sh->cabac);
    return ret;
}

void h265_hls_residual_coding(H265Slice *sh, int x0, int y0, int log2_trafo_size,
                              int scan_idx, int c_idx)
{
#define GET_COORD(offset, n)                                    \
    do {                                                        \
        x_c = (x_cg << 2) + scan_x_off[n];                      \
        y_c = (y_cg << 2) + scan_y_off[n];                      \
    } while (0)
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *frame = sh->frame;
    int transform_skip_flag = 0;
    int last_significant_coeff_x, last_significant_coeff_y;
    int last_scan_pos;
    int n_end;
    int num_coeff = 0;
    int greater1_ctx = 1;
    int num_last_subset;
    int x_cg_last_sig, y_cg_last_sig;
    const uint8_t *scan_x_cg, *scan_y_cg, *scan_x_off, *scan_y_off;
    ptrdiff_t stride = frame->linesize[c_idx];
    int hshift = sps->hshift[c_idx];
    int vshift = sps->vshift[c_idx];
    uint8_t *dst = &frame->data[c_idx][(y0 >> vshift) * stride +
                                      ((x0 >> hshift) << sps->pixel_shift)];
    int16_t *coeffs = (int16_t *)(c_idx ? sh->edge_emu_buffer2 : sh->edge_emu_buffer);
    uint8_t significant_coeff_group_flag[8][8];
    int explicit_rdpcm_flag = 0;
    int explicit_rdpcm_dir_flag = 0;
    int trafo_size = 1 << log2_trafo_size;
    int i;
    int qp, shift, add, scale, scale_m;
    static const uint8_t level_scale[] = { 40, 45, 51, 57, 64, 72 };
    const uint8_t *scale_matrix = NULL;
    uint8_t dc_scale;
    int pred_mode_intra = (c_idx == 0) ? sh->tu.intra_pred_mode :
                                         sh->tu.intra_pred_mode_c;

    if (log2_trafo_size > 2)
        memset(significant_coeff_group_flag, 0,
               sizeof(significant_coeff_group_flag));

    if (!sh->cu.cu_transquant_bypass_flag) {
        static const int qp_c[] = { 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37 };
        static const uint8_t rem6[51 + 4 * 6 + 1] = {
            0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2,
            3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
            4, 5, 0, 1, 2, 3, 4, 5, 0, 1
        };
        static const uint8_t div6[51 + 4 * 6 + 1] = {
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3,  3,  3,
            3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,  6,  6,
            7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10,
            10, 10, 11, 11, 11, 11, 11, 11, 12, 12
        };
        int qp_y = sh->qp_y;

        if (pps->transform_skip_enabled_flag &&
            log2_trafo_size <= pps->log2_max_transform_skip_block_size)
            transform_skip_flag = transform_skip_flag_decode(sh, c_idx);

        if (c_idx == 0) {
            qp = qp_y + sps->qp_bd_offset;
        } else {
            int qp_i, offset;
            if (c_idx == 1)
                offset = pps->cb_qp_offset + sh->slice_cb_qp_offset +
                         sh->tu.cu_qp_offset_cb;
            else
                offset = pps->cr_qp_offset + sh->slice_cr_qp_offset +
                         sh->tu.cu_qp_offset_cr;
            qp_i = H265_CLIP(qp_y + offset, -sps->qp_bd_offset, 57);
            if (sps->chroma_format_idc == 1) {
                if (qp_i < 30)
                    qp = qp_i;
                else if (qp_i > 43)
                    qp = qp_i - 6;
                else
                    qp = qp_c[qp_i - 30];
            } else {
                qp = qp_i > 51 ? 51 : qp_i;
            }
            qp += sps->qp_bd_offset;
        }

        shift = sps->bit_depth + log2_trafo_size - 5;
        add = 1 << (shift - 1);
        scale = level_scale[rem6[qp]] << (div6[qp]);
        scale_m = 16;
        dc_scale = 16;

        if (sps->scaling_list_enabled &&
            !(transform_skip_flag && log2_trafo_size > 2)) {
            const H265ScalingList *sl = pps->scaling_list_data_present_flag ?
                                        &pps->scaling_list : &sps->scaling_list;
            int matrix_id = sh->cu.pred_mode != H265_MODE_INTRA;
            matrix_id = 3 * matrix_id + c_idx;
            scale_matrix = sl->sl[log2_trafo_size - 2][matrix_id];
            if (log2_trafo_size >= 4)
                dc_scale = sl->sl_dc[log2_trafo_size - 4][matrix_id];
        }
    } else {
        shift = 0;
        add = 0;
        scale = 0;
        scale_m = 16;
        dc_scale = 16;
    }

    if (sh->cu.pred_mode == H265_MODE_INTER && sps->explicit_rdpcm_enabled &&
        (transform_skip_flag || sh->cu.cu_transquant_bypass_flag)) {
        explicit_rdpcm_flag = explicit_rdpcm_flag_decode(sh, c_idx);
        if (explicit_rdpcm_flag)
            explicit_rdpcm_dir_flag = explicit_rdpcm_dir_flag_decode(sh, c_idx);
    }

    last_significant_coeff_xy_prefix_decode(sh, c_idx, log2_trafo_size,
                                            &last_significant_coeff_x,
                                            &last_significant_coeff_y);

    if (last_significant_coeff_x > 3) {
        int suffix = last_significant_coeff_suffix_decode(sh, last_significant_coeff_x);
        last_significant_coeff_x = (1 << ((last_significant_coeff_x >> 1) - 1)) *
                                   (2 + (last_significant_coeff_x & 1)) + suffix;
    }
    if (last_significant_coeff_y > 3) {
        int suffix = last_significant_coeff_suffix_decode(sh, last_significant_coeff_y);
        last_significant_coeff_y = (1 << ((last_significant_coeff_y >> 1) - 1)) *
                                   (2 + (last_significant_coeff_y & 1)) + suffix;
    }

    if (scan_idx == H265_SCAN_VERT) {
        int t = last_significant_coeff_x;
        last_significant_coeff_x = last_significant_coeff_y;
        last_significant_coeff_y = t;
    }

    x_cg_last_sig = last_significant_coeff_x >> 2;
    y_cg_last_sig = last_significant_coeff_y >> 2;

    switch (scan_idx) {
    case H265_SCAN_DIAG: {
        int last_x_c = last_significant_coeff_x & 3;
        int last_y_c = last_significant_coeff_y & 3;
        scan_x_off = h265_diag_scan4x4_x;
        scan_y_off = h265_diag_scan4x4_y;
        num_coeff = diag_scan4x4_inv[last_y_c][last_x_c];
        if (trafo_size == 4) {
            scan_x_cg = scan_1x1;
            scan_y_cg = scan_1x1;
        } else if (trafo_size == 8) {
            num_coeff += diag_scan2x2_inv[y_cg_last_sig][x_cg_last_sig] << 4;
            scan_x_cg = diag_scan2x2_x;
            scan_y_cg = diag_scan2x2_y;
        } else if (trafo_size == 16) {
            num_coeff += diag_scan4x4_inv[y_cg_last_sig][x_cg_last_sig] << 4;
            scan_x_cg = h265_diag_scan4x4_x;
            scan_y_cg = h265_diag_scan4x4_y;
        } else {
            num_coeff += diag_scan8x8_inv[y_cg_last_sig][x_cg_last_sig] << 4;
            scan_x_cg = h265_diag_scan8x8_x;
            scan_y_cg = h265_diag_scan8x8_y;
        }
        break;
    }
    case H265_SCAN_HORIZ:
        scan_x_cg = horiz_scan2x2_x;
        scan_y_cg = horiz_scan2x2_y;
        scan_x_off = horiz_scan4x4_x;
        scan_y_off = horiz_scan4x4_y;
        num_coeff = horiz_scan8x8_inv[last_significant_coeff_y][last_significant_coeff_x];
        break;
    default:
        scan_x_cg = horiz_scan2x2_y;
        scan_y_cg = horiz_scan2x2_x;
        scan_x_off = horiz_scan4x4_y;
        scan_y_off = horiz_scan4x4_x;
        num_coeff = horiz_scan8x8_inv[last_significant_coeff_x][last_significant_coeff_y];
        break;
    }
    num_coeff++;
    num_last_subset = (num_coeff - 1) >> 4;

    if (!(last_significant_coeff_x == 0 && last_significant_coeff_y == 0 &&
          !transform_skip_flag && !sh->cu.cu_transquant_bypass_flag &&
          !sh->tu.cross_pf &&
          !(sh->cu.pred_mode == H265_MODE_INTRA && c_idx == 0 &&
            log2_trafo_size == 2))) {
        switch (log2_trafo_size) {
        case 2: memset(coeffs, 0, 4 * 4 * sizeof(int16_t)); break;
        case 3: memset(coeffs, 0, 8 * 8 * sizeof(int16_t)); break;
        case 4: memset(coeffs, 0, 16 * 16 * sizeof(int16_t)); break;
        default: memset(coeffs, 0, 32 * 32 * sizeof(int16_t)); break;
        }
    }

    for (i = num_last_subset; i >= 0; i--) {
        int n, m;
        int x_cg, y_cg, x_c, y_c, pos;
        int implicit_non_zero_coeff = 0;
        int64_t trans_coeff_level;
        int prev_sig = 0;
        int offset = i << 4;
        int rice_init = 0;
        uint8_t significant_coeff_flag_idx[16];
        uint8_t nb_significant_coeff_flag = 0;

        x_cg = scan_x_cg[i];
        y_cg = scan_y_cg[i];

        if ((i < num_last_subset) && (i > 0)) {
            int ctx_cg = 0;
            if (x_cg < (1 << (log2_trafo_size - 2)) - 1)
                ctx_cg += significant_coeff_group_flag[x_cg + 1][y_cg];
            if (y_cg < (1 << (log2_trafo_size - 2)) - 1)
                ctx_cg += significant_coeff_group_flag[x_cg][y_cg + 1];
            significant_coeff_group_flag[x_cg][y_cg] =
                significant_coeff_group_flag_decode(sh, c_idx, ctx_cg);
            implicit_non_zero_coeff = 1;
        } else {
            significant_coeff_group_flag[x_cg][y_cg] =
                ((x_cg == x_cg_last_sig && y_cg == y_cg_last_sig) ||
                 (x_cg == 0 && y_cg == 0));
        }

        last_scan_pos = num_coeff - offset - 1;
        if (i == num_last_subset) {
            n_end = last_scan_pos - 1;
            significant_coeff_flag_idx[0] = last_scan_pos;
            nb_significant_coeff_flag = 1;
        } else {
            n_end = 15;
        }

        if (x_cg < ((1 << log2_trafo_size) - 1) >> 2)
            prev_sig = !!significant_coeff_group_flag[x_cg + 1][y_cg];
        if (y_cg < ((1 << log2_trafo_size) - 1) >> 2)
            prev_sig += (!!significant_coeff_group_flag[x_cg][y_cg + 1] << 1);

        if (significant_coeff_group_flag[x_cg][y_cg] && n_end >= 0) {
            static const uint8_t ctx_idx_map[] = {
                0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8,
                1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            };
            const uint8_t *ctx_idx_map_p;
            int scf_offset = 0;
            if (sps->transform_skip_context_enabled &&
                (transform_skip_flag || sh->cu.cu_transquant_bypass_flag)) {
                ctx_idx_map_p = &ctx_idx_map[4 * 16];
                if (c_idx == 0)
                    scf_offset = 40;
                else
                    scf_offset = 14 + 27;
            } else {
                if (c_idx != 0)
                    scf_offset = 27;
                if (log2_trafo_size == 2) {
                    ctx_idx_map_p = &ctx_idx_map[0];
                } else {
                    ctx_idx_map_p = &ctx_idx_map[(prev_sig + 1) << 4];
                    if (c_idx == 0) {
                        if (x_cg > 0 || y_cg > 0)
                            scf_offset += 3;
                        if (log2_trafo_size == 3)
                            scf_offset += (scan_idx == H265_SCAN_DIAG) ? 9 : 15;
                        else
                            scf_offset += 21;
                    } else {
                        if (log2_trafo_size == 3)
                            scf_offset += 9;
                        else
                            scf_offset += 12;
                    }
                }
            }
            for (n = n_end; n > 0; n--) {
                x_c = scan_x_off[n];
                y_c = scan_y_off[n];
                if (significant_coeff_flag_decode(sh, x_c, y_c, scf_offset,
                                                  ctx_idx_map_p)) {
                    significant_coeff_flag_idx[nb_significant_coeff_flag] = n;
                    nb_significant_coeff_flag++;
                    implicit_non_zero_coeff = 0;
                }
            }
            if (implicit_non_zero_coeff == 0) {
                if (sps->transform_skip_context_enabled &&
                    (transform_skip_flag || sh->cu.cu_transquant_bypass_flag)) {
                    if (c_idx == 0)
                        scf_offset = 42;
                    else
                        scf_offset = 16 + 27;
                } else {
                    if (i == 0) {
                        if (c_idx == 0)
                            scf_offset = 0;
                        else
                            scf_offset = 27;
                    } else {
                        scf_offset = 2 + scf_offset;
                    }
                }
                if (significant_coeff_flag_decode_0(sh, c_idx, scf_offset) == 1) {
                    significant_coeff_flag_idx[nb_significant_coeff_flag] = 0;
                    nb_significant_coeff_flag++;
                }
            } else {
                significant_coeff_flag_idx[nb_significant_coeff_flag] = 0;
                nb_significant_coeff_flag++;
            }
        }

        n_end = nb_significant_coeff_flag;

        if (n_end) {
            int first_nz_pos_in_cg;
            int last_nz_pos_in_cg;
            int c_rice_param = 0;
            int first_greater1_coeff_idx = -1;
            uint8_t coeff_abs_level_greater1_flag[8];
            uint16_t coeff_sign_flag;
            int sum_abs = 0;
            int sign_hidden;
            int sb_type = 0;

            int ctx_set = (i > 0 && c_idx == 0) ? 2 : 0;

            if (sps->persistent_rice_adaptation_enabled) {
                if (!transform_skip_flag && !sh->cu.cu_transquant_bypass_flag)
                    sb_type = 2 * (c_idx == 0 ? 1 : 0);
                else
                    sb_type = 2 * (c_idx == 0 ? 1 : 0) + 1;
                c_rice_param = sh->stat_coeff[sb_type] / 4;
            }

            if (!(i == num_last_subset) && greater1_ctx == 0)
                ctx_set++;
            greater1_ctx = 1;
            last_nz_pos_in_cg = significant_coeff_flag_idx[0];

            for (m = 0; m < (n_end > 8 ? 8 : n_end); m++) {
                int inc = (ctx_set << 2) + greater1_ctx;
                coeff_abs_level_greater1_flag[m] =
                    coeff_abs_level_greater1_flag_decode(sh, c_idx, inc);
                if (coeff_abs_level_greater1_flag[m]) {
                    greater1_ctx = 0;
                    if (first_greater1_coeff_idx == -1)
                        first_greater1_coeff_idx = m;
                } else if (greater1_ctx > 0 && greater1_ctx < 3) {
                    greater1_ctx++;
                }
            }
            first_nz_pos_in_cg = significant_coeff_flag_idx[n_end - 1];

            if (sh->cu.cu_transquant_bypass_flag ||
                (sh->cu.pred_mode == H265_MODE_INTRA &&
                 sps->implicit_rdpcm_enabled && transform_skip_flag &&
                 (pred_mode_intra == 10 || pred_mode_intra == 26)) ||
                explicit_rdpcm_flag)
                sign_hidden = 0;
            else
                sign_hidden = (last_nz_pos_in_cg - first_nz_pos_in_cg >= 4);

            if (first_greater1_coeff_idx != -1)
                coeff_abs_level_greater1_flag[first_greater1_coeff_idx] +=
                    coeff_abs_level_greater2_flag_decode(sh, c_idx, ctx_set);
            if (!pps->sign_data_hiding_flag || !sign_hidden)
                coeff_sign_flag = coeff_sign_flag_decode(sh, nb_significant_coeff_flag) <<
                                  (16 - nb_significant_coeff_flag);
            else
                coeff_sign_flag = coeff_sign_flag_decode(sh, nb_significant_coeff_flag - 1) <<
                                  (16 - (nb_significant_coeff_flag - 1));

            for (m = 0; m < n_end; m++) {
                n = significant_coeff_flag_idx[m];
                GET_COORD(offset, n);
                if (m < 8) {
                    trans_coeff_level = 1 + coeff_abs_level_greater1_flag[m];
                    if (trans_coeff_level == ((m == first_greater1_coeff_idx) ? 3 : 2)) {
                        int remaining = coeff_abs_level_remaining_decode(sh, c_rice_param);
                        trans_coeff_level += remaining;
                        if (trans_coeff_level > (3 << c_rice_param))
                            c_rice_param = sps->persistent_rice_adaptation_enabled ?
                                           c_rice_param + 1 : H265_MIN(c_rice_param + 1, 4);
                        if (sps->persistent_rice_adaptation_enabled && !rice_init) {
                            int c_rice_p_init = sh->stat_coeff[sb_type] / 4;
                            if (remaining >= (3 << c_rice_p_init))
                                sh->stat_coeff[sb_type]++;
                            else if (2 * remaining < (1 << c_rice_p_init))
                                if (sh->stat_coeff[sb_type] > 0)
                                    sh->stat_coeff[sb_type]--;
                            rice_init = 1;
                        }
                    }
                } else {
                    int remaining = coeff_abs_level_remaining_decode(sh, c_rice_param);
                    trans_coeff_level = 1 + remaining;
                    if (trans_coeff_level > (3 << c_rice_param))
                        c_rice_param = sps->persistent_rice_adaptation_enabled ?
                                       c_rice_param + 1 : H265_MIN(c_rice_param + 1, 4);
                    if (sps->persistent_rice_adaptation_enabled && !rice_init) {
                        int c_rice_p_init = sh->stat_coeff[sb_type] / 4;
                        if (remaining >= (3 << c_rice_p_init))
                            sh->stat_coeff[sb_type]++;
                        else if (2 * remaining < (1 << c_rice_p_init))
                            if (sh->stat_coeff[sb_type] > 0)
                                sh->stat_coeff[sb_type]--;
                        rice_init = 1;
                    }
                }
                if (pps->sign_data_hiding_flag && sign_hidden) {
                    sum_abs += (int)trans_coeff_level;
                    if (n == first_nz_pos_in_cg && (sum_abs & 1))
                        trans_coeff_level = -trans_coeff_level;
                }
                if (coeff_sign_flag >> 15)
                    trans_coeff_level = -trans_coeff_level;
                coeff_sign_flag <<= 1;
                if (!sh->cu.cu_transquant_bypass_flag) {
                    if (sps->scaling_list_enabled &&
                        !(transform_skip_flag && log2_trafo_size > 2)) {
                        if (y_c || x_c || log2_trafo_size < 4) {
                            switch (log2_trafo_size) {
                            case 3: pos = (y_c << 3) + x_c; break;
                            case 4: pos = ((y_c >> 1) << 3) + (x_c >> 1); break;
                            case 5: pos = ((y_c >> 2) << 3) + (x_c >> 2); break;
                            default: pos = (y_c << 2) + x_c; break;
                            }
                            scale_m = scale_matrix[pos];
                        } else {
                            scale_m = dc_scale;
                        }
                    }
                    trans_coeff_level = (trans_coeff_level * (int64_t)scale *
                                         (int64_t)scale_m + add) >> shift;
                    if (trans_coeff_level < 0) {
                        if ((~trans_coeff_level) & 0xFffffffffff8000LL)
                            trans_coeff_level = -32768;
                    } else {
                        if (trans_coeff_level & 0xffffffffffff8000LL)
                            trans_coeff_level = 32767;
                    }
                }
                coeffs[y_c * trafo_size + x_c] = (int16_t)trans_coeff_level;
            }
        }
    }

    if (sh->cu.cu_transquant_bypass_flag) {
        if (explicit_rdpcm_flag || (sps->implicit_rdpcm_enabled &&
                                    (pred_mode_intra == 10 || pred_mode_intra == 26))) {
            int mode = sps->implicit_rdpcm_enabled ?
                       (pred_mode_intra == 26) : explicit_rdpcm_dir_flag;
            h265_transform_rdpcm(coeffs, log2_trafo_size, mode);
        }
    } else {
        if (transform_skip_flag) {
            int rot = sps->transform_skip_rotation_enabled &&
                      log2_trafo_size == 2 &&
                      sh->cu.pred_mode == H265_MODE_INTRA;
            if (rot) {
                for (i = 0; i < 8; i++) {
                    int16_t t = coeffs[i];
                    coeffs[i] = coeffs[16 - i - 1];
                    coeffs[16 - i - 1] = t;
                }
            }
            h265_dequant(coeffs, log2_trafo_size, sps->bit_depth);
            if (explicit_rdpcm_flag || (sps->implicit_rdpcm_enabled &&
                                        sh->cu.pred_mode == H265_MODE_INTRA &&
                                        (pred_mode_intra == 10 || pred_mode_intra == 26))) {
                int mode = explicit_rdpcm_flag ? explicit_rdpcm_dir_flag :
                           (pred_mode_intra == 26);
                h265_transform_rdpcm(coeffs, log2_trafo_size, mode);
            }
        } else if (sh->cu.pred_mode == H265_MODE_INTRA && c_idx == 0 &&
                   log2_trafo_size == 2) {
            h265_transform_4x4_luma(coeffs, sps->bit_depth);
        } else {
            int max_xy = H265_MAX(last_significant_coeff_x, last_significant_coeff_y);
            if (max_xy == 0) {
                h265_idct_dc(coeffs, log2_trafo_size, sps->bit_depth);
            } else {
                int col_limit = last_significant_coeff_x + last_significant_coeff_y + 4;
                if (max_xy < 4)
                    col_limit = H265_MIN(4, col_limit);
                else if (max_xy < 8)
                    col_limit = H265_MIN(8, col_limit);
                else if (max_xy < 12)
                    col_limit = H265_MIN(24, col_limit);
                h265_idct(coeffs, log2_trafo_size, col_limit, sps->bit_depth);
            }
        }
    }

    if (sh->tu.cross_pf) {
        int16_t *coeffs_y = (int16_t *)sh->edge_emu_buffer;
        for (i = 0; i < (trafo_size * trafo_size); i++)
            coeffs[i] = coeffs[i] + ((sh->tu.res_scale_val * coeffs_y[i]) >> 3);
    }

    h265_add_residual(dst, coeffs, stride, trafo_size, sps->pixel_shift,
                      sps->bit_depth);
}

void h265_hls_mvd_coding(H265Slice *sh)
{
    int x = abs_mvd_greater0_flag_decode(sh);
    int y = abs_mvd_greater0_flag_decode(sh);

    if (x)
        x += abs_mvd_greater1_flag_decode(sh);
    if (y)
        y += abs_mvd_greater1_flag_decode(sh);

    switch (x) {
    case 2: sh->pu.mvd.x = (int16_t)mvd_decode(sh); break;
    case 1: sh->pu.mvd.x = (int16_t)mvd_sign_flag_decode(sh); break;
    case 0: sh->pu.mvd.x = 0; break;
    }
    switch (y) {
    case 2: sh->pu.mvd.y = (int16_t)mvd_decode(sh); break;
    case 1: sh->pu.mvd.y = (int16_t)mvd_sign_flag_decode(sh); break;
    case 0: sh->pu.mvd.y = 0; break;
    }
}

/* Skip whole bytes inside the CABAC stream and reinitialise the arithmetic
 * decoder after them (used by PCM samples).  Returns the start of the skipped
 * bytes or NULL. */
const uint8_t *h265_cabac_skip_bytes(H265Slice *sh, int n)
{
    H265CABAC *c = &sh->cabac;
    const uint8_t *ptr = c->bytestream;

    if (c->low & 1)
        ptr--;
    if (c->low & 0x1FF)
        ptr--;
    if (ptr < c->bytestream_start || (int)(c->bytestream_end - ptr) < n)
        return NULL;
    if (h265_cabac_init(sh, ptr + n, (int)(c->bytestream_end - ptr - n)) < 0)
        return NULL;
    return ptr;
}

int h265_cabac_terminate(H265Slice *sh)
{
    return get_cabac_terminate(&sh->cabac);
}
