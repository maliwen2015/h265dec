/*
 * h265dec internal definitions.
 *
 * Algorithms and constant tables are derived from ITU-T Rec. H.265 and from
 * FFmpeg's HEVC decoder (LGPL-2.1+) which serves as the reference.
 */
#ifndef H265_INTERNAL_H
#define H265_INTERNAL_H

#define _WIN32_WINNT 0x0600
#ifndef _WIN32
#define _POSIX_C_SOURCE 200112L
#endif

#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

#include "../h265dec.h"

/* ------------------------------------------------------------------ */
/* misc helpers                                                        */
/* ------------------------------------------------------------------ */
#define H265_MIN(a, b) ((a) < (b) ? (a) : (b))
#define H265_MAX(a, b) ((a) > (b) ? (a) : (b))
#define H265_MIN3(a,b,c) H265_MIN(H265_MIN(a,b),c)
#define H265_MAX3(a,b,c) H265_MAX(H265_MAX(a,b),c)
#define H265_CLIP(a, amin, amax) ((a) < (amin) ? (amin) : (a) > (amax) ? (amax) : (a))
#define H265_ABS(a) ((a) >= 0 ? (a) : -(a))
#define H265_SIGN(a) ((a) > 0 ? 1 : (a) < 0 ? -1 : 0)
#define H265_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))
#define H265_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define H265_X86 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define H265_ARM64 1
#define H265_NEON 1
#endif

#ifdef _MSC_VER
#define H265_INLINE static __forceinline
#define H265_THREAD_LOCAL __declspec(thread)
#else
#define H265_INLINE static inline
#define H265_THREAD_LOCAL __thread
#endif

/* ------------------------------------------------------------------ */
/* constants                                                           */
/* ------------------------------------------------------------------ */
#define H265_MAX_VPS_COUNT              16
#define H265_MAX_SPS_COUNT              32
#define H265_MAX_PPS_COUNT              256
#define H265_MAX_DPB_SIZE               32
#define H265_MAX_REFS                   16
#define H265_MAX_ST_RPS                 64
#define H265_MAX_LONG_TERM_REF_PICS     32
#define H265_MAX_SUB_LAYERS             7
#define H265_MAX_SLICES                 256
#define H265_MAX_TILE_COLUMNS           64
#define H265_MAX_TILE_ROWS              64
#define H265_MAX_PALETTE_PREDICTOR_SIZE 128

#define H265_MAX_TB_SIZE                32
#define H265_MAX_PB_SIZE                64
#define H265_MAX_QP                     51
#define H265_DEFAULT_INTRA_TC_OFFSET    2

#define H265_CONTEXTS                   199
#define H265_STAT_COEFFS                4
#define H265_MRG_MAX_NUM_CANDS          5

#define H265_L0                         0
#define H265_L1                         1

#define H265_EPEL_EXTRA_BEFORE          1
#define H265_EPEL_EXTRA_AFTER           2
#define H265_EPEL_EXTRA                 3
#define H265_QPEL_EXTRA_BEFORE          3
#define H265_QPEL_EXTRA_AFTER           4
#define H265_QPEL_EXTRA                 7

#define H265_EDGE_EMU_BUFFER_STRIDE     80
#define H265_EDGE_EMU_PADDING           32

/* HEVC NAL unit types */
enum H265NalUnitType {
    H265_NAL_TRAIL_N    = 0,
    H265_NAL_TRAIL_R    = 1,
    H265_NAL_TSA_N      = 2,
    H265_NAL_TSA_R      = 3,
    H265_NAL_STSA_N     = 4,
    H265_NAL_STSA_R     = 5,
    H265_NAL_RADL_N     = 6,
    H265_NAL_RADL_R     = 7,
    H265_NAL_RASL_N     = 8,
    H265_NAL_RASL_R     = 9,
    H265_NAL_VCL_N10    = 10,
    H265_NAL_VCL_R11    = 11,
    H265_NAL_VCL_N12    = 12,
    H265_NAL_VCL_R13    = 13,
    H265_NAL_VCL_N14    = 14,
    H265_NAL_VCL_R15    = 15,
    H265_NAL_BLA_W_LP   = 16,
    H265_NAL_BLA_W_RADL = 17,
    H265_NAL_BLA_N_LP   = 18,
    H265_NAL_IDR_W_RADL = 19,
    H265_NAL_IDR_N_LP   = 20,
    H265_NAL_CRA_NUT    = 21,
    H265_NAL_IRAP_VCL22 = 22,
    H265_NAL_IRAP_VCL23 = 23,
    H265_NAL_RSV_VCL24  = 24,
    H265_NAL_RSV_VCL25  = 25,
    H265_NAL_RSV_VCL26  = 26,
    H265_NAL_RSV_VCL27  = 27,
    H265_NAL_RSV_VCL28  = 28,
    H265_NAL_RSV_VCL29  = 29,
    H265_NAL_RSV_VCL30  = 30,
    H265_NAL_RSV_VCL31  = 31,
    H265_NAL_VPS        = 32,
    H265_NAL_SPS        = 33,
    H265_NAL_PPS        = 34,
    H265_NAL_AUD        = 35,
    H265_NAL_EOS        = 36,
    H265_NAL_EOB        = 37,
    H265_NAL_FD         = 38,
    H265_NAL_PREFIX_SEI = 39,
    H265_NAL_SUFFIX_SEI = 40,
    H265_NAL_RSV_NVCL41 = 41,
    H265_NAL_RSV_NVCL47 = 47,
    H265_NAL_UNSPEC48   = 48,
    H265_NAL_UNSPEC63   = 63,
};

#define H265_IS_IDR(t) ((t) == H265_NAL_IDR_W_RADL || (t) == H265_NAL_IDR_N_LP)
#define H265_IS_BLA(t) ((t) == H265_NAL_BLA_W_LP || (t) == H265_NAL_BLA_W_RADL || \
                        (t) == H265_NAL_BLA_N_LP)
#define H265_IS_IRAP(t) ((t) >= H265_NAL_BLA_W_LP && (t) <= H265_NAL_IRAP_VCL23)
#define H265_IS_VCL(t) ((t) <= H265_NAL_RSV_VCL31)
#define H265_IS_NONREF(t) ((t) == H265_NAL_TRAIL_N || (t) == H265_NAL_TSA_N || \
                           (t) == H265_NAL_STSA_N || (t) == H265_NAL_RADL_N || \
                           (t) == H265_NAL_RASL_N || (t) == H265_NAL_VCL_N10 || \
                           (t) == H265_NAL_VCL_N12 || (t) == H265_NAL_VCL_N14)

/* slice types as coded in the bitstream */
#define H265_SLICE_B 0
#define H265_SLICE_P 1
#define H265_SLICE_I 2

/* PartMode */
enum H265PartMode {
    H265_PART_2Nx2N = 0,
    H265_PART_2NxN  = 1,
    H265_PART_Nx2N  = 2,
    H265_PART_NxN   = 3,
    H265_PART_2NxnU = 4,
    H265_PART_2NxnD = 5,
    H265_PART_nLx2N = 6,
    H265_PART_nRx2N = 7,
};

enum H265PredMode {
    H265_MODE_INTER = 0,
    H265_MODE_INTRA,
    H265_MODE_SKIP,
};

enum H265InterPredIdc {
    H265_PRED_L0 = 0,
    H265_PRED_L1,
    H265_PRED_BI,
};

enum H265PredFlag {
    H265_PF_INTRA = 0,
    H265_PF_L0,
    H265_PF_L1,
    H265_PF_BI,
};

enum H265IntraPredMode {
    H265_INTRA_PLANAR = 0,
    H265_INTRA_DC,
    H265_INTRA_ANGULAR_2,
    /* ... */
    H265_INTRA_ANGULAR_10 = 10,
    H265_INTRA_ANGULAR_26 = 26,
    H265_INTRA_ANGULAR_34 = 34,
};

enum H265SAOType {
    H265_SAO_NOT_APPLIED = 0,
    H265_SAO_BAND,
    H265_SAO_EDGE,
    H265_SAO_APPLIED
};

enum H265SAOEOClass {
    H265_SAO_EO_HORIZ = 0,
    H265_SAO_EO_VERT,
    H265_SAO_EO_135D,
    H265_SAO_EO_45D,
};

enum H265ScanType {
    H265_SCAN_DIAG = 0,
    H265_SCAN_HORIZ,
    H265_SCAN_VERT,
};

/* RPS list types */
enum H265RPSType {
    H265_ST_CURR_BEF = 0,
    H265_ST_CURR_AFT,
    H265_ST_FOLL,
    H265_LT_CURR,
    H265_LT_FOLL,
    H265_NB_RPS_TYPE,
};

/* frame flags */
#define H265_FRAME_FLAG_OUTPUT       (1 << 0)
#define H265_FRAME_FLAG_SHORT_REF    (1 << 1)
#define H265_FRAME_FLAG_LONG_REF     (1 << 2)
#define H265_FRAME_FLAG_UNAVAILABLE  (1 << 3)
#define H265_FRAME_FLAG_CORRUPT      (1 << 4)
#define H265_FRAME_FLAG_DECODED      (1 << 5)

/* boundary flags for the deblocking filter */
#define H265_BOUNDARY_LEFT_SLICE     (1 << 0)
#define H265_BOUNDARY_LEFT_TILE      (1 << 1)
#define H265_BOUNDARY_UPPER_SLICE    (1 << 2)
#define H265_BOUNDARY_UPPER_TILE     (1 << 3)

/* ------------------------------------------------------------------ */
/* tracing (very light)                                                */
/* ------------------------------------------------------------------ */
typedef struct H265Dbg {
    int bits, mb, mc, lf, sao, ps;
    int mbx, mby;
} H265Dbg;
extern H265Dbg h265_dbg_cfg;
void h265_dbg_init(void);
H265_INLINE const H265Dbg *h265_dbg(void) { return &h265_dbg_cfg; }

/* ------------------------------------------------------------------ */
/* synchronization                                                     */
/* ------------------------------------------------------------------ */
#ifdef _WIN32
typedef CRITICAL_SECTION h265_mutex;
#define h265_mutex_init(m)     InitializeCriticalSection(m)
#define h265_mutex_destroy(m)  DeleteCriticalSection(m)
#define h265_mutex_lock(m)     EnterCriticalSection(m)
#define h265_mutex_unlock(m)   LeaveCriticalSection(m)
typedef CONDITION_VARIABLE h265_cond;
#define h265_cond_init(c)      InitializeConditionVariable(c)
#define h265_cond_destroy(c)   ((void)0)
#define h265_cond_wait(c, m)   SleepConditionVariableCS((c), (m), INFINITE)
#define h265_cond_signal(c)    WakeConditionVariable(c)
#define h265_cond_broadcast(c) WakeAllConditionVariable(c)
#define H265_THREAD_RET        unsigned __stdcall
typedef HANDLE h265_thread;
#else
typedef pthread_mutex_t h265_mutex;
#define h265_mutex_init(m)     pthread_mutex_init(m, NULL)
#define h265_mutex_destroy(m)  pthread_mutex_destroy(m)
#define h265_mutex_lock(m)     pthread_mutex_lock(m)
#define h265_mutex_unlock(m)   pthread_mutex_unlock(m)
typedef pthread_cond_t h265_cond;
#define h265_cond_init(c)      pthread_cond_init(c, NULL)
#define h265_cond_destroy(c)   pthread_cond_destroy(c)
#define h265_cond_wait(c, m)   pthread_cond_wait(c, m)
#define h265_cond_signal(c)    pthread_cond_signal(c)
#define h265_cond_broadcast(c) pthread_cond_broadcast(c)
#define H265_THREAD_RET        void *
typedef pthread_t h265_thread;
#endif

#ifdef _WIN32
#define h265_atomic_inc(p)     InterlockedIncrement((volatile LONG *)(p))
#define h265_atomic_dec(p)     InterlockedDecrement((volatile LONG *)(p))
#define h265_atomic_set(p, v)  InterlockedExchange((volatile LONG *)(p), (LONG)(v))
#define h265_atomic_add(p, v)  InterlockedExchangeAdd((volatile LONG *)(p), (LONG)(v))
#define h265_atomic_get(p)     InterlockedCompareExchange((volatile LONG *)(p), 0, 0)
#else
#define h265_atomic_inc(p)     __sync_add_and_fetch((p), 1)
#define h265_atomic_dec(p)     __sync_sub_and_fetch((p), 1)
#define h265_atomic_set(p, v)  __sync_lock_test_and_set((p), (v))
#define h265_atomic_add(p, v)  __sync_fetch_and_add((p), (v))
#define h265_atomic_get(p)     __sync_add_and_fetch((p), 0)
#endif

/* ------------------------------------------------------------------ */
/* bit reader                                                          */
/* ------------------------------------------------------------------ */
typedef struct H265BitReader {
    const uint8_t *buffer;
    int size_bits;
    int pos;
    int error;
} H265BitReader;

H265_INLINE void h265_br_init(H265BitReader *br, const uint8_t *buf, int size_bits)
{
    br->buffer    = buf;
    br->size_bits = size_bits;
    br->pos       = 0;
    br->error     = 0;
}

H265_INLINE int h265_br_bits_left(const H265BitReader *br)
{
    return br->size_bits - br->pos;
}

static inline uint64_t h265_load_be64(const uint8_t *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
#ifdef _WIN32
    v = _byteswap_uint64(v);
#else
    v = __builtin_bswap64(v);
#endif
    return v;
}

H265_INLINE unsigned h265_br_get_bits(H265BitReader *br, int n)
{
    unsigned v;
    int pos;
    if (n <= 0)
        return 0;
    pos = br->pos;
    if (n <= 32 && pos + 32 <= br->size_bits) {
        uint64_t w = h265_load_be64(br->buffer + (pos >> 3));
        v = (unsigned)((w << (pos & 7)) >> (64 - n));
        br->pos = pos + n;
        return v;
    }
    if (pos + n > br->size_bits) {
        int avail = br->size_bits - pos;
        br->error = 1;
        v = avail > 0 ? h265_br_get_bits(br, avail) : 0;
        br->pos = br->size_bits;
        if (avail <= 0)
            return 0;
        return v << (n - avail);
    }
    {
        int i;
        v = 0;
        for (i = 0; i < n; i++) {
            int p = pos + i;
            v = (v << 1) | ((br->buffer[p >> 3] >> (7 - (p & 7))) & 1);
        }
    }
    br->pos = pos + n;
    return v;
}

H265_INLINE unsigned h265_br_get_bits1(H265BitReader *br)
{
    return h265_br_get_bits(br, 1);
}

H265_INLINE void h265_br_skip(H265BitReader *br, int n)
{
    if (n < 0)
        return;
    if (br->pos + n > br->size_bits) {
        br->error = 1;
        br->pos = br->size_bits;
        return;
    }
    br->pos += n;
}

H265_INLINE void h265_br_align(H265BitReader *br)
{
    int rem = br->pos & 7;
    if (rem)
        h265_br_skip(br, 8 - rem);
}

H265_INLINE int h265_br_byte_aligned(const H265BitReader *br)
{
    return !(br->pos & 7);
}

H265_INLINE unsigned h265_br_get_ue(H265BitReader *br)
{
    int leading = 0;
    while (leading < 32 && !h265_br_get_bits1(br))
        leading++;
    if (leading >= 32) {
        br->error = 1;
        return 0;
    }
    return ((1u << leading) - 1) + h265_br_get_bits(br, leading);
}

H265_INLINE int h265_br_get_se(H265BitReader *br)
{
    unsigned v = h265_br_get_ue(br);
    return (v & 1) ? (int)((v + 1) >> 1) : -(int)(v >> 1);
}

H265_INLINE int h265_br_more_rbsp_data(const H265BitReader *br)
{
    int pos = br->pos;
    int last = br->size_bits - 1;
    if (pos >= br->size_bits)
        return 0;
    while (last > pos && !((br->buffer[last >> 3] >> (7 - (last & 7))) & 1))
        last--;
    return pos < last;
}

/* ------------------------------------------------------------------ */
/* CABAC                                                               */
/* ------------------------------------------------------------------ */
#define H265_CABAC_BITS 16
#define H265_CABAC_MASK ((1 << H265_CABAC_BITS) - 1)

typedef struct H265CABAC {
    int low;
    int range;
    const uint8_t *bytestream_start;
    const uint8_t *bytestream;
    const uint8_t *bytestream_end;
} H265CABAC;

extern const uint8_t h265_cabac_tables[512 + 4*2*64 + 4*64 + 63];
#define H265_NORM_SHIFT_OFFSET  0
#define H265_LPS_RANGE_OFFSET   512
#define H265_MLPS_STATE_OFFSET  1024

/* ------------------------------------------------------------------ */
/* parameter sets                                                      */
/* ------------------------------------------------------------------ */
typedef struct H265Window {
    unsigned left_offset, right_offset, top_offset, bottom_offset;
} H265Window;

typedef struct H265PTL {
    uint8_t profile_space;
    uint8_t tier_flag;
    uint8_t profile_idc;
    uint8_t level_idc;
} H265PTL;

typedef struct H265ScalingList {
    uint8_t sl[4][6][64];
    uint8_t sl_dc[2][6];
} H265ScalingList;

typedef struct H265ShortTermRPS {
    int32_t delta_poc[32];
    uint32_t used;
    uint8_t delta_idx;
    uint8_t num_negative_pics;
    uint8_t num_delta_pocs;
    uint8_t rps_idx_num_delta_pocs;
    uint16_t abs_delta_rps;
    unsigned delta_rps_sign:1;
    unsigned rps_predict:1;
    unsigned use_delta:1;
} H265ShortTermRPS;

typedef struct H265LongTermRPS {
    int poc[32];
    uint8_t poc_msb_present[32];
    uint8_t used[32];
    uint8_t nb_refs;
} H265LongTermRPS;

typedef struct H265VPS {
    int refcount;
    int valid;
    unsigned vps_id;
    uint8_t vps_temporal_id_nesting_flag;
    int vps_max_layers;
    int vps_max_sub_layers;      /* minus1 + 1 */
    H265PTL ptl;
    int vps_sub_layer_ordering_info_present_flag;
    unsigned vps_max_dec_pic_buffering[H265_MAX_SUB_LAYERS];
    unsigned vps_num_reorder_pics[H265_MAX_SUB_LAYERS];
    unsigned vps_max_latency_increase[H265_MAX_SUB_LAYERS];
    int multilayer;
    int multilayer_auxiliary;
    int layer_id_in_nuh[2];
    int view_id[2];
} H265VPS;

typedef struct H265SPS {
    int refcount;
    int valid;
    unsigned sps_id;
    unsigned vps_id;
    int chroma_format_idc;
    int separate_colour_plane;

    int width, height;
    int bit_depth, bit_depth_chroma;
    int pixel_shift;
    unsigned log2_max_poc_lsb;
    int max_sub_layers;

    struct {
        int max_dec_pic_buffering;
        int num_reorder_pics;
        int max_latency_increase;
    } temporal_layer[H265_MAX_SUB_LAYERS];

    int log2_min_cb_size;
    int log2_diff_max_min_coding_block_size;
    int log2_min_tb_size;
    int log2_diff_max_min_transform_block_size;
    int log2_max_trafo_size;
    int log2_ctb_size;
    int log2_min_pu_size;
    int max_transform_hierarchy_depth_inter;
    int max_transform_hierarchy_depth_intra;

    int amp_enabled;
    int sao_enabled;
    int pcm_enabled;
    int pcm_loop_filter_disabled;
    struct {
        int bit_depth;
        int bit_depth_chroma;
        int log2_min_pcm_cb_size;
        int log2_max_pcm_cb_size;
    } pcm;

    int nb_st_rps;
    H265ShortTermRPS st_rps[H265_MAX_ST_RPS];

    int long_term_ref_pics_present;
    uint16_t lt_ref_pic_poc_lsb_sps[H265_MAX_LONG_TERM_REF_PICS];
    uint32_t used_by_curr_pic_lt;
    int num_long_term_ref_pics_sps;

    int temporal_mvp_enabled;
    int strong_intra_smoothing_enabled;

    int scaling_list_enabled;
    H265ScalingList scaling_list;

    int range_extension;
    int transform_skip_rotation_enabled;
    int transform_skip_context_enabled;
    int implicit_rdpcm_enabled;
    int explicit_rdpcm_enabled;
    int intra_smoothing_disabled;
    int persistent_rice_adaptation_enabled;
    int cabac_bypass_alignment_enabled;
    int motion_vector_resolution_control_idc;

    H265Window conf_win;
    H265Window output_window;
    int conformance_window;

    int vui_present;
    int sar_num, sar_den;
    int fps_num, fps_den;
    unsigned vui_num_units_in_tick;
    unsigned vui_time_scale;
    int video_full_range;
    int colour_primaries;
    int transfer_characteristics;
    int matrix_coeffs;
    int chroma_loc_type_top;
    int chroma_loc_type_bottom;
    int field_seq_flag;
    int frame_field_info_present;
    int neutral_chroma_indication;

    /* inferred */
    int ctb_width, ctb_height, ctb_size;
    int min_cb_width, min_cb_height;
    int min_tb_width, min_tb_height;
    int min_pu_width, min_pu_height;
    int tb_mask;
    int qp_bd_offset;
    int hshift[3], vshift[3];

    H265PTL ptl;
} H265SPS;

typedef struct H265PPS {
    int refcount;
    int valid;
    unsigned pps_id;
    unsigned sps_id;
    H265SPS *sps;

    int sign_data_hiding_flag;
    int cabac_init_present_flag;
    int num_ref_idx_l0_default_active;
    int num_ref_idx_l1_default_active;
    int pic_init_qp_minus26;
    int constrained_intra_pred_flag;
    int transform_skip_enabled_flag;
    int cu_qp_delta_enabled_flag;
    int diff_cu_qp_delta_depth;
    int cb_qp_offset;
    int cr_qp_offset;
    int pic_slice_level_chroma_qp_offsets_present_flag;
    int weighted_pred_flag;
    int weighted_bipred_flag;
    int output_flag_present_flag;
    int transquant_bypass_enable_flag;
    int dependent_slice_segments_enabled_flag;
    int tiles_enabled_flag;
    int entropy_coding_sync_enabled_flag;

    int num_tile_columns;
    int num_tile_rows;
    int uniform_spacing_flag;
    int loop_filter_across_tiles_enabled_flag;
    int seq_loop_filter_across_slices_enabled_flag;

    int deblocking_filter_control_present_flag;
    int deblocking_filter_override_enabled_flag;
    int disable_dbf;
    int beta_offset;
    int tc_offset;

    int scaling_list_data_present_flag;
    H265ScalingList scaling_list;
    int lists_modification_present_flag;
    int log2_parallel_merge_level;
    int num_extra_slice_header_bits;
    int slice_header_extension_present_flag;
    int log2_max_transform_skip_block_size;
    int pps_extension_present_flag;
    int pps_range_extensions_flag;
    int pps_scc_extension_flag;
    int cross_component_prediction_enabled_flag;
    int chroma_qp_offset_list_enabled_flag;
    int diff_cu_chroma_qp_offset_depth;
    int chroma_qp_offset_list_len_minus1;
    int8_t cb_qp_offset_list[6];
    int8_t cr_qp_offset_list[6];
    int log2_sao_offset_scale_luma;
    int log2_sao_offset_scale_chroma;

    int pps_curr_pic_ref_enabled_flag;
    int pps_slice_act_qp_offsets_present_flag;
    int8_t pps_act_y_qp_offset;
    int8_t pps_act_cb_qp_offset;
    int8_t pps_act_cr_qp_offset;

    /* inferred, owned */
    unsigned *column_width;
    unsigned *row_height;
    unsigned *col_bd;
    unsigned *row_bd;
    int *col_idxX;
    int *ctb_addr_rs_to_ts;
    int *ctb_addr_ts_to_rs;
    int *tile_id;
    int *tile_pos_rs;
    int *min_tb_addr_zs;
    int *min_tb_addr_zs_tab;
} H265PPS;

typedef struct H265ParamSets {
    H265VPS *vps_list[H265_MAX_VPS_COUNT];
    H265SPS *sps_list[H265_MAX_SPS_COUNT];
    H265PPS *pps_list[H265_MAX_PPS_COUNT];
} H265ParamSets;

/* ------------------------------------------------------------------ */
/* frame / block metadata                                              */
/* ------------------------------------------------------------------ */
typedef struct H265SAOParams {
    int type_idx[3];
    int offset_abs[3][4];
    int offset_sign[3][4];
    int band_position[3];
    int eo_class[3];
    int16_t offset_val[3][5];
} H265SAOParams;

typedef struct H265DBParams {
    int beta_offset;
    int tc_offset;
    int disabled;
} H265DBParams;

typedef struct H265Mv {
    int16_t x, y;
} H265Mv;

typedef struct H265MvField {
    H265Mv mv[2];
    int8_t ref_idx[2];
    int8_t pred_flag;
} H265MvField;

typedef struct H265RefList {
    int list[H265_MAX_REFS];          /* POC values */
    int isLongTerm[H265_MAX_REFS];
    int nb_refs;
} H265RefList;

typedef struct H265RefListTab {
    H265RefList refPicList[2];
} H265RefListTab;

typedef struct H265NeighbourAvailable {
    int cand_bottom_left;
    int cand_left;
    int cand_up;
    int cand_up_left;
    int cand_up_right;
    int cand_up_right_sap;
} H265NeighbourAvailable;

typedef struct H265CodingUnit {
    int x, y;
    int pred_mode;
    int part_mode;
    uint8_t intra_split_flag;
    uint8_t max_trafo_depth;
    uint8_t cu_transquant_bypass_flag;
} H265CodingUnit;

typedef struct H265PredictionUnit {
    int mpm_idx;
    int rem_intra_luma_pred_mode;
    uint8_t intra_pred_mode[4];
    H265Mv mvd;
    uint8_t merge_flag;
    uint8_t intra_pred_mode_c[4];
    uint8_t chroma_mode_c[4];
} H265PredictionUnit;

typedef struct H265TransformUnit {
    int cu_qp_delta;
    int res_scale_val;
    int intra_pred_mode;
    int intra_pred_mode_c;
    int chroma_mode_c;
    uint8_t is_cu_qp_delta_coded;
    uint8_t is_cu_chroma_qp_offset_coded;
    int8_t cu_qp_offset_cb;
    int8_t cu_qp_offset_cr;
    uint8_t cross_pf;
} H265TransformUnit;

typedef struct H265FilmGrainParams {
    int present;
    int model_id;
    int blending_mode_id;
    int log2_scale_factor;
    int component_model_present[3];
    int num_intensity_intervals[3];
    int num_model_values[3];
    int intensity_interval_lower_bound[3][256];
    int intensity_interval_upper_bound[3][256];
    int comp_model_value[3][256][6];
} H265FilmGrainParams;

typedef struct H265SEIMeta {
    int pic_struct;                /* -1 when absent */
    int recovery_present;
    int recovery_poc_cnt;
    int recovery_exact_match;
    int recovery_broken_link;

    int timecode_present;
    int timecode_num_clock_ts;
    int timecode_clock_timestamp_flag[3];
    int timecode_units_field_based[3];
    int timecode_counting_type[3];
    int timecode_full_timestamp[3];
    int timecode_discontinuity[3];
    int timecode_cnt_dropped[3];
    int timecode_n_frames[3];
    int timecode_seconds[3];
    int timecode_minutes[3];
    int timecode_hours[3];

    int mastering_present;
    unsigned display_primaries[3][2];
    unsigned white_point[2];
    unsigned max_luminance;
    unsigned min_luminance;

    int content_light_present;
    unsigned max_content_light_level;
    unsigned max_pic_average_light_level;

    int alternative_transfer_present;
    int preferred_transfer_characteristics;

    int display_orientation_present;
    int display_orientation_hflip;
    int display_orientation_vflip;
    unsigned display_orientation_rotation;

    int ambient_present;
    unsigned ambient_illuminance;
    unsigned ambient_light_x;
    unsigned ambient_light_y;

    int frame_packing_present;
    int frame_packing_type;
    int frame_packing_content;
    int frame_packing_current_frame_is_frame0;

    H265FilmGrainParams film_grain;
} H265SEIMeta;

typedef struct H265Frame {
    struct H265Decoder *owner;
    int in_use;
    int refcount;
    int pending_free;
    int seq;                       /* allocation sequence for output tie-break */
    int poc;
    int flags;
    int temporal_id;
    int nal_unit_type;
    int slice_type;
    int is_irap;
    int qp;
    int colour_plane_id;
    int layer_id;

    int64_t pts, dts;              /* timestamps of the packet that started it */
    H265SEIMeta sei;
    uint8_t *t35_data;
    int t35_size;
    uint8_t *unreg_data;
    int unreg_size;
    uint8_t *dovi_rpu;
    int dovi_rpu_size;
    uint8_t *alpha_data;
    int alpha_stride;
    int alpha_width;
    int alpha_height;

    /* pixel storage */
    uint8_t *data[3];
    int linesize[3];               /* bytes */
    int plane_w[3], plane_h[3];
    int width, height;             /* coded size */
    int display_w, display_h;
    int chroma_format;
    int bit_depth;
    int pixel_shift;
    int num_planes;

    /* layout of the current buffers, for reuse without reallocating */
    int alloc_w, alloc_h, alloc_chroma, alloc_bit_depth;
    int alloc_log2_min_cb, alloc_log2_ctb, alloc_sao;

    /* per-frame metadata */
    uint8_t *skip_flag;
    uint8_t *tab_ct_depth;
    uint8_t *cbf_luma;
    uint8_t *tab_ipm;
    uint8_t *is_pcm;
    uint8_t *filter_slice_edges;
    int32_t *tab_slice_address;
    int8_t *qp_y_tab;
    uint8_t *horizontal_bs;
    uint8_t *vertical_bs;
    H265SAOParams *sao;
    H265DBParams *deblock;
    H265MvField *tab_mvf;
    H265RefListTab *rpl_tab;

    /* parameter sets used by this picture (referenced) */
    H265SPS *sps;
    H265PPS *pps;

    /* SAO edge pixel storage */
    uint8_t *sao_pixel_buffer_h[3];
    uint8_t *sao_pixel_buffer_v[3];

    /* decode progress */
    int decode_done;
    int rows_done;
    int pending_slice_jobs;        /* slice-parallel decode coordination */
    int slice_error;
    h265_mutex mutex;
    h265_cond cond;
} H265Frame;

/* ------------------------------------------------------------------ */
/* slice                                                               */
/* ------------------------------------------------------------------ */
typedef struct H265Slice {
    struct H265Decoder *h;
    H265Frame *frame;
    H265PPS *pps;
    H265SPS *sps;

    unsigned pps_id;
    unsigned slice_segment_addr;
    unsigned slice_addr;
    int slice_type;
    int layer_id;

    int pic_order_cnt_lsb;
    int poc;

    uint8_t first_slice_in_pic_flag;
    uint8_t dependent_slice_segment_flag;
    uint8_t pic_output_flag;
    uint8_t colour_plane_id;

    int short_term_ref_pic_set_sps_flag;
    H265ShortTermRPS slice_rps;
    const H265ShortTermRPS *short_term_rps;
    H265LongTermRPS long_term_rps;
    unsigned list_entry_lx[2][32];
    uint8_t rpl_modification_flag[2];

    uint8_t no_output_of_prior_pics_flag;
    uint8_t slice_temporal_mvp_enabled_flag;
    uint8_t rpl_valid;
    unsigned nb_refs[2];
    H265RefList ref_list[2];
    H265Frame *ref_frame[2][H265_MAX_REFS];
    H265Frame *collocated_ref;
    uint8_t slice_sample_adaptive_offset_flag[3];
    uint8_t mvd_l1_zero_flag;
    uint8_t cabac_init_flag;
    uint8_t disable_deblocking_filter_flag;
    uint8_t slice_loop_filter_across_slices_enabled_flag;
    uint8_t collocated_list;
    unsigned collocated_ref_idx;

    int slice_qp_delta;
    int slice_cb_qp_offset;
    int slice_cr_qp_offset;
    int slice_act_y_qp_offset;
    int slice_act_cb_qp_offset;
    int slice_act_cr_qp_offset;
    uint8_t cu_chroma_qp_offset_enabled_flag;

    int beta_offset;
    int tc_offset;

    uint8_t max_num_merge_cand;
    uint8_t use_integer_mv_flag;

    unsigned *entry_point_offset;
    int *offset;
    int *size;
    int num_entry_point_offsets;
    int num_entry_point_alloc;

    int8_t slice_qp;
    uint8_t luma_log2_weight_denom;
    int16_t chroma_log2_weight_denom;
    int16_t luma_weight_l0[16], luma_weight_l1[16];
    int16_t chroma_weight_l0[16][2], chroma_weight_l1[16][2];
    int16_t luma_offset_l0[16], luma_offset_l1[16];
    int16_t chroma_offset_l0[16][2], chroma_offset_l1[16][2];

    int slice_ctb_addr_rs;
    unsigned data_offset;

    /* RBSP owned by this slice (unescaped) */
    uint8_t *rbsp;
    int rbsp_size;
    int *epb_pos;                  /* emulation-prevention byte positions */
    int nb_epb;
    int owns_rbsp;
    int owns_epb;

    /* decode state */
    H265CABAC cabac;
    uint8_t cabac_state[H265_CONTEXTS];
    uint8_t cabac_saved[H265_CONTEXTS];
    uint8_t stat_coeff[H265_STAT_COEFFS];
    uint8_t stat_coeff_saved[H265_STAT_COEFFS];

    int8_t qp_y;
    int8_t curr_qp_y;
    int qPy_pred;
    uint8_t first_qp_group;

    int ctb_left_flag;
    int ctb_up_flag;
    int ctb_up_right_flag;
    int ctb_up_left_flag;
    int end_of_tiles_x;
    int end_of_tiles_y;

    int ct_depth;
    H265CodingUnit cu;
    H265PredictionUnit pu;
    H265TransformUnit tu;
    H265NeighbourAvailable na;
    int boundary_flags;

    uint8_t *edge_emu_buffer;
    uint8_t *edge_emu_buffer2;
    int16_t *tmp;

    int slice_idx;
    int error;
    int row_progress;
} H265Slice;

/* ------------------------------------------------------------------ */
/* jobs / decoder                                                      */
/* ------------------------------------------------------------------ */
#define H265_JOB_IDLE   0
#define H265_JOB_QUEUED 1
#define H265_JOB_RUN   2
#define H265_JOB_DONE  3

typedef struct H265WPPContext {
    int refs;
    int rows;
    int ctb_width;
    int num_ctb;
    int *progress;
    int error;
    H265Slice **slices;           /* row slices; owned by the jobs */
    int *row_offset;
    int *row_size;
    uint8_t *rbsp;
    int rbsp_size;
    int *epb_pos;
    int nb_epb;
    unsigned *entry_point_offset;
    int *entry_offset;
    int *entry_size;
    h265_mutex mutex;
    h265_cond cond;
} H265WPPContext;

typedef struct H265Job {
    int state;
    int result;
    int64_t seq;
    H265Frame *frame;
    int nb_slices;
    int row_mode;
    int slice_parallel;           /* this job decodes exactly one slice */
    int frame_done;               /* this job applied the frame filters */
    H265WPPContext *wpp;
    int wpp_row;
    H265Slice **slices;
    H265Frame *held[2 * H265_MAX_REFS + 1];
    int nb_held;
    struct H265Decoder *h;
} H265Job;

struct H265Decoder;

typedef struct H265ThreadPool H265ThreadPool;

typedef struct H265FramePool {
    H265Frame *frames;
    int count;
    int capacity;
} H265FramePool;

typedef struct H265Decoder {
    h265_param_t param;
    H265ParamSets ps;

    H265VPS *vps;
    H265SPS *sps;
    H265SPS *sei_sps;              /* newest parsed SPS, for prefix SEI */
    H265PPS *pps;
    H265Slice cur_sh;              /* header of the slice being parsed */
    H265Slice *slices[H265_MAX_SLICES];
    int nb_slices;
    H265SEIMeta sei;               /* metadata of the current access unit */
    H265Frame *cur_frame;
    H265Frame *collocated_ref;

    int nal_unit_type;
    int temporal_id;
    int poc;
    int poc_tid0;
    int first_nal_type;
    int last_eos;
    int recovery_poc;
    int no_rasl_output_flag;
    int eos;

    H265RefList rps[H265_NB_RPS_TYPE];

    H265FramePool pool;
    H265Frame *DPB[H265_MAX_DPB_SIZE];
    int nb_dpb;

    /* lazily-generated H.274 grain patterns, owned by the decoder */
    int8_t *film_grain_db;
    uint32_t film_grain_ready[6];

    int64_t packet_pts, packet_dts;
    int64_t decode_seq;

    /* auxiliary-layer decoder for independent alpha/depth layers */
    int ignore_layer_id;
    struct H265Decoder *aux_dec;
    H265Frame *aux_frames[64];
    int aux_count;

    /* raw user-data side data waiting for the next picture */
    uint8_t *pending_t35;
    int pending_t35_size;
    uint8_t *pending_unreg;
    int pending_unreg_size;
    uint8_t *pending_dovi;
    int pending_dovi_size;

    int header_dirty;
    h265_stream_info info;

    h265_mutex lock;
    h265_cond cond;
    h265_cond work_cond;
    int shutting_down;

    /* threaded decode */
    int threaded;
    int nb_threads;
    int nb_threads_running;
    h265_thread thread_handles[32];
    H265ThreadPool *threads;
    H265Job *jobs;
    int nb_jobs;
    int64_t submit_seq;
    int64_t commit_seq;
    int jobs_in_flight;

    int draining;
    int failed;
    int au_open;

    /* input parser: retains the final, possibly incomplete, NAL */
    uint8_t *inbuf;
    int inbuf_len;
    int inbuf_cap;
    int64_t inbuf_pts, inbuf_dts;

    H265Frame *out_q[64];
    int out_count;
} H265Decoder;

/* ------------------------------------------------------------------ */
/* function prototypes                                                 */
/* ------------------------------------------------------------------ */
void h265_vps_free(H265VPS *vps);
void h265_sps_free(H265SPS *sps);
void h265_pps_free(H265PPS *pps);
H265VPS *h265_vps_ref(H265VPS *v);
H265SPS *h265_sps_ref(H265SPS *s);
H265PPS *h265_pps_ref(H265PPS *p);
void h265_vps_unref(H265VPS **v);
void h265_sps_unref(H265SPS **s);
void h265_pps_unref(H265PPS **p);

int h265_parse_vps(H265Decoder *h, const uint8_t *rbsp, int size);
int h265_parse_sps(H265Decoder *h, const uint8_t *rbsp, int size);
int h265_parse_pps(H265Decoder *h, const uint8_t *rbsp, int size);
uint8_t *h265_rbsp_extract_ex(const uint8_t *nal, int len, int *out_size,
                              int **out_epb, int *out_nb);
int h265_parse_sei(H265Decoder *h, const uint8_t *rbsp, int size, int nal_type);
void h265_sei_reset_au(H265Decoder *h);
int h265_set_side_blob(uint8_t **dst, int *dst_size, const uint8_t *data,
                       int size);
void h265_free_side_blob(uint8_t **data, int *size);

int h265_parse_short_term_rps(H265BitReader *br, H265ShortTermRPS *rps,
                              const H265SPS *sps, int is_slice_header);
int h265_compute_poc(unsigned log2_max_poc_lsb, int poc_tid0, int poc_lsb,
                     int nal_unit_type);

int h265_parse_slice_header(H265Decoder *h, H265Slice *sh, const uint8_t *rbsp,
                            int size);
int h265_slice_compute_entry_offsets(H265Slice *sh);
int h265_decode_slice(H265Slice *sh);
int h265_decode_wpp_row(H265Slice *sh, H265WPPContext *ctx, int row);
H265Slice *h265_slice_clone(H265Slice *src);
void h265_wpp_unref(H265WPPContext *ctx);
int h265_frame_nb_refs(H265Slice *sh);
int h265_frame_start(H265Decoder *h, H265Slice *sh);
int h265_slice_rpl(H265Decoder *h, H265Slice *sh);
void h265_frame_finish(H265Decoder *h, H265Slice *sh);
void h265_output_frames(H265Decoder *h, int max_output, int max_dpb, int discard);
void h265_set_qPy(H265Slice *sh, int xBase, int yBase, int log2_cb_size);
void h265_deblocking_boundary_strengths(H265Slice *sh, int x0, int y0,
                                        int log2_trafo_size);
void h265_hevc_h_loop_filter_luma(uint8_t *pix, ptrdiff_t stride, int beta,
                                  const int32_t *tc, const uint8_t *no_p,
                                  const uint8_t *no_q, int pixel_shift, int bit_depth);
void h265_hevc_v_loop_filter_luma(uint8_t *pix, ptrdiff_t stride, int beta,
                                  const int32_t *tc, const uint8_t *no_p,
                                  const uint8_t *no_q, int pixel_shift, int bit_depth);
void h265_hevc_h_loop_filter_chroma(uint8_t *pix, ptrdiff_t stride,
                                    const int32_t *tc, const uint8_t *no_p,
                                    const uint8_t *no_q, int pixel_shift, int bit_depth);
void h265_hevc_v_loop_filter_chroma(uint8_t *pix, ptrdiff_t stride,
                                    const int32_t *tc, const uint8_t *no_p,
                                    const uint8_t *no_q, int pixel_shift, int bit_depth);

/* frame management */
int h265_frame_pool_init(H265Decoder *h);
void h265_frame_pool_free(H265Decoder *h);
H265Frame *h265_frame_alloc(H265Decoder *h);
void h265_frame_unref(H265Frame *f);
void h265_frame_addref(H265Frame *f);
void h265_frame_reset_meta(H265Frame *f);
void h265_frame_clear_aux(H265Frame *f);
int h265_frame_alloc_aux(H265Frame *f, H265SPS *sps);
H265Frame *h265_frame_alloc_display(H265Decoder *h, const H265Frame *src);
int h265_film_grain_supported(const H265FilmGrainParams *p,
                              const H265SPS *sps);
int h265_film_grain_apply(H265Decoder *h, H265Frame *dst,
                          const H265Frame *src,
                          const H265FilmGrainParams *p);
void h265_frame_wait_ref(H265Frame *f);
void h265_frame_wait_rows(H265Frame *f, int rows);
void h265_frame_report_rows(H265Frame *f, int rows);
void h265_frame_mark_done(H265Frame *f);
void h265_frame_filter(H265Decoder *h, H265Slice *sh);
void h265_filter_row(H265Slice *sh, int y0, int last);
void h265_slice_free(H265Slice *sh);
void h265_job_release(H265Job *job);
void h265_thread_commit_locked(H265Decoder *h);

/* cabac */
int h265_cabac_init(H265Slice *sh, const uint8_t *data, int size);
int h265_cabac_init_states(H265Slice *sh);
int h265_cabac_reinit(H265Slice *sh);
const uint8_t *h265_cabac_skip_bytes(H265Slice *sh, int n);
int h265_cabac_terminate(H265Slice *sh);
void h265_save_states(H265Slice *sh, int ctb_addr_ts);
int h265_end_of_slice_flag_decode(H265Slice *sh);
int h265_sao_merge_flag_decode(H265Slice *sh);
int h265_sao_type_idx_decode(H265Slice *sh);
int h265_sao_band_position_decode(H265Slice *sh);
int h265_sao_offset_abs_decode(H265Slice *sh, int bit_depth);
int h265_sao_offset_sign_decode(H265Slice *sh);
int h265_sao_eo_class_decode(H265Slice *sh);
int h265_cu_transquant_bypass_flag_decode(H265Slice *sh);
int h265_skip_flag_decode(H265Slice *sh, uint8_t *skip_flag, int x0, int y0,
                          int x_cb, int y_cb, int min_cb_width);
int h265_pred_mode_decode(H265Slice *sh);
int h265_split_coding_unit_flag_decode(H265Slice *sh, uint8_t *tab_ct_depth,
                                       const H265SPS *sps, int ct_depth,
                                       int x0, int y0);
int h265_part_mode_decode(H265Slice *sh, const H265SPS *sps, int log2_cb_size);
int h265_pcm_flag_decode(H265Slice *sh);
int h265_prev_intra_luma_pred_flag_decode(H265Slice *sh);
int h265_mpm_idx_decode(H265Slice *sh);
int h265_rem_intra_luma_pred_mode_decode(H265Slice *sh);
int h265_intra_chroma_pred_mode_decode(H265Slice *sh);
int h265_merge_idx_decode(H265Slice *sh);
int h265_merge_flag_decode(H265Slice *sh);
int h265_inter_pred_idc_decode(H265Slice *sh, int nPbW, int nPbH);
int h265_ref_idx_lx_decode(H265Slice *sh, int num_ref_idx_lx);
int h265_mvp_lx_flag_decode(H265Slice *sh);
int h265_no_residual_syntax_flag_decode(H265Slice *sh);
int h265_split_transform_flag_decode(H265Slice *sh, int log2_trafo_size);
int h265_cbf_cb_cr_decode(H265Slice *sh, int trafo_depth);
int h265_cbf_luma_decode(H265Slice *sh, int trafo_depth);
int h265_log2_res_scale_abs(H265Slice *sh, int idx);
int h265_res_scale_sign_flag(H265Slice *sh, int idx);
int h265_cu_qp_delta_abs(H265Slice *sh);
int h265_cu_qp_delta_sign_flag(H265Slice *sh);
int h265_cu_chroma_qp_offset_flag(H265Slice *sh);
int h265_cu_chroma_qp_offset_idx(H265Slice *sh, int chroma_qp_offset_list_len_minus1);
void h265_hls_residual_coding(H265Slice *sh, int x0, int y0, int log2_trafo_size,
                              int scan_idx, int c_idx);
void h265_hls_mvd_coding(H265Slice *sh);

/* intra / inter prediction */
void h265_set_neighbour_available(H265Slice *sh, int x0, int y0, int nPbW,
                                  int nPbH, int log2_ctb_size);
void h265_intra_pred(H265Slice *sh, int x0, int y0, int log2_size, int c_idx);
void h265_luma_mv_merge_mode(H265Slice *sh, int x0, int y0, int nPbW, int nPbH,
                             int log2_cb_size, int part_idx, int merge_idx,
                             H265MvField *mv);
void h265_luma_mv_mvp_mode(H265Slice *sh, int x0, int y0, int nPbW, int nPbH,
                           int log2_cb_size, int part_idx, int merge_idx,
                           H265MvField *mv, int mvp_lx_flag, int LX);

/* transforms */
void h265_dequant(int16_t *coeffs, int log2_size, int bit_depth);
void h265_idct(int16_t *coeffs, int log2_size, int col_limit, int bit_depth);
void h265_idct_dc(int16_t *coeffs, int log2_size, int bit_depth);
void h265_transform_4x4_luma(int16_t *coeffs, int bit_depth);
void h265_transform_rdpcm(int16_t *coeffs, int log2_size, int mode);
void h265_add_residual(uint8_t *dst, const int16_t *res, ptrdiff_t stride,
                       int size, int pixel_shift, int bit_depth);
void h265_put_pcm(uint8_t *dst, ptrdiff_t stride, int width, int height,
                  H265BitReader *gb, int pcm_bit_depth, int pixel_shift,
                  int bit_depth);

/* motion compensation */
void h265_emulated_edge_mc(uint8_t *dst, const uint8_t *src, ptrdiff_t dst_stride,
                           ptrdiff_t src_stride, int block_w, int block_h,
                           int src_x, int src_y, int w, int h, int pixel_shift);
void h265_mc_luma_uni(H265Slice *sh, uint8_t *dst, ptrdiff_t dststride,
                      const H265Frame *ref, const H265Mv *mv, int x_off,
                      int y_off, int block_w, int block_h,
                      int luma_weight, int luma_offset);
void h265_mc_luma_bi(H265Slice *sh, uint8_t *dst, ptrdiff_t dststride,
                     const H265Frame *ref0, const H265Mv *mv0, int x_off,
                     int y_off, int block_w, int block_h,
                     const H265Frame *ref1, const H265Mv *mv1,
                     H265MvField *current_mv);
void h265_mc_chroma_uni(H265Slice *sh, uint8_t *dst0, ptrdiff_t dststride,
                        const uint8_t *src0, ptrdiff_t srcstride, int reflist,
                        int x_off, int y_off, int block_w, int block_h,
                        const H265MvField *current_mv, int chroma_weight,
                        int chroma_offset);
void h265_mc_chroma_bi(H265Slice *sh, uint8_t *dst0, ptrdiff_t dststride,
                       const H265Frame *ref0, const H265Frame *ref1,
                       int x_off, int y_off, int block_w, int block_h,
                       const H265MvField *current_mv, int cidx);

/* filters */
void h265_deblock_frame(H265Slice *sh);
void h265_deblock_ctb_row(H265Slice *sh, int y0);
void h265_sao_frame(H265Slice *sh);
void h265_sao_ctb_row(H265Slice *sh, int y0);

/* ctu decoding */
int h265_decode_ctu_line(H265Slice *sh, int ctb_addr_ts, int *next_ts);

/* dsp init */
#ifdef H265_X86
void h265_add_residual_sse2(uint8_t *dst, const int16_t *res, ptrdiff_t stride,
                            int size, int pixel_shift, int bit_depth);
void h265_avg_pixels_sse2(uint8_t *dst, ptrdiff_t dst_stride,
                          const uint8_t *src1, ptrdiff_t src1_stride,
                          const uint8_t *src2, ptrdiff_t src2_stride,
                          int width, int height, int pixel_shift);
void h265_copy_pixels_sse2(uint8_t *dst, ptrdiff_t dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int width, int height, int pixel_shift);
void h265_add_residual_avx2(uint8_t *dst, const int16_t *res, ptrdiff_t stride,
                            int size, int pixel_shift, int bit_depth);
void h265_avg_pixels_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                          const uint8_t *src1, ptrdiff_t src1_stride,
                          const uint8_t *src2, ptrdiff_t src2_stride,
                          int width, int height, int pixel_shift);
void h265_sao_band_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                        const uint8_t *src, ptrdiff_t src_stride,
                        const int *offset_table, int width, int height,
                        int pixel_shift, int bit_depth);
void h265_sao_edge_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                        const uint8_t *src, ptrdiff_t src_stride,
                        int a_stride, int b_stride,
                        const int16_t *sao_offset_val, int width, int height,
                        int pixel_shift, int bit_depth);
void h265_mc_filter_h_avx2(int16_t *dst, int dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int height, int width, const int8_t *filter,
                           int taps, int shift, int pixel_shift);
void h265_mc_filter_v_avx2(int16_t *dst, int dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int height, int width, const int8_t *filter,
                           int taps, int shift, int pixel_shift,
                           int signed_src);
void h265_copy_pixels_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int width, int height, int pixel_shift);
void h265_mc_uni_write_avx2(uint8_t *dst8, ptrdiff_t dst_stride,
                            const int16_t *src, int src_stride,
                            int width, int height, int shift, int add,
                            int bit_depth, int wx, int ox);
void h265_put_pixels_avx2(int16_t *dst, int dst_stride, const uint8_t *src,
                          ptrdiff_t src_stride, int width, int height,
                          int shift, int pixel_shift);
void h265_mc_weighted_int_avx2(uint8_t *dst8, ptrdiff_t dst_stride,
                              const uint8_t *src, ptrdiff_t src_stride,
                              int width, int height, int wx, int ox,
                              int denom, int bit_depth, int pixel_shift);
void h265_mc_combine_bi_avx2(uint8_t *dst8, ptrdiff_t dst_stride,
                             const int16_t *first, const int16_t *second,
                             int src_stride, int width, int height, int shift,
                             int add, int bit_depth, int weighted,
                             int wx0, int wx1);
#endif

#ifdef H265_NEON
void h265_add_residual_neon(uint8_t *dst, const int16_t *res, ptrdiff_t stride,
                            int size, int pixel_shift, int bit_depth);
void h265_avg_pixels_neon(uint8_t *dst, ptrdiff_t dst_stride,
                          const uint8_t *src1, ptrdiff_t src1_stride,
                          const uint8_t *src2, ptrdiff_t src2_stride,
                          int width, int height, int pixel_shift);
void h265_copy_pixels_neon(uint8_t *dst, ptrdiff_t dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int width, int height, int pixel_shift);
void h265_put_pixels_neon(int16_t *dst, int dst_stride, const uint8_t *src,
                          ptrdiff_t src_stride, int width, int height,
                          int shift, int pixel_shift);
void h265_mc_filter_h_neon(int16_t *dst, int dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int height, int width, const int8_t *filter,
                           int taps, int shift, int pixel_shift);
void h265_mc_filter_v_neon(int16_t *dst, int dst_stride,
                           const uint8_t *src, ptrdiff_t src_stride,
                           int height, int width, const int8_t *filter,
                           int taps, int shift, int pixel_shift,
                           int signed_src);
void h265_mc_uni_write_neon(uint8_t *dst8, ptrdiff_t dst_stride,
                            const int16_t *src, int src_stride,
                            int width, int height, int shift, int add,
                            int bit_depth, int wx, int ox);
void h265_mc_combine_bi_neon(uint8_t *dst8, ptrdiff_t dst_stride,
                             const int16_t *first, const int16_t *second,
                             int src_stride, int width, int height, int shift,
                             int add, int bit_depth, int weighted,
                             int wx0, int wx1);
void h265_mc_weighted_int_neon(uint8_t *dst8, ptrdiff_t dst_stride,
                               const uint8_t *src, ptrdiff_t src_stride,
                               int width, int height, int wx, int ox,
                               int denom, int bit_depth, int pixel_shift);
void h265_sao_band_neon(uint8_t *dst, ptrdiff_t dst_stride,
                        const uint8_t *src, ptrdiff_t src_stride,
                        const int *offset_table, int width, int height,
                        int pixel_shift, int bit_depth);
void h265_sao_edge_neon(uint8_t *dst, ptrdiff_t dst_stride,
                        const uint8_t *src, ptrdiff_t src_stride,
                        int a_stride, int b_stride,
                        const int16_t *sao_offset_val, int width, int height,
                        int pixel_shift, int bit_depth);
#endif

void h265_dsp_init(void);
void h265_cpu_init(void);
extern int h265_have_sse2;
extern int h265_have_avx2;
extern int h265_have_neon;

/* threading */
int h265_thread_init(H265Decoder *h);
void h265_thread_uninit(H265Decoder *h);
int h265_job_submit(H265Decoder *h, H265Job *job);
void h265_job_wait_all(H265Decoder *h);

/* debug / tables */
extern const uint8_t h265_diag_scan4x4_x[16];
extern const uint8_t h265_diag_scan4x4_y[16];
extern const uint8_t h265_diag_scan8x8_x[64];
extern const uint8_t h265_diag_scan8x8_y[64];
extern const int8_t h265_epel_filters[8][4];
extern const int8_t h265_qpel_filters[4][16];

#endif /* H265_INTERNAL_H */
