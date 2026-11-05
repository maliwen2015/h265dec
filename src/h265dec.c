/*
 * h265dec public API and NAL dispatch.
 *
 * Algorithms and constant tables are derived from ITU-T Rec. H.265 and from
 * FFmpeg's HEVC decoder (LGPL-2.1+).
 */
#include "h265_internal.h"


H265Dbg h265_dbg_cfg;
static int h265_dbg_initialized;

int h265_set_side_blob(uint8_t **dst, int *dst_size, const uint8_t *data,
                       int size)
{
    uint8_t *p;
    if (size <= 0) {
        h265_free_side_blob(dst, dst_size);
        return 0;
    }
    p = (uint8_t *)malloc((size_t)size);
    if (!p)
        return -1;
    memcpy(p, data, (size_t)size);
    free(*dst);
    *dst = p;
    *dst_size = size;
    return 0;
}

void h265_free_side_blob(uint8_t **data, int *size)
{
    free(*data);
    *data = NULL;
    *size = 0;
}

void h265_dbg_init(void)
{
    const char *e;
    if (h265_dbg_initialized)
        return;
    h265_dbg_initialized = 1;
    memset(&h265_dbg_cfg, 0, sizeof(h265_dbg_cfg));
    e = getenv("H265_DEBUG_BITS"); if (e) h265_dbg_cfg.bits = atoi(e);
    e = getenv("H265_DEBUG_MB");   if (e) h265_dbg_cfg.mb = atoi(e);
    e = getenv("H265_DEBUG_MC");   if (e) h265_dbg_cfg.mc = atoi(e);
    e = getenv("H265_DEBUG_LF");   if (e) h265_dbg_cfg.lf = atoi(e);
    e = getenv("H265_DEBUG_SAO");  if (e) h265_dbg_cfg.sao = atoi(e);
    e = getenv("H265_DEBUG_PS");   if (e) h265_dbg_cfg.ps = atoi(e);
    e = getenv("H265_DEBUG_X");    if (e) h265_dbg_cfg.mbx = atoi(e);
    e = getenv("H265_DEBUG_Y");    if (e) h265_dbg_cfg.mby = atoi(e);
}

/* exported by h265_nal.c */
int h265_nal_start_code(const uint8_t *data, int size, int pos, int *sc_len);
int h265_split_annexb(const uint8_t *data, int size, int *pos,
                      const uint8_t **out, int *out_len);
int h265_split_avcc(const uint8_t *data, int size, int *pos,
                    const uint8_t **out, int *out_len, int nal_length_size);
uint8_t *h265_rbsp_extract(const uint8_t *nal, int len, int *out_size);
uint8_t *h265_rbsp_extract_ex(const uint8_t *nal, int len, int *out_size,
                              int **out_epb, int *out_nb);
int h265_nal_type(const uint8_t *nal, int len);
int h265_nal_layer_id(const uint8_t *nal, int len);
int h265_nal_temporal_id(const uint8_t *nal, int len);

extern void *h265_malloc_aligned(size_t size, int align);
extern void h265_free_aligned(void *p);

void h265_slice_free(H265Slice *sh)
{
    if (!sh)
        return;
    if (sh->owns_rbsp)
        free(sh->rbsp);
    if (sh->owns_epb)
        free(sh->epb_pos);
    free(sh->entry_point_offset);
    free(sh->offset);
    free(sh->size);
    h265_free_aligned(sh->edge_emu_buffer);
    h265_free_aligned(sh->edge_emu_buffer2);
    free(sh->tmp);
    free(sh);
}

static void update_info(H265Decoder *h, H265SPS *sps)
{
    h265_stream_info *info = &h->info;
    int max_sub = sps->max_sub_layers - 1;

    memset(info, 0, sizeof(*info));
    info->profile = sps->ptl.profile_idc;
    info->level = sps->ptl.level_idc;
    info->coded_width = sps->width;
    info->coded_height = sps->height;
    info->width = sps->width - (int)sps->output_window.left_offset -
                  (int)sps->output_window.right_offset;
    info->height = sps->height - (int)sps->output_window.top_offset -
                   (int)sps->output_window.bottom_offset;
    info->chroma_format = sps->chroma_format_idc;
    info->bit_depth = sps->bit_depth;
    info->frame_rate_num = sps->fps_num;
    info->frame_rate_den = sps->fps_den;
    info->bitrate = 0;
    info->sar_width = sps->sar_num;
    info->sar_height = sps->sar_den;
    info->max_num_reorder_frames = sps->temporal_layer[max_sub].num_reorder_pics;
    info->max_dec_frame_buffering = sps->temporal_layer[max_sub].max_dec_pic_buffering;
    info->temporal_id_nesting = 1;
    info->separate_colour_plane = sps->separate_colour_plane;
    info->video_full_range = sps->video_full_range;
    info->colour_primaries = sps->colour_primaries;
    info->transfer_characteristics = sps->transfer_characteristics;
    info->matrix_coeffs = sps->matrix_coeffs;
    info->chroma_sample_location = sps->chroma_loc_type_top;
    info->field_seq_flag = sps->field_seq_flag;
}

/* Set up one job per WPP CTU row.  Returns 1 when the access unit was handed
 * to the worker pool, 0 to fall back to the sequential decoder. */
static int start_wpp_jobs(H265Decoder *h, H265Slice *sh)
{
    H265SPS *sps;
    H265PPS *pps;
    H265WPPContext *ctx;
    H265Slice **rows;
    H265Slice ***job_arrays = NULL;
    int nr, r;

    if (!sh || !h->threaded || h->nb_slices != 1 ||
        !h->param.wpp_threads)
        return 0;
    sps = sh->sps;
    pps = sh->pps;
    if (!sps || !pps || sh->dependent_slice_segment_flag ||
        sh->slice_ctb_addr_rs != 0 ||
        !pps->entropy_coding_sync_enabled_flag ||
        pps->tiles_enabled_flag)
        return 0;
    nr = sps->ctb_height;
    if (sps->ctb_width < 2 || nr < 2 ||
        sh->num_entry_point_offsets != nr - 1)
        return 0;
    if (h265_slice_compute_entry_offsets(sh) < 0)
        return 0;

    rows = (H265Slice **)calloc((size_t)nr, sizeof(H265Slice *));
    if (!rows)
        return 0;
    rows[0] = sh;
    for (r = 1; r < nr; r++) {
        rows[r] = h265_slice_clone(sh);
        if (!rows[r])
            goto fail_rows;
        rows[r]->slice_ctb_addr_rs = r * sps->ctb_width;
        rows[r]->row_progress = 0;
    }

    ctx = (H265WPPContext *)calloc(1, sizeof(*ctx));
    if (!ctx)
        goto fail_rows;
    h265_mutex_init(&ctx->mutex);
    h265_cond_init(&ctx->cond);
    ctx->rows = nr;
    ctx->ctb_width = sps->ctb_width;
    ctx->num_ctb = sps->ctb_size;
    ctx->progress = (int *)calloc((size_t)nr, sizeof(int));
    ctx->row_offset = (int *)malloc((size_t)nr * sizeof(int));
    ctx->row_size = (int *)malloc((size_t)nr * sizeof(int));
    ctx->slices = rows;
    job_arrays = (H265Slice ***)calloc((size_t)nr, sizeof(H265Slice **));
    if (!ctx->progress || !ctx->row_offset || !ctx->row_size || !job_arrays)
        goto fail_ctx;
    for (r = 0; r < nr; r++) {
        job_arrays[r] = (H265Slice **)malloc(sizeof(H265Slice *));
        if (!job_arrays[r])
            goto fail_ctx;
    }

    /* Move the shared bitstream/entry-point ownership into the context. */
    ctx->rbsp = sh->rbsp;
    ctx->rbsp_size = sh->rbsp_size;
    ctx->epb_pos = sh->epb_pos;
    ctx->nb_epb = sh->nb_epb;
    ctx->entry_point_offset = sh->entry_point_offset;
    ctx->entry_offset = sh->offset;
    ctx->entry_size = sh->size;
    sh->owns_rbsp = 0;
    sh->owns_epb = 0;
    sh->entry_point_offset = NULL;
    sh->offset = NULL;
    sh->size = NULL;
    sh->num_entry_point_offsets = 0;
    sh->num_entry_point_alloc = 0;

    for (r = 0; r < nr; r++) {
        rows[r]->owns_rbsp = 0;
        rows[r]->owns_epb = 0;
        rows[r]->entry_point_offset = NULL;
        rows[r]->offset = NULL;
        rows[r]->size = NULL;
        rows[r]->num_entry_point_offsets = 0;
        rows[r]->row_progress = 0;
        ctx->row_offset[r] = ctx->entry_offset[r];
        ctx->row_size[r] = ctx->entry_size[r];
    }

    ctx->refs = nr;
    h->cur_frame->pending_slice_jobs = nr;
    h->cur_frame->slice_error = 0;

    for (r = 0; r < nr; r++) {
        H265Job job;
        H265Slice *hs = rows[r];
        int li, ri, k;
        memset(&job, 0, sizeof(job));
        job.h = h;
        job.frame = h->cur_frame;
        job.nb_slices = 1;
        job.row_mode = 0;
        job.slice_parallel = 1;
        job.wpp = ctx;
        job.wpp_row = r;
        job.slices = job_arrays[r];
        job.slices[0] = hs;
        h265_frame_addref(h->cur_frame);
        for (li = 0; li < 2; li++)
            for (ri = 0; ri < hs->ref_list[li].nb_refs; ri++) {
                H265Frame *rf = hs->ref_frame[li][ri];
                if (rf && job.nb_held < (int)H265_ARRAY_ELEMS(job.held)) {
                    h265_frame_addref(rf);
                    job.held[job.nb_held++] = rf;
                }
            }
        if (h265_job_submit(h, &job) < 0) {
            free(job.slices);
            for (k = 0; k < job.nb_held; k++)
                h265_frame_unref(job.held[k]);
            h265_frame_unref(h->cur_frame);
            break;
        }
        job_arrays[r] = NULL;
    }
    free(job_arrays);
    if (r == nr)
        return 1;

    /* The pool could not accept every row; release the context and fall back. */
    h265_wpp_unref(ctx);
    return 0;

fail_ctx:
    for (r = 0; job_arrays && r < nr; r++)
        free(job_arrays[r]);
    free(job_arrays);
    h265_mutex_destroy(&ctx->mutex);
    h265_cond_destroy(&ctx->cond);
    free(ctx->progress);
    free(ctx->row_offset);
    free(ctx->row_size);
    free(ctx);
fail_rows:
    for (r = 1; r < nr; r++)
        h265_slice_free(rows[r]);
    free(rows);
    return 0;
}

/* Finish the current access unit: filter and release the slice objects. */
static void h265_au_close(H265Decoder *h)
{
    int i;
    H265Slice *last;

    if (!h->au_open)
        return;
    last = h->nb_slices ? h->slices[h->nb_slices - 1] : NULL;
    if (h->cur_frame) {
        if (h->threaded) {
            int has_dependent = 0;
            for (i = 0; i < h->nb_slices; i++)
                if (h->slices[i]->dependent_slice_segment_flag)
                    has_dependent = 1;
            if (has_dependent) {
                /* A dependent segment inherits CABAC state from the previous
                 * segment, so decode this access unit in order on the owner
                 * thread instead of handing it to the pool. */
                for (i = 0; i < h->nb_slices; i++) {
                    h->slices[i]->row_progress = 0;
                    if (h265_decode_slice(h->slices[i]) < 0) {
                        h->slices[i]->error = 1;
                        h->cur_frame->flags |= H265_FRAME_FLAG_CORRUPT;
                    }
                }
                h265_frame_finish(h, last ? last : &h->cur_sh);
                for (i = 0; i < h->nb_slices; i++)
                    h265_slice_free(h->slices[i]);
                h->nb_slices = 0;
                h->au_open = 0;
                h->cur_frame = NULL;
                h265_sei_reset_au(h);
                return;
            }

            if (start_wpp_jobs(h, h->nb_slices == 1 ? h->slices[0] : NULL)) {
                h->nb_slices = 0;
                h->au_open = 0;
                h->cur_frame = NULL;
                h265_sei_reset_au(h);
                return;
            }

            if (h->nb_slices > 1) {
                /* Independent slices are decoded by separate workers.  The
                 * last one to finish applies the frame filters; the others
                 * only release their references. */
                H265Slice **arrays[H265_MAX_SLICES];
                int prepared = 1;
                for (i = 0; i < h->nb_slices; i++) {
                    arrays[i] = (H265Slice **)malloc(sizeof(H265Slice *));
                    if (!arrays[i]) {
                        prepared = 0;
                        while (i-- > 0)
                            free(arrays[i]);
                        break;
                    }
                    arrays[i][0] = h->slices[i];
                }
                if (prepared) {
                    h->cur_frame->pending_slice_jobs = h->nb_slices;
                    h->cur_frame->slice_error = 0;
                    for (i = 0; i < h->nb_slices; i++) {
                        H265Job job;
                        H265Slice *hs = h->slices[i];
                        int li, ri, k;
                        memset(&job, 0, sizeof(job));
                        job.h = h;
                        job.frame = h->cur_frame;
                        job.nb_slices = 1;
                        job.row_mode = 0;
                        job.slice_parallel = 1;
                        job.slices = arrays[i];
                        h265_frame_addref(h->cur_frame);
                        for (li = 0; li < 2; li++)
                            for (ri = 0; ri < hs->ref_list[li].nb_refs; ri++) {
                                H265Frame *rf = hs->ref_frame[li][ri];
                                if (rf && job.nb_held < (int)H265_ARRAY_ELEMS(job.held)) {
                                    h265_frame_addref(rf);
                                    job.held[job.nb_held++] = rf;
                                }
                            }
                        if (h265_job_submit(h, &job) < 0) {
                            free(job.slices);
                            for (k = 0; k < job.nb_held; k++)
                                h265_frame_unref(job.held[k]);
                            h265_slice_free(hs);
                            h265_frame_unref(h->cur_frame);
                        }
                    }
                    h->nb_slices = 0;
                    h->au_open = 0;
                    h->cur_frame = NULL;
                    h265_sei_reset_au(h);
                    return;
                }
            }

            {
                H265Job job;
                memset(&job, 0, sizeof(job));
                job.h = h;
                job.frame = h->cur_frame;
                job.nb_slices = h->nb_slices;
                job.row_mode = (h->nb_slices == 1 && h->pps &&
                                !h->pps->tiles_enabled_flag);
                job.slices = (H265Slice **)malloc(H265_MAX_SLICES * sizeof(H265Slice *));
                if (job.slices) {
                    int si, li, ri;
                    for (i = 0; i < h->nb_slices; i++)
                        job.slices[i] = h->slices[i];
                    h265_frame_addref(h->cur_frame);
                    for (si = 0; si < h->nb_slices; si++) {
                        H265Slice *hs = h->slices[si];
                        for (li = 0; li < 2; li++)
                            for (ri = 0; ri < hs->ref_list[li].nb_refs; ri++) {
                                H265Frame *rf = hs->ref_frame[li][ri];
                                if (rf && job.nb_held < (int)H265_ARRAY_ELEMS(job.held)) {
                                    h265_frame_addref(rf);
                                    job.held[job.nb_held++] = rf;
                                }
                            }
                    }
                    if (h265_job_submit(h, &job) < 0) {
                        free(job.slices);
                        for (i = 0; i < job.nb_held; i++)
                            h265_frame_unref(job.held[i]);
                        for (i = 0; i < h->nb_slices; i++)
                            h265_slice_free(h->slices[i]);
                    }
                } else {
                    for (i = 0; i < h->nb_slices; i++)
                        h265_slice_free(h->slices[i]);
                }
            }
            h->nb_slices = 0;
            h->au_open = 0;
            h->cur_frame = NULL;
            h265_sei_reset_au(h);
            return;
        }
        if (last && last->error)
            h->cur_frame->flags |= H265_FRAME_FLAG_CORRUPT;
        h265_frame_finish(h, last ? last : &h->cur_sh);
        h->cur_frame = NULL;
    }
    for (i = 0; i < h->nb_slices; i++)
        h265_slice_free(h->slices[i]);
    h->nb_slices = 0;
    h->au_open = 0;
    h265_sei_reset_au(h);
}

static H265Slice *slice_new(H265Decoder *h, uint8_t *rbsp, int rbsp_size,
                            int *epb_pos, int nb_epb)
{
    H265Slice *sh = (H265Slice *)calloc(1, sizeof(H265Slice));
    if (!sh) {
        free(rbsp);
        free(epb_pos);
        return NULL;
    }
    sh->h = h;
    sh->rbsp = rbsp;
    sh->rbsp_size = rbsp_size;
    sh->layer_id = ((((const uint8_t *)rbsp)[0] & 1) << 5) |
                   (((const uint8_t *)rbsp)[1] >> 3);
    sh->epb_pos = epb_pos;
    sh->nb_epb = nb_epb;
    sh->owns_rbsp = 1;
    sh->owns_epb = 1;
    sh->edge_emu_buffer = (uint8_t *)h265_malloc_aligned(131072, 32);
    sh->edge_emu_buffer2 = (uint8_t *)h265_malloc_aligned(131072, 32);
    sh->tmp = (int16_t *)malloc(64 * 64 * sizeof(int16_t));
    if (!sh->edge_emu_buffer || !sh->edge_emu_buffer2 || !sh->tmp) {
        h265_slice_free(sh);
        return NULL;
    }
    return sh;
}

H265Slice *h265_slice_clone(H265Slice *src)
{
    H265Slice *sh;
    if (!src)
        return NULL;
    sh = (H265Slice *)malloc(sizeof(H265Slice));
    if (!sh)
        return NULL;
    *sh = *src;
    sh->owns_rbsp = 0;
    sh->owns_epb = 0;
    sh->entry_point_offset = NULL;
    sh->offset = NULL;
    sh->size = NULL;
    sh->num_entry_point_offsets = 0;
    sh->num_entry_point_alloc = 0;
    sh->edge_emu_buffer = (uint8_t *)h265_malloc_aligned(131072, 32);
    sh->edge_emu_buffer2 = (uint8_t *)h265_malloc_aligned(131072, 32);
    sh->tmp = (int16_t *)malloc(64 * 64 * sizeof(int16_t));
    if (!sh->edge_emu_buffer || !sh->edge_emu_buffer2 || !sh->tmp) {
        free(sh->edge_emu_buffer);
        free(sh->edge_emu_buffer2);
        free(sh->tmp);
        free(sh);
        return NULL;
    }
    memset(&sh->cabac, 0, sizeof(sh->cabac));
    memset(sh->cabac_state, 0, sizeof(sh->cabac_state));
    memset(sh->cabac_saved, 0, sizeof(sh->cabac_saved));
    memset(sh->stat_coeff, 0, sizeof(sh->stat_coeff));
    memset(sh->stat_coeff_saved, 0, sizeof(sh->stat_coeff_saved));
    return sh;
}

void h265_wpp_unref(H265WPPContext *ctx)
{
    int i;
    if (!ctx)
        return;
    if (h265_atomic_dec(&ctx->refs) != 0)
        return;
    for (i = 0; i < ctx->rows; i++)
        h265_slice_free(ctx->slices[i]);
    h265_mutex_destroy(&ctx->mutex);
    h265_cond_destroy(&ctx->cond);
    free(ctx->progress);
    free(ctx->slices);
    free(ctx->row_offset);
    free(ctx->row_size);
    free(ctx->rbsp);
    free(ctx->epb_pos);
    free(ctx->entry_point_offset);
    free(ctx->entry_offset);
    free(ctx->entry_size);
    free(ctx);
}

static int decode_one_slice(H265Decoder *h, H265Slice *sh)
{
    if (!sh)
        return -1;
    if (h265_decode_slice(sh) < 0) {
        sh->error = 1;
        if (h->cur_frame)
            h->cur_frame->flags |= H265_FRAME_FLAG_CORRUPT;
        return -1;
    }
    return 0;
}

static int handle_vcl(H265Decoder *h, int nal_type, int temporal_id,
                      const uint8_t *rbsp, int rbsp_size, int *epb_pos,
                      int nb_epb)
{
    H265Slice *sh;
    int first;

    if (rbsp_size < 3) {
        free((void *)rbsp);
        free(epb_pos);
        return -1;
    }
    first = rbsp[2] >> 7;
    h->nal_unit_type = nal_type;
    h->temporal_id = temporal_id;

    if (first) {
        h265_au_close(h);
        sh = slice_new(h, (uint8_t *)rbsp, rbsp_size, epb_pos, nb_epb);
        if (!sh)
            return -1;
        epb_pos = NULL;
        if (h265_parse_slice_header(h, sh, rbsp, rbsp_size) < 0) {
            h265_slice_free(sh);
            return -1;
        }
        if (h265_frame_start(h, sh) < 0) {
            h265_slice_free(sh);
            return -1;
        }
        if (h->nb_slices < H265_MAX_SLICES)
            h->slices[h->nb_slices++] = sh;
        else
            h265_slice_free(sh);
        h->au_open = 1;
        if (!h->threaded)
            decode_one_slice(h, sh);
    } else {
        if (!h->au_open || !h->cur_frame) {
            return -1;
        }
        sh = slice_new(h, (uint8_t *)rbsp, rbsp_size, epb_pos, nb_epb);
        if (!sh)
            return -1;
        epb_pos = NULL;
        if (h->nb_slices > 0) {
            H265Slice *prev = h->slices[h->nb_slices - 1];
            uint8_t firstbits = rbsp[2] >> 7;
            int dependent = 0;
            (void)firstbits;
            /* Copy the independent header so that a dependent segment inherits
             * all the fields it is not allowed to re-send.  Detect the
             * dependent flag from the bitstream only after parsing it. */
            {
                uint8_t *e1 = sh->edge_emu_buffer;
                uint8_t *e2 = sh->edge_emu_buffer2;
                int16_t *tmp = sh->tmp;
                *sh = h->cur_sh;
                sh->edge_emu_buffer = e1;
                sh->edge_emu_buffer2 = e2;
                sh->tmp = tmp;
            }
            sh->h = h;
            sh->rbsp = (uint8_t *)rbsp;
            sh->rbsp_size = rbsp_size;
            sh->entry_point_offset = NULL;
            sh->offset = NULL;
            sh->size = NULL;
            sh->num_entry_point_alloc = 0;
            sh->num_entry_point_offsets = 0;
            memset(&sh->cabac, 0, sizeof(sh->cabac));
            sh->error = 0;
            sh->frame = h->cur_frame;
            if (h265_parse_slice_header(h, sh, rbsp, rbsp_size) < 0) {
                h265_slice_free(sh);
                return -1;
            }
            dependent = sh->dependent_slice_segment_flag;
            if (dependent) {
                memcpy(sh->cabac_state, prev->cabac_state, H265_CONTEXTS);
                memcpy(sh->stat_coeff, prev->stat_coeff, H265_STAT_COEFFS);
            } else if (sh->slice_type != H265_SLICE_I) {
                if (h265_slice_rpl(h, sh) < 0) {
                    h265_slice_free(sh);
                    return -1;
                }
            }
            sh->rpl_valid = 1;
            if (h->nb_slices < H265_MAX_SLICES)
                h->slices[h->nb_slices++] = sh;
            else
                h265_slice_free(sh);
            if (!h->threaded)
                decode_one_slice(h, sh);
            if (!dependent) {
                h->cur_sh = *sh;
                h->cur_sh.frame = h->cur_frame;
            }
        } else {
            h265_slice_free(sh);
        }
    }
    return 0;
}

void *h265_dec_open(h265_param_t *param)
{
    H265Decoder *h;
    h265_param_t def;

    h265_dbg_init();
    memset(&def, 0, sizeof(def));
    if (!param)
        param = &def;
    h = (H265Decoder *)calloc(1, sizeof(H265Decoder));
    if (!h)
        return NULL;
    h->param = *param;
    if (h->param.threads < 0)
        h->param.threads = 0;
    if (h->param.threads > 16)
        h->param.threads = 16;
    h->threaded = h->param.threads > 1;
    h->recovery_poc = INT_MAX;
    h265_sei_reset_au(h);

    h265_mutex_init(&h->lock);
    h265_cond_init(&h->cond);
    h265_cond_init(&h->work_cond);
    if (h->threaded && h265_thread_init(h) < 0) {
        h->threaded = 0;
    }
    if (h265_frame_pool_init(h) < 0) {
        h265_dec_close(h);
        return NULL;
    }
    h265_dsp_init();
    return h;
}

void h265_dec_close(void *decoder)
{
    H265Decoder *h = (H265Decoder *)decoder;
    int i;

    if (!h)
        return;

    h265_mutex_lock(&h->lock);
    h265_au_close(h);
    h265_job_wait_all(h);
    h265_free_side_blob(&h->pending_t35, &h->pending_t35_size);
    h265_free_side_blob(&h->pending_unreg, &h->pending_unreg_size);
    h265_free_side_blob(&h->pending_dovi, &h->pending_dovi_size);
    h265_mutex_unlock(&h->lock);
    h265_thread_uninit(h);

    for (i = 0; i < h->aux_count; i++)
        h265_frame_unref(h->aux_frames[i]);
    h->aux_count = 0;
    if (h->aux_dec) {
        h265_dec_close(h->aux_dec);
        h->aux_dec = NULL;
    }

    for (i = 0; i < h->nb_dpb; i++)
        h265_frame_unref(h->DPB[i]);
    h->nb_dpb = 0;
    for (i = 0; i < h->out_count; i++)
        h265_frame_unref(h->out_q[i]);
    h->out_count = 0;

    h265_frame_pool_free(h);
    free(h->inbuf);
    h->inbuf = NULL;
    free(h->film_grain_db);
    h->film_grain_db = NULL;

    for (i = 0; i < H265_MAX_VPS_COUNT; i++)
        h265_vps_unref(&h->ps.vps_list[i]);
    for (i = 0; i < H265_MAX_SPS_COUNT; i++)
        h265_sps_unref(&h->ps.sps_list[i]);
    for (i = 0; i < H265_MAX_PPS_COUNT; i++)
        h265_pps_unref(&h->ps.pps_list[i]);

    h265_mutex_destroy(&h->lock);
    h265_cond_destroy(&h->cond);
    h265_cond_destroy(&h->work_cond);
    free(h);
}

static int parse_parameter_nal(H265Decoder *h, const uint8_t *nal, int len)
{
    int type, rbsp_size = 0;
    uint8_t *rbsp;
    int ret = 0;

    if (len < 2)
        return 0;
    type = h265_nal_type(nal, len);
    if (type != H265_NAL_VPS && type != H265_NAL_SPS &&
        type != H265_NAL_PPS && type != H265_NAL_PREFIX_SEI &&
        type != H265_NAL_SUFFIX_SEI)
        return 0;

    rbsp = h265_rbsp_extract(nal, len, &rbsp_size);
    if (!rbsp)
        return -1;
    if (type == H265_NAL_VPS) {
        if (h265_parse_vps(h, rbsp, rbsp_size) < 0)
            ret = -1;
    } else if (type == H265_NAL_SPS) {
        if (h265_parse_sps(h, rbsp, rbsp_size) < 0)
            ret = -1;
        else
            h->header_dirty = 1;
    } else if (type == H265_NAL_PPS) {
        if (h265_parse_pps(h, rbsp, rbsp_size) < 0)
            ret = -1;
    } else {
        h265_parse_sei(h, rbsp, rbsp_size, type);
    }
    free(rbsp);
    return ret;
}

int h265_dec_set_extradata(void *decoder, const void *data, int size)
{
    H265Decoder *h = (H265Decoder *)decoder;
    const uint8_t *p = (const uint8_t *)data;
    int ret = 0;

    if (!h || !p || size <= 0)
        return H265_DEC_ERROR;

    h265_mutex_lock(&h->lock);
    if (size >= 23 && p[0] == 1) {
        /* ISO/IEC 14496-15 hvcC configuration record. */
        int num_arrays = p[22];
        int pos = 23;
        int i, j;

        h->param.nal_length_size = (p[21] & 3) + 1;
        for (i = 0; i < num_arrays && ret == 0; i++) {
            int num_nalus;
            if (pos + 3 > size) {
                ret = -1;
                break;
            }
            pos++;                              /* array_completeness + type */
            num_nalus = (p[pos] << 8) | p[pos + 1];
            pos += 2;
            for (j = 0; j < num_nalus; j++) {
                int nal_size;
                if (pos + 2 > size) {
                    ret = -1;
                    break;
                }
                nal_size = (p[pos] << 8) | p[pos + 1];
                pos += 2;
                if (nal_size < 0 || nal_size > size - pos) {
                    ret = -1;
                    break;
                }
                if (parse_parameter_nal(h, p + pos, nal_size) < 0)
                    ret = -1;
                pos += nal_size;
            }
        }
    } else {
        /* Annex-B parameter-set blob. */
        int pos = 0;
        const uint8_t *nal;
        int len;
        while (h265_split_annexb(p, size, &pos, &nal, &len)) {
            if (parse_parameter_nal(h, nal, len) < 0)
                ret = -1;
        }
    }
    h265_mutex_unlock(&h->lock);
    return ret < 0 ? H265_DEC_ERROR : 0;
}

static void aux_cache_push(H265Decoder *h, H265Frame *f)
{
    if (h->aux_count >= (int)H265_ARRAY_ELEMS(h->aux_frames)) {
        h265_frame_unref(h->aux_frames[0]);
        memmove(&h->aux_frames[0], &h->aux_frames[1],
                (H265_ARRAY_ELEMS(h->aux_frames) - 1) * sizeof(H265Frame *));
        h->aux_count--;
    }
    h->aux_frames[h->aux_count++] = f;
}

static void h265_feed_aux(H265Decoder *h, const uint8_t *nal, int len)
{
    uint8_t *buf;
    int total, is_vcl, pos = 0;
    h265_packet_t pkt;
    h265_stream_info info;
    h265_picture_t pic;

    if (!h->aux_dec) {
        h265_param_t p;
        memset(&p, 0, sizeof(p));
        p.threads = 1;
        h->aux_dec = (H265Decoder *)h265_dec_open(&p);
        if (!h->aux_dec)
            return;
        h->aux_dec->ignore_layer_id = 1;
    }

    is_vcl = H265_IS_VCL(h265_nal_type(nal, len));
    total = 3 + len + (is_vcl ? 8 : 0);
    buf = (uint8_t *)malloc((size_t)total);
    if (!buf)
        return;
    buf[pos++] = 0;
    buf[pos++] = 0;
    buf[pos++] = 1;
    memcpy(buf + pos, nal, (size_t)len);
    pos += len;
    if (is_vcl) {
        /* AUD closes the auxiliary access unit immediately, so its frame is
         * available when the base-layer picture is output. */
        buf[pos++] = 0;
        buf[pos++] = 0;
        buf[pos++] = 1;
        buf[pos++] = 0x46;
        buf[pos++] = 0x01;
        /* A following start code makes the parser consume the AUD (and close
         * the auxiliary AU) immediately instead of retaining the last NAL. */
        buf[pos++] = 0;
        buf[pos++] = 0;
        buf[pos++] = 1;
    }
    memset(&pkt, 0, sizeof(pkt));
    pkt.data = buf;
    pkt.len = pos;
    if (h265_dec_send_packet(h->aux_dec, &pkt) < 0) {
        free(buf);
        return;
    }
    free(buf);
    if (is_vcl) {
        h265_mutex_lock(&h->aux_dec->lock);
        h265_au_close(h->aux_dec);
        h265_output_frames(h->aux_dec, 0, 0, 0);
        h265_mutex_unlock(&h->aux_dec->lock);
    }
    while (h265_dec_recv_frame(h->aux_dec, &info, &pic) ==
           H265_DEC_GOT_FRAME) {
        if (pic.dec_frame)
            aux_cache_push(h, (H265Frame *)pic.dec_frame);
    }
}

static void attach_alpha(H265Decoder *h, H265Frame *f)
{
    int i;
    for (i = 0; i < h->aux_count; i++) {
        H265Frame *af = h->aux_frames[i];
        int w, hh, bytes;
        int ox, oy;
        uint8_t *src;
        if (!af || af->poc != f->poc)
            continue;
        w = H265_MIN(f->display_w, af->plane_w[0]);
        hh = H265_MIN(f->display_h, af->plane_h[0]);
        bytes = 1 << f->pixel_shift;
        ox = af->sps ? (int)af->sps->output_window.left_offset : 0;
        oy = af->sps ? (int)af->sps->output_window.top_offset : 0;
        if (w > 0 && hh > 0) {
            int y;
            f->alpha_stride = w * bytes;
            f->alpha_width = w;
            f->alpha_height = hh;
            f->alpha_data = (uint8_t *)malloc((size_t)f->alpha_stride * hh);
            if (f->alpha_data) {
                src = af->data[0] + (size_t)oy * af->linesize[0] +
                      (size_t)ox * bytes;
                for (y = 0; y < hh; y++)
                    memcpy(f->alpha_data + (size_t)y * f->alpha_stride,
                           src + (size_t)y * af->linesize[0],
                           (size_t)f->alpha_stride);
            }
        }
        h265_frame_unref(af);
        memmove(&h->aux_frames[i], &h->aux_frames[i + 1],
                (h->aux_count - i - 1) * sizeof(H265Frame *));
        h->aux_count--;
        return;
    }
}

static int process_nal(H265Decoder *h, const uint8_t *nal, int len)
{
    int type, temporal_id, rbsp_size = 0;
    uint8_t *rbsp;
    int ret = 0;

    if (len < 2)
        return 0;
    if (h265_nal_layer_id(nal, len) > 0 && !h->ignore_layer_id) {
        /* Independent auxiliary layers are decoded with a dedicated decoder
         * and exposed as an alpha/depth plane.  Inter-layer-predicted MV-HEVC
         * views are not reconstructed. */
        H265SPS *sps = h->sps ? h->sps : h->sei_sps;
        H265VPS *vps = NULL;
        if (sps && sps->vps_id < H265_MAX_VPS_COUNT)
            vps = h->ps.vps_list[sps->vps_id];
        if (vps && vps->multilayer_auxiliary)
            h265_feed_aux(h, nal, len);
        return 0;
    }
    type = h265_nal_type(nal, len);
    temporal_id = h265_nal_temporal_id(nal, len);
    h->nal_unit_type = type;
    h->temporal_id = temporal_id;

    if (type == H265_NAL_VPS || type == H265_NAL_SPS ||
        type == H265_NAL_PPS || type == H265_NAL_PREFIX_SEI ||
        type == H265_NAL_SUFFIX_SEI) {
        /* A prefix SEI after VCL data belongs to the next access unit. */
        if (type == H265_NAL_PREFIX_SEI && h->au_open) {
            h265_au_close(h);
            h265_thread_commit_locked(h);
        }
        rbsp = h265_rbsp_extract(nal, len, &rbsp_size);
        if (!rbsp)
            return -1;
        if (type == H265_NAL_VPS) {
            if (h265_parse_vps(h, rbsp, rbsp_size) < 0)
                ret = -1;
        } else if (type == H265_NAL_SPS) {
            if (h265_parse_sps(h, rbsp, rbsp_size) < 0)
                ret = -1;
            else
                h->header_dirty = 1;
        } else if (type == H265_NAL_PPS) {
            if (h265_parse_pps(h, rbsp, rbsp_size) < 0)
                ret = -1;
        } else {
            h265_parse_sei(h, rbsp, rbsp_size, type);
        }
        free(rbsp);
        return ret;
    }

    if (type == 62) {               /* Dolby Vision RPU */
        if (len > 2) {
            if (h->cur_frame)
                h265_set_side_blob(&h->cur_frame->dovi_rpu,
                                   &h->cur_frame->dovi_rpu_size,
                                   nal + 2, len - 2);
            else
                h265_set_side_blob(&h->pending_dovi, &h->pending_dovi_size,
                                   nal + 2, len - 2);
        }
        return 0;
    }
    if (type == H265_NAL_AUD) {
        h265_au_close(h);
        return 0;
    }
    if (type == H265_NAL_EOS || type == H265_NAL_EOB) {
        h->last_eos = 1;
        h265_au_close(h);
        return 0;
    }
    if (H265_IS_VCL(type)) {
        /* A packet may contain more than one access unit.  The first bit after
         * the two-byte NAL header is first_slice_in_pic_flag. */
        if (len > 2 && (nal[2] & 0x80) && h->au_open) {
            h265_au_close(h);
            h265_thread_commit_locked(h);
        }
        if ((type == H265_NAL_RASL_N || type == H265_NAL_RASL_R) &&
            h->no_rasl_output_flag)
            return 0;
        {
            int *epb = NULL, nb_epb = 0;
            rbsp = h265_rbsp_extract_ex(nal, len, &rbsp_size, &epb, &nb_epb);
            if (!rbsp)
                return -1;
            /* handle_vcl() takes ownership of rbsp/epb (the slice or its
             * error path frees them). */
            if (handle_vcl(h, type, temporal_id, rbsp, rbsp_size, epb,
                           nb_epb) < 0)
                ret = -1;
        }
    }
    return ret;
}

static int append_input(H265Decoder *h, const uint8_t *data, int len)
{
    int need;
    if (len <= 0)
        return 0;
    if (h->inbuf_len > INT_MAX - len)
        return -1;
    need = h->inbuf_len + len;
    if (need > h->inbuf_cap) {
        int cap = h->inbuf_cap ? h->inbuf_cap : 65536;
        uint8_t *p;
        while (cap < need) {
            if (cap > INT_MAX / 2) {
                cap = need;
                break;
            }
            cap *= 2;
        }
        p = (uint8_t *)realloc(h->inbuf, (size_t)cap);
        if (!p)
            return -1;
        h->inbuf = p;
        h->inbuf_cap = cap;
    }
    memcpy(h->inbuf + h->inbuf_len, data, (size_t)len);
    h->inbuf_len = need;
    return 0;
}

/* Parse every complete NAL in the input buffer.  In non-final mode the last
 * Annex-B NAL is retained because it may continue in the next packet. */
static int process_input(H265Decoder *h, int final, int64_t pts, int64_t dts,
                         int had_pending)
{
    int pos = 0, first = 1, ret = 0;

    if (h->param.nal_length_size) {
        int nls = h->param.nal_length_size;
        while (pos + nls <= h->inbuf_len) {
            int n = 0, i;
            for (i = 0; i < nls; i++)
                n = (n << 8) | h->inbuf[pos + i];
            if (n <= 0 || n > h->inbuf_len - pos - nls)
                break;
            h->packet_pts = (first && had_pending) ? h->inbuf_pts : pts;
            h->packet_dts = (first && had_pending) ? h->inbuf_dts : dts;
            if (process_nal(h, h->inbuf + pos + nls, n) < 0)
                ret = -1;
            first = 0;
            pos += nls + n;
        }
    } else {
        for (;;) {
            int sc_len = 0, next_sc_len = 0;
            int start = h265_nal_start_code(h->inbuf, h->inbuf_len, pos, &sc_len);
            int payload, end, next;
            if (start < 0)
                break;
            payload = start + 3;
            next = h265_nal_start_code(h->inbuf, h->inbuf_len, payload,
                                       &next_sc_len);
            if (next < 0) {
                if (!final) {
                    pos = start;
                    break;
                }
                end = h->inbuf_len;
            } else {
                end = next;
            }
            while (end > payload && h->inbuf[end - 1] == 0)
                end--;
            if (end > payload) {
                h->packet_pts = (first && had_pending) ? h->inbuf_pts : pts;
                h->packet_dts = (first && had_pending) ? h->inbuf_dts : dts;
                if (process_nal(h, h->inbuf + payload, end - payload) < 0)
                    ret = -1;
                first = 0;
            }
            if (next < 0) {
                pos = h->inbuf_len;
                break;
            }
            pos = next;
        }
    }

    if (pos > 0) {
        memmove(h->inbuf, h->inbuf + pos, (size_t)(h->inbuf_len - pos));
        h->inbuf_len -= pos;
    }
    if (first) {
        /* Nothing was completed: the retained bytes keep their original PTS. */
    } else {
        h->inbuf_pts = pts;
        h->inbuf_dts = dts;
    }
    if (final) {
        h->inbuf_len = 0;
    }
    return ret;
}

int h265_dec_send_packet(void *decoder, h265_packet_t *packet)
{
    H265Decoder *h = (H265Decoder *)decoder;
    int had_pending, ret;

    if (!h || !packet || !packet->data || packet->len <= 0)
        return H265_DEC_ERROR;

    h265_mutex_lock(&h->lock);
    h265_thread_commit_locked(h);
    if (h->inbuf_len == 0) {
        h->inbuf_pts = packet->pts;
        h->inbuf_dts = packet->dts;
    }
    had_pending = h->inbuf_len > 0;
    if (append_input(h, packet->data, packet->len) < 0) {
        h265_mutex_unlock(&h->lock);
        return H265_DEC_ERROR;
    }
    ret = process_input(h, 0, packet->pts, packet->dts, had_pending);
    /* Do not close the access unit here: a caller may split one picture across
     * several packets (one NAL per packet).  The next first-slice NAL, an AUD,
     * EOS/EOB or h265_dec_flush() closes it. */
    h265_thread_commit_locked(h);
    h265_mutex_unlock(&h->lock);
    return ret < 0 ? H265_DEC_ERROR : 0;
}

static int fill_picture(H265Decoder *h, H265Frame *f, h265_picture_t *pic)
{
    H265SPS *sps = f->sps;
    int i;
    int ow_left = sps ? (int)sps->output_window.left_offset : 0;
    int ow_top = sps ? (int)sps->output_window.top_offset : 0;

    memset(pic, 0, sizeof(*pic));
    pic->num_planes = f->num_planes;
    pic->bytes_per_sample = f->pixel_shift ? 2 : 1;
    pic->bit_depth = f->bit_depth;
    pic->chroma_format = f->chroma_format;
    for (i = 0; i < f->num_planes; i++) {
        int sh = sps ? sps->hshift[i] : 0;
        int sv = sps ? sps->vshift[i] : 0;
        pic->planes[i] = f->data[i] + (ow_top >> sv) * f->linesize[i] +
                         ((ow_left >> sh) << f->pixel_shift);
        pic->widths[i] = f->display_w >> sh;
        pic->lines[i] = f->display_h >> sv;
        pic->strides[i] = f->linesize[i];
    }
    pic->pic_order_count = f->poc;
    pic->type = 3 - f->slice_type;
    pic->qp = f->qp;
    pic->pts = f->pts;
    pic->dts = f->dts;
    pic->b_decode_error = !!(f->flags & H265_FRAME_FLAG_CORRUPT);
    pic->temporal_id = f->temporal_id;
    pic->nal_unit_type = f->nal_unit_type;
    pic->is_irap = f->is_irap;

    pic->video_full_range = sps ? sps->video_full_range : 0;
    pic->colour_primaries = sps ? sps->colour_primaries : 2;
    pic->transfer_characteristics = sps ? sps->transfer_characteristics : 2;
    pic->matrix_coeffs = sps ? sps->matrix_coeffs : 2;
    pic->chroma_sample_location = sps ? sps->chroma_loc_type_top : -1;
    pic->field_seq_flag = sps ? sps->field_seq_flag : 0;
    pic->colour_plane_id = f->colour_plane_id;
    pic->layer_id = f->layer_id;
    pic->pic_struct = f->sei.pic_struct;
    {
        int ps = f->sei.pic_struct;
        int field = (ps == 1 || ps == 2 || ps == 9 || ps == 10 ||
                     ps == 11 || ps == 12);
        pic->interlaced = field || (sps && sps->field_seq_flag);
        pic->top_field_first = (ps == 1 || ps == 9 || ps == 11);
        pic->repeat_pict = ps == 7 ? 1 : ps == 8 ? 2 : 0;
    }

    pic->recovery_frame_cnt = f->sei.recovery_present ?
                              f->sei.recovery_poc_cnt : -1;
    pic->broken_recovery = f->sei.recovery_broken_link;
    pic->timecode_valid = f->sei.timecode_present &&
                          f->sei.timecode_num_clock_ts > 0 &&
                          f->sei.timecode_clock_timestamp_flag[0];
    if (pic->timecode_valid) {
        pic->timecode_hours = f->sei.timecode_hours[0];
        pic->timecode_minutes = f->sei.timecode_minutes[0];
        pic->timecode_seconds = f->sei.timecode_seconds[0];
        pic->timecode_frames = f->sei.timecode_n_frames[0];
        pic->timecode_drop = f->sei.timecode_cnt_dropped[0];
    }
    pic->mastering_display_present = f->sei.mastering_present;
    if (f->sei.mastering_present) {
        memcpy(pic->display_primaries, f->sei.display_primaries,
               sizeof(pic->display_primaries));
        memcpy(pic->white_point, f->sei.white_point, sizeof(pic->white_point));
        pic->max_luminance = f->sei.max_luminance;
        pic->min_luminance = f->sei.min_luminance;
    }
    pic->content_light_present = f->sei.content_light_present;
    if (f->sei.content_light_present) {
        pic->max_content_light_level = f->sei.max_content_light_level;
        pic->max_pic_average_light_level = f->sei.max_pic_average_light_level;
    }
    pic->alternative_transfer_present = f->sei.alternative_transfer_present;
    pic->preferred_transfer_characteristics =
        f->sei.preferred_transfer_characteristics;
    pic->display_orientation_present = f->sei.display_orientation_present;
    if (f->sei.display_orientation_present) {
        pic->display_orientation_hflip = f->sei.display_orientation_hflip;
        pic->display_orientation_vflip = f->sei.display_orientation_vflip;
        pic->display_orientation_rotation = f->sei.display_orientation_rotation;
    }
    pic->ambient_viewing_environment_present = f->sei.ambient_present;
    if (f->sei.ambient_present) {
        pic->ambient_illuminance = f->sei.ambient_illuminance;
        pic->ambient_light_x = f->sei.ambient_light_x;
        pic->ambient_light_y = f->sei.ambient_light_y;
    }
    pic->frame_packing_present = f->sei.frame_packing_present;
    if (f->sei.frame_packing_present) {
        pic->frame_packing_type = f->sei.frame_packing_type;
        pic->frame_packing_content = f->sei.frame_packing_content;
        pic->frame_packing_current_frame_is_frame0 =
            f->sei.frame_packing_current_frame_is_frame0;
    }

    pic->itu_t_t35 = f->t35_data;
    pic->itu_t_t35_size = f->t35_size;
    pic->user_data_unregistered = f->unreg_data;
    pic->user_data_unregistered_size = f->unreg_size;
    pic->dovi_rpu = f->dovi_rpu;
    pic->dovi_rpu_size = f->dovi_rpu_size;

    if (!f->alpha_data)
        attach_alpha(h, f);
    pic->alpha_plane = f->alpha_data;
    pic->alpha_stride = f->alpha_stride;
    pic->alpha_width = f->alpha_width;
    pic->alpha_height = f->alpha_height;

    pic->dec_frame = f;
    return 0;
}

int h265_dec_recv_frame(void *decoder, h265_stream_info *info,
                        h265_picture_t *out_frame)
{
    H265Decoder *h = (H265Decoder *)decoder;
    H265Frame *f;
    int ret;

    if (!h || !out_frame)
        return H265_DEC_ERROR;
    h265_mutex_lock(&h->lock);
    h265_thread_commit_locked(h);

    if (h->out_count <= 0) {
        if (h->draining)
            ret = H265_DEC_END;
        else
            ret = H265_DEC_DEFAULT;
        h265_mutex_unlock(&h->lock);
        return ret;
    }

    f = h->out_q[0];
    memmove(&h->out_q[0], &h->out_q[1], (h->out_count - 1) * sizeof(H265Frame *));
    h->out_count--;

    if (h265_film_grain_supported(&f->sei.film_grain, f->sps)) {
        H265Frame *g = h265_frame_alloc_display(h, f);
        if (g) {
            if (h265_film_grain_apply(h, g, f, &f->sei.film_grain) == 0) {
                h265_frame_unref(f);
                f = g;
            } else {
                h265_frame_unref(g);
            }
        }
    }

    fill_picture(h, f, out_frame);
    if (h->sps) {
        update_info(h, h->sps);
        if (info)
            *info = h->info;
        h->header_dirty = 0;
    }
    h265_mutex_unlock(&h->lock);
    /* f has one reference belonging to the caller */
    return H265_DEC_GOT_FRAME;
}

int h265_dec_flush(void *decoder, h265_stream_info *info,
                   h265_picture_t *out_frame)
{
    H265Decoder *h = (H265Decoder *)decoder;

    if (!h || !out_frame)
        return H265_DEC_ERROR;
    h265_mutex_lock(&h->lock);
    if (h->inbuf_len > 0)
        process_input(h, 1, h->inbuf_pts, h->inbuf_dts, 1);
    h265_au_close(h);
    h265_job_wait_all(h);
    h->header_dirty = 0;
    h->draining = 1;
    h265_output_frames(h, 0, 0, 0);
    h265_mutex_unlock(&h->lock);
    return h265_dec_recv_frame(decoder, info, out_frame);
}

void h265_dec_frame_unref(void *decoder, h265_picture_t *out_frame)
{
    (void)decoder;
    if (out_frame && out_frame->dec_frame) {
        h265_frame_unref((H265Frame *)out_frame->dec_frame);
        out_frame->dec_frame = NULL;
    }
}
