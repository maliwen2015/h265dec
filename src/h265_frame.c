/*
 * HEVC frame pool, pixel buffers and per-frame metadata.
 */
#include "h265_internal.h"


static H265Frame *g_pool_base;

void *h265_malloc_aligned(size_t size, int align)
{
    void *p;
#ifdef _WIN32
    p = _aligned_malloc(size, align);
#else
    if (posix_memalign(&p, align, size))
        p = NULL;
#endif
    return p;
}

void h265_free_aligned(void *p)
{
    if (!p)
        return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void h265_frame_clear_side_data(H265Frame *f)
{
    free(f->t35_data);
    free(f->unreg_data);
    free(f->dovi_rpu);
    free(f->alpha_data);
    f->t35_data = NULL;
    f->unreg_data = NULL;
    f->dovi_rpu = NULL;
    f->alpha_data = NULL;
    f->t35_size = 0;
    f->unreg_size = 0;
    f->dovi_rpu_size = 0;
    f->alpha_stride = 0;
    f->alpha_width = 0;
    f->alpha_height = 0;
}

void h265_frame_clear_aux(H265Frame *f)
{
    int i;

    h265_free_aligned(f->data[0]);
    h265_free_aligned(f->data[1]);
    h265_free_aligned(f->data[2]);
    f->data[0] = f->data[1] = f->data[2] = NULL;
    f->linesize[0] = f->linesize[1] = f->linesize[2] = 0;
    f->alloc_w = f->alloc_h = 0;
    f->alloc_chroma = f->alloc_bit_depth = 0;
    f->alloc_log2_min_cb = f->alloc_log2_ctb = -1;
    f->alloc_sao = 0;

    free(f->skip_flag);
    free(f->tab_ct_depth);
    free(f->cbf_luma);
    free(f->tab_ipm);
    free(f->is_pcm);
    free(f->filter_slice_edges);
    free(f->tab_slice_address);
    free(f->qp_y_tab);
    free(f->horizontal_bs);
    free(f->vertical_bs);
    free(f->sao);
    free(f->deblock);
    free(f->tab_mvf);
    free(f->rpl_tab);
    f->skip_flag = NULL;
    f->tab_ct_depth = NULL;
    f->cbf_luma = NULL;
    f->tab_ipm = NULL;
    f->is_pcm = NULL;
    f->filter_slice_edges = NULL;
    f->tab_slice_address = NULL;
    f->qp_y_tab = NULL;
    f->horizontal_bs = NULL;
    f->vertical_bs = NULL;
    f->sao = NULL;
    f->deblock = NULL;
    f->tab_mvf = NULL;
    f->rpl_tab = NULL;

    for (i = 0; i < 3; i++) {
        free(f->sao_pixel_buffer_h[i]);
        free(f->sao_pixel_buffer_v[i]);
        f->sao_pixel_buffer_h[i] = NULL;
        f->sao_pixel_buffer_v[i] = NULL;
    }
    h265_frame_clear_side_data(f);
}

static int alloc_plane(uint8_t **out, int *linesize, int w, int h,
                       int pixel_shift)
{
    int stride = H265_ALIGN(w << pixel_shift, 64) + 64;
    size_t size = (size_t)stride * (h + 8);
    uint8_t *p = (uint8_t *)h265_malloc_aligned(size, 32);
    if (!p)
        return -1;
    memset(p, 0, size);
    *out = p;
    *linesize = stride;
    return 0;
}

int h265_frame_alloc_aux(H265Frame *f, H265SPS *sps)
{
    int pic_size_in_ctb = (sps->min_cb_width + 1) * (sps->min_cb_height + 1);
    int ctb_count = sps->ctb_width * sps->ctb_height;
    int min_pu_size = sps->min_pu_width * sps->min_pu_height;
    int bs_width = (sps->width >> 2) + 1;
    int bs_height = (sps->height >> 2) + 1;
    int i;
    int reuse;

    reuse = f->data[0] != NULL &&
            f->alloc_w == sps->width && f->alloc_h == sps->height &&
            f->alloc_chroma == sps->chroma_format_idc &&
            f->alloc_bit_depth == sps->bit_depth &&
            f->alloc_log2_min_cb == sps->log2_min_cb_size &&
            f->alloc_log2_ctb == sps->log2_ctb_size &&
            f->alloc_sao == !!sps->sao_enabled;

    if (reuse) {
        h265_frame_clear_side_data(f);
    } else {
        h265_frame_clear_aux(f);
    }

    f->width = sps->width;
    f->height = sps->height;
    f->chroma_format = sps->chroma_format_idc;
    f->bit_depth = sps->bit_depth;
    f->pixel_shift = sps->pixel_shift;
    f->num_planes = sps->chroma_format_idc ? 3 : 1;

    if (!reuse) {
        if (alloc_plane(&f->data[0], &f->linesize[0], sps->width, sps->height,
                        sps->pixel_shift) < 0)
            goto fail;
        if (sps->chroma_format_idc) {
            if (alloc_plane(&f->data[1], &f->linesize[1],
                            sps->width >> sps->hshift[1],
                            sps->height >> sps->vshift[1], sps->pixel_shift) < 0)
                goto fail;
            if (alloc_plane(&f->data[2], &f->linesize[2],
                            sps->width >> sps->hshift[2],
                            sps->height >> sps->vshift[2], sps->pixel_shift) < 0)
                goto fail;
        }

        f->skip_flag = (uint8_t *)malloc(sps->min_cb_width * sps->min_cb_height);
        f->tab_ct_depth = (uint8_t *)malloc(sps->min_cb_width * sps->min_cb_height);
        f->cbf_luma = (uint8_t *)malloc(sps->min_tb_width * sps->min_tb_height);
        f->tab_ipm = (uint8_t *)malloc(min_pu_size);
        f->is_pcm = (uint8_t *)malloc(min_pu_size);
        f->filter_slice_edges = (uint8_t *)malloc(ctb_count);
        f->tab_slice_address = (int32_t *)malloc(pic_size_in_ctb * sizeof(int32_t));
        f->qp_y_tab = (int8_t *)malloc(pic_size_in_ctb * sizeof(int8_t));
        f->horizontal_bs = (uint8_t *)malloc(bs_width * bs_height);
        f->vertical_bs = (uint8_t *)malloc(bs_width * bs_height);
        f->sao = (H265SAOParams *)malloc(ctb_count * sizeof(H265SAOParams));
        f->deblock = (H265DBParams *)malloc(ctb_count * sizeof(H265DBParams));
        f->tab_mvf = (H265MvField *)malloc(min_pu_size * sizeof(H265MvField));
        f->rpl_tab = (H265RefListTab *)malloc(ctb_count * sizeof(H265RefListTab));

        if (!f->skip_flag || !f->tab_ct_depth || !f->cbf_luma || !f->tab_ipm ||
            !f->is_pcm || !f->filter_slice_edges || !f->tab_slice_address ||
            !f->qp_y_tab || !f->horizontal_bs || !f->vertical_bs || !f->sao ||
            !f->deblock || !f->tab_mvf || !f->rpl_tab)
            goto fail;

        if (sps->sao_enabled) {
            int c_count = sps->chroma_format_idc ? 3 : 1;
            for (i = 0; i < c_count; i++) {
                int w = sps->width >> sps->hshift[i];
                int h = sps->height >> sps->vshift[i];
                size_t hs = (size_t)w * 2 * sps->ctb_height << sps->pixel_shift;
                size_t vs = (size_t)h * 2 * sps->ctb_width << sps->pixel_shift;
                f->sao_pixel_buffer_h[i] = (uint8_t *)malloc(hs);
                f->sao_pixel_buffer_v[i] = (uint8_t *)malloc(vs);
                if (!f->sao_pixel_buffer_h[i] || !f->sao_pixel_buffer_v[i])
                    goto fail;
            }
        }
    }

    f->plane_w[0] = sps->width;
    f->plane_h[0] = sps->height;
    f->plane_w[1] = sps->width >> sps->hshift[1];
    f->plane_h[1] = sps->height >> sps->vshift[1];
    f->plane_w[2] = sps->width >> sps->hshift[2];
    f->plane_h[2] = sps->height >> sps->vshift[2];

    f->display_w = sps->width - (int)sps->output_window.left_offset -
                   (int)sps->output_window.right_offset;
    f->display_h = sps->height - (int)sps->output_window.top_offset -
                   (int)sps->output_window.bottom_offset;
    h265_sps_unref(&f->sps);
    f->sps = h265_sps_ref(sps);
    f->alloc_w = sps->width;
    f->alloc_h = sps->height;
    f->alloc_chroma = sps->chroma_format_idc;
    f->alloc_bit_depth = sps->bit_depth;
    f->alloc_log2_min_cb = sps->log2_min_cb_size;
    f->alloc_log2_ctb = sps->log2_ctb_size;
    f->alloc_sao = !!sps->sao_enabled;
    return 0;

fail:
    h265_frame_clear_aux(f);
    return -1;
}

H265Frame *h265_frame_alloc_display(H265Decoder *h, const H265Frame *src)
{
    H265Frame *f = h265_frame_alloc(h);
    int i;

    if (!f)
        return NULL;
    h265_frame_clear_aux(f);
    f->width = src->width;
    f->height = src->height;
    f->display_w = src->display_w;
    f->display_h = src->display_h;
    f->chroma_format = src->chroma_format;
    f->bit_depth = src->bit_depth;
    f->pixel_shift = src->pixel_shift;
    f->num_planes = src->num_planes;
    f->sps = h265_sps_ref(src->sps);
    f->pps = h265_pps_ref(src->pps);
    f->poc = src->poc;
    f->temporal_id = src->temporal_id;
    f->nal_unit_type = src->nal_unit_type;
    f->slice_type = src->slice_type;
    f->is_irap = src->is_irap;
    f->colour_plane_id = src->colour_plane_id;
    f->layer_id = src->layer_id;
    f->qp = src->qp;
    f->pts = src->pts;
    f->dts = src->dts;
    f->sei = src->sei;
    f->flags = src->flags;
    h265_set_side_blob(&f->t35_data, &f->t35_size, src->t35_data,
                       src->t35_size);
    h265_set_side_blob(&f->unreg_data, &f->unreg_size, src->unreg_data,
                       src->unreg_size);
    h265_set_side_blob(&f->dovi_rpu, &f->dovi_rpu_size, src->dovi_rpu,
                       src->dovi_rpu_size);
    if (src->alpha_data && src->alpha_stride > 0 && src->alpha_height > 0) {
        size_t bytes = (size_t)src->alpha_stride * src->alpha_height;
        f->alpha_data = (uint8_t *)malloc(bytes);
        if (f->alpha_data) {
            memcpy(f->alpha_data, src->alpha_data, bytes);
            f->alpha_stride = src->alpha_stride;
            f->alpha_width = src->alpha_width;
            f->alpha_height = src->alpha_height;
        }
    }
    for (i = 0; i < src->num_planes; i++) {
        if (alloc_plane(&f->data[i], &f->linesize[i], src->plane_w[i],
                        src->plane_h[i], src->pixel_shift) < 0) {
            h265_frame_clear_aux(f);
            h265_sps_unref(&f->sps);
            h265_pps_unref(&f->pps);
            f->in_use = 0;
            f->refcount = 0;
            return NULL;
        }
        f->plane_w[i] = src->plane_w[i];
        f->plane_h[i] = src->plane_h[i];
        memset(f->data[i], 0, (size_t)f->linesize[i] * src->plane_h[i]);
    }
    return f;
}

void h265_frame_reset_meta(H265Frame *f)
{
    H265SPS *sps = f->sps;
    H265PPS *pps = f->pps;
    int pic_size_in_ctb, ctb_count, min_pu_size, bs_w, bs_h, i;

    if (!sps)
        return;

    pic_size_in_ctb = (sps->min_cb_width + 1) * (sps->min_cb_height + 1);
    ctb_count = sps->ctb_width * sps->ctb_height;
    min_pu_size = sps->min_pu_width * sps->min_pu_height;
    bs_w = (sps->width >> 2) + 1;
    bs_h = (sps->height >> 2) + 1;

    memset(f->cbf_luma, 0, sps->min_tb_width * sps->min_tb_height);
    memset(f->is_pcm, 0, min_pu_size);
    memset(f->filter_slice_edges, 0, ctb_count);
    memset(f->tab_slice_address, 0xff, pic_size_in_ctb * sizeof(int32_t));
    memset(f->horizontal_bs, 0, bs_w * bs_h);
    memset(f->vertical_bs, 0, bs_w * bs_h);

    for (i = 0; i < 3; i++) {
        if (f->sao_pixel_buffer_h[i]) {
            int w = sps->width >> sps->hshift[i];
            int h = sps->height >> sps->vshift[i];
            memset(f->sao_pixel_buffer_h[i], 0,
                   (size_t)w * 2 * sps->ctb_height << sps->pixel_shift);
            memset(f->sao_pixel_buffer_v[i], 0,
                   (size_t)h * 2 * sps->ctb_width << sps->pixel_shift);
        }
    }
    (void)pps;
}

int h265_frame_pool_init(H265Decoder *h)
{
    int n = 288, i;
    h->pool.frames = (H265Frame *)calloc(n, sizeof(H265Frame));
    g_pool_base = h->pool.frames;
    if (!h->pool.frames)
        return -1;
    h->pool.count = n;
    h->pool.capacity = n;
    for (i = 0; i < n; i++) {
        H265Frame *f = &h->pool.frames[i];
        h265_mutex_init(&f->mutex);
        h265_cond_init(&f->cond);
        f->owner = h;
        f->in_use = 0;
        f->refcount = 0;
        f->seq = i;
    }
    return 0;
}

static int frame_referenced(H265Decoder *h, H265Frame *f)
{
    int ji, jk, si, li, ri;
    if (!h->jobs)
        return 0;
    for (ji = 0; ji < h->nb_jobs; ji++) {
        H265Job *jb = &h->jobs[ji];
        if (jb->state == H265_JOB_IDLE)
            continue;
        if (jb->frame == f)
            return 1;
        for (jk = 0; jk < jb->nb_held; jk++)
            if (jb->held[jk] == f)
                return 1;
        for (si = 0; si < jb->nb_slices; si++) {
            H265Slice *hs = jb->slices[si];
            if (!hs)
                continue;
            if (hs->frame == f)
                return 1;
            for (li = 0; li < 2; li++)
                for (ri = 0; ri < hs->ref_list[li].nb_refs; ri++)
                    if (hs->ref_frame[li][ri] == f)
                        return 1;
        }
    }
    return 0;
}

void h265_frame_pool_free(H265Decoder *h)
{
    int i;
    if (!h->pool.frames)
        return;
    for (i = 0; i < h->pool.capacity; i++) {
        H265Frame *f = &h->pool.frames[i];
        h265_frame_clear_aux(f);
        h265_sps_unref(&f->sps);
        h265_pps_unref(&f->pps);
        h265_mutex_destroy(&f->mutex);
        h265_cond_destroy(&f->cond);
    }
    free(h->pool.frames);
    h->pool.frames = NULL;
    h->pool.count = 0;
}

H265Frame *h265_frame_alloc(H265Decoder *h)
{
    int i, seq = (int)h->decode_seq;
    H265Frame *best = NULL;

    for (i = 0; i < h->pool.capacity; i++) {
        H265Frame *f = &h->pool.frames[i];
        if (f->in_use)
            continue;
        if (frame_referenced(h, f))
            continue;
        if (f->pending_free) {
            h265_sps_unref(&f->sps);
            h265_pps_unref(&f->pps);
            f->pending_free = 0;
        }
        best = f;
        break;
    }
    if (!best)
        return NULL;

    best->in_use = 1;
    best->refcount = 1;
    best->pending_free = 0;
    best->flags = 0;
    best->poc = 0;
    best->temporal_id = 0;
    best->nal_unit_type = 0;
    best->slice_type = H265_SLICE_I;
    best->is_irap = 0;
    best->qp = 26;
    best->decode_done = 0;
    best->rows_done = 0;
    best->pending_slice_jobs = 0;
    best->slice_error = 0;
    best->seq = seq;
    return best;
}

void h265_frame_addref(H265Frame *f)
{
    h265_atomic_inc(&f->refcount);
}

void h265_frame_unref(H265Frame *f)
{
    if (!f)
        return;
    if (h265_atomic_dec(&f->refcount) == 0) {
        f->flags = 0;
        f->in_use = 0;
        f->decode_done = 0;
        f->pending_free = 1;
    }
}

void h265_frame_wait_ref(H265Frame *f)
{
    if (!f || f->decode_done)
        return;
    h265_mutex_lock(&f->mutex);
    while (!f->decode_done)
        h265_cond_wait(&f->cond, &f->mutex);
    h265_mutex_unlock(&f->mutex);
}

void h265_frame_mark_done(H265Frame *f)
{
    h265_mutex_lock(&f->mutex);
    f->decode_done = 1;
    h265_cond_broadcast(&f->cond);
    h265_mutex_unlock(&f->mutex);
}

void h265_frame_report_rows(H265Frame *f, int rows)
{
    h265_mutex_lock(&f->mutex);
    if (rows > f->rows_done)
        f->rows_done = rows;
    h265_cond_broadcast(&f->cond);
    h265_mutex_unlock(&f->mutex);
}

void h265_frame_wait_rows(H265Frame *f, int rows)
{
    if (!f || f->decode_done)
        return;
    if (rows > f->height)
        rows = f->height;
    if (rows <= 0)
        return;
    h265_mutex_lock(&f->mutex);
    while (!f->decode_done && f->rows_done < rows) {
        h265_cond_wait(&f->cond, &f->mutex);
    }
    h265_mutex_unlock(&f->mutex);
}
