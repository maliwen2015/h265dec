/*
 * Frame-level worker pool.
 */
#include "h265_internal.h"


#ifdef _WIN32
#include <process.h>
typedef unsigned (__stdcall *h265_thread_fn)(void *);
#endif

struct H265ThreadPool {
    int reserved;
};

static void run_job(H265Decoder *h, H265Job *job)
{
    int i;
    H265Slice *last = NULL;
    int err = 0;

    if (job->wpp) {
        H265Slice *sh = job->slices[0];
        sh->row_progress = 0;
        if (h265_decode_wpp_row(sh, job->wpp, job->wpp_row) < 0) {
            sh->error = 1;
            err = 1;
        }
        last = sh;
    } else {
        for (i = 0; i < job->nb_slices; i++) {
            H265Slice *sh = job->slices[i];
            if (!sh)
                continue;
            sh->row_progress = job->row_mode;
            if (h265_decode_slice(sh) < 0) {
                sh->error = 1;
                err = 1;
            }
            last = sh;
        }
    }

    if (job->slice_parallel && job->frame) {
        int left;
        h265_mutex_lock(&job->frame->mutex);
        if (err)
            job->frame->slice_error = 1;
        left = --job->frame->pending_slice_jobs;
        h265_mutex_unlock(&job->frame->mutex);
        if (left == 0) {
            if (job->frame->slice_error)
                job->frame->flags |= H265_FRAME_FLAG_CORRUPT;
            if (last)
                h265_frame_filter(h, last);
            else
                h265_frame_mark_done(job->frame);
            job->frame_done = 1;
        }
    } else {
        if (err && job->frame)
            job->frame->flags |= H265_FRAME_FLAG_CORRUPT;
        if (job->frame) {
            if (last)
                h265_frame_filter(h, last);
            else
                h265_frame_mark_done(job->frame);
            job->frame_done = 1;
        }
    }

    {
        int k;
        for (k = 0; k < job->nb_held; k++) {
            h265_frame_unref(job->held[k]);
        }
        job->nb_held = 0;
    }
}

#ifdef _WIN32
static unsigned __stdcall worker_main(void *arg)
#else
static void *worker_main(void *arg)
#endif
{
    H265Decoder *h = (H265Decoder *)arg;

    h265_mutex_lock(&h->lock);
    while (!h->shutting_down) {
        H265Job *job = NULL;
        int i;
        int64_t best_seq = INT64_MAX;
        for (i = 0; i < h->nb_jobs; i++) {
            if (h->jobs[i].state == H265_JOB_QUEUED &&
                h->jobs[i].seq < best_seq) {
                job = &h->jobs[i];
                best_seq = h->jobs[i].seq;
            }
        }
        if (!job) {
            h265_cond_wait(&h->work_cond, &h->lock);
            continue;
        }
        job->state = H265_JOB_RUN;
        h265_mutex_unlock(&h->lock);
        run_job(h, job);
        h265_mutex_lock(&h->lock);
        job->state = H265_JOB_DONE;
        h265_cond_broadcast(&h->cond);
    }
    h265_mutex_unlock(&h->lock);
    return 0;
}

int h265_thread_init(H265Decoder *h)
{
    int i;

    if (!h->threaded)
        return 0;
    if (h->param.threads < 2)
        return 0;
    h->nb_threads = h->param.threads;
    if (h->nb_threads > 32)
        h->nb_threads = 32;
    h->nb_jobs = h->nb_threads + 2;
    h->jobs = (H265Job *)calloc(h->nb_jobs, sizeof(H265Job));
    if (!h->jobs)
        return -1;
    for (i = 0; i < h->nb_jobs; i++) {
        h->jobs[i].state = H265_JOB_IDLE;
        h->jobs[i].seq = -1;
    }
    h->submit_seq = 0;
    h->commit_seq = 0;
    h->jobs_in_flight = 0;
    h->threads = (H265ThreadPool *)calloc(1, sizeof(H265ThreadPool));
    if (!h->threads)
        return -1;
    h->nb_threads_running = 0;
    for (i = 0; i < h->nb_threads; i++) {
#ifdef _WIN32
        h->thread_handles[i] = (h265_thread)0;
#else
        h->thread_handles[i] = 0;
#endif
    }
    for (i = 0; i < h->nb_threads; i++) {
#ifdef _WIN32
        uintptr_t th = _beginthreadex(NULL, 16 * 1024 * 1024, worker_main, h, 0, NULL);
        if (!th)
            break;
        h->thread_handles[i] = (h265_thread)th;
#else
        if (pthread_create(&h->thread_handles[i], NULL, worker_main, h) != 0)
            break;
#endif
        h->nb_threads_running++;
    }
    return 0;
}

void h265_thread_uninit(H265Decoder *h)
{
    int i;

    if (!h->threads)
        return;
    h265_mutex_lock(&h->lock);
    h->shutting_down = 1;
    h265_cond_broadcast(&h->cond);
    h265_cond_broadcast(&h->work_cond);
    h265_mutex_unlock(&h->lock);
    for (i = 0; i < h->nb_threads_running; i++) {
#ifdef _WIN32
        WaitForSingleObject((HANDLE)h->thread_handles[i], INFINITE);
        CloseHandle((HANDLE)h->thread_handles[i]);
#else
        pthread_join(h->thread_handles[i], NULL);
#endif
    }
    free(h->threads);
    h->threads = NULL;
    free(h->jobs);
    h->jobs = NULL;
    h->nb_jobs = 0;
    h->nb_threads_running = 0;
}

void h265_job_release(H265Job *job)
{
    int i;
    if (job->frame) {
        h265_frame_unref(job->frame);
    }
    for (i = 0; i < job->nb_held; i++) {
        h265_frame_unref(job->held[i]);
    }
    job->nb_held = 0;
    if (job->slices) {
        if (!job->wpp)
            for (i = 0; i < job->nb_slices; i++)
                h265_slice_free(job->slices[i]);
        free(job->slices);
    }
    job->slices = NULL;
    if (job->wpp) {
        h265_wpp_unref(job->wpp);
        job->wpp = NULL;
    }
    job->frame = NULL;
    job->nb_slices = 0;
    job->frame_done = 0;
    job->slice_parallel = 0;
    job->state = H265_JOB_IDLE;
}

void h265_thread_commit_locked(H265Decoder *h)
{
    if (!h->threaded || !h->jobs)
        return;
    for (;;) {
        int i, found = 0;
        H265SPS *sps = NULL;
        for (i = 0; i < h->nb_jobs; i++) {
            H265Job *job = &h->jobs[i];
            if (job->state == H265_JOB_DONE && job->seq == h->commit_seq) {
                int frame_done = job->frame_done;
                if (job->frame)
                    sps = job->frame->sps;
                h265_job_release(job);
                h->jobs_in_flight--;
                h->commit_seq++;
                found = 1;
                if (frame_done && sps) {
                    h265_output_frames(h,
                        sps->temporal_layer[sps->max_sub_layers - 1].num_reorder_pics,
                        sps->temporal_layer[sps->max_sub_layers - 1].max_dec_pic_buffering,
                        0);
                }
                break;
            }
        }
        if (!found)
            break;
    }
    h265_cond_broadcast(&h->cond);
}

int h265_job_submit(H265Decoder *h, H265Job *job)
{
    H265Job *slot = NULL;
    int i;

    if (!h->threaded)
        return -1;

    for (;;) {
        h265_thread_commit_locked(h);
        for (i = 0; i < h->nb_jobs; i++) {
            if (h->jobs[i].state == H265_JOB_IDLE) {
                slot = &h->jobs[i];
                break;
            }
        }
        if (slot)
            break;
        h265_cond_wait(&h->cond, &h->lock);
    }

    slot->state = H265_JOB_QUEUED;
    slot->result = 0;
    slot->seq = h->submit_seq++;
    slot->frame = job->frame;
    slot->nb_slices = job->nb_slices;
    slot->row_mode = job->row_mode;
    slot->slice_parallel = job->slice_parallel;
    slot->frame_done = 0;
    slot->wpp = job->wpp;
    slot->wpp_row = job->wpp_row;
    slot->slices = job->slices;
    slot->nb_held = job->nb_held;
    memcpy(slot->held, job->held, sizeof(slot->held));
    slot->h = h;
    h->jobs_in_flight++;
    h265_cond_signal(&h->work_cond);
    return 0;
}

void h265_job_wait_all(H265Decoder *h)
{
    if (!h->threaded || !h->jobs)
        return;
    while (h->jobs_in_flight > 0) {
        h265_thread_commit_locked(h);
        if (h->jobs_in_flight > 0)
            h265_cond_wait(&h->cond, &h->lock);
    }
}
