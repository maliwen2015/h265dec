/*
 * HEVC deblocking filter and boundary-strength derivation.
 *
 * Derived from FFmpeg's hevc/filter.c, h26x/h2656_deblock_template.c and
 * hevc/dsp_template.c (LGPL-2.1+).
 */
#include "h265_internal.h"

#ifdef H265_X86
#include <immintrin.h>
#define DEBLOCK_AVX2 __attribute__((target("avx2")))
#endif
#ifdef H265_NEON
#include <arm_neon.h>
#endif

int g_lf_x, g_lf_y;

static const uint8_t tctable[54] = {
    0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1,  1,  2,  2,  2,  2,  3,  3,  3,  3, 4, 4, 4,
    5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24
};

static const uint8_t betatable[52] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  7,  8,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
    38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64
};

static int chroma_tc(H265PPS *pps, H265SPS *sps, int qp_y, int c_idx,
                     int tc_offset)
{
    static const int qp_c[] = { 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37 };
    int qp, qp_i, offset, idxt;

    if (c_idx == 1)
        offset = pps->cb_qp_offset;
    else
        offset = pps->cr_qp_offset;

    qp_i = H265_CLIP(qp_y + offset, 0, 57);
    if (sps->chroma_format_idc == 1) {
        if (qp_i < 30)
            qp = qp_i;
        else if (qp_i > 43)
            qp = qp_i - 6;
        else
            qp = qp_c[qp_i - 30];
    } else {
        qp = H265_CLIP(qp_i, 0, 51);
    }
    idxt = H265_CLIP(qp + H265_DEFAULT_INTRA_TC_OFFSET + tc_offset, 0, 53);
    return tctable[idxt];
}

static int get_qPy_pred(H265Slice *sh, int xBase, int yBase, int log2_cb_size)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    int ctb_size_mask = (1 << sps->log2_ctb_size) - 1;
    int MinCuQpDeltaSizeMask = (1 << (sps->log2_ctb_size -
                                     pps->diff_cu_qp_delta_depth)) - 1;
    int xQgBase = xBase - (xBase & MinCuQpDeltaSizeMask);
    int yQgBase = yBase - (yBase & MinCuQpDeltaSizeMask);
    int min_cb_width = sps->min_cb_width;
    int x_cb = xQgBase >> sps->log2_min_cb_size;
    int y_cb = yQgBase >> sps->log2_min_cb_size;
    int availableA = (xBase & ctb_size_mask) && (xQgBase & ctb_size_mask);
    int availableB = (yBase & ctb_size_mask) && (yQgBase & ctb_size_mask);
    int qPy_pred, qPy_a, qPy_b;

    if (sh->first_qp_group || (!xQgBase && !yQgBase)) {
        sh->first_qp_group = !sh->tu.is_cu_qp_delta_coded;
        qPy_pred = sh->slice_qp;
    } else {
        qPy_pred = sh->qPy_pred;
    }

    if (availableA == 0)
        qPy_a = qPy_pred;
    else
        qPy_a = f->qp_y_tab[(x_cb - 1) + y_cb * min_cb_width];

    if (availableB == 0)
        qPy_b = qPy_pred;
    else
        qPy_b = f->qp_y_tab[x_cb + (y_cb - 1) * min_cb_width];

    return (qPy_a + qPy_b + 1) >> 1;
}

void h265_set_qPy(H265Slice *sh, int xBase, int yBase, int log2_cb_size)
{
    H265SPS *sps = sh->sps;
    int qp_y = get_qPy_pred(sh, xBase, yBase, log2_cb_size);

    if (sh->tu.cu_qp_delta != 0) {
        int off = sps->qp_bd_offset;
        int v = qp_y + sh->tu.cu_qp_delta + 52 + 2 * off;
        int m = 52 + off;
        v = v % m;
        if (v < 0)
            v += m;
        sh->qp_y = (int8_t)(v - off);
    } else {
        sh->qp_y = (int8_t)qp_y;
    }
}

static int get_qPy(H265SPS *sps, const int8_t *qp_y_tab, int xC, int yC)
{
    int x = xC >> sps->log2_min_cb_size;
    int y = yC >> sps->log2_min_cb_size;
    return qp_y_tab[x + y * sps->min_cb_width];
}

/* ------------------------------------------------------------------ */
/* pixel filters                                                       */
/* ------------------------------------------------------------------ */
#define GETP(p, off) (pixel_shift ? ((uint16_t *)((p) + (off)))[0] : (p)[(off)])
#define SETP(p, off, v) do {                                             \
    if (pixel_shift)                                                     \
        ((uint16_t *)((p) + (off)))[0] = (uint16_t)(v);                  \
    else                                                                 \
        (p)[(off)] = (uint8_t)(v);                                       \
} while (0)

H265_INLINE int clip3(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

#ifdef H265_X86
DEBLOCK_AVX2 static __m128i db_load4(const uint8_t *p, int ystride,
                                     int pixel_shift)
{
    if (!pixel_shift) {
        if (ystride == 1)
            return _mm_cvtepu8_epi32(_mm_loadu_si32(p));
        return _mm_setr_epi32(p[0], p[ystride], p[2 * ystride],
                              p[3 * ystride]);
    }
    if (ystride == 2)
        return _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)p));
    {
        const uint16_t *q = (const uint16_t *)p;
        int s = ystride >> 1;
        return _mm_setr_epi32(q[0], q[s], q[2 * s], q[3 * s]);
    }
}

DEBLOCK_AVX2 static void db_store4(uint8_t *p, int ystride, int pixel_shift,
                                   __m128i v)
{
    int t[4];
    int i;
    if (!pixel_shift && ystride == 1) {
        __m128i p8 = _mm_packs_epi32(v, v);
        p8 = _mm_packus_epi16(p8, p8);
        _mm_storeu_si32(p, p8);
        return;
    }
    if (pixel_shift && ystride == 2) {
        __m128i p16 = _mm_packs_epi32(v, v);
        _mm_storel_epi64((__m128i *)p, p16);
        return;
    }
    _mm_storeu_si128((__m128i *)t, v);
    for (i = 0; i < 4; i++) {
        if (!pixel_shift)
            p[i * ystride] = (uint8_t)t[i];
        else
            ((uint16_t *)(p + i * ystride))[0] = (uint16_t)t[i];
    }
}

DEBLOCK_AVX2 static void db_t8b(const __m128i in[8], __m128i out[8])
{
    __m128i t0 = _mm_unpacklo_epi8(in[0], in[1]);
    __m128i t1 = _mm_unpacklo_epi8(in[2], in[3]);
    __m128i t2 = _mm_unpacklo_epi8(in[4], in[5]);
    __m128i t3 = _mm_unpacklo_epi8(in[6], in[7]);
    __m128i u0 = _mm_unpacklo_epi16(t0, t1);
    __m128i u1 = _mm_unpackhi_epi16(t0, t1);
    __m128i u2 = _mm_unpacklo_epi16(t2, t3);
    __m128i u3 = _mm_unpackhi_epi16(t2, t3);
    __m128i v0 = _mm_unpacklo_epi32(u0, u2);
    __m128i v1 = _mm_unpackhi_epi32(u0, u2);
    __m128i v2 = _mm_unpacklo_epi32(u1, u3);
    __m128i v3 = _mm_unpackhi_epi32(u1, u3);
    out[0] = v0;
    out[1] = _mm_srli_si128(v0, 8);
    out[2] = v1;
    out[3] = _mm_srli_si128(v1, 8);
    out[4] = v2;
    out[5] = _mm_srli_si128(v2, 8);
    out[6] = v3;
    out[7] = _mm_srli_si128(v3, 8);
}

DEBLOCK_AVX2 static void db_t8w(const __m128i in[8], __m128i out[8])
{
    __m128i t0 = _mm_unpacklo_epi16(in[0], in[1]);
    __m128i t1 = _mm_unpackhi_epi16(in[0], in[1]);
    __m128i t2 = _mm_unpacklo_epi16(in[2], in[3]);
    __m128i t3 = _mm_unpackhi_epi16(in[2], in[3]);
    __m128i t4 = _mm_unpacklo_epi16(in[4], in[5]);
    __m128i t5 = _mm_unpackhi_epi16(in[4], in[5]);
    __m128i t6 = _mm_unpacklo_epi16(in[6], in[7]);
    __m128i t7 = _mm_unpackhi_epi16(in[6], in[7]);
    __m128i u0 = _mm_unpacklo_epi32(t0, t2);
    __m128i u1 = _mm_unpackhi_epi32(t0, t2);
    __m128i u2 = _mm_unpacklo_epi32(t1, t3);
    __m128i u3 = _mm_unpackhi_epi32(t1, t3);
    __m128i u4 = _mm_unpacklo_epi32(t4, t6);
    __m128i u5 = _mm_unpackhi_epi32(t4, t6);
    __m128i u6 = _mm_unpacklo_epi32(t5, t7);
    __m128i u7 = _mm_unpackhi_epi32(t5, t7);
    out[0] = _mm_unpacklo_epi64(u0, u4);
    out[1] = _mm_unpackhi_epi64(u0, u4);
    out[2] = _mm_unpacklo_epi64(u1, u5);
    out[3] = _mm_unpackhi_epi64(u1, u5);
    out[4] = _mm_unpacklo_epi64(u2, u6);
    out[5] = _mm_unpackhi_epi64(u2, u6);
    out[6] = _mm_unpacklo_epi64(u3, u7);
    out[7] = _mm_unpackhi_epi64(u3, u7);
}

DEBLOCK_AVX2 static __m128i db_clip16(__m128i v, __m128i lo, __m128i hi)
{
    return _mm_min_epi16(_mm_max_epi16(v, lo), hi);
}

/* Filter one 8-sample luma edge (all 8 lines at once) with 16-bit lanes.
 * Only used for 8/10-bit; 12-bit keeps the 32-bit scalar-lane path. */
DEBLOCK_AVX2 static void db_luma_strong_avx2(uint8_t *pix, int xstride,
                                             int ystride, int tc, int tc2,
                                             int tc3, int no_p, int no_q,
                                             int pixel_shift, int maxv);
DEBLOCK_AVX2 static void db_luma_weak_avx2(uint8_t *pix, int xstride,
                                           int ystride, int tc, int no_p,
                                           int no_q, int nd_p, int nd_q,
                                           int pixel_shift, int maxv);

DEBLOCK_AVX2 static void db_luma_filter8(uint8_t *pix, int xstride, int ystride,
                                         int beta, const int32_t *tc,
                                         const uint8_t *no_p, const uint8_t *no_q,
                                         int pixel_shift, int bit_depth)
{
    const int step = pixel_shift ? 2 : 1;
    const int vertical = xstride == step;
    const int maxv = (1 << bit_depth) - 1;
    const int bshift = bit_depth - 8;
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vmax = _mm_set1_epi16((short)maxv);
    const __m128i vfour = _mm_set1_epi16(4);
    const __m128i vtwo = _mm_set1_epi16(2);
    const __m128i vone = _mm_set1_epi16(1);
    __m128i v[8], t[8], out[8];
    int16_t dpv[8], dqv[8], dvv[8], apqv[8], pqv[8];
    int16_t tcs[8], nps[8], nqs[8], ndps[8], ndqs[8];
    int mode[2];
    int j, k, l;

    if (vertical) {
        uint8_t *p = pix - 4 * xstride;
        if (!pixel_shift) {
            for (k = 0; k < 8; k++)
                t[k] = _mm_loadl_epi64(
                    (const __m128i *)(p + (ptrdiff_t)k * ystride));
            db_t8b(t, v);
            for (k = 0; k < 8; k++)
                v[k] = _mm_cvtepu8_epi16(v[k]);
        } else {
            for (k = 0; k < 8; k++)
                t[k] = _mm_loadu_si128(
                    (const __m128i *)(p + (ptrdiff_t)k * ystride));
            db_t8w(t, v);
        }
    } else {
        for (k = 0; k < 8; k++) {
            const uint8_t *q = pix + (ptrdiff_t)(k - 4) * xstride;
            if (!pixel_shift)
                v[k] = _mm_cvtepu8_epi16(
                    _mm_loadl_epi64((const __m128i *)q));
            else
                v[k] = _mm_loadu_si128((const __m128i *)q);
        }
    }

    {
        __m128i p3 = v[0], p2 = v[1], p1 = v[2], p0 = v[3];
        __m128i q0 = v[4], q1 = v[5], q2 = v[6], q3 = v[7];
        __m128i dp = _mm_abs_epi16(_mm_add_epi16(
            _mm_sub_epi16(p2, _mm_slli_epi16(p1, 1)), p0));
        __m128i dq = _mm_abs_epi16(_mm_add_epi16(
            _mm_sub_epi16(q2, _mm_slli_epi16(q1, 1)), q0));
        __m128i dv = _mm_add_epi16(dp, dq);
        __m128i ap = _mm_add_epi16(
            _mm_abs_epi16(_mm_sub_epi16(p3, p0)),
            _mm_abs_epi16(_mm_sub_epi16(q3, q0)));
        __m128i pq = _mm_abs_epi16(_mm_sub_epi16(p0, q0));

        _mm_storeu_si128((__m128i *)dpv, dp);
        _mm_storeu_si128((__m128i *)dqv, dq);
        _mm_storeu_si128((__m128i *)dvv, dv);
        _mm_storeu_si128((__m128i *)apqv, ap);
        _mm_storeu_si128((__m128i *)pqv, pq);

        for (j = 0; j < 2; j++) {
            int l0 = j * 4;
            int tcv = tc[j] << bshift;
            int wthr = (beta + (beta >> 1)) >> 3;
            int ndp = dpv[l0] + dpv[l0 + 3] < wthr ? 2 : 1;
            int ndq = dqv[l0] + dqv[l0 + 3] < wthr ? 2 : 1;
            mode[j] = 0;
            if (dvv[l0] + dvv[l0 + 3] < beta) {
                int beta3 = beta >> 3;
                int beta2 = beta >> 2;
                int tc25 = (tcv * 5 + 1) >> 1;
                if (apqv[l0] < beta3 && pqv[l0] < tc25 &&
                    apqv[l0 + 3] < beta3 && pqv[l0 + 3] < tc25 &&
                    (dvv[l0] << 1) < beta2 && (dvv[l0 + 3] << 1) < beta2)
                    mode[j] = 2;
                else
                    mode[j] = 1;
            }
            for (l = l0; l < l0 + 4; l++) {
                tcs[l] = (int16_t)(mode[j] == 2 ? tcv * 2 : tcv);
                nps[l] = no_p[j] ? -1 : 0;
                nqs[l] = no_q[j] ? -1 : 0;
                ndps[l] = ndp > 1 ? -1 : 0;
                ndqs[l] = ndq > 1 ? -1 : 0;
            }
        }

        if (!mode[0] && !mode[1])
            return;

        if (mode[0] == 2 && mode[1] == 2) {
            __m128i tc3 = _mm_loadu_si128((const __m128i *)tcs);
            __m128i npv = _mm_loadu_si128((const __m128i *)nps);
            __m128i nqv = _mm_loadu_si128((const __m128i *)nqs);
            __m128i ntc = _mm_sub_epi16(vzero, tc3);
            __m128i np0, np1, np2, nq0, nq1, nq2;

            np0 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                _mm_add_epi16(p2, _mm_slli_epi16(p1, 1)),
                _mm_add_epi16(_mm_slli_epi16(p0, 1),
                              _mm_slli_epi16(q0, 1))),
                _mm_add_epi16(q1, vfour)), 3);
            np1 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                _mm_add_epi16(p2, p1), _mm_add_epi16(p0, q0)), vtwo), 2);
            np2 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                _mm_add_epi16(_mm_slli_epi16(p3, 1),
                              _mm_add_epi16(_mm_slli_epi16(p2, 1), p2)),
                _mm_add_epi16(p1, _mm_add_epi16(p0, q0))), vfour), 3);

            np0 = _mm_add_epi16(p0, db_clip16(_mm_sub_epi16(np0, p0), ntc, tc3));
            np1 = _mm_add_epi16(p1, db_clip16(_mm_sub_epi16(np1, p1), ntc, tc3));
            np2 = _mm_add_epi16(p2, db_clip16(_mm_sub_epi16(np2, p2), ntc, tc3));

            nq0 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                _mm_add_epi16(q2, _mm_slli_epi16(q1, 1)),
                _mm_add_epi16(_mm_slli_epi16(q0, 1),
                              _mm_slli_epi16(p0, 1))),
                _mm_add_epi16(p1, vfour)), 3);
            nq1 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                _mm_add_epi16(q2, q1), _mm_add_epi16(q0, p0)), vtwo), 2);
            nq2 = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                _mm_add_epi16(_mm_slli_epi16(q3, 1),
                              _mm_add_epi16(_mm_slli_epi16(q2, 1), q2)),
                _mm_add_epi16(q1, _mm_add_epi16(q0, p0))), vfour), 3);

            nq0 = _mm_add_epi16(q0, db_clip16(_mm_sub_epi16(nq0, q0), ntc, tc3));
            nq1 = _mm_add_epi16(q1, db_clip16(_mm_sub_epi16(nq1, q1), ntc, tc3));
            nq2 = _mm_add_epi16(q2, db_clip16(_mm_sub_epi16(nq2, q2), ntc, tc3));

            v[1] = _mm_blendv_epi8(np2, p2, npv);
            v[2] = _mm_blendv_epi8(np1, p1, npv);
            v[3] = _mm_blendv_epi8(np0, p0, npv);
            v[4] = _mm_blendv_epi8(nq0, q0, nqv);
            v[5] = _mm_blendv_epi8(nq1, q1, nqv);
            v[6] = _mm_blendv_epi8(nq2, q2, nqv);
        } else if (mode[0] == 1 && mode[1] == 1) {
            __m128i tcv16 = _mm_loadu_si128((const __m128i *)tcs);
            __m128i tc2v = _mm_srai_epi16(tcv16, 1);
            __m128i npv = _mm_loadu_si128((const __m128i *)nps);
            __m128i nqv = _mm_loadu_si128((const __m128i *)nqs);
            __m128i ndpv = _mm_loadu_si128((const __m128i *)ndps);
            __m128i ndqv = _mm_loadu_si128((const __m128i *)ndqs);
            __m128i ntc = _mm_sub_epi16(vzero, tcv16);
            __m128i ntc2 = _mm_sub_epi16(vzero, tc2v);
            __m128i x0 = _mm_sub_epi16(q0, p0);
            __m128i x1 = _mm_sub_epi16(q1, p1);
            __m128i nine = _mm_add_epi16(_mm_slli_epi16(x0, 3), x0);
            __m128i three = _mm_add_epi16(_mm_slli_epi16(x1, 1), x1);
            __m128i delta = _mm_srai_epi16(_mm_add_epi16(
                _mm_sub_epi16(nine, three), _mm_set1_epi16(8)), 4);
            __m128i ten = _mm_mullo_epi16(tcv16, _mm_set1_epi16(10));
            __m128i valid = _mm_cmpgt_epi16(ten, _mm_abs_epi16(delta));
            __m128i pmask = _mm_andnot_si128(npv, valid);
            __m128i qmask = _mm_andnot_si128(nqv, valid);
            __m128i np0, np1, nq0, nq1;

            delta = db_clip16(delta, ntc, tcv16);
            np0 = db_clip16(_mm_add_epi16(p0, delta), vzero, vmax);
            nq0 = db_clip16(_mm_sub_epi16(q0, delta), vzero, vmax);

            np1 = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(
                _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(p2, p0), vone), 1),
                p1), delta), 1);
            nq1 = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(
                _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(q2, q0), vone), 1),
                q1), _mm_sub_epi16(vzero, delta)), 1);
            np1 = db_clip16(np1, ntc2, tc2v);
            nq1 = db_clip16(nq1, ntc2, tc2v);
            np1 = db_clip16(_mm_add_epi16(p1, np1), vzero, vmax);
            nq1 = db_clip16(_mm_add_epi16(q1, nq1), vzero, vmax);

            np1 = _mm_blendv_epi8(p1, np1, ndpv);
            nq1 = _mm_blendv_epi8(q1, nq1, ndqv);

            v[2] = _mm_blendv_epi8(p1, np1, pmask);
            v[3] = _mm_blendv_epi8(p0, np0, pmask);
            v[4] = _mm_blendv_epi8(q0, nq0, qmask);
            v[5] = _mm_blendv_epi8(q1, nq1, qmask);
        } else {
            for (j = 0; j < 2; j++) {
                uint8_t *p = pix + (ptrdiff_t)j * 4 * ystride;
                int tcv = tc[j] << bshift;
                if (mode[j] == 2)
                    db_luma_strong_avx2(p, xstride, ystride, tcv * 2, tcv * 2,
                                        tcv * 2, no_p[j], no_q[j],
                                        pixel_shift, maxv);
                else if (mode[j] == 1) {
                    int wthr = (beta + (beta >> 1)) >> 3;
                    int ndp = dpv[j * 4] + dpv[j * 4 + 3] < wthr ? 2 : 1;
                    int ndq = dqv[j * 4] + dqv[j * 4 + 3] < wthr ? 2 : 1;
                    db_luma_weak_avx2(p, xstride, ystride, tcv, no_p[j],
                                      no_q[j], ndp, ndq, pixel_shift, maxv);
                }
            }
            return;
        }
    }

    if (vertical) {
        uint8_t *p = pix - 4 * xstride;
        if (!pixel_shift) {
            for (k = 0; k < 8; k++)
                t[k] = _mm_packus_epi16(v[k], v[k]);
            db_t8b(t, out);
            for (k = 0; k < 8; k++)
                _mm_storel_epi64((__m128i *)(p + (ptrdiff_t)k * ystride),
                                 out[k]);
        } else {
            db_t8w(v, out);
            for (k = 0; k < 8; k++)
                _mm_storeu_si128((__m128i *)(p + (ptrdiff_t)k * ystride),
                                 out[k]);
        }
    } else {
        for (k = 0; k < 8; k++) {
            uint8_t *q = pix + (ptrdiff_t)(k - 4) * xstride;
            if (!pixel_shift)
                _mm_storel_epi64((__m128i *)q,
                                 _mm_packus_epi16(v[k], v[k]));
            else
                _mm_storeu_si128((__m128i *)q, v[k]);
        }
    }
}

DEBLOCK_AVX2 static __m256i db_clip16_256(__m256i v, __m256i lo, __m256i hi)
{
    return _mm256_min_epi16(_mm256_max_epi16(v, lo), hi);
}

/* Filter two adjacent luma edges at once.  Lanes 0..7 hold edge 0,
 * lanes 8..15 edge 1.  Each edge has two independent 4-line groups. */
DEBLOCK_AVX2 static void db_luma_body16(__m256i v[8],
                                        const int beta[2],
                                        const int32_t tc[2][2],
                                        const uint8_t no_p[2][2],
                                        const uint8_t no_q[2][2],
                                        int bshift, int maxv)
{
    const __m256i vzero = _mm256_setzero_si256();
    const __m256i vmax = _mm256_set1_epi16((short)maxv);
    const __m256i vfour = _mm256_set1_epi16(4);
    const __m256i vtwo = _mm256_set1_epi16(2);
    const __m256i vone = _mm256_set1_epi16(1);
    __m256i p3 = v[0], p2 = v[1], p1 = v[2], p0 = v[3];
    __m256i q0 = v[4], q1 = v[5], q2 = v[6], q3 = v[7];
    __m256i dp = _mm256_abs_epi16(_mm256_add_epi16(
        _mm256_sub_epi16(p2, _mm256_slli_epi16(p1, 1)), p0));
    __m256i dq = _mm256_abs_epi16(_mm256_add_epi16(
        _mm256_sub_epi16(q2, _mm256_slli_epi16(q1, 1)), q0));
    __m256i dv = _mm256_add_epi16(dp, dq);
    __m256i ap = _mm256_add_epi16(
        _mm256_abs_epi16(_mm256_sub_epi16(p3, p0)),
        _mm256_abs_epi16(_mm256_sub_epi16(q3, q0)));
    __m256i pq = _mm256_abs_epi16(_mm256_sub_epi16(p0, q0));
    int16_t dpv[16], dqv[16], dvv[16], apv[16], pqv[16];
    int16_t tcs[16], nps[16], nqs[16], ndps[16], ndqs[16], m1s[16], m2s[16];
    int mode[4];
    int any1 = 0, any2 = 0;
    int g, l;

    _mm256_storeu_si256((__m256i *)dpv, dp);
    _mm256_storeu_si256((__m256i *)dqv, dq);
    _mm256_storeu_si256((__m256i *)dvv, dv);
    _mm256_storeu_si256((__m256i *)apv, ap);
    _mm256_storeu_si256((__m256i *)pqv, pq);

    for (g = 0; g < 4; g++) {
        int e = g >> 1, j = g & 1, l0 = g * 4;
        int tcv = tc[e][j] << bshift;
        int wthr = (beta[e] + (beta[e] >> 1)) >> 3;
        int ndp = dpv[l0] + dpv[l0 + 3] < wthr ? 2 : 1;
        int ndq = dqv[l0] + dqv[l0 + 3] < wthr ? 2 : 1;
        mode[g] = 0;
        if (dvv[l0] + dvv[l0 + 3] < beta[e]) {
            int beta3 = beta[e] >> 3;
            int beta2 = beta[e] >> 2;
            int tc25 = (tcv * 5 + 1) >> 1;
            if (apv[l0] < beta3 && pqv[l0] < tc25 &&
                apv[l0 + 3] < beta3 && pqv[l0 + 3] < tc25 &&
                (dvv[l0] << 1) < beta2 && (dvv[l0 + 3] << 1) < beta2)
                mode[g] = 2;
            else
                mode[g] = 1;
        }
        for (l = l0; l < l0 + 4; l++) {
            tcs[l] = (int16_t)(mode[g] == 2 ? tcv * 2 : tcv);
            nps[l] = no_p[e][j] ? -1 : 0;
            nqs[l] = no_q[e][j] ? -1 : 0;
            ndps[l] = ndp > 1 ? -1 : 0;
            ndqs[l] = ndq > 1 ? -1 : 0;
            m1s[l] = mode[g] == 1 ? -1 : 0;
            m2s[l] = mode[g] == 2 ? -1 : 0;
        }
        if (mode[g] == 1)
            any1 = 1;
        else if (mode[g] == 2)
            any2 = 1;
    }

    if (!any1 && !any2)
        return;

    if (any1) {
        __m256i tcv16 = _mm256_loadu_si256((const __m256i *)tcs);
        __m256i tc2v = _mm256_srai_epi16(tcv16, 1);
        __m256i npv = _mm256_loadu_si256((const __m256i *)nps);
        __m256i nqv = _mm256_loadu_si256((const __m256i *)nqs);
        __m256i ndpv = _mm256_loadu_si256((const __m256i *)ndps);
        __m256i ndqv = _mm256_loadu_si256((const __m256i *)ndqs);
        __m256i m1 = _mm256_loadu_si256((const __m256i *)m1s);
        __m256i ntc = _mm256_sub_epi16(vzero, tcv16);
        __m256i ntc2 = _mm256_sub_epi16(vzero, tc2v);
        __m256i x0 = _mm256_sub_epi16(q0, p0);
        __m256i x1 = _mm256_sub_epi16(q1, p1);
        __m256i nine = _mm256_add_epi16(_mm256_slli_epi16(x0, 3), x0);
        __m256i three = _mm256_add_epi16(_mm256_slli_epi16(x1, 1), x1);
        __m256i delta = _mm256_srai_epi16(_mm256_add_epi16(
            _mm256_sub_epi16(nine, three), _mm256_set1_epi16(8)), 4);
        __m256i ten = _mm256_mullo_epi16(tcv16, _mm256_set1_epi16(10));
        __m256i valid = _mm256_cmpgt_epi16(ten, _mm256_abs_epi16(delta));
        __m256i pmask = _mm256_andnot_si256(npv, valid);
        __m256i qmask = _mm256_andnot_si256(nqv, valid);
        __m256i np0, np1, nq0, nq1;

        delta = db_clip16_256(delta, ntc, tcv16);
        np0 = db_clip16_256(_mm256_add_epi16(p0, delta), vzero, vmax);
        nq0 = db_clip16_256(_mm256_sub_epi16(q0, delta), vzero, vmax);
        np1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(
            _mm256_srai_epi16(_mm256_add_epi16(
                _mm256_add_epi16(p2, p0), vone), 1), p1), delta), 1);
        nq1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(
            _mm256_srai_epi16(_mm256_add_epi16(
                _mm256_add_epi16(q2, q0), vone), 1), q1),
            _mm256_sub_epi16(vzero, delta)), 1);
        np1 = db_clip16_256(np1, ntc2, tc2v);
        nq1 = db_clip16_256(nq1, ntc2, tc2v);
        np1 = db_clip16_256(_mm256_add_epi16(p1, np1), vzero, vmax);
        nq1 = db_clip16_256(_mm256_add_epi16(q1, nq1), vzero, vmax);
        np1 = _mm256_blendv_epi8(p1, np1, ndpv);
        nq1 = _mm256_blendv_epi8(q1, nq1, ndqv);
        pmask = _mm256_and_si256(pmask, m1);
        qmask = _mm256_and_si256(qmask, m1);
        v[2] = _mm256_blendv_epi8(v[2], np1, pmask);
        v[3] = _mm256_blendv_epi8(v[3], np0, pmask);
        v[4] = _mm256_blendv_epi8(v[4], nq0, qmask);
        v[5] = _mm256_blendv_epi8(v[5], nq1, qmask);
    }

    if (any2) {
        __m256i tc3 = _mm256_loadu_si256((const __m256i *)tcs);
        __m256i npv = _mm256_loadu_si256((const __m256i *)nps);
        __m256i nqv = _mm256_loadu_si256((const __m256i *)nqs);
        __m256i m2 = _mm256_loadu_si256((const __m256i *)m2s);
        __m256i ntc = _mm256_sub_epi16(vzero, tc3);
        __m256i np0, np1, np2, nq0, nq1, nq2;

        np0 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(
            _mm256_add_epi16(p2, _mm256_slli_epi16(p1, 1)),
            _mm256_add_epi16(_mm256_slli_epi16(p0, 1),
                             _mm256_slli_epi16(q0, 1))),
            _mm256_add_epi16(q1, vfour)), 3);
        np1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(
            _mm256_add_epi16(p2, p1), _mm256_add_epi16(p0, q0)), vtwo), 2);
        np2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_slli_epi16(p3, 1),
                             _mm256_add_epi16(_mm256_slli_epi16(p2, 1), p2)),
            _mm256_add_epi16(p1, _mm256_add_epi16(p0, q0))), vfour), 3);
        np0 = _mm256_add_epi16(p0,
            db_clip16_256(_mm256_sub_epi16(np0, p0), ntc, tc3));
        np1 = _mm256_add_epi16(p1,
            db_clip16_256(_mm256_sub_epi16(np1, p1), ntc, tc3));
        np2 = _mm256_add_epi16(p2,
            db_clip16_256(_mm256_sub_epi16(np2, p2), ntc, tc3));

        nq0 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(
            _mm256_add_epi16(q2, _mm256_slli_epi16(q1, 1)),
            _mm256_add_epi16(_mm256_slli_epi16(q0, 1),
                             _mm256_slli_epi16(p0, 1))),
            _mm256_add_epi16(p1, vfour)), 3);
        nq1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(
            _mm256_add_epi16(q2, q1), _mm256_add_epi16(q0, p0)), vtwo), 2);
        nq2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_slli_epi16(q3, 1),
                             _mm256_add_epi16(_mm256_slli_epi16(q2, 1), q2)),
            _mm256_add_epi16(q1, _mm256_add_epi16(q0, p0))), vfour), 3);
        nq0 = _mm256_add_epi16(q0,
            db_clip16_256(_mm256_sub_epi16(nq0, q0), ntc, tc3));
        nq1 = _mm256_add_epi16(q1,
            db_clip16_256(_mm256_sub_epi16(nq1, q1), ntc, tc3));
        nq2 = _mm256_add_epi16(q2,
            db_clip16_256(_mm256_sub_epi16(nq2, q2), ntc, tc3));

        np2 = _mm256_blendv_epi8(np2, p2, npv);
        np1 = _mm256_blendv_epi8(np1, p1, npv);
        np0 = _mm256_blendv_epi8(np0, p0, npv);
        nq0 = _mm256_blendv_epi8(nq0, q0, nqv);
        nq1 = _mm256_blendv_epi8(nq1, q1, nqv);
        nq2 = _mm256_blendv_epi8(nq2, q2, nqv);
        v[1] = _mm256_blendv_epi8(v[1], np2, m2);
        v[2] = _mm256_blendv_epi8(v[2], np1, m2);
        v[3] = _mm256_blendv_epi8(v[3], np0, m2);
        v[4] = _mm256_blendv_epi8(v[4], nq0, m2);
        v[5] = _mm256_blendv_epi8(v[5], nq1, m2);
        v[6] = _mm256_blendv_epi8(v[6], nq2, m2);
    }
}

/* Two horizontally adjacent luma edges (x and x+8) on the same boundary. */
DEBLOCK_AVX2 static void db_luma_filter16_h(uint8_t *pix, int xstride,
                                            int beta0, const int32_t *tc0,
                                            const uint8_t *np0,
                                            const uint8_t *nq0, int beta1,
                                            const int32_t *tc1,
                                            const uint8_t *np1,
                                            const uint8_t *nq1,
                                            int pixel_shift, int bit_depth)
{
    __m256i v[8];
    int beta[2];
    int32_t tc[2][2];
    uint8_t no_p[2][2], no_q[2][2];
    int maxv = (1 << bit_depth) - 1;
    int bshift = bit_depth - 8;
    int k;

    beta[0] = beta0 << bshift;
    beta[1] = beta1 << bshift;
    tc[0][0] = tc0[0]; tc[0][1] = tc0[1];
    tc[1][0] = tc1[0]; tc[1][1] = tc1[1];
    no_p[0][0] = np0[0]; no_p[0][1] = np0[1];
    no_p[1][0] = np1[0]; no_p[1][1] = np1[1];
    no_q[0][0] = nq0[0]; no_q[0][1] = nq0[1];
    no_q[1][0] = nq1[0]; no_q[1][1] = nq1[1];

    for (k = 0; k < 8; k++) {
        const uint8_t *row = pix + (ptrdiff_t)(k - 4) * xstride;
        v[k] = pixel_shift
            ? _mm256_loadu_si256((const __m256i *)row)
            : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)row));
    }
    db_luma_body16(v, beta, tc, no_p, no_q, bshift, maxv);
    for (k = 0; k < 8; k++) {
        uint8_t *row = pix + (ptrdiff_t)(k - 4) * xstride;
        if (!pixel_shift) {
            _mm_storeu_si128((__m128i *)row,
                _mm_packus_epi16(_mm256_castsi256_si128(v[k]),
                                 _mm256_extracti128_si256(v[k], 1)));
        } else {
            _mm256_storeu_si256((__m256i *)row, v[k]);
        }
    }
}

/* Two vertically adjacent luma edges (x and x+8) on the same rows. */
DEBLOCK_AVX2 static void db_luma_filter16_v(uint8_t *pix, ptrdiff_t stride,
                                            int beta0, const int32_t *tc0,
                                            const uint8_t *np0,
                                            const uint8_t *nq0, int beta1,
                                            const int32_t *tc1,
                                            const uint8_t *np1,
                                            const uint8_t *nq1,
                                            int pixel_shift, int bit_depth)
{
    const int step = pixel_shift ? 2 : 1;
    uint8_t *base = pix - 4 * step;
    __m128i lo[8], hi[8], colsA[8], colsB[8], rowsA[8], rowsB[8];
    __m256i v[8];
    int beta[2];
    int32_t tc[2][2];
    uint8_t no_p[2][2], no_q[2][2];
    int maxv = (1 << bit_depth) - 1;
    int bshift = bit_depth - 8;
    int k;

    beta[0] = beta0 << bshift;
    beta[1] = beta1 << bshift;
    tc[0][0] = tc0[0]; tc[0][1] = tc0[1];
    tc[1][0] = tc1[0]; tc[1][1] = tc1[1];
    no_p[0][0] = np0[0]; no_p[0][1] = np0[1];
    no_p[1][0] = np1[0]; no_p[1][1] = np1[1];
    no_q[0][0] = nq0[0]; no_q[0][1] = nq0[1];
    no_q[1][0] = nq1[0]; no_q[1][1] = nq1[1];

    for (k = 0; k < 8; k++) {
        const uint8_t *p = base + (ptrdiff_t)k * stride;
        if (!pixel_shift) {
            lo[k] = _mm_loadu_si128((const __m128i *)p);
            hi[k] = _mm_srli_si128(lo[k], 8);
        } else {
            lo[k] = _mm_loadu_si128((const __m128i *)p);
            hi[k] = _mm_loadu_si128((const __m128i *)(p + 16));
        }
    }
    if (!pixel_shift) {
        db_t8b(lo, colsA);
        db_t8b(hi, colsB);
        for (k = 0; k < 8; k++)
            v[k] = _mm256_cvtepu8_epi16(
                _mm_unpacklo_epi64(colsA[k], colsB[k]));
    } else {
        db_t8w(lo, colsA);
        db_t8w(hi, colsB);
        for (k = 0; k < 8; k++)
            v[k] = _mm256_set_m128i(colsB[k], colsA[k]);
    }

    db_luma_body16(v, beta, tc, no_p, no_q, bshift, maxv);

    if (!pixel_shift) {
        for (k = 0; k < 8; k++) {
            __m128i t = _mm_packus_epi16(_mm256_castsi256_si128(v[k]),
                                         _mm256_extracti128_si256(v[k], 1));
            lo[k] = t;
            hi[k] = _mm_srli_si128(t, 8);
        }
        db_t8b(lo, rowsA);
        db_t8b(hi, rowsB);
        for (k = 0; k < 8; k++)
            _mm_storeu_si128((__m128i *)(base + (ptrdiff_t)k * stride),
                             _mm_unpacklo_epi64(rowsA[k], rowsB[k]));
    } else {
        for (k = 0; k < 8; k++) {
            lo[k] = _mm256_castsi256_si128(v[k]);
            hi[k] = _mm256_extracti128_si256(v[k], 1);
        }
        db_t8w(lo, rowsA);
        db_t8w(hi, rowsB);
        for (k = 0; k < 8; k++)
            _mm256_storeu_si256((__m256i *)(base + (ptrdiff_t)k * stride),
                                _mm256_set_m128i(rowsB[k], rowsA[k]));
    }
}

DEBLOCK_AVX2 static __m128i db_clip3(__m128i v, int lo, int hi)
{
    v = _mm_max_epi32(v, _mm_set1_epi32(lo));
    return _mm_min_epi32(v, _mm_set1_epi32(hi));
}

DEBLOCK_AVX2 static void db_luma_strong_avx2(uint8_t *pix, int xstride,
                                             int ystride, int tc, int tc2,
                                             int tc3, int no_p, int no_q,
                                             int pixel_shift, int maxv)
{
    __m128i p3 = db_load4(pix - 4 * xstride, ystride, pixel_shift);
    __m128i p2 = db_load4(pix - 3 * xstride, ystride, pixel_shift);
    __m128i p1 = db_load4(pix - 2 * xstride, ystride, pixel_shift);
    __m128i p0 = db_load4(pix - 1 * xstride, ystride, pixel_shift);
    __m128i q0 = db_load4(pix, ystride, pixel_shift);
    __m128i q1 = db_load4(pix + 1 * xstride, ystride, pixel_shift);
    __m128i q2 = db_load4(pix + 2 * xstride, ystride, pixel_shift);
    __m128i q3 = db_load4(pix + 3 * xstride, ystride, pixel_shift);
    (void)maxv;

    if (!no_p) {
        __m128i v0 = _mm_srai_epi32(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(p2, _mm_slli_epi32(p1, 1)),
                          _mm_add_epi32(_mm_slli_epi32(p0, 1),
                                        _mm_slli_epi32(q0, 1))),
            _mm_add_epi32(q1, _mm_set1_epi32(4))), 3);
        __m128i v1 = _mm_srai_epi32(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(p2, p1), _mm_add_epi32(p0, q0)),
            _mm_set1_epi32(2)), 2);
        __m128i v2 = _mm_srai_epi32(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(_mm_slli_epi32(p3, 1),
                                        _mm_mullo_epi32(p2, _mm_set1_epi32(3))),
                          _mm_add_epi32(p1, _mm_add_epi32(p0, q0))),
            _mm_set1_epi32(4)), 3);
        {
            __m128i np0 = _mm_add_epi32(p0,
                db_clip3(_mm_sub_epi32(v0, p0), -tc3, tc3));
            __m128i np1 = _mm_add_epi32(p1,
                db_clip3(_mm_sub_epi32(v1, p1), -tc2, tc2));
            __m128i np2 = _mm_add_epi32(p2,
                db_clip3(_mm_sub_epi32(v2, p2), -tc, tc));
            db_store4(pix - 1 * xstride, ystride, pixel_shift, np0);
            db_store4(pix - 2 * xstride, ystride, pixel_shift, np1);
            db_store4(pix - 3 * xstride, ystride, pixel_shift, np2);
        }
    }
    if (!no_q) {
        __m128i v0 = _mm_srai_epi32(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(p1, _mm_slli_epi32(p0, 1)),
                          _mm_add_epi32(_mm_slli_epi32(q0, 1),
                                        _mm_slli_epi32(q1, 1))),
            _mm_add_epi32(q2, _mm_set1_epi32(4))), 3);
        __m128i v1 = _mm_srai_epi32(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(p0, q0), _mm_add_epi32(q1, q2)),
            _mm_set1_epi32(2)), 2);
        __m128i v2 = _mm_srai_epi32(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(_mm_slli_epi32(q3, 1),
                                        _mm_mullo_epi32(q2, _mm_set1_epi32(3))),
                          _mm_add_epi32(q1, _mm_add_epi32(q0, p0))),
            _mm_set1_epi32(4)), 3);
        {
            __m128i nq0 = _mm_add_epi32(q0,
                db_clip3(_mm_sub_epi32(v0, q0), -tc3, tc3));
            __m128i nq1 = _mm_add_epi32(q1,
                db_clip3(_mm_sub_epi32(v1, q1), -tc2, tc2));
            __m128i nq2 = _mm_add_epi32(q2,
                db_clip3(_mm_sub_epi32(v2, q2), -tc, tc));
            db_store4(pix, ystride, pixel_shift, nq0);
            db_store4(pix + 1 * xstride, ystride, pixel_shift, nq1);
            db_store4(pix + 2 * xstride, ystride, pixel_shift, nq2);
        }
    }
}

DEBLOCK_AVX2 static void db_luma_weak_avx2(uint8_t *pix, int xstride,
                                           int ystride, int tc, int no_p,
                                           int no_q, int nd_p, int nd_q,
                                           int pixel_shift, int maxv)
{
    int tc_2 = tc >> 1;
    __m128i p2 = db_load4(pix - 3 * xstride, ystride, pixel_shift);
    __m128i p1 = db_load4(pix - 2 * xstride, ystride, pixel_shift);
    __m128i p0 = db_load4(pix - 1 * xstride, ystride, pixel_shift);
    __m128i q0 = db_load4(pix, ystride, pixel_shift);
    __m128i q1 = db_load4(pix + 1 * xstride, ystride, pixel_shift);
    __m128i q2 = db_load4(pix + 2 * xstride, ystride, pixel_shift);
    __m128i delta0 = _mm_srai_epi32(_mm_add_epi32(
        _mm_sub_epi32(_mm_mullo_epi32(_mm_sub_epi32(q0, p0),
                                      _mm_set1_epi32(9)),
                      _mm_mullo_epi32(_mm_sub_epi32(q1, p1),
                                      _mm_set1_epi32(3))),
        _mm_set1_epi32(8)), 4);
    __m128i mask = _mm_cmpgt_epi32(_mm_set1_epi32(10 * tc),
                                   _mm_abs_epi32(delta0));
    delta0 = db_clip3(delta0, -tc, tc);

    if (!no_p) {
        __m128i np = _mm_add_epi32(p0, delta0);
        np = _mm_min_epi32(_mm_max_epi32(np, _mm_setzero_si128()),
                           _mm_set1_epi32(maxv));
        db_store4(pix - 1 * xstride, ystride, pixel_shift,
                  _mm_blendv_epi8(p0, np, mask));
    }
    if (!no_q) {
        __m128i nq = _mm_sub_epi32(q0, delta0);
        nq = _mm_min_epi32(_mm_max_epi32(nq, _mm_setzero_si128()),
                           _mm_set1_epi32(maxv));
        db_store4(pix, ystride, pixel_shift,
                  _mm_blendv_epi8(q0, nq, mask));
    }
    if (!no_p && nd_p > 1) {
        __m128i dp = _mm_srai_epi32(_mm_add_epi32(
            _mm_sub_epi32(_mm_srai_epi32(
                _mm_add_epi32(_mm_add_epi32(p2, p0), _mm_set1_epi32(1)), 1),
                p1), delta0), 1);
        dp = db_clip3(dp, -tc_2, tc_2);
        __m128i np = _mm_add_epi32(p1, dp);
        np = _mm_min_epi32(_mm_max_epi32(np, _mm_setzero_si128()),
                           _mm_set1_epi32(maxv));
        db_store4(pix - 2 * xstride, ystride, pixel_shift,
                  _mm_blendv_epi8(p1, np, mask));
    }
    if (!no_q && nd_q > 1) {
        __m128i dq = _mm_srai_epi32(_mm_sub_epi32(
            _mm_srai_epi32(
                _mm_add_epi32(_mm_add_epi32(q2, q0), _mm_set1_epi32(1)), 1),
            _mm_add_epi32(q1, delta0)), 1);
        dq = db_clip3(dq, -tc_2, tc_2);
        __m128i nq = _mm_add_epi32(q1, dq);
        nq = _mm_min_epi32(_mm_max_epi32(nq, _mm_setzero_si128()),
                           _mm_set1_epi32(maxv));
        db_store4(pix + 1 * xstride, ystride, pixel_shift,
                  _mm_blendv_epi8(q1, nq, mask));
    }
}

DEBLOCK_AVX2 static void db_chroma_avx2(uint8_t *pix, int xstride,
                                        int ystride, int tcv, int no_p,
                                        int no_q, int pixel_shift, int maxv)
{
    __m128i p1 = db_load4(pix - 2 * xstride, ystride, pixel_shift);
    __m128i p0 = db_load4(pix - 1 * xstride, ystride, pixel_shift);
    __m128i q0 = db_load4(pix, ystride, pixel_shift);
    __m128i q1 = db_load4(pix + 1 * xstride, ystride, pixel_shift);
    __m128i delta0 = _mm_srai_epi32(_mm_add_epi32(
        _mm_add_epi32(_mm_slli_epi32(_mm_sub_epi32(q0, p0), 2),
                      _mm_sub_epi32(p1, q1)), _mm_set1_epi32(4)), 3);
    delta0 = db_clip3(delta0, -tcv, tcv);
    if (!no_p) {
        __m128i np = _mm_add_epi32(p0, delta0);
        np = _mm_min_epi32(_mm_max_epi32(np, _mm_setzero_si128()),
                           _mm_set1_epi32(maxv));
        db_store4(pix - 1 * xstride, ystride, pixel_shift, np);
    }
    if (!no_q) {
        __m128i nq = _mm_sub_epi32(q0, delta0);
        nq = _mm_min_epi32(_mm_max_epi32(nq, _mm_setzero_si128()),
                           _mm_set1_epi32(maxv));
        db_store4(pix, ystride, pixel_shift, nq);
    }
}
#endif

static void loop_filter_luma_strong(uint8_t *pix, int xstride, int ystride,
                                    int tc, int tc2, int tc3, int no_p,
                                    int no_q, int pixel_shift, int maxv)
{
    int d;
    for (d = 0; d < 4; d++) {
        int p3 = GETP(pix, -4 * xstride);
        int p2 = GETP(pix, -3 * xstride);
        int p1 = GETP(pix, -2 * xstride);
        int p0 = GETP(pix, -1 * xstride);
        int q0 = GETP(pix, 0 * xstride);
        int q1 = GETP(pix, 1 * xstride);
        int q2 = GETP(pix, 2 * xstride);
        int q3 = GETP(pix, 3 * xstride);
        if (!no_p) {
            SETP(pix, -1 * xstride, p0 + clip3(((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) - p0, -tc3, tc3));
            SETP(pix, -2 * xstride, p1 + clip3(((p2 + p1 + p0 + q0 + 2) >> 2) - p1, -tc2, tc2));
            SETP(pix, -3 * xstride, p2 + clip3(((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) - p2, -tc, tc));
        }
        if (!no_q) {
            SETP(pix, 0 * xstride, q0 + clip3(((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) - q0, -tc3, tc3));
            SETP(pix, 1 * xstride, q1 + clip3(((p0 + q0 + q1 + q2 + 2) >> 2) - q1, -tc2, tc2));
            SETP(pix, 2 * xstride, q2 + clip3(((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) - q2, -tc, tc));
        }
        pix += ystride;
    }
}

static void loop_filter_luma_weak(uint8_t *pix, int xstride, int ystride,
                                  int tc, int beta, int no_p, int no_q,
                                  int nd_p, int nd_q, int pixel_shift, int maxv)
{
    int tc_2 = tc >> 1;
    int d;
    (void)beta;
    for (d = 0; d < 4; d++) {
        int p2 = GETP(pix, -3 * xstride);
        int p1 = GETP(pix, -2 * xstride);
        int p0 = GETP(pix, -1 * xstride);
        int q0 = GETP(pix, 0 * xstride);
        int q1 = GETP(pix, 1 * xstride);
        int q2 = GETP(pix, 2 * xstride);
        int delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if (H265_ABS(delta0) < 10 * tc) {
            delta0 = clip3(delta0, -tc, tc);
            if (!no_p)
                SETP(pix, -1 * xstride, clip3(p0 + delta0, 0, maxv));
            if (!no_q)
                SETP(pix, 0 * xstride, clip3(q0 - delta0, 0, maxv));
            if (!no_p && nd_p > 1) {
                int deltap1 = clip3((((p2 + p0 + 1) >> 1) - p1 + delta0) >> 1, -tc_2, tc_2);
                SETP(pix, -2 * xstride, clip3(p1 + deltap1, 0, maxv));
            }
            if (!no_q && nd_q > 1) {
                int deltaq1 = clip3((((q2 + q0 + 1) >> 1) - q1 - delta0) >> 1, -tc_2, tc_2);
                SETP(pix, 1 * xstride, clip3(q1 + deltaq1, 0, maxv));
            }
        }
        pix += ystride;
    }
}

#ifdef H265_NEON
/* NEON 8-line luma/chroma deblocking.  This mirrors the SSE/AVX2 8-lane
 * path; each vector holds one 8-sample edge. */
static void db_t8b_neon(const uint8x16_t in[8], uint8x16_t out[8])
{
    uint8x16_t t0 = vzip1q_u8(in[0], in[1]);
    uint8x16_t t1 = vzip1q_u8(in[2], in[3]);
    uint8x16_t t2 = vzip1q_u8(in[4], in[5]);
    uint8x16_t t3 = vzip1q_u8(in[6], in[7]);
    uint8x16_t u0 = vreinterpretq_u8_u16(vzip1q_u16(
        vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t1)));
    uint8x16_t u1 = vreinterpretq_u8_u16(vzip2q_u16(
        vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t1)));
    uint8x16_t u2 = vreinterpretq_u8_u16(vzip1q_u16(
        vreinterpretq_u16_u8(t2), vreinterpretq_u16_u8(t3)));
    uint8x16_t u3 = vreinterpretq_u8_u16(vzip2q_u16(
        vreinterpretq_u16_u8(t2), vreinterpretq_u16_u8(t3)));
    uint8x16_t v0 = vreinterpretq_u8_u32(vzip1q_u32(
        vreinterpretq_u32_u8(u0), vreinterpretq_u32_u8(u2)));
    uint8x16_t v1 = vreinterpretq_u8_u32(vzip2q_u32(
        vreinterpretq_u32_u8(u0), vreinterpretq_u32_u8(u2)));
    uint8x16_t v2 = vreinterpretq_u8_u32(vzip1q_u32(
        vreinterpretq_u32_u8(u1), vreinterpretq_u32_u8(u3)));
    uint8x16_t v3 = vreinterpretq_u8_u32(vzip2q_u32(
        vreinterpretq_u32_u8(u1), vreinterpretq_u32_u8(u3)));
    out[0] = v0;
    out[1] = vextq_u8(v0, v0, 8);
    out[2] = v1;
    out[3] = vextq_u8(v1, v1, 8);
    out[4] = v2;
    out[5] = vextq_u8(v2, v2, 8);
    out[6] = v3;
    out[7] = vextq_u8(v3, v3, 8);
}

static void db_t8w_neon(const uint16x8_t in[8], uint16x8_t out[8])
{
    uint16x8_t t0 = vzip1q_u16(in[0], in[1]);
    uint16x8_t t1 = vzip2q_u16(in[0], in[1]);
    uint16x8_t t2 = vzip1q_u16(in[2], in[3]);
    uint16x8_t t3 = vzip2q_u16(in[2], in[3]);
    uint16x8_t t4 = vzip1q_u16(in[4], in[5]);
    uint16x8_t t5 = vzip2q_u16(in[4], in[5]);
    uint16x8_t t6 = vzip1q_u16(in[6], in[7]);
    uint16x8_t t7 = vzip2q_u16(in[6], in[7]);
    uint16x8_t u0 = vreinterpretq_u16_u32(vzip1q_u32(
        vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2)));
    uint16x8_t u1 = vreinterpretq_u16_u32(vzip2q_u32(
        vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2)));
    uint16x8_t u2 = vreinterpretq_u16_u32(vzip1q_u32(
        vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3)));
    uint16x8_t u3 = vreinterpretq_u16_u32(vzip2q_u32(
        vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3)));
    uint16x8_t u4 = vreinterpretq_u16_u32(vzip1q_u32(
        vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6)));
    uint16x8_t u5 = vreinterpretq_u16_u32(vzip2q_u32(
        vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6)));
    uint16x8_t u6 = vreinterpretq_u16_u32(vzip1q_u32(
        vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7)));
    uint16x8_t u7 = vreinterpretq_u16_u32(vzip2q_u32(
        vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7)));
    out[0] = vreinterpretq_u16_u64(vzip1q_u64(
        vreinterpretq_u64_u16(u0), vreinterpretq_u64_u16(u4)));
    out[1] = vreinterpretq_u16_u64(vzip2q_u64(
        vreinterpretq_u64_u16(u0), vreinterpretq_u64_u16(u4)));
    out[2] = vreinterpretq_u16_u64(vzip1q_u64(
        vreinterpretq_u64_u16(u1), vreinterpretq_u64_u16(u5)));
    out[3] = vreinterpretq_u16_u64(vzip2q_u64(
        vreinterpretq_u64_u16(u1), vreinterpretq_u64_u16(u5)));
    out[4] = vreinterpretq_u16_u64(vzip1q_u64(
        vreinterpretq_u64_u16(u2), vreinterpretq_u64_u16(u6)));
    out[5] = vreinterpretq_u16_u64(vzip2q_u64(
        vreinterpretq_u64_u16(u2), vreinterpretq_u64_u16(u6)));
    out[6] = vreinterpretq_u16_u64(vzip1q_u64(
        vreinterpretq_u64_u16(u3), vreinterpretq_u64_u16(u7)));
    out[7] = vreinterpretq_u16_u64(vzip2q_u64(
        vreinterpretq_u64_u16(u3), vreinterpretq_u64_u16(u7)));
}

H265_INLINE int16x8_t db_clip16_neon(int16x8_t v, int16x8_t lo,
                                            int16x8_t hi)
{
    return vminq_s16(vmaxq_s16(v, lo), hi);
}

static void db_load8_neon(const uint8_t *pix, int xstride, int ystride,
                          int pixel_shift, int16x8_t v[8])
{
    const int step = pixel_shift ? 2 : 1;
    int k;

    if (xstride == step) {
        const uint8_t *p = pix - 4 * xstride;
        if (!pixel_shift) {
            uint8x16_t in[8], out[8];
            for (k = 0; k < 8; k++)
                in[k] = vld1q_u8(p + (ptrdiff_t)k * ystride);
            db_t8b_neon(in, out);
            for (k = 0; k < 8; k++)
                v[k] = vreinterpretq_s16_u16(
                    vmovl_u8(vget_low_u8(out[k])));
        } else {
            uint16x8_t in[8], out[8];
            for (k = 0; k < 8; k++)
                in[k] = vld1q_u16(
                    (const uint16_t *)(p + (ptrdiff_t)k * ystride));
            db_t8w_neon(in, out);
            for (k = 0; k < 8; k++)
                v[k] = vreinterpretq_s16_u16(out[k]);
        }
    } else {
        for (k = 0; k < 8; k++) {
            const uint8_t *q = pix + (ptrdiff_t)(k - 4) * xstride;
            if (!pixel_shift)
                v[k] = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(q)));
            else
                v[k] = vreinterpretq_s16_u16(
                    vld1q_u16((const uint16_t *)q));
        }
    }
}

static void db_store8_neon(uint8_t *pix, int xstride, int ystride,
                           int pixel_shift, const int16x8_t v[8])
{
    const int step = pixel_shift ? 2 : 1;
    int k;

    if (xstride == step) {
        uint8_t *p = pix - 4 * xstride;
        if (!pixel_shift) {
            uint8x16_t in[8], out[8];
            for (k = 0; k < 8; k++)
                in[k] = vcombine_u8(vqmovun_s16(v[k]), vdup_n_u8(0));
            db_t8b_neon(in, out);
            for (k = 0; k < 8; k++)
                vst1_u8(p + (ptrdiff_t)k * ystride, vget_low_u8(out[k]));
        } else {
            uint16x8_t in[8], out[8];
            for (k = 0; k < 8; k++)
                in[k] = vreinterpretq_u16_s16(v[k]);
            db_t8w_neon(in, out);
            for (k = 0; k < 8; k++)
                vst1q_u16((uint16_t *)(p + (ptrdiff_t)k * ystride), out[k]);
        }
    } else {
        for (k = 0; k < 8; k++) {
            uint8_t *q = pix + (ptrdiff_t)(k - 4) * xstride;
            if (!pixel_shift)
                vst1_u8(q, vqmovun_s16(v[k]));
            else
                vst1q_u16((uint16_t *)q, vreinterpretq_u16_s16(v[k]));
        }
    }
}

static void h265_luma_filter8_neon(uint8_t *pix, int xstride, int ystride,
                                   int beta, const int32_t *tc,
                                   const uint8_t *no_p, const uint8_t *no_q,
                                   int pixel_shift, int bit_depth)
{
    const int maxv = (1 << bit_depth) - 1;
    const int bshift = bit_depth - 8;
    const int16x8_t vzero = vdupq_n_s16(0);
    const int16x8_t vmax = vdupq_n_s16((short)maxv);
    const int16x8_t vfour = vdupq_n_s16(4);
    const int16x8_t vtwo = vdupq_n_s16(2);
    const int16x8_t vone = vdupq_n_s16(1);
    int16x8_t v[8];
    int16_t dpv[8], dqv[8], dvv[8], apv[8], pqv[8];
    int16_t tcs[8], nps[8], nqs[8], ndps[8], ndqs[8], m1s[8], m2s[8];
    int mode[2];
    int any1 = 0, any2 = 0;
    int g, l;

    db_load8_neon(pix, xstride, ystride, pixel_shift, v);

    {
        int16x8_t p3 = v[0], p2 = v[1], p1 = v[2], p0 = v[3];
        int16x8_t q0 = v[4], q1 = v[5], q2 = v[6], q3 = v[7];
        int16x8_t dp = vabsq_s16(vaddq_s16(
            vsubq_s16(p2, vshlq_n_s16(p1, 1)), p0));
        int16x8_t dq = vabsq_s16(vaddq_s16(
            vsubq_s16(q2, vshlq_n_s16(q1, 1)), q0));
        int16x8_t dv = vaddq_s16(dp, dq);
        int16x8_t ap = vaddq_s16(vabsq_s16(vsubq_s16(p3, p0)),
                                 vabsq_s16(vsubq_s16(q3, q0)));
        int16x8_t pq = vabsq_s16(vsubq_s16(p0, q0));

        vst1q_s16(dpv, dp);
        vst1q_s16(dqv, dq);
        vst1q_s16(dvv, dv);
        vst1q_s16(apv, ap);
        vst1q_s16(pqv, pq);

        for (g = 0; g < 2; g++) {
            int l0 = g * 4;
            int tcv = tc[g] << bshift;
            int wthr = (beta + (beta >> 1)) >> 3;
            int ndp = dpv[l0] + dpv[l0 + 3] < wthr ? 2 : 1;
            int ndq = dqv[l0] + dqv[l0 + 3] < wthr ? 2 : 1;
            mode[g] = 0;
            if (dvv[l0] + dvv[l0 + 3] < beta) {
                int beta3 = beta >> 3;
                int beta2 = beta >> 2;
                int tc25 = (tcv * 5 + 1) >> 1;
                if (apv[l0] < beta3 && pqv[l0] < tc25 &&
                    apv[l0 + 3] < beta3 && pqv[l0 + 3] < tc25 &&
                    (dvv[l0] << 1) < beta2 && (dvv[l0 + 3] << 1) < beta2)
                    mode[g] = 2;
                else
                    mode[g] = 1;
            }
            for (l = l0; l < l0 + 4; l++) {
                tcs[l] = (int16_t)(mode[g] == 2 ? tcv * 2 : tcv);
                nps[l] = no_p[g] ? -1 : 0;
                nqs[l] = no_q[g] ? -1 : 0;
                ndps[l] = ndp > 1 ? -1 : 0;
                ndqs[l] = ndq > 1 ? -1 : 0;
                m1s[l] = mode[g] == 1 ? -1 : 0;
                m2s[l] = mode[g] == 2 ? -1 : 0;
            }
            if (mode[g] == 1)
                any1 = 1;
            else if (mode[g] == 2)
                any2 = 1;
        }

        if (!any1 && !any2)
            return;

        if (any1) {
            int16x8_t tcv16 = vld1q_s16(tcs);
            int16x8_t tc2v = vshrq_n_s16(tcv16, 1);
            int16x8_t npv = vld1q_s16(nps);
            int16x8_t nqv = vld1q_s16(nqs);
            int16x8_t ndpv = vld1q_s16(ndps);
            int16x8_t ndqv = vld1q_s16(ndqs);
            int16x8_t m1 = vld1q_s16(m1s);
            int16x8_t ntc = vnegq_s16(tcv16);
            int16x8_t ntc2 = vnegq_s16(tc2v);
            int16x8_t x0 = vsubq_s16(q0, p0);
            int16x8_t x1 = vsubq_s16(q1, p1);
            int16x8_t nine = vaddq_s16(vshlq_n_s16(x0, 3), x0);
            int16x8_t three = vaddq_s16(vshlq_n_s16(x1, 1), x1);
            int16x8_t delta = vshrq_n_s16(vaddq_s16(
                vsubq_s16(nine, three), vdupq_n_s16(8)), 4);
            int16x8_t ten = vmulq_s16(tcv16, vdupq_n_s16(10));
            uint16x8_t valid = vcgtq_s16(ten, vabsq_s16(delta));
            uint16x8_t pmask = vbicq_u16(valid, vreinterpretq_u16_s16(npv));
            uint16x8_t qmask = vbicq_u16(valid, vreinterpretq_u16_s16(nqv));
            int16x8_t np0, np1, nq0, nq1;

            delta = db_clip16_neon(delta, ntc, tcv16);
            np0 = db_clip16_neon(vaddq_s16(p0, delta), vzero, vmax);
            nq0 = db_clip16_neon(vsubq_s16(q0, delta), vzero, vmax);
            np1 = vshrq_n_s16(vaddq_s16(vsubq_s16(vshrq_n_s16(
                vaddq_s16(vaddq_s16(p2, p0), vone), 1), p1), delta), 1);
            nq1 = vshrq_n_s16(vaddq_s16(vsubq_s16(vshrq_n_s16(
                vaddq_s16(vaddq_s16(q2, q0), vone), 1), q1),
                vnegq_s16(delta)), 1);
            np1 = db_clip16_neon(np1, ntc2, tc2v);
            nq1 = db_clip16_neon(nq1, ntc2, tc2v);
            np1 = db_clip16_neon(vaddq_s16(p1, np1), vzero, vmax);
            nq1 = db_clip16_neon(vaddq_s16(q1, nq1), vzero, vmax);
            np1 = vbslq_s16(vreinterpretq_u16_s16(ndpv), np1, p1);
            nq1 = vbslq_s16(vreinterpretq_u16_s16(ndqv), nq1, q1);
            pmask = vandq_u16(pmask, vreinterpretq_u16_s16(m1));
            qmask = vandq_u16(qmask, vreinterpretq_u16_s16(m1));
            v[2] = vbslq_s16(pmask, np1, v[2]);
            v[3] = vbslq_s16(pmask, np0, v[3]);
            v[4] = vbslq_s16(qmask, nq0, v[4]);
            v[5] = vbslq_s16(qmask, nq1, v[5]);
        }

        if (any2) {
            int16x8_t tc3 = vld1q_s16(tcs);
            int16x8_t npv = vld1q_s16(nps);
            int16x8_t nqv = vld1q_s16(nqs);
            int16x8_t m2 = vld1q_s16(m2s);
            int16x8_t ntc = vnegq_s16(tc3);
            int16x8_t np0, np1, np2, nq0, nq1, nq2;

            np0 = vshrq_n_s16(vaddq_s16(vaddq_s16(
                vaddq_s16(p2, vshlq_n_s16(p1, 1)),
                vaddq_s16(vshlq_n_s16(p0, 1), vshlq_n_s16(q0, 1))),
                vaddq_s16(q1, vfour)), 3);
            np1 = vshrq_n_s16(vaddq_s16(vaddq_s16(
                vaddq_s16(p2, p1), vaddq_s16(p0, q0)), vtwo), 2);
            np2 = vshrq_n_s16(vaddq_s16(vaddq_s16(
                vaddq_s16(vshlq_n_s16(p3, 1),
                          vaddq_s16(vshlq_n_s16(p2, 1), p2)),
                vaddq_s16(p1, vaddq_s16(p0, q0))), vfour), 3);
            np0 = vaddq_s16(p0,
                db_clip16_neon(vsubq_s16(np0, p0), ntc, tc3));
            np1 = vaddq_s16(p1,
                db_clip16_neon(vsubq_s16(np1, p1), ntc, tc3));
            np2 = vaddq_s16(p2,
                db_clip16_neon(vsubq_s16(np2, p2), ntc, tc3));

            nq0 = vshrq_n_s16(vaddq_s16(vaddq_s16(
                vaddq_s16(q2, vshlq_n_s16(q1, 1)),
                vaddq_s16(vshlq_n_s16(q0, 1), vshlq_n_s16(p0, 1))),
                vaddq_s16(p1, vfour)), 3);
            nq1 = vshrq_n_s16(vaddq_s16(vaddq_s16(
                vaddq_s16(q2, q1), vaddq_s16(q0, p0)), vtwo), 2);
            nq2 = vshrq_n_s16(vaddq_s16(vaddq_s16(
                vaddq_s16(vshlq_n_s16(q3, 1),
                          vaddq_s16(vshlq_n_s16(q2, 1), q2)),
                vaddq_s16(q1, vaddq_s16(q0, p0))), vfour), 3);
            nq0 = vaddq_s16(q0,
                db_clip16_neon(vsubq_s16(nq0, q0), ntc, tc3));
            nq1 = vaddq_s16(q1,
                db_clip16_neon(vsubq_s16(nq1, q1), ntc, tc3));
            nq2 = vaddq_s16(q2,
                db_clip16_neon(vsubq_s16(nq2, q2), ntc, tc3));

            np2 = vbslq_s16(vreinterpretq_u16_s16(npv), p2, np2);
            np1 = vbslq_s16(vreinterpretq_u16_s16(npv), p1, np1);
            np0 = vbslq_s16(vreinterpretq_u16_s16(npv), p0, np0);
            nq0 = vbslq_s16(vreinterpretq_u16_s16(nqv), q0, nq0);
            nq1 = vbslq_s16(vreinterpretq_u16_s16(nqv), q1, nq1);
            nq2 = vbslq_s16(vreinterpretq_u16_s16(nqv), q2, nq2);
            v[1] = vbslq_s16(vreinterpretq_u16_s16(m2), np2, v[1]);
            v[2] = vbslq_s16(vreinterpretq_u16_s16(m2), np1, v[2]);
            v[3] = vbslq_s16(vreinterpretq_u16_s16(m2), np0, v[3]);
            v[4] = vbslq_s16(vreinterpretq_u16_s16(m2), nq0, v[4]);
            v[5] = vbslq_s16(vreinterpretq_u16_s16(m2), nq1, v[5]);
            v[6] = vbslq_s16(vreinterpretq_u16_s16(m2), nq2, v[6]);
        }
    }

    db_store8_neon(pix, xstride, ystride, pixel_shift, v);
}

static void h265_chroma_filter_neon(uint8_t *pix, int xstride, int ystride,
                                    const int32_t *tc, const uint8_t *no_p,
                                    const uint8_t *no_q, int pixel_shift,
                                    int bit_depth)
{
    const int maxv = (1 << bit_depth) - 1;
    int16x8_t tcv;
    int16x8_t v[8];
    int16x8_t p1, p0, q0, q1, delta, np0, nq0;
    int16x8_t npv, nqv;

    if (!tc[0] && !tc[1])
        return;
    tcv = vcombine_s16(vdup_n_s16((short)(tc[0] << (bit_depth - 8))),
                       vdup_n_s16((short)(tc[1] << (bit_depth - 8))));
    db_load8_neon(pix, xstride, ystride, pixel_shift, v);
    p1 = v[2]; p0 = v[3]; q0 = v[4]; q1 = v[5];

    delta = vshrq_n_s16(vaddq_s16(vaddq_s16(
        vshlq_n_s16(vsubq_s16(q0, p0), 2), vsubq_s16(p1, q1)),
        vdupq_n_s16(4)), 3);
    delta = db_clip16_neon(delta, vnegq_s16(tcv), tcv);
    np0 = db_clip16_neon(vaddq_s16(p0, delta), vdupq_n_s16(0),
                         vdupq_n_s16((short)maxv));
    nq0 = db_clip16_neon(vsubq_s16(q0, delta), vdupq_n_s16(0),
                         vdupq_n_s16((short)maxv));
    npv = vcombine_s16(vdup_n_s16(no_p[0] ? -1 : 0),
                       vdup_n_s16(no_p[1] ? -1 : 0));
    nqv = vcombine_s16(vdup_n_s16(no_q[0] ? -1 : 0),
                       vdup_n_s16(no_q[1] ? -1 : 0));
    v[3] = vbslq_s16(vreinterpretq_u16_s16(npv), v[3], np0);
    v[4] = vbslq_s16(vreinterpretq_u16_s16(nqv), v[4], nq0);
    db_store8_neon(pix, xstride, ystride, pixel_shift, v);
}
#endif /* H265_NEON */

static void hevc_loop_filter_luma(uint8_t *pix, int xstride, int ystride,
                                  int beta, const int32_t *tc,
                                  const uint8_t *no_p, const uint8_t *no_q,
                                  int pixel_shift, int bit_depth)
{
    int j;
    int maxv = (1 << bit_depth) - 1;
    beta <<= bit_depth - 8;

#ifdef H265_X86
    if (h265_have_avx2 && bit_depth <= 10) {
        db_luma_filter8(pix, xstride, ystride, beta, tc, no_p, no_q,
                        pixel_shift, bit_depth);
        return;
    }
#endif
#ifdef H265_NEON
    if (h265_have_neon && bit_depth <= 10) {
        h265_luma_filter8_neon(pix, xstride, ystride, beta, tc, no_p, no_q,
                               pixel_shift, bit_depth);
        return;
    }
#endif

    for (j = 0; j < 2; j++) {
        uint8_t *p = pix + j * 4 * ystride;
        int dp0 = H265_ABS(GETP(p, -3 * xstride) - 2 * GETP(p, -2 * xstride) + GETP(p, -1 * xstride));
        int dq0 = H265_ABS(GETP(p, 2 * xstride) - 2 * GETP(p, 1 * xstride) + GETP(p, 0 * xstride));
        int dp3 = H265_ABS(GETP(p, -3 * xstride + 3 * ystride) - 2 * GETP(p, -2 * xstride + 3 * ystride) + GETP(p, -1 * xstride + 3 * ystride));
        int dq3 = H265_ABS(GETP(p, 2 * xstride + 3 * ystride) - 2 * GETP(p, 1 * xstride + 3 * ystride) + GETP(p, 0 * xstride + 3 * ystride));
        int d0 = dp0 + dq0;
        int d3 = dp3 + dq3;
        int tcv = tc[j] << (bit_depth - 8);
        int np = no_p[j];
        int nq = no_q[j];

        if (d0 + d3 < beta) {
            int beta_3 = beta >> 3;
            int beta_2 = beta >> 2;
            int tc25 = ((tcv * 5 + 1) >> 1);
            if (H265_ABS(GETP(p, -4 * xstride) - GETP(p, -1 * xstride)) +
                    H265_ABS(GETP(p, 3 * xstride) - GETP(p, 0 * xstride)) < beta_3 &&
                H265_ABS(GETP(p, -1 * xstride) - GETP(p, 0 * xstride)) < tc25 &&
                H265_ABS(GETP(p, -4 * xstride + 3 * ystride) - GETP(p, -1 * xstride + 3 * ystride)) +
                    H265_ABS(GETP(p, 3 * xstride + 3 * ystride) - GETP(p, 0 * xstride + 3 * ystride)) < beta_3 &&
                H265_ABS(GETP(p, -1 * xstride + 3 * ystride) - GETP(p, 0 * xstride + 3 * ystride)) < tc25 &&
                (d0 << 1) < beta_2 && (d3 << 1) < beta_2) {
                int tc2 = tcv << 1;
#ifdef H265_X86
                if (h265_have_avx2)
                    db_luma_strong_avx2(p, xstride, ystride, tc2, tc2, tc2,
                                        np, nq, pixel_shift, maxv);
                else
#endif
                loop_filter_luma_strong(p, xstride, ystride, tc2, tc2, tc2,
                                        np, nq, pixel_shift, maxv);
            } else {
                int nd_p = 1, nd_q = 1;
                if (dp0 + dp3 < ((beta + (beta >> 1)) >> 3))
                    nd_p = 2;
                if (dq0 + dq3 < ((beta + (beta >> 1)) >> 3))
                    nd_q = 2;
#ifdef H265_X86
                if (h265_have_avx2)
                    db_luma_weak_avx2(p, xstride, ystride, tcv, np, nq,
                                      nd_p, nd_q, pixel_shift, maxv);
                else
#endif
                loop_filter_luma_weak(p, xstride, ystride, tcv, beta, np, nq,
                                      nd_p, nd_q, pixel_shift, maxv);
            }
        }
    }
}

static void hevc_loop_filter_chroma(uint8_t *pix, int xstride, int ystride,
                                    const int32_t *tc, const uint8_t *no_p,
                                    const uint8_t *no_q, int pixel_shift,
                                    int bit_depth)
{
    int j;
    int maxv = (1 << bit_depth) - 1;

#ifdef H265_NEON
    if (h265_have_neon) {
        h265_chroma_filter_neon(pix, xstride, ystride, tc, no_p, no_q,
                                pixel_shift, bit_depth);
        return;
    }
#endif
    for (j = 0; j < 2; j++) {
        uint8_t *p = pix + j * 4 * ystride;
        int tcv = tc[j] << (bit_depth - 8);
        if (tcv > 0) {
            int no_pv = no_p[j];
            int no_qv = no_q[j];
            int d;
#ifdef H265_X86
            if (h265_have_avx2) {
                db_chroma_avx2(p, xstride, ystride, tcv, no_pv, no_qv,
                               pixel_shift, maxv);
                continue;
            }
#endif
            for (d = 0; d < 4; d++) {
                int p1 = GETP(p, -2 * xstride);
                int p0 = GETP(p, -1 * xstride);
                int q0 = GETP(p, 0 * xstride);
                int q1 = GETP(p, 1 * xstride);
                int delta0 = clip3((((q0 - p0) * 4) + p1 - q1 + 4) >> 3, -tcv, tcv);
                if (!no_pv)
                    SETP(p, -1 * xstride, clip3(p0 + delta0, 0, maxv));
                if (!no_qv)
                    SETP(p, 0 * xstride, clip3(q0 - delta0, 0, maxv));
                p += ystride;
            }
        }
    }
}

void h265_hevc_h_loop_filter_luma(uint8_t *pix, ptrdiff_t stride, int beta,
                                  const int32_t *tc, const uint8_t *no_p,
                                  const uint8_t *no_q, int pixel_shift, int bit_depth)
{
    hevc_loop_filter_luma(pix, (int)stride, pixel_shift ? 2 : 1, beta, tc,
                          no_p, no_q, pixel_shift, bit_depth);
}

void h265_hevc_v_loop_filter_luma(uint8_t *pix, ptrdiff_t stride, int beta,
                                  const int32_t *tc, const uint8_t *no_p,
                                  const uint8_t *no_q, int pixel_shift, int bit_depth)
{
    hevc_loop_filter_luma(pix, pixel_shift ? 2 : 1, (int)stride, beta, tc,
                          no_p, no_q, pixel_shift, bit_depth);
}

void h265_hevc_h_loop_filter_chroma(uint8_t *pix, ptrdiff_t stride,
                                    const int32_t *tc, const uint8_t *no_p,
                                    const uint8_t *no_q, int pixel_shift, int bit_depth)
{
    hevc_loop_filter_chroma(pix, (int)stride, pixel_shift ? 2 : 1, tc,
                            no_p, no_q, pixel_shift, bit_depth);
}

void h265_hevc_v_loop_filter_chroma(uint8_t *pix, ptrdiff_t stride,
                                    const int32_t *tc, const uint8_t *no_p,
                                    const uint8_t *no_q, int pixel_shift, int bit_depth)
{
    hevc_loop_filter_chroma(pix, pixel_shift ? 2 : 1, (int)stride, tc,
                            no_p, no_q, pixel_shift, bit_depth);
}

/* ------------------------------------------------------------------ */
/* boundary strengths                                                  */
/* ------------------------------------------------------------------ */
H265_INLINE const H265RefList *frame_ref_list(const H265Frame *f,
                                                     const H265PPS *pps,
                                                     int x0, int y0)
{
    const H265SPS *sps = f->sps;
    int x_cb = x0 >> sps->log2_ctb_size;
    int y_cb = y0 >> sps->log2_ctb_size;
    int ctb_addr_ts = pps->ctb_addr_rs_to_ts[y_cb * sps->ctb_width + x_cb];
    return &f->rpl_tab[ctb_addr_ts].refPicList[0];
}

static
#if defined(__GNUC__)
__attribute__((always_inline))
#endif
inline int boundary_strength(H265Slice *sh, const H265MvField *curr,
                             const H265MvField *neigh,
                             const H265RefList *neigh_refPicList)
{
    const H265RefList *cur = sh->ref_list;

    if (curr->pred_flag == H265_PF_BI && neigh->pred_flag == H265_PF_BI) {
        if (cur[0].list[curr->ref_idx[0]] == neigh_refPicList[0].list[neigh->ref_idx[0]] &&
            cur[0].list[curr->ref_idx[0]] == cur[1].list[curr->ref_idx[1]] &&
            neigh_refPicList[0].list[neigh->ref_idx[0]] == neigh_refPicList[1].list[neigh->ref_idx[1]]) {
            if ((H265_ABS(neigh->mv[0].x - curr->mv[0].x) >= 4 ||
                 H265_ABS(neigh->mv[0].y - curr->mv[0].y) >= 4 ||
                 H265_ABS(neigh->mv[1].x - curr->mv[1].x) >= 4 ||
                 H265_ABS(neigh->mv[1].y - curr->mv[1].y) >= 4) &&
                (H265_ABS(neigh->mv[1].x - curr->mv[0].x) >= 4 ||
                 H265_ABS(neigh->mv[1].y - curr->mv[0].y) >= 4 ||
                 H265_ABS(neigh->mv[0].x - curr->mv[1].x) >= 4 ||
                 H265_ABS(neigh->mv[0].y - curr->mv[1].y) >= 4))
                return 1;
            return 0;
        } else if (neigh_refPicList[0].list[neigh->ref_idx[0]] == cur[0].list[curr->ref_idx[0]] &&
                   neigh_refPicList[1].list[neigh->ref_idx[1]] == cur[1].list[curr->ref_idx[1]]) {
            if (H265_ABS(neigh->mv[0].x - curr->mv[0].x) >= 4 ||
                H265_ABS(neigh->mv[0].y - curr->mv[0].y) >= 4 ||
                H265_ABS(neigh->mv[1].x - curr->mv[1].x) >= 4 ||
                H265_ABS(neigh->mv[1].y - curr->mv[1].y) >= 4)
                return 1;
            return 0;
        } else if (neigh_refPicList[1].list[neigh->ref_idx[1]] == cur[0].list[curr->ref_idx[0]] &&
                   neigh_refPicList[0].list[neigh->ref_idx[0]] == cur[1].list[curr->ref_idx[1]]) {
            if (H265_ABS(neigh->mv[1].x - curr->mv[0].x) >= 4 ||
                H265_ABS(neigh->mv[1].y - curr->mv[0].y) >= 4 ||
                H265_ABS(neigh->mv[0].x - curr->mv[1].x) >= 4 ||
                H265_ABS(neigh->mv[0].y - curr->mv[1].y) >= 4)
                return 1;
            return 0;
        }
        return 1;
    } else if (curr->pred_flag != H265_PF_BI && neigh->pred_flag != H265_PF_BI) {
        H265Mv A, B;
        int ref_A, ref_B;
        if (curr->pred_flag & 1) {
            A = curr->mv[0];
            ref_A = cur[0].list[curr->ref_idx[0]];
        } else {
            A = curr->mv[1];
            ref_A = cur[1].list[curr->ref_idx[1]];
        }
        if (neigh->pred_flag & 1) {
            B = neigh->mv[0];
            ref_B = neigh_refPicList[0].list[neigh->ref_idx[0]];
        } else {
            B = neigh->mv[1];
            ref_B = neigh_refPicList[1].list[neigh->ref_idx[1]];
        }
        if (ref_A == ref_B) {
            if (H265_ABS(A.x - B.x) >= 4 || H265_ABS(A.y - B.y) >= 4)
                return 1;
            return 0;
        }
        return 1;
    }
    return 1;
}

void h265_deblocking_boundary_strengths(H265Slice *sh, int x0, int y0,
                                        int log2_trafo_size)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    const H265MvField *tab_mvf = f->tab_mvf;
    int log2_min_pu_size = sps->log2_min_pu_size;
    int log2_min_tu_size = sps->log2_min_tb_size;
    int min_pu_width = sps->min_pu_width;
    int min_tu_width = sps->min_tb_width;
    int bs_width = (sps->width >> 2) + 1;
    int is_intra = tab_mvf[(y0 >> log2_min_pu_size) * min_pu_width +
                           (x0 >> log2_min_pu_size)].pred_flag == H265_PF_INTRA;
    int boundary_upper, boundary_left;
    int i, j, bs;

    boundary_upper = y0 > 0 && !(y0 & 7);
    if (boundary_upper &&
        ((!sh->slice_loop_filter_across_slices_enabled_flag &&
          sh->boundary_flags & H265_BOUNDARY_UPPER_SLICE &&
          (y0 % (1 << sps->log2_ctb_size)) == 0) ||
         (!pps->loop_filter_across_tiles_enabled_flag &&
          sh->boundary_flags & H265_BOUNDARY_UPPER_TILE &&
          (y0 % (1 << sps->log2_ctb_size)) == 0)))
        boundary_upper = 0;

    if (boundary_upper) {
        const H265RefList *rpl_top = (sh->boundary_flags & H265_BOUNDARY_UPPER_SLICE) ?
                                     frame_ref_list(f, pps, x0, y0 - 1) : sh->ref_list;
        int yp_pu = (y0 - 1) >> log2_min_pu_size;
        int yq_pu = y0 >> log2_min_pu_size;
        int yp_tu = (y0 - 1) >> log2_min_tu_size;
        int yq_tu = y0 >> log2_min_tu_size;

        for (i = 0; i < (1 << log2_trafo_size); i += 4) {
            int x_pu = (x0 + i) >> log2_min_pu_size;
            int x_tu = (x0 + i) >> log2_min_tu_size;
            const H265MvField *top = &tab_mvf[yp_pu * min_pu_width + x_pu];
            const H265MvField *curr = &tab_mvf[yq_pu * min_pu_width + x_pu];
            uint8_t top_cbf_luma = f->cbf_luma[yp_tu * min_tu_width + x_tu];
            uint8_t curr_cbf_luma = f->cbf_luma[yq_tu * min_tu_width + x_tu];

            if (curr->pred_flag == H265_PF_INTRA || top->pred_flag == H265_PF_INTRA)
                bs = 2;
            else if (curr_cbf_luma || top_cbf_luma)
                bs = 1;
            else
                bs = boundary_strength(sh, curr, top, rpl_top);
            f->horizontal_bs[((x0 + i) + y0 * bs_width) >> 2] = (uint8_t)bs;
        }
    }

    boundary_left = x0 > 0 && !(x0 & 7);
    if (boundary_left &&
        ((!sh->slice_loop_filter_across_slices_enabled_flag &&
          sh->boundary_flags & H265_BOUNDARY_LEFT_SLICE &&
          (x0 % (1 << sps->log2_ctb_size)) == 0) ||
         (!pps->loop_filter_across_tiles_enabled_flag &&
          sh->boundary_flags & H265_BOUNDARY_LEFT_TILE &&
          (x0 % (1 << sps->log2_ctb_size)) == 0)))
        boundary_left = 0;

    if (boundary_left) {
        const H265RefList *rpl_left = (sh->boundary_flags & H265_BOUNDARY_LEFT_SLICE) ?
                                      frame_ref_list(f, pps, x0 - 1, y0) : sh->ref_list;
        int xp_pu = (x0 - 1) >> log2_min_pu_size;
        int xq_pu = x0 >> log2_min_pu_size;
        int xp_tu = (x0 - 1) >> log2_min_tu_size;
        int xq_tu = x0 >> log2_min_tu_size;

        for (i = 0; i < (1 << log2_trafo_size); i += 4) {
            int y_pu = (y0 + i) >> log2_min_pu_size;
            int y_tu = (y0 + i) >> log2_min_tu_size;
            const H265MvField *left = &tab_mvf[y_pu * min_pu_width + xp_pu];
            const H265MvField *curr = &tab_mvf[y_pu * min_pu_width + xq_pu];
            uint8_t left_cbf_luma = f->cbf_luma[y_tu * min_tu_width + xp_tu];
            uint8_t curr_cbf_luma = f->cbf_luma[y_tu * min_tu_width + xq_tu];

            if (curr->pred_flag == H265_PF_INTRA || left->pred_flag == H265_PF_INTRA)
                bs = 2;
            else if (curr_cbf_luma || left_cbf_luma)
                bs = 1;
            else
                bs = boundary_strength(sh, curr, left, rpl_left);
            f->vertical_bs[(x0 + (y0 + i) * bs_width) >> 2] = (uint8_t)bs;
        }
    }

    if (log2_trafo_size > log2_min_pu_size && !is_intra) {
        for (j = 8; j < (1 << log2_trafo_size); j += 8) {
            int yp_pu = (y0 + j - 1) >> log2_min_pu_size;
            int yq_pu = (y0 + j) >> log2_min_pu_size;
            for (i = 0; i < (1 << log2_trafo_size); i += 4) {
                int x_pu = (x0 + i) >> log2_min_pu_size;
                const H265MvField *top = &tab_mvf[yp_pu * min_pu_width + x_pu];
                const H265MvField *curr = &tab_mvf[yq_pu * min_pu_width + x_pu];
                bs = boundary_strength(sh, curr, top, sh->ref_list);
                f->horizontal_bs[((x0 + i) + (y0 + j) * bs_width) >> 2] = (uint8_t)bs;
            }
        }
        for (j = 0; j < (1 << log2_trafo_size); j += 4) {
            int y_pu = (y0 + j) >> log2_min_pu_size;
            for (i = 8; i < (1 << log2_trafo_size); i += 8) {
                int xp_pu = (x0 + i - 1) >> log2_min_pu_size;
                int xq_pu = (x0 + i) >> log2_min_pu_size;
                const H265MvField *left = &tab_mvf[y_pu * min_pu_width + xp_pu];
                const H265MvField *curr = &tab_mvf[y_pu * min_pu_width + xq_pu];
                bs = boundary_strength(sh, curr, left, sh->ref_list);
                f->vertical_bs[((x0 + i) + (y0 + j) * bs_width) >> 2] = (uint8_t)bs;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* whole-frame deblocking                                              */
/* ------------------------------------------------------------------ */
static int get_pcm(H265SPS *sps, const uint8_t *is_pcm, int x, int y)
{
    int log2_min_pu_size = sps->log2_min_pu_size;
    int x_pu, y_pu;

    if (x < 0 || y < 0)
        return 2;
    x_pu = x >> log2_min_pu_size;
    y_pu = y >> log2_min_pu_size;
    if (x_pu >= sps->min_pu_width || y_pu >= sps->min_pu_height)
        return 2;
    return is_pcm[y_pu * sps->min_pu_width + x_pu];
}

#define TC_CALC(qp, bs, tc_offset) \
    tctable[H265_CLIP((qp) + H265_DEFAULT_INTRA_TC_OFFSET * ((bs) - 1) + \
                      ((tc_offset) & -2), 0, H265_MAX_QP + H265_DEFAULT_INTRA_TC_OFFSET)]

static void deblocking_filter_CTB(H265Slice *sh, int x0, int y0)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    H265Frame *f = sh->frame;
    int pixel_shift = sps->pixel_shift;
    int bit_depth = sps->bit_depth;
    uint8_t *src;
    int x, y;
    int chroma, beta;
    int32_t c_tc[2], tc[2];
    uint8_t no_p[2] = { 0 };
    uint8_t no_q[2] = { 0 };
    int log2_ctb_size = sps->log2_ctb_size;
    int x_end, x_end2, y_end;
    int ctb_size = 1 << log2_ctb_size;
    int ctb = (x0 >> log2_ctb_size) + (y0 >> log2_ctb_size) * sps->ctb_width;
    int bs_width = (sps->width >> 2) + 1;
    int cur_tc_offset = f->deblock[ctb].tc_offset;
    int cur_beta_offset = f->deblock[ctb].beta_offset;
    int left_tc_offset, left_beta_offset;
    int tc_offset = 0, beta_offset = 0;
    int pcmf = (sps->pcm_enabled && sps->pcm_loop_filter_disabled) ||
               pps->transquant_bypass_enable_flag;

    if (f->deblock[ctb].disabled)
        return;

    if (x0) {
        left_tc_offset = f->deblock[ctb - 1].tc_offset;
        left_beta_offset = f->deblock[ctb - 1].beta_offset;
    } else {
        left_tc_offset = 0;
        left_beta_offset = 0;
    }

    x_end = x0 + ctb_size;
    if (x_end > sps->width)
        x_end = sps->width;
    y_end = y0 + ctb_size;
    if (y_end > sps->height)
        y_end = sps->height;

    tc_offset = cur_tc_offset;
    beta_offset = cur_beta_offset;

    x_end2 = x_end;
    if (x_end2 != sps->width)
        x_end2 -= 8;
    for (y = y0; y < y_end; y += 8) {
        for (x = x0 ? x0 : 8; x < x_end; x += 8) {
            const int bs0 = f->vertical_bs[(x + y * bs_width) >> 2];
            const int bs1 = f->vertical_bs[(x + (y + 4) * bs_width) >> 2];
            if (!bs0 && !bs1)
                continue;
#ifdef H265_X86
            if (h265_have_avx2 && bit_depth <= 10 && x + 8 < x_end) {
                const int xb = x + 8;
                const int bs2 = f->vertical_bs[(xb + y * bs_width) >> 2];
                const int bs3 = f->vertical_bs[(xb + (y + 4) * bs_width) >> 2];
                if (bs2 || bs3) {
                    const int qp0 = (get_qPy(sps, f->qp_y_tab, x - 1, y) +
                                     get_qPy(sps, f->qp_y_tab, x, y) + 1) >> 1;
                    const int qp1 = (get_qPy(sps, f->qp_y_tab, xb - 1, y) +
                                     get_qPy(sps, f->qp_y_tab, xb, y) + 1) >> 1;
                    const int b0 = betatable[H265_CLIP(qp0 + beta_offset, 0,
                                                      H265_MAX_QP)];
                    const int b1 = betatable[H265_CLIP(qp1 + beta_offset, 0,
                                                      H265_MAX_QP)];
                    int32_t tc0[2], tc1[2];
                    uint8_t np0[2] = { 0 }, nq0[2] = { 0 };
                    uint8_t np1[2] = { 0 }, nq1[2] = { 0 };
                    tc0[0] = bs0 ? TC_CALC(qp0, bs0, tc_offset) : 0;
                    tc0[1] = bs1 ? TC_CALC(qp0, bs1, tc_offset) : 0;
                    tc1[0] = bs2 ? TC_CALC(qp1, bs2, tc_offset) : 0;
                    tc1[1] = bs3 ? TC_CALC(qp1, bs3, tc_offset) : 0;
                    if (pcmf) {
                        np0[0] = get_pcm(sps, f->is_pcm, x - 1, y);
                        np0[1] = get_pcm(sps, f->is_pcm, x - 1, y + 4);
                        nq0[0] = get_pcm(sps, f->is_pcm, x, y);
                        nq0[1] = get_pcm(sps, f->is_pcm, x, y + 4);
                        np1[0] = get_pcm(sps, f->is_pcm, xb - 1, y);
                        np1[1] = get_pcm(sps, f->is_pcm, xb - 1, y + 4);
                        nq1[0] = get_pcm(sps, f->is_pcm, xb, y);
                        nq1[1] = get_pcm(sps, f->is_pcm, xb, y + 4);
                    }
                    src = &f->data[0][y * f->linesize[0] + (x << pixel_shift)];
                    g_lf_x = x; g_lf_y = y;
                    db_luma_filter16_v(src, f->linesize[0], b0, tc0, np0, nq0,
                                       b1, tc1, np1, nq1, pixel_shift,
                                       bit_depth);
                    x += 8;
                    continue;
                }
            }
#endif
            {
                const int qp = (get_qPy(sps, f->qp_y_tab, x - 1, y) +
                                get_qPy(sps, f->qp_y_tab, x, y) + 1) >> 1;
                beta = betatable[H265_CLIP(qp + beta_offset, 0, H265_MAX_QP)];
                tc[0] = bs0 ? TC_CALC(qp, bs0, tc_offset) : 0;
                tc[1] = bs1 ? TC_CALC(qp, bs1, tc_offset) : 0;
                src = &f->data[0][y * f->linesize[0] + (x << pixel_shift)];
                g_lf_x = x; g_lf_y = y;
                if (pcmf) {
                    no_p[0] = get_pcm(sps, f->is_pcm, x - 1, y);
                    no_p[1] = get_pcm(sps, f->is_pcm, x - 1, y + 4);
                    no_q[0] = get_pcm(sps, f->is_pcm, x, y);
                    no_q[1] = get_pcm(sps, f->is_pcm, x, y + 4);
                }
                h265_hevc_v_loop_filter_luma(src, f->linesize[0], beta, tc,
                                             no_p, no_q, pixel_shift, bit_depth);
            }
        }
        if (!y)
            continue;

        for (x = x0 ? x0 - 8 : 0; x < x_end2; x += 8) {
            const int bs0 = f->horizontal_bs[(x + y * bs_width) >> 2];
            const int bs1 = f->horizontal_bs[((x + 4) + y * bs_width) >> 2];
            if (!bs0 && !bs1)
                continue;
#ifdef H265_X86
            if (h265_have_avx2 && bit_depth <= 10 && x + 8 < x_end2) {
                const int xb = x + 8;
                const int bs2 = f->horizontal_bs[(xb + y * bs_width) >> 2];
                const int bs3 = f->horizontal_bs[((xb + 4) + y * bs_width) >> 2];
                if (bs2 || bs3) {
                    const int qp0 = (get_qPy(sps, f->qp_y_tab, x, y - 1) +
                                     get_qPy(sps, f->qp_y_tab, x, y) + 1) >> 1;
                    const int qp1 = (get_qPy(sps, f->qp_y_tab, xb, y - 1) +
                                     get_qPy(sps, f->qp_y_tab, xb, y) + 1) >> 1;
                    const int toff0 = x >= x0 ? cur_tc_offset : left_tc_offset;
                    const int boff0 = x >= x0 ? cur_beta_offset : left_beta_offset;
                    const int toff1 = xb >= x0 ? cur_tc_offset : left_tc_offset;
                    const int boff1 = xb >= x0 ? cur_beta_offset : left_beta_offset;
                    const int b0 = betatable[H265_CLIP(qp0 + boff0, 0, H265_MAX_QP)];
                    const int b1 = betatable[H265_CLIP(qp1 + boff1, 0, H265_MAX_QP)];
                    int32_t tc0[2], tc1[2];
                    uint8_t np0[2] = { 0 }, nq0[2] = { 0 };
                    uint8_t np1[2] = { 0 }, nq1[2] = { 0 };
                    tc0[0] = bs0 ? TC_CALC(qp0, bs0, toff0) : 0;
                    tc0[1] = bs1 ? TC_CALC(qp0, bs1, toff0) : 0;
                    tc1[0] = bs2 ? TC_CALC(qp1, bs2, toff1) : 0;
                    tc1[1] = bs3 ? TC_CALC(qp1, bs3, toff1) : 0;
                    if (pcmf) {
                        np0[0] = get_pcm(sps, f->is_pcm, x, y - 1);
                        np0[1] = get_pcm(sps, f->is_pcm, x + 4, y - 1);
                        nq0[0] = get_pcm(sps, f->is_pcm, x, y);
                        nq0[1] = get_pcm(sps, f->is_pcm, x + 4, y);
                        np1[0] = get_pcm(sps, f->is_pcm, xb, y - 1);
                        np1[1] = get_pcm(sps, f->is_pcm, xb + 4, y - 1);
                        nq1[0] = get_pcm(sps, f->is_pcm, xb, y);
                        nq1[1] = get_pcm(sps, f->is_pcm, xb + 4, y);
                    }
                    src = &f->data[0][y * f->linesize[0] + (x << pixel_shift)];
                    g_lf_x = x; g_lf_y = y;
                    db_luma_filter16_h(src, f->linesize[0], b0, tc0, np0, nq0,
                                       b1, tc1, np1, nq1, pixel_shift,
                                       bit_depth);
                    x += 8;
                    continue;
                }
            }
#endif
            {
                const int qp = (get_qPy(sps, f->qp_y_tab, x, y - 1) +
                                get_qPy(sps, f->qp_y_tab, x, y) + 1) >> 1;
                tc_offset = x >= x0 ? cur_tc_offset : left_tc_offset;
                beta_offset = x >= x0 ? cur_beta_offset : left_beta_offset;
                beta = betatable[H265_CLIP(qp + beta_offset, 0, H265_MAX_QP)];
                tc[0] = bs0 ? TC_CALC(qp, bs0, tc_offset) : 0;
                tc[1] = bs1 ? TC_CALC(qp, bs1, tc_offset) : 0;
                src = &f->data[0][y * f->linesize[0] + (x << pixel_shift)];
                g_lf_x = x; g_lf_y = y;
                if (pcmf) {
                    no_p[0] = get_pcm(sps, f->is_pcm, x, y - 1);
                    no_p[1] = get_pcm(sps, f->is_pcm, x + 4, y - 1);
                    no_q[0] = get_pcm(sps, f->is_pcm, x, y);
                    no_q[1] = get_pcm(sps, f->is_pcm, x + 4, y);
                }
                h265_hevc_h_loop_filter_luma(src, f->linesize[0], beta, tc,
                                             no_p, no_q, pixel_shift, bit_depth);
            }
        }
    }

    if (sps->chroma_format_idc) {
        for (chroma = 1; chroma <= 2; chroma++) {
            int h = 1 << sps->hshift[chroma];
            int v = 1 << sps->vshift[chroma];

            for (y = y0; y < y_end; y += (8 * v)) {
                for (x = x0 ? x0 : 8 * h; x < x_end; x += (8 * h)) {
                    const int bs0 = f->vertical_bs[(x + y * bs_width) >> 2];
                    const int bs1 = f->vertical_bs[(x + (y + (4 * v)) * bs_width) >> 2];
                    if ((bs0 == 2) || (bs1 == 2)) {
                        const int qp0 = (get_qPy(sps, f->qp_y_tab, x - 1, y) +
                                         get_qPy(sps, f->qp_y_tab, x, y) + 1) >> 1;
                        const int qp1 = (get_qPy(sps, f->qp_y_tab, x - 1, y + (4 * v)) +
                                         get_qPy(sps, f->qp_y_tab, x, y + (4 * v)) + 1) >> 1;
                        c_tc[0] = (bs0 == 2) ? chroma_tc(pps, sps, qp0, chroma, tc_offset) : 0;
                        c_tc[1] = (bs1 == 2) ? chroma_tc(pps, sps, qp1, chroma, tc_offset) : 0;
                        src = &f->data[chroma][(y >> sps->vshift[chroma]) * f->linesize[chroma] +
                                               ((x >> sps->hshift[chroma]) << pixel_shift)];
                        if (pcmf) {
                            no_p[0] = get_pcm(sps, f->is_pcm, x - 1, y);
                            no_p[1] = get_pcm(sps, f->is_pcm, x - 1, y + (4 * v));
                            no_q[0] = get_pcm(sps, f->is_pcm, x, y);
                            no_q[1] = get_pcm(sps, f->is_pcm, x, y + (4 * v));
                        }
                        h265_hevc_v_loop_filter_chroma(src, f->linesize[chroma],
                                                       c_tc, no_p, no_q, pixel_shift, bit_depth);
                    }
                }
                if (!y)
                    continue;

                tc_offset = x0 ? left_tc_offset : cur_tc_offset;
                x_end2 = x_end;
                if (x_end != sps->width)
                    x_end2 = x_end - 8 * h;
                for (x = x0 ? x0 - 8 * h : 0; x < x_end2; x += (8 * h)) {
                    const int bs0 = f->horizontal_bs[(x + y * bs_width) >> 2];
                    const int bs1 = f->horizontal_bs[((x + 4 * h) + y * bs_width) >> 2];
                    if ((bs0 == 2) || (bs1 == 2)) {
                        const int qp0 = bs0 == 2 ? (get_qPy(sps, f->qp_y_tab, x, y - 1) +
                                                    get_qPy(sps, f->qp_y_tab, x, y) + 1) >> 1 : 0;
                        const int qp1 = bs1 == 2 ? (get_qPy(sps, f->qp_y_tab, x + (4 * h), y - 1) +
                                                    get_qPy(sps, f->qp_y_tab, x + (4 * h), y) + 1) >> 1 : 0;
                        c_tc[0] = bs0 == 2 ? chroma_tc(pps, sps, qp0, chroma, tc_offset) : 0;
                        c_tc[1] = bs1 == 2 ? chroma_tc(pps, sps, qp1, chroma, cur_tc_offset) : 0;
                        src = &f->data[chroma][(y >> sps->vshift[1]) * f->linesize[chroma] +
                                               ((x >> sps->hshift[1]) << pixel_shift)];
                        if (pcmf) {
                            no_p[0] = get_pcm(sps, f->is_pcm, x, y - 1);
                            no_p[1] = get_pcm(sps, f->is_pcm, x + (4 * h), y - 1);
                            no_q[0] = get_pcm(sps, f->is_pcm, x, y);
                            no_q[1] = get_pcm(sps, f->is_pcm, x + (4 * h), y);
                        }
                        h265_hevc_h_loop_filter_chroma(src, f->linesize[chroma],
                                                       c_tc, no_p, no_q, pixel_shift, bit_depth);
                    }
                }
            }
        }
    }
}

void h265_deblock_frame(H265Slice *sh)
{
    H265SPS *sps = sh->sps;
    H265PPS *pps = sh->pps;
    int ctb_count = sps->ctb_width * sps->ctb_height;
    int ctb_size = 1 << sps->log2_ctb_size;
    int ts;

    if (sh->h->param.disable_lf)
        return;
    for (ts = 0; ts < ctb_count; ts++) {
        int rs = pps->ctb_addr_ts_to_rs[ts];
        int x0 = (rs % sps->ctb_width) * ctb_size;
        int y0 = (rs / sps->ctb_width) * ctb_size;
        deblocking_filter_CTB(sh, x0, y0);
    }
}

void h265_deblock_ctb_row(H265Slice *sh, int y0)
{
    H265SPS *sps = sh->sps;
    int ctb_size = 1 << sps->log2_ctb_size;
    int x;
    if (sh->h->param.disable_lf)
        return;
    for (x = 0; x < sps->ctb_width; x++)
        deblocking_filter_CTB(sh, x * ctb_size, y0);
}
