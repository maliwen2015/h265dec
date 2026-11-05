/*
 * H.274 film grain synthesis (model_id 0, 8-bit 4:2:0), matching the
 * reference implementation in FFmpeg's libavcodec/h274.c.
 *
 * The constant tables are in h265_film_grain_tables.h (SMPTE RDD 5-2006,
 * LGPL-2.1+).
 */
#include "h265_internal.h"
#include "h265_film_grain_tables.h"

#define GRAIN_DIM 64
#define GRAIN_SLICES (13 * 13)
#define GRAIN_SLICE_BYTES (GRAIN_DIM * GRAIN_DIM)

static int grain_clip(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static void prng_shift(uint32_t *state)
{
    uint32_t x = *state;
    uint8_t feedback = 1u ^ (x >> 2) ^ (x >> 30);
    *state = (x << 1) | (feedback & 1u);
}

static void init_slice_c(int8_t out[64][64], uint8_t h, uint8_t v)
{
    static const uint8_t deblock_factors[13] = {
        64, 71, 77, 84, 90, 96, 103, 109, 116, 122, 128, 128, 128
    };
    int16_t tmp[64][64];
    const uint8_t deblock_coeff = deblock_factors[v];
    const uint8_t freq_h = ((h + 3) << 2) - 1;
    const uint8_t freq_v = ((v + 3) << 2) - 1;
    uint32_t seed = Seed_LUT[h + v * 13];

    for (int l = 0; l <= freq_v; l++) {
        for (int k = 0; k <= freq_h; k += 4) {
            uint16_t offset = seed % 2048;
            out[l][k + 0] = Gaussian_LUT[offset + 0];
            out[l][k + 1] = Gaussian_LUT[offset + 1];
            out[l][k + 2] = Gaussian_LUT[offset + 2];
            out[l][k + 3] = Gaussian_LUT[offset + 3];
            prng_shift(&seed);
        }
    }

    out[0][0] = 0;

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x <= freq_v; x++) {
            int32_t sum = 0;
            for (int p = 0; p <= freq_h; p++)
                sum += R64T[y][p] * out[x][p];
            tmp[y][x] = (int16_t)((sum + 128) >> 8);
        }
    }

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            int32_t sum = 0;
            for (int p = 0; p <= freq_v; p++)
                sum += tmp[x][p] * R64T[y][p];
            out[y][x] = (int8_t)grain_clip((sum + 128) >> 8, -127, 127);
        }
    }

    for (int y = 0; y < 64; y += 8) {
        for (int x = 0; x < 64; x++) {
            out[y + 0][x] = (int8_t)((out[y + 0][x] * deblock_coeff) >> 7);
            out[y + 7][x] = (int8_t)((out[y + 7][x] * deblock_coeff) >> 7);
        }
    }
}

static int8_t *grain_slice(H265Decoder *h, int hh, int vv)
{
    int bit = hh * 13 + vv;
    int8_t *db = h->film_grain_db;

    if (!db) {
        db = (int8_t *)calloc(GRAIN_SLICES * GRAIN_SLICE_BYTES, 1);
        if (!db)
            return NULL;
        h->film_grain_db = db;
        memset(h->film_grain_ready, 0, sizeof(h->film_grain_ready));
    }
    if (!(h->film_grain_ready[bit >> 5] & (1u << (bit & 31)))) {
        init_slice_c((int8_t (*)[64])(db + (size_t)bit * GRAIN_SLICE_BYTES),
                     (uint8_t)hh, (uint8_t)vv);
        h->film_grain_ready[bit >> 5] |= 1u << (bit & 31);
    }
    return db + (size_t)bit * GRAIN_SLICE_BYTES;
}

static uint16_t avg_8x8(const uint8_t *in, int stride)
{
    uint16_t avg[8] = {0};
    int y, x;
    for (y = 0; y < 8; y++) {
        for (x = 0; x < 8; x++)
            avg[x] += in[x];
        in += stride;
    }
    return (uint16_t)((avg[0] + avg[1] + avg[2] + avg[3] +
                       avg[4] + avg[5] + avg[6] + avg[7]) >> 6);
}

static void synth_grain_8x8(int8_t *out, int out_stride, int16_t scale,
                            uint8_t shift, const int8_t *db)
{
    int y, x;
    for (y = 0; y < 8; y++) {
        for (x = 0; x < 8; x++)
            out[x] = (int8_t)((scale * db[x]) >> shift);
        out += out_stride;
        db += 64;
    }
}

static void deblock_8x8(int8_t *out, int out_stride)
{
    int y;
    for (y = 0; y < 8; y++) {
        int8_t l1 = out[-2], l0 = out[-1];
        int8_t r0 = out[0], r1 = out[1];
        out[0]  = (int8_t)((l0 + r0 * 2 + r1) >> 2);
        out[-1] = (int8_t)((r0 + l0 * 2 + l1) >> 2);
        out += out_stride;
    }
}

static void generate(H265Decoder *h, int8_t *out, int out_stride,
                     const uint8_t *in, int in_stride,
                     const H265FilmGrainParams *p, int c, int invert,
                     int deblock, int y_offset, int x_offset)
{
    const uint8_t shift = (uint8_t)(p->log2_scale_factor + 6);
    const uint16_t avg = avg_8x8(in, in_stride);
    int16_t scale;
    uint8_t hh, vv;
    int8_t s = -1;
    int i, y;

    for (i = 0; i < p->num_intensity_intervals[c]; i++) {
        if (avg >= p->intensity_interval_lower_bound[c][i] &&
            avg <= p->intensity_interval_upper_bound[c][i]) {
            s = (int8_t)i;
            break;
        }
    }
    if (s < 0) {
        for (y = 0; y < 8; y++)
            memset(out + y * out_stride, 0, 8);
        return;
    }

    hh = (uint8_t)(grain_clip(p->comp_model_value[c][s][1], 2, 14) - 2);
    vv = (uint8_t)(grain_clip(p->comp_model_value[c][s][2], 2, 14) - 2);
    {
        int8_t *db = grain_slice(h, hh, vv);
        if (!db) {
            for (y = 0; y < 8; y++)
                memset(out + y * out_stride, 0, 8);
            return;
        }
        scale = (int16_t)p->comp_model_value[c][s][0];
        if (invert)
            scale = (int16_t)-scale;
        synth_grain_8x8(out, out_stride, scale, shift,
                        db + y_offset * 64 + x_offset);
    }
    if (deblock)
        deblock_8x8(out, out_stride);
}

static void add_8x8_clip(uint8_t *out, const uint8_t *a, const int8_t *b,
                         int n)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = (uint8_t)grain_clip(a[i] + b[i], 0, 255);
}

int h265_film_grain_supported(const H265FilmGrainParams *p, const H265SPS *sps)
{
    return p && p->present && p->model_id == 0 && sps &&
           sps->chroma_format_idc == 1 && sps->bit_depth == 8;
}

int h265_film_grain_apply(H265Decoder *h, H265Frame *dst, const H265Frame *src,
                          const H265FilmGrainParams *params)
{
    H265FilmGrainParams local = *params;
    int c;

    if (!h265_film_grain_supported(params, src->sps))
        return -1;

    for (c = 0; c < 3; c++) {
        static const uint8_t color_offset[3] = { 0, 85, 170 };
        uint32_t seed = Seed_LUT[(src->poc + color_offset[c]) % 256];
        int width = src->plane_w[c];
        int height = src->plane_h[c];
        uint8_t *out = dst->data[c];
        int out_stride = dst->linesize[c];
        int8_t *grain = (int8_t *)dst->data[c];
        const uint8_t *in = src->data[c];
        int in_stride = src->linesize[c];
        int y, x;

        if (!local.component_model_present[c]) {
            for (y = 0; y < height; y++)
                memcpy(out + (size_t)y * out_stride, in + (size_t)y * in_stride,
                       (size_t)width);
            continue;
        }

        if (c > 0) {
            int i;
            for (i = 0; i < local.num_intensity_intervals[c]; i++) {
                local.comp_model_value[c][i][0] >>= 1;
                local.comp_model_value[c][i][1] *= 2;
                local.comp_model_value[c][i][2] *= 2;
            }
        }

        for (y = 0; y < height; y += 16) {
            for (x = 0; x < width; x += 16) {
                uint16_t x_offset = (uint16_t)((seed >> 16) % 52);
                uint16_t y_offset = (uint16_t)((seed & 0xffff) % 56);
                int invert = seed & 1;
                x_offset &= 0xfffc;
                y_offset &= 0xfff8;
                prng_shift(&seed);
                for (int yy = 0; yy < 16 && y + yy < height; yy += 8) {
                    for (int xx = 0; xx < 16 && x + xx < width; xx += 8) {
                        generate(h, grain + (size_t)(y + yy) * out_stride + x + xx,
                                 out_stride,
                                 in + (size_t)(y + yy) * in_stride + x + xx,
                                 in_stride, &local, c, invert, (x + xx) > 0,
                                 y_offset + yy, x_offset + xx);
                    }
                }
            }
        }

        for (y = 0; y < height; y++)
            add_8x8_clip(out + (size_t)y * out_stride,
                         in + (size_t)y * in_stride,
                         grain + (size_t)y * out_stride, width);
    }
    return 0;
}
