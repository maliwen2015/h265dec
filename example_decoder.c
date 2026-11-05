#include "h265dec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double wall_time_ms(void)
{
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return (double)now.QuadPart * 1000.0 / freq.QuadPart;
    }
}
#else
#include <sys/time.h>
static double wall_time_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

static int save_yuv_frame(FILE *fp, const h265_picture_t *pic)
{
    int i;
    if (!fp)
        return 0;
    for (i = 0; i < pic->lines[0]; i++)
        fwrite(pic->planes[0] + (size_t)i * pic->strides[0],
               pic->bytes_per_sample, pic->widths[0], fp);
    if (pic->num_planes < 3)
        return 0;
    for (i = 0; i < pic->lines[1]; i++)
        fwrite(pic->planes[1] + (size_t)i * pic->strides[1],
               pic->bytes_per_sample, pic->widths[1], fp);
    for (i = 0; i < pic->lines[2]; i++)
        fwrite(pic->planes[2] + (size_t)i * pic->strides[2],
               pic->bytes_per_sample, pic->widths[2], fp);
    if (pic->alpha_plane) {
        for (i = 0; i < pic->alpha_height; i++)
            fwrite(pic->alpha_plane + (size_t)i * pic->alpha_stride,
                   pic->bytes_per_sample, pic->alpha_width, fp);
    }
    return 0;
}

static int start_code_len(const uint8_t *d, int i, int size)
{
    if (i + 3 > size || d[i] || d[i + 1])
        return 0;
    if (d[i + 2] == 1)
        return 3;
    if (i + 4 <= size && d[i + 2] == 0 && d[i + 3] == 1)
        return 4;
    return 0;
}

/* Locate access-unit boundaries in an Annex-B HEVC stream.  The first bit of
 * the slice header after the two-byte NAL header is first_slice_in_pic_flag. */
static int find_access_units(const uint8_t *data, int size, int **out)
{
    int cap = 64, nb = 0, i;
    int *pos = (int *)malloc(cap * sizeof(int));
    int *scs = (int *)malloc(cap * sizeof(int));
    int *starts;
    int last_vcl = 0;

    if (!pos || !scs) {
        free(pos);
        free(scs);
        *out = NULL;
        return 0;
    }
    for (i = 0; i + 3 <= size; ) {
        int sc = start_code_len(data, i, size);
        if (!sc) {
            i++;
            continue;
        }
        if (i + sc >= size)
            break;
        {
            int off = i + sc;
            int type = (data[off] >> 1) & 0x3f;
            int first = (type <= 31) && off + 2 < size && (data[off + 2] & 0x80);
            if (type <= 31) {
                if (first && last_vcl) {
                    if (nb == cap) {
                        cap *= 2;
                        pos = (int *)realloc(pos, cap * sizeof(int));
                        scs = (int *)realloc(scs, cap * sizeof(int));
                    }
                    pos[nb] = off;
                    scs[nb] = i;
                    nb++;
                } else if (!nb) {
                    pos[nb] = off;
                    scs[nb] = i;
                    nb++;
                }
                last_vcl = 1;
            }
        }
        i += sc + 2;
    }

    starts = (int *)malloc((nb + 1) * sizeof(int));
    if (!starts) {
        free(pos);
        free(scs);
        *out = NULL;
        return 0;
    }
    for (i = 0; i < nb; i++)
        starts[i] = scs[i];
    /* include leading parameter sets before the first VCL NAL */
    if (nb == 0 || scs[0] > 0)
        starts[0] = 0;
    starts[nb] = size;
    free(pos);
    free(scs);
    *out = starts;
    return nb + 1;
}

int main(int argc, char **argv)
{
    const char *in_path, *out_path;
    int threads = 0;
    FILE *in, *out = NULL;
    uint8_t *data;
    long size;
    int *au;
    int nb_au, i;
    h265_param_t param;
    void *dec;
    h265_stream_info info;
    h265_picture_t pic;
    int frames = 0, errors = 0;
    double t0, t1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.h265> <output.yuv|-> [threads]\n", argv[0]);
        return 1;
    }
    in_path = argv[1];
    out_path = argv[2];
    if (argc > 3)
        threads = atoi(argv[3]);

    in = fopen(in_path, "rb");
    if (!in) {
        fprintf(stderr, "cannot open %s\n", in_path);
        return 1;
    }
    fseek(in, 0, SEEK_END);
    size = ftell(in);
    fseek(in, 0, SEEK_SET);
    data = (uint8_t *)malloc((size_t)size + 1);
    if (!data) {
        fclose(in);
        return 1;
    }
    if (fread(data, 1, (size_t)size, in) != (size_t)size) {
        fclose(in);
        free(data);
        return 1;
    }
    fclose(in);

    if (strcmp(out_path, "-") != 0) {
        out = fopen(out_path, "wb");
        if (!out) {
            fprintf(stderr, "cannot open %s\n", out_path);
            free(data);
            return 1;
        }
        setvbuf(out, NULL, _IOFBF, 1 << 20);
    }

    memset(&param, 0, sizeof(param));
    param.threads = threads;
    param.disable_lf = getenv("H265_NO_LF") != NULL;
    param.disable_sao = getenv("H265_NO_SAO") != NULL;
    param.wpp_threads = getenv("H265_WPP_THREADS") != NULL;
    dec = h265_dec_open(&param);
    if (!dec) {
        fprintf(stderr, "cannot open decoder\n");
        return 1;
    }

    nb_au = find_access_units(data, (int)size, &au);
    t0 = wall_time_ms();
    for (i = 0; i + 1 < nb_au; i++) {
        h265_packet_t pkt;
        int start = au[i];
        int end = au[i + 1];
        pkt.data = data + start;
        pkt.len = end - start;
        pkt.pts = i;
        pkt.dts = i;
        pkt.opaque = NULL;
        if (h265_dec_send_packet(dec, &pkt) < 0)
            errors++;
        while (h265_dec_recv_frame(dec, &info, &pic) == H265_DEC_GOT_FRAME) {
            frames++;
            if (pic.b_decode_error)
                errors++;
            if (getenv("H265_PRINT_META"))
                fprintf(stderr, "meta pts=%lld poc=%d color=%d/%d/%d range=%d "
                        "mastering=%d cll=%u/%u tc=%d orientation=%d\n",
                        (long long)pic.pts, pic.pic_order_count,
                        pic.colour_primaries, pic.transfer_characteristics,
                        pic.matrix_coeffs, pic.video_full_range,
                        pic.mastering_display_present,
                        pic.max_content_light_level,
                        pic.max_pic_average_light_level,
                        pic.timecode_valid, pic.display_orientation_present);
            if (getenv("H265_PRINT_META"))
                fprintf(stderr, "     field: ps=%d interlaced=%d top_first=%d repeat=%d\n",
                        pic.pic_struct, pic.interlaced, pic.top_field_first,
                        pic.repeat_pict);
            if (getenv("H265_PRINT_META"))
                fprintf(stderr, "     side: t35=%d unreg=%d dovi=%d\n",
                        pic.itu_t_t35_size, pic.user_data_unregistered_size,
                        pic.dovi_rpu_size);
            save_yuv_frame(out, &pic);
            h265_dec_frame_unref(dec, &pic);
        }
    }
    while (h265_dec_flush(dec, &info, &pic) == H265_DEC_GOT_FRAME) {
        frames++;
        if (pic.b_decode_error)
            errors++;
        save_yuv_frame(out, &pic);
        h265_dec_frame_unref(dec, &pic);
    }
    t1 = wall_time_ms();
    h265_dec_close(dec);
    free(au);
    free(data);
    if (out)
        fclose(out);

    fprintf(stderr, "frames=%d errors=%d time=%.1fms fps=%.1f\n", frames, errors,
            t1 - t0, (t1 > t0) ? frames * 1000.0 / (t1 - t0) : 0.0);
    return 0;
}
