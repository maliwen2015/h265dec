/*
 * hvcC/AVCC end-to-end check.
 *
 * Reads a length-prefixed (AVCC) stream and its hvcC extradata, feeds one NAL
 * per send_packet() call, and writes planar YUV for comparison with FFmpeg.
 * This exercises hvcC parsing, per-NAL packet splitting and multi-slice access
 * units that span several packets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "h265dec.h"

static unsigned char *read_file(const char *path, int *size)
{
    FILE *f = fopen(path, "rb");
    long n;
    unsigned char *p;

    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    p = (unsigned char *)malloc((size_t)n + 1);
    if (!p) {
        fclose(f);
        return NULL;
    }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (int)n;
    return p;
}

static void write_plane(FILE *out, const h265_picture_t *pic, int plane)
{
    int y;
    for (y = 0; y < pic->lines[plane]; y++)
        fwrite(pic->planes[plane] + (size_t)y * pic->strides[plane], 1,
               (size_t)pic->widths[plane] * pic->bytes_per_sample, out);
}

int main(int argc, char **argv)
{
    int extradata_size, data_size, pos = 0, frames = 0, errors = 0;
    unsigned char *extradata, *data;
    FILE *out;
    h265_param_t p;
    h265_packet_t pkt;
    h265_stream_info info;
    h265_picture_t pic;
    void *dec;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <hvcc.bin> <in.avcc> <out.yuv>\n", argv[0]);
        return 2;
    }
    extradata = read_file(argv[1], &extradata_size);
    data = read_file(argv[2], &data_size);
    if (!extradata || !data)
        return 2;
    out = fopen(argv[3], "wb");
    if (!out)
        return 2;

    memset(&p, 0, sizeof(p));
    dec = h265_dec_open(&p);
    if (!dec)
        return 2;
    if (h265_dec_set_extradata(dec, extradata, extradata_size) < 0) {
        fprintf(stderr, "hvcC parse failed\n");
        return 1;
    }

    while (pos + 4 <= data_size) {
        int start = pos;
        int nal_size = (data[pos] << 24) | (data[pos + 1] << 16) |
                       (data[pos + 2] << 8) | data[pos + 3];
        pos += 4;
        if (nal_size <= 0 || nal_size > data_size - pos)
            break;
        memset(&pkt, 0, sizeof(pkt));
        pkt.data = data + start;
        pkt.len = nal_size + 4;
        pos += nal_size;
        if (h265_dec_send_packet(dec, &pkt) < 0)
            errors++;
        while (h265_dec_recv_frame(dec, &info, &pic) == H265_DEC_GOT_FRAME) {
            frames++;
            if (pic.b_decode_error)
                errors++;
            write_plane(out, &pic, 0);
            if (pic.num_planes > 1) {
                write_plane(out, &pic, 1);
                write_plane(out, &pic, 2);
            }
            h265_dec_frame_unref(dec, &pic);
        }
    }
    while (h265_dec_flush(dec, &info, &pic) == H265_DEC_GOT_FRAME) {
        frames++;
        if (pic.b_decode_error)
            errors++;
        write_plane(out, &pic, 0);
        if (pic.num_planes > 1) {
            write_plane(out, &pic, 1);
            write_plane(out, &pic, 2);
        }
        h265_dec_frame_unref(dec, &pic);
    }

    fclose(out);
    h265_dec_close(dec);
    free(extradata);
    free(data);
    fprintf(stderr, "avcc: frames=%d errors=%d\n", frames, errors);
    return frames > 0 && errors == 0 ? 0 : 1;
}
