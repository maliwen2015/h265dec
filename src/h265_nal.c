/*
 * HEVC NAL unit splitting and RBSP extraction.
 */
#include "h265_internal.h"

int h265_nal_start_code(const uint8_t *data, int size, int pos, int *sc_len)
{
    int i;
    for (i = pos; i + 2 < size; i++) {
        if (data[i] == 0 && data[i + 1] == 0) {
            if (data[i + 2] == 1) {
                *sc_len = (i > pos && data[i - 1] == 0) ? 4 : 3;
                return i;
            }
        }
    }
    return -1;
}

/* Locate the next NAL unit of an Annex-B stream.  Returns the offset of the
 * payload in *out_pos and its size in *out_len, or 0 at end of data. */
int h265_split_annexb(const uint8_t *data, int size, int *pos,
                      const uint8_t **out, int *out_len)
{
    int start, i, next, next_start, sc_len = 3;

    start = *pos;
    while (start + 3 <= size) {
        if (data[start] == 0 && data[start + 1] == 0 && data[start + 2] == 1) {
            start += 3;
            break;
        }
        if (start + 4 <= size && data[start] == 0 && data[start + 1] == 0 &&
            data[start + 2] == 0 && data[start + 3] == 1) {
            start += 4;
            break;
        }
        start++;
    }
    if (start + 2 > size)
        return 0;

    next = size;
    next_start = size;
    for (i = start; i + 2 < size; i++) {
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
            sc_len = (i > 0 && data[i - 1] == 0) ? 4 : 3;
            next_start = i - (sc_len == 4 ? 1 : 0);
            next = next_start;
            break;
        }
    }

    while (next > start && data[next - 1] == 0)
        next--;

    if (next <= start) {
        *pos = next_start > start ? next_start : size;
        return 0;
    }

    *out = data + start;
    *out_len = next - start;
    *pos = next_start;
    return 1;
}

int h265_split_avcc(const uint8_t *data, int size, int *pos,
                    const uint8_t **out, int *out_len, int nal_length_size)
{
    int p = *pos, len = 0, i;

    if (p + nal_length_size > size)
        return 0;
    for (i = 0; i < nal_length_size; i++)
        len = (len << 8) | data[p + i];
    p += nal_length_size;
    if (len <= 0 || p + len > size)
        return 0;
    *out = data + p;
    *out_len = len;
    *pos = p + len;
    return 1;
}

/* Remove emulation-prevention bytes.  The returned buffer is padded with
 * eight zero bytes for the CABAC over-read. */
uint8_t *h265_rbsp_extract_ex(const uint8_t *nal, int len, int *out_size,
                              int **out_epb, int *out_nb)
{
    uint8_t *buf;
    int *epb = NULL;
    int nb = 0, cap = 0;
    int i, j = 0, zeros = 0;

    if (len < 0)
        return NULL;
    buf = (uint8_t *)malloc((size_t)len + 16);
    if (!buf)
        return NULL;

    for (i = 0; i < len; i++) {
        if (zeros >= 2 && nal[i] == 0x03) {
            if (out_epb) {
                if (nb == cap) {
                    int ncap = cap ? cap * 2 : 16;
                    int *p = (int *)realloc(epb, (size_t)ncap * sizeof(int));
                    if (!p) {
                        free(epb);
                        free(buf);
                        return NULL;
                    }
                    epb = p;
                    cap = ncap;
                }
                /* Position of the second zero, in unescaped coordinates,
                 * matching FFmpeg's skipped_bytes_pos convention. */
                epb[nb++] = j - 1;
            }
            zeros = 0;
            continue;
        }
        buf[j++] = nal[i];
        if (nal[i] == 0)
            zeros++;
        else
            zeros = 0;
    }
    memset(buf + j, 0, 8);
    *out_size = j;
    if (out_epb) {
        *out_epb = epb;
        *out_nb = nb;
    } else {
        free(epb);
    }
    return buf;
}

uint8_t *h265_rbsp_extract(const uint8_t *nal, int len, int *out_size)
{
    return h265_rbsp_extract_ex(nal, len, out_size, NULL, NULL);
}

int h265_nal_type(const uint8_t *nal, int len)
{
    if (len < 2)
        return -1;
    return (nal[0] >> 1) & 0x3f;
}

int h265_nal_layer_id(const uint8_t *nal, int len)
{
    if (len < 2)
        return 0;
    return ((nal[0] & 1) << 5) | (nal[1] >> 3);
}

int h265_nal_temporal_id(const uint8_t *nal, int len)
{
    if (len < 2)
        return 0;
    return (nal[1] & 7) - 1;
}
