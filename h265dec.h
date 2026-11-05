/*
 * h265dec - a self-contained HEVC/H.265 decoder (8/9/10/12-bit,
 * 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4) with frame threading and SIMD kernels.
 *
 * Public API. The interface deliberately mirrors the h264dec API.
 *
 * Algorithms and constant tables are derived from ITU-T Rec. H.265 and from
 * FFmpeg's HEVC decoder (LGPL-2.1+) used as a reference implementation.
 */
#ifndef H265DEC_H
#define H265DEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes */
#define H265_DEC_ERROR       -1   /* decoding error */
#define H265_DEC_DEFAULT      0   /* no output available */
#define H265_DEC_GOT_FRAME    1   /* got a decoded picture */
#define H265_DEC_GOT_HEADER   2   /* got a new sequence header */
#define H265_DEC_END          3   /* end of stream */

/* Stream information */
typedef struct h265_stream_info
{
    int profile;               /* profile_idc */
    int level;                 /* level_idc */
    int width;                 /* display width (cropping applied) */
    int height;                /* display height */
    int coded_width;
    int coded_height;
    int chroma_format;         /* 0 = 4:0:0, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4 */
    int bit_depth;             /* 8, 9, 10 or 12 */
    int frame_rate_num;
    int frame_rate_den;
    int bitrate;
    int sar_width;
    int sar_height;
    int max_num_reorder_frames;
    int max_dec_frame_buffering;
    int temporal_id_nesting;
    int separate_colour_plane;
    int video_full_range;
    int colour_primaries;
    int transfer_characteristics;
    int matrix_coeffs;
    int chroma_sample_location;    /* top field, -1 when not signalled */
    int field_seq_flag;
} h265_stream_info;

/* Input packet: any slice of an Annex-B byte stream, or of a length-prefixed
 * (AVCC/HVCC) stream when nal_length_size is set.  Packets may contain
 * several access units, one NAL, or a fragment of a NAL; the decoder keeps the
 * unconsumed tail and finishes it on the next packet or h265_dec_flush(). */
struct h265_packet_t
{
    const uint8_t* data;
    int len;
    int64_t pts;
    int64_t dts;
    void* opaque;
};
typedef struct h265_packet_t h265_packet_t;

/* Decoded picture.  Planes are the top-left of the displayed (cropped) area.
 * bytes_per_sample is 1 for 8-bit and 2 for the 16-bit planar formats. */
struct h265_picture_t
{
    uint8_t* planes[3];
    int widths[3];
    int lines[3];
    int strides[3];            /* bytes */
    int pic_order_count;
    int type;                  /* 1 = I, 2 = P, 3 = B */
    int qp;
    int64_t pts;
    int64_t dts;
    int num_planes;
    int bytes_per_sample;
    int bit_depth;
    int chroma_format;
    int b_decode_error;
    int temporal_id;
    int nal_unit_type;         /* type of the first VCL NAL of the picture */
    int is_irap;

    /* VUI colour description */
    int video_full_range;
    int colour_primaries;
    int transfer_characteristics;
    int matrix_coeffs;
    int chroma_sample_location;    /* top field, -1 when not signalled */
    int field_seq_flag;
    int colour_plane_id;           /* -1 unless separate_colour_plane_flag */
    int layer_id;                  /* nuh_layer_id; only base layer is decoded */
    int interlaced;                /* 1 = field-coded/interlaced source */
    int top_field_first;
    int repeat_pict;               /* frame doubling/tripling: 0, 1 or 2 */

    /* SEI metadata.  -1 means the message was not present in the access unit. */
    int pic_struct;                /* H.265 Table D.2.4 */
    int recovery_frame_cnt;        /* -1 when absent */
    int broken_recovery;
    int timecode_valid;
    int timecode_hours;
    int timecode_minutes;
    int timecode_seconds;
    int timecode_frames;
    int timecode_drop;

    int mastering_display_present;
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
    unsigned display_orientation_rotation;   /* anticlockwise, 1/65536 degree */
    int ambient_viewing_environment_present;
    unsigned ambient_illuminance;
    unsigned ambient_light_x;
    unsigned ambient_light_y;
    int frame_packing_present;
    int frame_packing_type;
    int frame_packing_content;
    int frame_packing_current_frame_is_frame0;

    /* Raw side data, valid until h265_dec_frame_unref().
     * itu_t_t35 contains the registered user-data payload (HDR10+, A53 CC,
     * AFD, ...); unregistered is user_data_unregistered; dovi_rpu is the
     * payload of an HEVC NAL unit of type 62. */
    const uint8_t* itu_t_t35;
    int itu_t_t35_size;
    const uint8_t* user_data_unregistered;
    int user_data_unregistered_size;
    const uint8_t* dovi_rpu;
    int dovi_rpu_size;

    /* Auxiliary alpha/depth plane (luma of an independently coded layer 1),
     * valid until h265_dec_frame_unref(). */
    const uint8_t* alpha_plane;
    int alpha_stride;
    int alpha_width;
    int alpha_height;

    void* dec_frame;
    void* opaque;
};
typedef struct h265_picture_t h265_picture_t;

/* Decoder parameters */
typedef struct h265_param_t
{
    int threads;               /* worker threads (0/1 = no extra threads) */
    int output_order;          /* 0 = display order, 1 = decode order */
    int disable_lf;            /* 1 = skip deblocking filter */
    int disable_sao;           /* 1 = skip sample adaptive offset */
    int skip_non_ref;          /* 1 = do not output non-reference frames */
    int nal_length_size;       /* 0 = Annex-B start codes (default),
                                  otherwise 1/2/4 byte NAL length prefix */
    int max_threads;           /* clamp, 0 = default */
    int wpp_threads;           /* 1 = parallelize WPP CTU rows within a
                                  picture.  Off by default because frame-level
                                  parallelism can be faster on B-frame
                                  content. */
} h265_param_t;

#ifdef H265DECLIB_SHARED
#define H265DEC_EXPORT __attribute__((visibility("default")))
#else
#define H265DEC_EXPORT
#endif

H265DEC_EXPORT void* h265_dec_open(h265_param_t* param);
/* Feed an hvcC configuration record (as stored in MP4/MKV extradata) or an
 * Annex-B parameter-set blob.  Returns 0 on success.  When an hvcC record is
 * supplied, nal_length_size is taken from the record and subsequent packets
 * must use length-prefixed NAL units. */
H265DEC_EXPORT int   h265_dec_set_extradata(void* decoder, const void* data, int size);
H265DEC_EXPORT int   h265_dec_send_packet(void* decoder, h265_packet_t* packet);
H265DEC_EXPORT int   h265_dec_recv_frame(void* decoder, h265_stream_info* info,
                                         h265_picture_t* out_frame);
H265DEC_EXPORT int   h265_dec_flush(void* decoder, h265_stream_info* info,
                                    h265_picture_t* out_frame);
H265DEC_EXPORT void  h265_dec_frame_unref(void* decoder, h265_picture_t* out_frame);
H265DEC_EXPORT void  h265_dec_close(void* decoder);

#ifdef __cplusplus
}
#endif

#endif /* H265DEC_H */
