# h265dec

A from-scratch HEVC/H.265 decoder in C99 with a frame-threaded worker pool and
SSE2/AVX2/NEON kernels.  It decodes 8/9/10/12-bit 4:0:0, 4:2:0, 4:2:2 and 4:4:4
streams, applies H.274 film grain and reproduces FFmpeg's output byte for byte
on the tested corpus.

The public API deliberately mirrors the companion `h264dec`:

```c
h265_param_t p; memset(&p, 0, sizeof p);
p.threads = 8;              /* 0/1 = single threaded */
p.wpp_threads = 0;          /* 1 = opt-in WPP CTU-row parallelism */
p.nal_length_size = 0;      /* 0 = Annex-B, or 1/2/4 = AVCC */
void *dec = h265_dec_open(&p);
h265_dec_set_extradata(dec, hvcc, hvcc_size);   /* optional hvcC/Annex-B */

h265_packet_t pkt = { data, len, pts, dts, NULL };   /* NAL/AU or fragment */
h265_dec_send_packet(dec, &pkt);
h265_picture_t pic;
h265_stream_info info;
while (h265_dec_recv_frame(dec, &info, &pic) == H265_DEC_GOT_FRAME) {
    /* pic.planes[0..2], pic.widths, pic.lines, pic.strides,
       pic.bytes_per_sample (1 or 2), pic.bit_depth, pic.qp, ... */
    h265_dec_frame_unref(dec, &pic);
}
h265_dec_flush(dec, &info, &pic);   /* until H265_DEC_END */
h265_dec_close(dec);
```

`example_decoder.c` splits an Annex-B file into access units, decodes them and
writes planar YUV.  The reference point is FFmpeg 9.0.2.

## Build

MSYS2 / MinGW-w64, or gcc/clang on POSIX (`-lpthread` on glibc older than
2.34):

```sh
mingw32-make -j8           # libh265dec.a + h265dec_example
mingw32-make check         # corpus, hvcC/AVCC and FFmpeg comparisons
mingw32-make clean
```

The baseline build is plain x86-64.  Vector kernels carry their own
`target("sse2")` / `target("avx2")` attributes and are selected at run time
from CPUID/XGETBV (`src/h265_cpu.c`).  `H265_DISABLE_SIMD=1` forces the scalar
path; both paths are bit-exact.

AArch64 is supported as a first-class target.  NEON is mandatory there, so
`src/h265_dsp_neon.c` is selected automatically (`H265_NEON` is defined for
`__aarch64__` and `_M_ARM64`).  Cross-compile with:

```sh
make clean
make -j8 CC=aarch64-linux-gnu-gcc      # Linux arm64
```

Native builds work on Linux and macOS (pthreads) and on Windows on ARM64
(Win32 thread API).  `tests/check_arm64.sh` cross-compiles and compares the
NEON output against the C fallback under `qemu-aarch64-static`, including
2/4/8-thread runs.

## What is implemented

* Full HEVC bitstream structure: VPS, SPS, PPS, AUD, EOS/EOB, prefix/suffix
  SEI, slice segment headers and dependent slice segments, short-term and
  long-term reference picture sets, POC types 0 and 1.
* CABAC and every HEVC syntax element: coding quadtree, all partition modes
  (2Nx2N, 2NxN, Nx2N, NxN, AMP), intra modes 0..34 with 3-tap and strong
  smoothing, merge/AMVP with temporal MVP, weighted uni/bi prediction.
* All transform sizes (4x4, 8x8, 16x16, 32x32), 4x4 luma DST, transform skip,
  RDPCM, scaling lists (SPS and PPS), cross-component prediction, PCM.
* Deblocking filter with boundary strengths derived per TU/PU and the HEVC
  alpha/beta/tc tables, plus sample adaptive offset (band and edge offsets).
* Reference picture management with DPB output/display reordering, IDR/BLA/CRA
  handling, `no_output_of_prior_pics_flag`, and dropping RASL pictures when
  they have no valid reference.
* Tiles, WPP entry points, WPP CTU-row parallelism (`wpp_threads`), arbitrary
  slice counts, multiple slices per picture.
* Field-coded sequences: `field_seq_flag`, picture timing `pic_struct`, and
  `interlaced` / `top_field_first` / `repeat_pict` output flags.  Field pictures
  are decoded as pictures, matching FFmpeg's HEVC output.
* Separate colour planes are decoded with the monochrome syntax, as FFmpeg
  does; `colour_plane_id` identifies the returned plane.
* Raw SEI/user-data side data: ITU-T T.35 registered payloads (HDR10+, A53 CC,
  AFD, ...), unregistered user data, and Dolby Vision RPU NAL units.
* MV-HEVC base-view decoding plus independently coded auxiliary alpha/depth
  layers (AuxId signalling), exposed as `alpha_plane`.
* 8/9/10/12-bit pixel pipelines for monochrome and 4:2:0/4:2:2/4:4:4.
* Annex-B and hvcC/AVCC input, including several NALs per packet, one NAL per
  packet and NALs split across packet boundaries.
* VUI colour description and SEI metadata: mastering display, content light,
  timecode, recovery point, display orientation, ambient viewing and frame
  packing are exposed on `h265_picture_t`.
* H.274 film grain synthesis (model 0, 8-bit 4:2:0), matching FFmpeg's output.
* Frame-level multithreading.  A picture is split into CTU rows; workers
  publish filtered rows so following pictures can start before the previous
  one finishes.  Independent slices of one picture are also dispatched to
  separate workers, with the last one applying deblocking/SAO.  WPP CTU rows
  can additionally be decoded as wavefront jobs via `wpp_threads=1`; this is
  opt-in because frame-level parallelism is often faster on B-frame content.
  Output is bit-identical at every thread count.
* SSE2 kernels for residual addition and pixel copy/average; AVX2 variants for
  those paths plus SAO band and edge offsets (16 pixels per iteration),
  luma deblocking (a whole 8-line edge per call) and chroma deblocking,
  half-pel luma/chroma motion compensation with 16-bit dot products,
  motion-compensation output conversion and bi-prediction blending, and both
  IDCT passes.  AArch64 has the equivalent NEON kernels: residual addition,
  copy/average, integer and sub-pixel motion compensation, weighted
  prediction, bi-prediction blending, SAO band/edge, 8-line luma and chroma
  deblocking, and both IDCT passes.  All are dispatched at run time.

Not implemented: SCC/RExt palette, ACT and intra block copy (FFmpeg's native
decoder rejects SCC too), inter-view/inter-layer prediction for MV-HEVC
multiview streams (base view and independent auxiliary layers are decoded),
and hardware acceleration.  Independent slices are
parallel and WPP rows are available through `wpp_threads`; sequential WPP
remains the default for throughput.  Film grain synthesis follows FFmpeg and
therefore covers model 0 with 8-bit 4:2:0.

## Architecture

| file | contents |
| --- | --- |
| `src/h265_nal.c` | Annex-B / AVCC splitter, emulation-prevention removal |
| `src/h265_ps.c` | VPS/SPS/PPS, scaling lists, short-term RPS, VUI |
| `src/h265_slice.c` | slice header, POC, DPB/RPS, reference lists, output order |
| `src/h265_cabac.c` | CABAC engine, context init, syntax elements, residual decoder |
| `src/h265_ctu.c` | CTU/CU/PU/TU recursion, PCM, slice data loop |
| `src/h265_pred.c` | intra prediction |
| `src/h265_mv.c` | merge and AMVP candidate derivation |
| `src/h265_mc.c` | luma/chroma motion compensation and weighted prediction |
| `src/h265_trans.c` | dequantisation, inverse transforms, residual addition |
| `src/h265_deblock.c` | deblocking filter and boundary strengths |
| `src/h265_sao.c` | SAO band/edge filters and edge buffering |
| `src/h265_film_grain.c` | H.274 film grain synthesis |
| `src/h265_frame.c` | picture pool, planes, per-frame metadata, row progress |
| `src/h265_thread.c` | frame-threaded job pool |
| `src/h265dec.c` | public API, NAL dispatch, access-unit handling |
| `src/h265_dsp_sse2.c`, `src/h265_dsp_avx2.c`, `src/h265_dsp_neon.c` | vector kernels |

The decoder keeps per-picture arrays for skip flags, coding-tree depth, coded
block flags, intra modes, QP, boundary strengths, SAO/deblock parameters,
motion fields and per-CTB reference lists.  A picture is fully independent of
later pictures' mutable state, which is what makes frame threading safe.

## Tests

`tests/gen_streams.sh` encodes the 320x240 corpus with libx265: B-pyramid,
weighted bi-prediction (a fade makes x265 signal per-PU weight tables),
multiple slices, 8/2 tiles, WPP on/off, AMP, lossless, scaling lists,
constrained intra, every chroma format at 8, 10 and 12 bits, monochrome,
temporal layers, open GOP and field-coded (interlaced signalling) output.  It also injects an H.274 film-grain SEI.
`tests/compare.sh` decodes each stream with both decoders and compares the raw
YUV bytes:

```sh
sh tests/gen_streams.sh
THREADS=1  sh tests/compare.sh
THREADS=4  sh tests/compare.sh
THREADS=12 sh tests/compare.sh
```

On this machine all 33 test streams (including 12-bit, film grain, field-coded
and the P-only 720p WPP clip) are bit-exact at 1, 4 and 12 slots; the 12-slot
run is repeated with
`H265_WPP_THREADS=1` to exercise WPP row parallelism.  `tests/avcc_check.sh`
additionally feeds hvcC
extradata and AVCC NALs, one NAL per packet, for a B-frame and a multi-slice
stream.  The 720p and 1080p benchmark clips produced by `tests/gen_bench.sh`
are also bit-exact at every tested thread count.
Damaged input is survivable: truncated, bit-flipped and byte-injected variants
of the corpus decode without a crash or hang.

For AArch64, `tests/check_arm64.sh` cross-compiles with
`aarch64-linux-gnu-gcc` and runs the corpus under `qemu-aarch64-static`,
comparing the NEON output against the C fallback (`H265_DISABLE_SIMD=1`) and
checking 2/4/8-thread output against the single-thread result.  The full
corpus plus the 720p benchmark clips are bit-exact on AArch64 as well.

## Performance

Measured on the same Windows machine, both sides discarding pictures, medians
of three runs.  A 1280x720 300-frame High stream encoded with libx265:

| threads | h265dec fps | FFmpeg 9.0.2 fps | ratio |
| --- | ---: | ---: | ---: |
| 1 | 456 | 485 | 0.94 |
| 2 | 740 | 711 | 1.04 |
| 4 | 992 | 1079 | 0.92 |
| 8 | 1089 | 1163 | 0.94 |
| 12 | 1127 | 1288 | 0.88 |
| 16 | 1135 | 1333 | 0.85 |

A second 1920x1080 90-frame clip: 206 fps on one slot, 443 fps on eight and
458 fps on sixteen (FFmpeg 200/511/497).  A 720p P-only 300-frame clip (no
B-frame parallelism): 336 fps on one slot, 508 fps on eight and 682 fps on
sixteen (FFmpeg 354/583/794).

`tests/bench_compare.sh` reproduces these numbers (it also accepts
`FFMPEG_FLAGS`, e.g. `FFMPEG_FLAGS="-cpuflags 0"` for an FFmpeg build without
SIMD, and honours `H265_DISABLE_SIMD=1` for this decoder's C fallback):

```sh
CLIP=testdata/bench_720p.h265 REPS=3 sh tests/bench_compare.sh
CLIP=testdata/bench_1080p.h265 REPS=3 sh tests/bench_compare.sh
CLIP=testdata/bench_p720.h265 REPS=3 sh tests/bench_compare.sh
```

Against FFmpeg 9.0.2 with all SIMD disabled (`-cpuflags 0`), measured in the
same session with this decoder on its normal path and on its own C fallback
(`H265_DISABLE_SIMD=1`), medians of five alternating runs:

| clip | threads | h265dec | h265dec no-SIMD | FFmpeg 9.0.2 no-SIMD | SIMD / FFmpeg no-SIMD | no-SIMD / FFmpeg no-SIMD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p | 1 | 396 | 292 | 289 | 1.37 | 1.01 |
| 720p | 4 | 943 | 694 | 682 | 1.38 | 1.02 |
| 720p | 8 | 1007 | 805 | 811 | 1.24 | 0.99 |
| 1080p | 1 | 183 | 146 | 122 | 1.50 | 1.20 |
| 1080p | 4 | 398 | 302 | 277 | 1.44 | 1.09 |
| 1080p | 8 | 419 | 334 | 310 | 1.35 | 1.07 |
| P720 | 1 | 322 | 226 | 226 | 1.42 | 1.00 |
| P720 | 4 | 452 | 358 | 371 | 1.22 | 0.96 |
| P720 | 8 | 513 | 402 | 423 | 1.21 | 0.95 |

With SIMD this decoder is 1.21–1.50x faster than FFmpeg without SIMD; on the
pure C path the two are within a few percent (0.95–1.20x).  SIMD buys this
decoder 1.2–1.5x over its own scalar path, so the remaining gap to FFmpeg's
assembly comes from its hand-written kernels rather than the algorithms.
Details are in `ANALYSIS.md`.

Single-thread time is still dominated by CABAC entropy decoding; the
reconstruction kernels are now on par with FFmpeg's assembly.  The
motion-compensation filters use `pmaddubsw`/`pmaddwd` 16-bit dot products,
integer-MV weighted prediction and edge emulation work directly on reference
pixels, deblocking filters a whole 8-line edge per call with 16-bit lanes, SAO
runs 16 pixels per iteration, the IDCT handles eight columns per pass, and
decoded frames keep their plane and metadata allocations when a slot is
reused.  Frame-level workers publish filtered rows with a one-row lag and use
a separate work condition variable so idle workers are not woken for every
completed job.

## Licence

The algorithms and constant tables are derived from ITU-T Rec. H.265 and from
FFmpeg's HEVC decoder and H.274 film-grain implementation (LGPL-2.1+), which
are used as the reference implementation.  **If you ship this, you carry the
LGPL obligations of that derived code.**
