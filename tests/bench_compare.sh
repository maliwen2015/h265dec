#!/bin/sh
# Compare software decode throughput against FFmpeg for one clip.
# Both sides discard decoded pictures.  Prints medians of REPS runs.
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-./h265dec_example}
CLIP=${CLIP:-testdata/bench_720p.h265}
REPS=${REPS:-3}
THREADS=${THREADS:-"1 2 4 8 12 16"}
FFMPEG_FLAGS=${FFMPEG_FLAGS:-}
FRAMES=$(ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$CLIP")

median() {
    sort -n | awk '{ a[NR] = $1 } END {
        if (NR == 0) { print "0"; exit }
        if (NR % 2) print a[(NR + 1) / 2];
        else print (a[NR / 2] + a[NR / 2 + 1]) / 2;
    }'
}

printf 'frames=%s\n' "$FRAMES"
printf '%-8s %12s %12s %8s\n' threads "h265dec" "ffmpeg" "ratio"
for t in $THREADS; do
    ours=""
    ref=""
    i=0
    while [ "$i" -lt "$REPS" ]; do
        v=$($BIN "$CLIP" - "$t" 2>&1 |
            sed -n 's/.*fps=\([0-9.][0-9.]*\).*/\1/p' | tail -1)
        ours="$ours
$v"
        v=$(ffmpeg -hide_banner $FFMPEG_FLAGS -loglevel info -threads "$t" -benchmark \
            -i "$CLIP" -f null - 2>&1 |
            sed -n 's/.*bench: utime=.* rtime=\([0-9.][0-9.]*\).*/\1/p' |
            tail -1)
        ref="$ref
$(awk -v f="$FRAMES" -v r="$v" 'BEGIN { print (r > 0) ? f / r : 0 }')"
        i=$((i + 1))
    done
    o=$(printf '%s\n' "$ours" | sed '/^$/d' | median)
    r=$(printf '%s\n' "$ref" | sed '/^$/d' | median)
    ratio=$(awk -v o="$o" -v r="$r" 'BEGIN { print (r > 0) ? o / r : 0 }')
    printf '%-8s %12.1f %12.1f %8.3f\n' "$t" "$o" "$r" "$ratio"
done
