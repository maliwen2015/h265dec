#!/bin/sh
# Throughput of this decoder against FFmpeg on the benchmark clips.  Both
# sides discard the pictures; medians of REPS runs.  Informational.
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-./h265dec_example}
CLIP=${CLIP:-testdata/bench_720p.h265}
REPS=${REPS:-3}
THREADS=${THREADS:-"1 4 8 12"}
for t in $THREADS; do
    i=0
    while [ $i -lt $REPS ]; do
        $BIN "$CLIP" - "$t" 2>&1 | sed "s/^/ours t=$t /"
        i=$((i + 1))
    done
done
echo "ffmpeg reference:"
ffmpeg -hide_banner -loglevel info -threads 1 -benchmark -i "$CLIP" -f null - 2>&1 |
    grep -E "bench: utime" | sed 's/^/ffmpeg /'
