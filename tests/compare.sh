#!/bin/sh
# Decode every testdata/*.h265 with this decoder and with FFmpeg, compare the
# raw YUV bytes.  DATA=dir THREADS=n BIN=./h265dec_example
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-./h265dec_example}
DATA=${DATA:-testdata}
THREADS=${THREADS:-1}
TMP=${TMP:-testdata/tmp}
mkdir -p "$TMP"
fail=0
total=0
for f in "$DATA"/*.h265; do
    [ -e "$f" ] || continue
    base=$(basename "$f" .h265)
    pix=$(ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt \
          -of default=nw=1:nk=1 "$f")
    [ -n "$pix" ] || pix=yuv420p
    total=$((total + 1))
    ffmpeg -hide_banner -loglevel error -y -i "$f" -f rawvideo \
        -pix_fmt "$pix" "$TMP/ref.yuv"
    "$BIN" "$f" "$TMP/our.yuv" "$THREADS" >/dev/null 2>"$TMP/err.txt" || {
        echo "FAIL $base decoder exited nonzero"; cat "$TMP/err.txt"; fail=1; continue; }
    if cmp -s "$TMP/ref.yuv" "$TMP/our.yuv"; then
        echo "PASS $base ($pix, $THREADS slot(s))"
    else
        echo "FAIL $base: output differs"
        fail=1
    fi
done
if [ $fail -ne 0 ]; then
    echo "compare: FAILED ($total stream(s))"
    exit 1
fi
echo "compare: all $total streams bit-exact at $THREADS slot(s)"
