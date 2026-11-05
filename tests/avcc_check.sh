#!/bin/sh
# hvcC extradata + length-prefixed AVCC input, one NAL per packet.
set -e
cd "$(dirname "$0")/.."
mkdir -p testdata/tmp/avcc
BIN=${BIN:-testdata/tmp/avcc/avcc_check}
STREAMS=${STREAMS:-"b_frames slices"}

gcc -O2 -std=c99 -I. tests/avcc_check.c libh265dec.a -o "$BIN"

for name in $STREAMS; do
    SRC=testdata/$name.h265
    WORK=testdata/tmp/avcc/$name
    [ -e "$SRC" ] || { echo "avcc: $SRC missing"; exit 1; }
    mkdir -p "$WORK"
    ffmpeg -hide_banner -loglevel error -y -i "$SRC" -c copy "$WORK/in.mp4"
    ${PYTHON:-py} - "$WORK/in.mp4" "$WORK/hvcc.bin" "$SRC" "$WORK/in.avcc" <<'PY'
import struct, sys
mp4, hvcc, annexb, avcc = sys.argv[1:5]
data = open(mp4, 'rb').read()
i = data.find(b'hvcC')
if i < 4:
    raise SystemExit('hvcC not found')
size = struct.unpack('>I', data[i - 4:i])[0]
open(hvcc, 'wb').write(data[i + 4:i - 4 + size])

src = open(annexb, 'rb').read()
starts = []
p = 0
while True:
    j = src.find(b'\x00\x00\x01', p)
    if j < 0:
        break
    starts.append(j)
    p = j + 3
out = bytearray()
for k, s in enumerate(starts):
    b = s + 3
    e = starts[k + 1] if k + 1 < len(starts) else len(src)
    while e > b and src[e - 1] == 0:
        e -= 1
    nal = src[b:e]
    out += len(nal).to_bytes(4, 'big') + nal
open(avcc, 'wb').write(out)
PY
    "$BIN" "$WORK/hvcc.bin" "$WORK/in.avcc" "$WORK/our.yuv" 2>"$WORK/err.txt"
    PIX=$(ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt \
          -of default=nw=1:nk=1 "$SRC")
    ffmpeg -hide_banner -loglevel error -y -i "$SRC" -f rawvideo \
        -pix_fmt "$PIX" "$WORK/ref.yuv"
    if cmp -s "$WORK/ref.yuv" "$WORK/our.yuv"; then
        echo "PASS avcc/$name ($PIX)"
    else
        echo "FAIL avcc/$name: output differs"
        exit 1
    fi
done
echo "avcc: hvcC/AVCC streams bit-exact"
