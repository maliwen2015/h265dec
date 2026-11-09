#!/bin/sh
# Generate the HEVC test corpus with libx265 (FFmpeg).  Needs ffmpeg with
# libx265.  Every stream is small enough to commit; references are generated
# on demand by tests/compare.sh.
set -e
cd "$(dirname "$0")/.."
mkdir -p testdata
SRC="testsrc2=size=320x240:rate=30:duration=1"

gen() {
    name=$1; pix=$2; params=$3; src=${4:-$SRC}
    echo "gen $name ($pix)"
    ffmpeg -hide_banner -loglevel error -y -f lavfi -i "$src" \
        -c:v libx265 -x265-params "$params" -pix_fmt "$pix" \
        -f hevc "testdata/$name.h265"
}

gen b_frames        yuv420p    "keyint=60:min-keyint=1:bframes=4:b-adapt=0"
gen slices          yuv420p    "keyint=30:min-keyint=1:bframes=3:slices=4"
gen slices8         yuv420p    "keyint=30:min-keyint=1:bframes=0:slices=8"
gen slices422_10    yuv422p10le "keyint=30:min-keyint=1:bframes=3:slices=2"
gen tiles           yuv420p    "keyint=30:min-keyint=1:bframes=0:tile-columns=2:tile-rows=1"
gen tiles_slices    yuv420p    "keyint=30:min-keyint=1:bframes=0:tile-columns=2:slices=4"
gen nowpp           yuv420p    "keyint=30:min-keyint=1:bframes=0:no-wpp=1"
gen amp             yuv420p    "keyint=30:min-keyint=1:bframes=0:amp=1"
gen lossless        yuv420p    "keyint=30:min-keyint=1:bframes=0:lossless=1"
gen cqm             yuv420p    "keyint=30:min-keyint=1:bframes=3:scaling-list=default"
gen constrained     yuv420p    "keyint=30:min-keyint=1:bframes=0:constrained-intra=1"
gen intra           yuv420p    "keyint=1:bframes=0"
gen h8_422          yuv422p    "keyint=30:min-keyint=1:bframes=3"
gen h8_444          yuv444p    "keyint=30:min-keyint=1:bframes=3"
gen h8_gray         gray       "keyint=30:min-keyint=1:bframes=3"
gen h10_420         yuv420p10le "keyint=30:min-keyint=1:bframes=3"
gen h10_422         yuv422p10le "keyint=30:min-keyint=1:bframes=3"
gen h10_444         yuv444p10le "keyint=30:min-keyint=1:bframes=3"
gen h10_gray        gray10le   "keyint=30:min-keyint=1:bframes=3"
gen h12_420         yuv420p12le "keyint=30:min-keyint=1:bframes=3"
gen h12_422         yuv422p12le "keyint=30:min-keyint=1:bframes=3"
gen h12_444         yuv444p12le "keyint=30:min-keyint=1:bframes=3"
gen h12_gray        gray12le   "keyint=30:min-keyint=1:bframes=3"
gen cqm10           yuv420p10le "keyint=30:min-keyint=1:bframes=3:scaling-list=default"
gen wpp_tmvp        yuv420p    "keyint=30:min-keyint=1:bframes=4:b-pyramid=1:temporal-layers=1"
gen opengop         yuv420p    "keyint=25:min-keyint=25:open-gop=1:bframes=4:b-pyramid=1"
gen sao             yuv420p    "keyint=30:min-keyint=1:bframes=3:sao=1"
gen field           yuv420p    "keyint=30:min-keyint=1:bframes=2:interlace=1"
# fade in/out forces x265 to signal per-PU weighted bi-prediction tables
gen weighted        yuv420p    "keyint=30:min-keyint=1:bframes=4:weightp=2:weightb=1" \
    "testsrc2=size=320x240:rate=30:duration=1,fade=t=in:st=0:d=0.5,fade=t=out:st=0.5:d=0.5"

# Inject an H.274 film-grain SEI (model 0, 8-bit 4:2:0) into b_frames.
${PYTHON:-py} - <<'PY'
import re

def ue(v):
    n = v + 1
    b = bin(n)[2:]
    return '0' * (len(b) - 1) + b

def se(v):
    return ue(2 * abs(v) - (1 if v > 0 else 0))

bits = '0' + '00' + '0' + '00' + '0010' + '110'
bits += '00000000' + '010' + '00000000' + '11111111' + se(20) + se(4) + se(4)
bits += '00000000' + '010' + '00000000' + '11111111' + se(8) + se(4) + se(4)
bits += '1'
bits += '0' * ((-len(bits)) % 8)
payload = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
escaped = bytearray()
zeros = 0
for b in payload:
    if zeros >= 2 and b <= 3:
        escaped.append(3)
        zeros = 0
    escaped.append(b)
    zeros = zeros + 1 if b == 0 else 0
sei = bytes([0x4e, 0x01, 19, len(escaped)]) + bytes(escaped) + b'\x80'

data = open('testdata/b_frames.h265', 'rb').read()
pos = [
    m.start() for m in re.finditer(b'\x00\x00\x01', data)
]
insert = None
for start in pos:
    p = start + 3
    if p + 1 < len(data) and ((data[p] >> 1) & 0x3f) <= 31:
        insert = start
        break
if insert is None:
    raise SystemExit('no VCL NAL found')
open('testdata/grain.h265', 'wb').write(
    data[:insert] + b'\x00\x00\x01' + sei + data[insert:])
PY
echo "gen grain (yuv420p)"
echo "done"
