#!/bin/sh
# Long benchmark clips: 720p and 1080p, 10 seconds each.
set -e
cd "$(dirname "$0")/.."
mkdir -p testdata
echo "gen bench_720p"
ffmpeg -hide_banner -loglevel error -y -f lavfi \
    -i "testsrc2=size=1280x720:rate=30:duration=10" \
    -c:v libx265 -x265-params \
    "preset=medium:keyint=60:min-keyint=1:bframes=4:b-pyramid=1" \
    -pix_fmt yuv420p -f hevc testdata/bench_720p.h265
echo "gen bench_1080p"
ffmpeg -hide_banner -loglevel error -y -f lavfi \
    -i "testsrc2=size=1920x1080:rate=30:duration=10" \
    -c:v libx265 -x265-params \
    "preset=fast:keyint=60:min-keyint=1:bframes=4:b-pyramid=1" \
    -pix_fmt yuv420p -f hevc testdata/bench_1080p.h265
echo "done"
