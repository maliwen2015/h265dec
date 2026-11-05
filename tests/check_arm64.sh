#!/bin/sh
# AArch64 NEON validation.  Cross-compiles the decoder for arm64 and compares
# the NEON kernels against the C fallback (H265_DISABLE_SIMD=1) under QEMU.
#
# Needs: aarch64-linux-gnu-gcc and qemu-aarch64-static.
# On native AArch64 hardware set QEMU= to run the binaries directly.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
WORK=${WORK:-/tmp/h265dec-arm64}
CROSS=${CROSS:-aarch64-linux-gnu-gcc}
if [ -z "${QEMU+x}" ]; then
    QEMU="qemu-aarch64-static -L /usr/aarch64-linux-gnu"
fi

rm -rf "$WORK"
mkdir -p "$WORK"
cp -r src example_decoder.c h265dec.h Makefile "$WORK"/
cp testdata/*.h265 "$WORK"/
make -C "$WORK" -j8 CC="$CROSS" >/dev/null

cd "$WORK"
fail=0
total=0
for f in *.h265; do
    case "$f" in
        bench_*) continue ;;
    esac
    total=$((total + 1))
    $QEMU ./h265dec_example "$f" out_neon.yuv 1 >/dev/null 2>&1 || {
        echo "FAIL $f (neon run)"; fail=1; continue; }
    H265_DISABLE_SIMD=1 $QEMU ./h265dec_example "$f" out_scalar.yuv 1 \
        >/dev/null 2>&1 || { echo "FAIL $f (scalar run)"; fail=1; continue; }
    if cmp -s out_neon.yuv out_scalar.yuv; then
        echo "PASS $f"
    else
        echo "FAIL $f (output differs)"
        fail=1
    fi
done

for f in bench_720p.h265 bench_p720.h265; do
    [ -e "$f" ] || continue
    echo "== multithread $f"
    $QEMU ./h265dec_example "$f" out_1.yuv 1 >/dev/null 2>&1
    for t in 2 4 8; do
        $QEMU ./h265dec_example "$f" out_t.yuv "$t" >/dev/null 2>&1
        if cmp -s out_1.yuv out_t.yuv; then
            echo "PASS $f $t threads"
        else
            echo "FAIL $f $t threads"
            fail=1
        fi
    done
    H265_WPP_THREADS=1 $QEMU ./h265dec_example "$f" out_w.yuv 4 >/dev/null 2>&1
    if cmp -s out_1.yuv out_w.yuv; then
        echo "PASS $f WPP 4 threads"
    else
        echo "FAIL $f WPP 4 threads"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "arm64: FAILED ($total streams)"
    exit 1
fi
echo "arm64: NEON bit-exact against the C fallback ($total streams)"
