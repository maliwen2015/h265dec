#!/bin/sh
# Damaged-stream safety: truncation, bit flips and byte injection.  A decoder
# exit above 128 (a signal) or a timeout fails the run.
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-./h265dec_example}
DATA=${DATA:-testdata}
SLOTS=${SLOTS:-4}
TMP=${TMP:-testdata/fuzz}
mkdir -p "$TMP"
${PYTHON:-py} - "$DATA" "$TMP" <<'PY'
import os, random, sys
src, out = sys.argv[1], sys.argv[2]
random.seed(1234)
names = [f for f in os.listdir(src) if f.endswith('.h265')][:6]
for name in names:
    data = bytearray(open(os.path.join(src, name), 'rb').read())
    for k in range(6):
        e = bytearray(data)
        if k % 3 == 0:
            e = e[:random.randint(8, max(8, len(e) - 1))]
        elif k % 3 == 1:
            for _ in range(8):
                i = random.randrange(min(len(e), 256))
                e[i] ^= 1 << random.randrange(8)
        else:
            for _ in range(8):
                i = random.randrange(len(e))
                e[i:i] = bytes([random.randrange(256)])
        open(os.path.join(out, '%s_%d.h265' % (name[:-5], k)), 'wb').write(e)
PY
fail=0
for f in "$TMP"/*.h265; do
    timeout 20 "$BIN" "$f" - "$SLOTS" >/dev/null 2>&1
    rc=$?
    if [ $rc -ge 128 ] || [ $rc -eq 124 ]; then
        echo "FAIL $f rc=$rc"
        fail=1
    fi
done
[ $fail -eq 0 ] && echo "fuzz: no crashes or hangs"
exit $fail
