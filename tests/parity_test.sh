#!/usr/bin/env bash
# Parity test: build the same corpus with the Python reference and the Rust
# binary, then assert that retrieval returns identical top-K doc_ids on a
# small set of queries.
#
# Run from repo root:
#   bash tests/parity_test.sh
#
# Requires the Python reference dependencies and a release sift binary.
# Override PYTHON, SIFT_BIN, or SIFT_REGRESSION_DATA for non-default paths.
set -euo pipefail

PY=${PYTHON:-python3}
DATA=${SIFT_REGRESSION_DATA:-tests/data}
SIFT=${SIFT_BIN:-target/release/sift}
DS=${DS:-scifact}

command -v "$PY" >/dev/null || { echo "python not found: $PY" >&2; exit 2; }
[ -x "$SIFT" ] || { echo "sift binary not found: $SIFT" >&2; exit 2; }
[ -f "$DATA/$DS/corpus.jsonl" ] || {
  echo "missing corpus: $DATA/$DS/corpus.jsonl (set SIFT_REGRESSION_DATA)" >&2
  exit 2
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "[1/4] python build ($DS)"
$PY -c "
import sys; sys.path.insert(0, '.')
from sift_build.cli import main
sys.argv = ['sift', 'build', '--input', '$DATA/$DS/corpus.jsonl',
            '--out', '$TMP/py.sift', '--format', 'beir', '--quiet']
main()
"

echo "[2/4] rust build ($DS)"
# Match the legacy Python reference recipe explicitly. The production Rust
# defaults include normalization, title/subword weighting, bigrams, and f16.
$SIFT build --input "$DATA/$DS/corpus.jsonl" --out "$TMP/rs.sift" --format beir \
  --k-expand 10 --threshold 0.65 --title-weight 1 --subword-weight 1 \
  --no-normalize --no-bigrams --f16-postings=false --no-payload >/dev/null

echo "[3/4] starting two servers"
SIFT_ARTIFACTS=$TMP/py.sift.dir SIFT_BIND=127.0.0.1:18091 \
  bash -c "mkdir -p $TMP/py.sift.dir && cp -r $TMP/py.sift $TMP/py.sift.dir/$DS.sift && $SIFT serve" >/dev/null 2>&1 &
PIDP=$!
SIFT_ARTIFACTS=$TMP/rs.sift.dir SIFT_BIND=127.0.0.1:18092 \
  bash -c "mkdir -p $TMP/rs.sift.dir && cp -r $TMP/rs.sift $TMP/rs.sift.dir/$DS.sift && $SIFT serve" >/dev/null 2>&1 &
PIDR=$!
trap "kill $PIDP $PIDR 2>/dev/null; rm -rf $TMP" EXIT
sleep 1

echo "[4/4] comparing top-5 for canned queries"
N_MATCH=0; N_TOTAL=0
for q in "cancer mortality" "rapamycin diabetes" "physician" "covid lung" \
         "antibiotic resistance" "vitamin deficiency" "muscle pain" \
         "cell death" "breast cancer" "vaccine efficacy"; do
  py=$(curl -sS -X POST http://127.0.0.1:18091/search -H 'content-type: application/json' \
    -d "{\"index\":\"$DS\",\"q\":\"$q\",\"k\":5}" \
    | $PY -c "import sys,json;d=json.load(sys.stdin);print(' '.join(h['doc_id'] for h in d['hits']))")
  rs=$(curl -sS -X POST http://127.0.0.1:18092/search -H 'content-type: application/json' \
    -d "{\"index\":\"$DS\",\"q\":\"$q\",\"k\":5}" \
    | $PY -c "import sys,json;d=json.load(sys.stdin);print(' '.join(h['doc_id'] for h in d['hits']))")
  N_TOTAL=$((N_TOTAL+1))
  if [ "$py" = "$rs" ]; then
    N_MATCH=$((N_MATCH+1))
    printf "  ✓ %-26s\n" "$q"
  else
    printf "  ✗ %-26s\n      py:   %s\n      rust: %s\n" "$q" "$py" "$rs"
  fi
done
echo
echo "RESULT: $N_MATCH / $N_TOTAL queries had byte-identical top-5"
[ "$N_MATCH" -eq "$N_TOTAL" ] || exit 1
