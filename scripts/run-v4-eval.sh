#!/bin/bash
# Full v4 model evaluation suite
# Run after training completes and model is served
#
# Prerequisites:
#   - Model served via ollama (cve-backport:q8_0) or llama-server
#   - SSH tunnel if running remotely: ssh -L 11434:localhost:11434 ai01
#
# Usage:
#   bash scripts/run-v4-eval.sh [--url http://localhost:11434] [--model cve-backport:q8_0]

set -euo pipefail

URL="${1:-http://localhost:11434}"
MODEL="${2:-cve-backport:q8_0}"
OUTDIR="eval-results-v4-$(date +%Y%m%d)"

mkdir -p "$OUTDIR"

echo "============================================================"
echo "CVE Backport Model v4 — Full Evaluation"
echo "Server: $URL"
echo "Model: $MODEL"
echo "Output: $OUTDIR/"
echo "============================================================"
echo

# 1. Data quality validation
echo "--- [1/4] Data Quality Validation ---"
python3 modules/domain/cve-backport/eval/validate_data.py \
    modules/domain/cve-backport/data/cve-backport.jsonl \
    --strict 2>&1 | tee "$OUTDIR/01-validate-data.txt"
echo

# 2. Codegen recall (100 examples — apples-to-apples with v3)
echo "--- [2/4] Codegen Recall (n=100) ---"
python3 modules/domain/cve-backport/eval/test_recall.py \
    --url "$URL" --model "$MODEL" \
    --n 100 \
    -o "$OUTDIR/02-recall.json" 2>&1 | tee "$OUTDIR/02-recall.txt"
echo

# 3. Code safety
echo "--- [3/4] Code Safety ---"
python3 modules/domain/cve-backport/eval/test_code_safety.py \
    --url "$URL" --model "$MODEL" \
    -o "$OUTDIR/03-safety.json" 2>&1 | tee "$OUTDIR/03-safety.txt"
echo

# 4. Test generation quality (from module-owned 5-turn examples)
echo "--- [4/4] Test Generation (n=100) ---"
python3 modules/domain/cve-backport/eval/test_generation_eval.py \
    --url "$URL" --model "$MODEL" \
    --eval modules/domain/cve-backport/data/cve-backport.jsonl \
    --n 100 --min-score 0.20 \
    -o "$OUTDIR/04-test-gen.json" 2>&1 | tee "$OUTDIR/04-test-gen.txt"
echo

echo "============================================================"
echo "Evaluation complete. Results in $OUTDIR/"
echo "============================================================"
ls -la "$OUTDIR/"
