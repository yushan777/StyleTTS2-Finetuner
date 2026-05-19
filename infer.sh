#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEXT="${1:-Hello, you look nice today!.}"
REF="${2:-training_datasets/gillian-45/audio_normalized_24khz/PHO_fricative_sh_0002.wav}"
CHECKPOINT="${3:-output/gillian-45/epoch_2nd_00059.pth}"
OUTPUT="${4:-test_output.wav}"

source venv/bin/activate

python3 infer.py \
    --text       "$TEXT" \
    --ref        "$REF" \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUTPUT"
