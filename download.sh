#!/usr/bin/env bash
# Download a model from HuggingFace Hub using standard HTTP (no xet backend).
#
# Usage:
#   ./download.sh [MODEL_ID] [OPTIONS]
#   ./download.sh                        # reads model from config.yaml
#   ./download.sh --gguf-only            # download only *.gguf files (for ollama)
#
# Options:
#   --revision REV      Git revision / tag / commit
#   --token TOKEN       HuggingFace access token
#   --models-dir DIR    Local destination root (default: ./models)
#   --gguf-only         Only download *.gguf files (skips safetensors weights)
#   --include GLOB      Extra --include glob forwarded to hf download
#
# Environment variables:
#   HF_TOKEN      HuggingFace access token (alternative to --token)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"
MODELS_DIR="${SCRIPT_DIR}/models"

# Disable the xet storage backend; fall back to standard HTTP downloads.
export HF_HUB_DISABLE_XET=1

# ── helpers ────────────────────────────────────────────────────────────────────

die()         { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" &>/dev/null || die "'$1' is not in PATH."; }

# Parse a dot-separated key from config.yaml (no yq dependency — uses uv run).
yaml_get() {
    local key="$1"
    uv run --quiet python3 - <<PYEOF
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
keys = '${key}'.split('.')
val = cfg
for k in keys:
    val = (val or {}).get(k)
print(val if val is not None else '', end='')
PYEOF
}

# ── argument parsing ────────────────────────────────────────────────────────────

MODEL_ID=""
REVISION=""
HF_TOKEN="${HF_TOKEN:-}"
GGUF_ONLY=0
EXTRA_INCLUDE=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --revision)   REVISION="$2";               shift 2 ;;
        --token)      HF_TOKEN="$2";               shift 2 ;;
        --models-dir) MODELS_DIR="$2";             shift 2 ;;
        --gguf-only)  GGUF_ONLY=1;                 shift   ;;
        --include)    EXTRA_INCLUDE+=("$2");        shift 2 ;;
        -*)           die "Unknown flag: $1" ;;
        *)            MODEL_ID="$1";               shift   ;;
    esac
done

require_cmd uv

# Fall back to config.yaml when no model is given on the command line.
if [[ -z "$MODEL_ID" ]]; then
    [[ -f "$CONFIG" ]] || die "No model specified and ${CONFIG} not found."
    MODEL_ID="$(yaml_get model.name)"
    [[ -n "$MODEL_ID" ]] || die "model.name is not set in ${CONFIG}."
fi

if [[ -z "$REVISION" && -f "$CONFIG" ]]; then
    REVISION="$(yaml_get model.revision 2>/dev/null || true)"
fi

# ── download ───────────────────────────────────────────────────────────────────

REPO_NAME="${MODEL_ID##*/}"           # e.g. "Llama-3.1-8B-Instruct"
LOCAL_DIR="${MODELS_DIR}/${REPO_NAME}"

echo "Model   : ${MODEL_ID}"
echo "Revision: ${REVISION:-latest}"
echo "Dest    : ${LOCAL_DIR}"
echo

mkdir -p "${LOCAL_DIR}"

HF_ARGS=(
    download
    "${MODEL_ID}"
    --local-dir "${LOCAL_DIR}"
)

[[ -n "$REVISION" && "$REVISION" != "main" ]] && HF_ARGS+=(--revision "${REVISION}")
[[ -n "$HF_TOKEN" ]] && HF_ARGS+=(--token "${HF_TOKEN}")

# Filter to GGUF files only (useful for the ollama backend — skips large safetensors)
if [[ "$GGUF_ONLY" -eq 1 ]]; then
    HF_ARGS+=(--include "*.gguf")
fi

for pat in "${EXTRA_INCLUDE[@]}"; do
    HF_ARGS+=(--include "$pat")
done

uv run hf "${HF_ARGS[@]}"

echo
echo "Done. Model saved to: ${LOCAL_DIR}"
echo
echo "To serve from the local copy, set in config.yaml:"
echo "  model:"
echo "    local_path: ${LOCAL_DIR}"
if [[ "$GGUF_ONLY" -eq 1 ]]; then
    echo
    echo "GGUF-only download — use with:  uv run serve.py --backend ollama"
fi
