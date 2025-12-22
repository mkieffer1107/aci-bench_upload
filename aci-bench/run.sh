#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by flags)
HF_USERNAME_DEFAULT="mkieffer"
HF_REPO_NAME_DEFAULT="ACI-Bench"
PRIVATE_DEFAULT="false"

HF_USERNAME="$HF_USERNAME_DEFAULT"
HF_REPO_NAME="$HF_REPO_NAME_DEFAULT"
PRIVATE="$PRIVATE_DEFAULT"

usage() {
  cat <<'EOF'
Usage: ./run.sh [--username <HF_USERNAME>] [--repo <HF_REPO_NAME>] [--private <true|false>]

Examples:
  ./run.sh
  ./run.sh --username mkieffer --repo ACI-Bench --private true
EOF
}

# simple long-flag parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    --username)
      HF_USERNAME="${2:-}"; shift 2 ;;
    --repo|--reponame|--repo-name)
      HF_REPO_NAME="${2:-}"; shift 2 ;;
    --private)
      PRIVATE="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

# Normalize PRIVATE to lowercase
PRIVATE="$(echo "$PRIVATE" | tr '[:upper:]' '[:lower:]')"
if [[ "$PRIVATE" != "true" && "$PRIVATE" != "false" ]]; then
  echo "Error: --private must be 'true' or 'false' (got '$PRIVATE')" >&2
  exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download ACI-Bench CSV splits
downloads=(
  "test1.csv|https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/clinicalnlp_taskB_test1.csv"
  "test2.csv|https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/clinicalnlp_taskC_test2.csv"
  "test3.csv|https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/clef_taskC_test3.csv"
  "train.csv|https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/train.csv"
  "valid.csv|https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/valid.csv"
)

for entry in "${downloads[@]}"; do
  IFS='|' read -r file url <<< "$entry"
  if [[ -z "${file}" || -z "${url}" ]]; then
    echo "Internal error: malformed download spec: $entry" >&2
    exit 1
  fi

  if [[ ! -f "data/${file}" ]]; then
    echo "Downloading ${file}..."
    curl -L -o "data/${file}" "${url}"
  else
    echo "${file} already exists, skipping download."
  fi
done

echo "Downloads complete."

# Run the uploader, forwarding the args
python3 upload_to_hf.py \
  --hf_username "$HF_USERNAME" \
  --hf_repo_name "$HF_REPO_NAME" \
  --private "$PRIVATE"
