#!/usr/bin/env bash
# Fetch upstream Magenta DrumsRNN bundle without adding it to Git.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="$script_dir/models"
model_path="$model_dir/drum_kit_rnn.mag"
tmp_path="$model_path.part"
url="https://download.magenta.tensorflow.org/models/drum_kit_rnn.mag"

mkdir -p "$model_dir"

if [[ -s "$model_path" ]]; then
  printf 'Existing model: %s\n' "$model_path"
else
  command -v curl >/dev/null || { echo 'Error: curl is required' >&2; exit 1; }
  trap 'rm -f "$tmp_path"' EXIT
  echo "Downloading from: $url"
  curl --fail --location --retry 3 --connect-timeout 15 \
    --output "$tmp_path" "$url"
  [[ -s "$tmp_path" ]] || { echo 'Error: downloaded file is empty' >&2; exit 1; }
  mv "$tmp_path" "$model_path"
  trap - EXIT
fi

printf 'File: %s\n' "$model_path"
printf 'Bytes: '
wc -c < "$model_path" | tr -d '[:space:]'
printf '\nSHA-256: '
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$model_path"
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$model_path"
else
  echo 'Neither shasum nor sha256sum is available'
fi

echo 'Note: size/hash displayed only; no trusted upstream reference hash is known yet.'
