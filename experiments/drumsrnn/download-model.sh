#!/usr/bin/env bash
# Fetch Google's Magenta.js-converted DrumsRNN checkpoint with normal TLS verification.
# The legacy .mag host currently presents a mismatched TLS certificate.
set -euo pipefail

command -v curl >/dev/null || { echo "Error: curl is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "Error: python3 is required" >&2; exit 1; }
command -v shasum >/dev/null || { echo "Error: shasum is required" >&2; exit 1; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="$script_dir/models/drum_kit_rnn_tfjs"
base_url="https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/drum_kit_rnn"

mkdir -p "$model_dir"

download_file() {
  local name="$1"
  local output="$model_dir/$name"
  local partial="$output.part"
  if [[ -s "$output" ]]; then
    printf 'Already downloaded: %s\n' "$name"
    return
  fi
  printf 'Downloading: %s\n' "$name"
  rm -f "$partial"
  if ! curl --fail --show-error --location --retry 3 --connect-timeout 15 \
    --output "$partial" "$base_url/$name"; then
    rm -f "$partial"
    return 1
  fi
  if [[ ! -s "$partial" ]]; then
    rm -f "$partial"
    echo "Error: empty download: $name" >&2
    return 1
  fi
  mv "$partial" "$output"
}

download_file "config.json"
download_file "weights_manifest.json"

shards="$(python3 - "$model_dir" <<'PY'
import json
import pathlib
import re
import sys

folder = pathlib.Path(sys.argv[1])
config = json.loads((folder / "config.json").read_text())
if config.get("type") != "MusicRNN":
    raise SystemExit("Unexpected checkpoint type (expected MusicRNN)")
manifest = json.loads((folder / "weights_manifest.json").read_text())
if not isinstance(manifest, list) or not manifest:
    raise SystemExit("Missing/invalid weights manifest groups")
paths = []
for group in manifest:
    if not isinstance(group, dict) or not group.get("weights"):
        raise SystemExit("Invalid manifest group without weights")
    for name in group.get("paths", []):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise SystemExit("Invalid shard filename in manifest")
        paths.append(name)
if not paths:
    raise SystemExit("Manifest contains no binary shards")
print("\n".join(dict.fromkeys(paths)))
PY
)"

while IFS= read -r shard; do
  [[ -n "$shard" ]] || continue
  download_file "$shard"
done <<< "$shards"

printf '\nCheckpoint folder: %s\n' "$model_dir"
printf 'Downloaded file sizes and SHA-256:\n'
for name in config.json weights_manifest.json; do
  printf '  %s: ' "$name"
  wc -c < "$model_dir/$name" | tr -d '[:space:]'
  printf ' bytes\n'
  shasum -a 256 "$model_dir/$name"
done
while IFS= read -r shard; do
  [[ -n "$shard" ]] || continue
  printf '  %s: ' "$shard"
  wc -c < "$model_dir/$shard" | tr -d '[:space:]'
  printf ' bytes\n'
  shasum -a 256 "$model_dir/$shard"
done <<< "$shards"

echo "Note: this is a TF.js-converted checkpoint, NOT the legacy .mag bundle."
echo "SHA-256 values are recorded for reproducibility, not verified against an upstream reference."
