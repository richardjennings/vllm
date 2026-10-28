#!/bin/bash
# Stage the files this branch changes since BASE_COMMIT as the Docker build
# context for overlay/Dockerfile, and list them.
# Usage: overlay/build-context.sh OUTPUT_DIR
#
# Only added or modified Python files under vllm/ can be copied over the
# official image. Any other change needs a full image build and fails here.
set -euo pipefail

out=${1:?usage: overlay/build-context.sh OUTPUT_DIR}
cd "$(git rev-parse --show-toplevel)"
. overlay/base.env

if ! git merge-base --is-ancestor "$BASE_COMMIT" HEAD; then
    echo "HEAD does not contain BASE_COMMIT $BASE_COMMIT" >&2
    exit 1
fi

files=()
refused=0
while IFS=$'\t' read -r status path; do
    case "$path" in
        tests/* | docs/* | overlay/* | .github/workflows/overlay-image.yml)
            continue ;;
        vllm/*.py)
            if [[ $status == [AM] ]]; then
                files+=("$path")
                continue
            fi ;;
    esac
    echo "cannot overlay: $status $path" >&2
    refused=1
done < <(git diff --no-renames --name-status "$BASE_COMMIT" HEAD)
if (( refused )); then
    echo "these changes need a full image build" >&2
    exit 1
fi
if (( ${#files[@]} == 0 )); then
    echo "no Python changes since $BASE_COMMIT" >&2
    exit 1
fi

rm -rf "$out"
for path in "${files[@]}"; do
    mkdir -p "$out/$(dirname "$path")"
    cp "$path" "$out/$path"
done
printf '%s\n' "${files[@]}"
