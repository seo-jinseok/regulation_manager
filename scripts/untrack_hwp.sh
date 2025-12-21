#!/usr/bin/env bash
set -euo pipefail

if ! git ls-files -- '*.hwp' | grep -q .; then
  echo "No tracked .hwp files found."
  exit 0
fi

git ls-files -z -- '*.hwp' | xargs -0 git rm --cached --
echo "Removed tracked .hwp files from index. Files remain on disk."
