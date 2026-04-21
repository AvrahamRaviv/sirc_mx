#!/usr/bin/env bash
set -euo pipefail

# Push pending commits, bump minor git tag (vMAJOR.MINOR), push tag.
LATEST=$(git tag --list 'v[0-9]*.[0-9]*' --sort=-v:refname | head -n1 || true)
if [ -z "$LATEST" ]; then
  NEW="v1.2"
else
  MAJOR=$(echo "$LATEST" | sed -E 's/^v([0-9]+)\.([0-9]+).*/\1/')
  MINOR=$(echo "$LATEST" | sed -E 's/^v([0-9]+)\.([0-9]+).*/\2/')
  NEW="v${MAJOR}.$((MINOR + 1))"
fi

git push "$@"
git tag -a "$NEW" -m "Release $NEW"
git push origin "$NEW"
echo "Released $NEW"
