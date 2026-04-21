#!/usr/bin/env bash
set -euo pipefail

# Bump minor version in setup.py, commit, push.
NEW=$(python3 - <<'PY'
import re, pathlib
p = pathlib.Path("setup.py")
s = p.read_text()
m = re.search(r'version="(\d+)\.(\d+)"', s)
major, minor = int(m.group(1)), int(m.group(2))
new = f"{major}.{minor + 1}"
p.write_text(re.sub(r'version="\d+\.\d+"', f'version="{new}"', s))
print(new)
PY
)

git add setup.py
git commit -m "Bump version to ${NEW}"
git push "$@"
echo "Pushed v${NEW}"
