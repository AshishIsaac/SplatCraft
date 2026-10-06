#!/usr/bin/env bash
# SplatCraft (macOS / Linux): finds Python 3.10 - 3.12 and runs run.py. Arguments are passed through,
# e.g.  ./start.sh --host 0.0.0.0   ./start.sh --cpu   ./start.sh --home /data/splatcraft
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in ${PYTHON:-} python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 &&
     "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
    PY="$c"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3.10 - 3.12 was not found."
  case "$(uname -s)" in
    Darwin) echo "Install it with:  brew install python@3.12   (or from https://www.python.org)" ;;
    *)      echo "Install it with:  sudo apt install python3 python3-venv python3-tk   (or your distro's equivalent)" ;;
  esac
  exit 1
fi

if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  echo "$PY is missing the venv module. On Debian/Ubuntu run:  sudo apt install python3-venv"
  exit 1
fi

exec "$PY" run.py "$@"
