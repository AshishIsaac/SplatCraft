#!/usr/bin/env sh
# First start: download the LPIPS weights into the /workspace volume (benchmark scenes are fetched on
# demand by `splatcraft data download`). SPLATCRAFT_SKIP_SETUP=1 skips it (CI, viewing only).
set -e
if [ -z "$SPLATCRAFT_SKIP_SETUP" ] && [ ! -d "$SPLATCRAFT_HOME/models/torch" ]; then
  echo "First start: downloading the LPIPS weights into $SPLATCRAFT_HOME (once) ..."
  splatcraft models download || echo "LPIPS weights unavailable - evaluation will retry later"
fi
exec splatcraft "$@"
