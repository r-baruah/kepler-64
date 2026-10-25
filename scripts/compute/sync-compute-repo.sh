#!/usr/bin/env bash
# Copy the engine sources and inputs into a checkout of the compute repository
# (docs/v1/08_DISTRIBUTED_COMPUTE.md §5). Results and runs are never copied
# back automatically; reduced outputs are reviewed and committed by hand.
#
#   scripts/compute/sync-compute-repo.sh <path-to-kepler-64-compute-checkout>
set -euo pipefail
src="$(cd "$(dirname "$0")/../.." && pwd)/engine"
dst="${1:?usage: $0 <compute-repo-checkout>}"
mkdir -p "$dst/engine"
# Only sources and data: every build directory (target, target-*) and generated
# packages (pkg/) stay local.
rsync -a --delete \
  --exclude 'target/' --exclude 'target-*/' --exclude 'pkg/' --exclude 'node_modules/' \
  --exclude 'runs/' --exclude 'results/' \
  "$src/" "$dst/engine/"
echo "synced engine/ from $(git -C "$src" rev-parse --short HEAD) into $dst/engine"
git -C "$src" rev-parse HEAD > "$dst/ENGINE_COMMIT"
