#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PENZAI_MEM="${PENZAI_MEM:-xrt}"
export PENZAI_PORT="${PENZAI_PORT:-29091}"

exec "$DIR/kr260-fake-smoke.sh"
