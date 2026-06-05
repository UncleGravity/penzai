#!/usr/bin/env bash
# run.sh - build, deploy, and verify the KR260 DMA loopback app.
set -euo pipefail
cd "$(dirname "$0")"

./build.sh
./deploy.sh
./test.sh
