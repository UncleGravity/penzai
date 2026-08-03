#!/usr/bin/env bash
# Run declarative out-of-context synthesis probes on the Vivado VM.

set -euo pipefail
cd "$(dirname "$0")"

normalize_text_dir() {
  local dir="$1" file tmp
  for file in "$dir"/*; do
    [[ -f "$file" ]] || continue
    case "$file" in
      *.log|*.rpt|*.tsv|*.txt)
        tmp="${file}.lf.$$"
        LC_ALL=C tr -d '\r' < "$file" > "$tmp"
        mv "$tmp" "$file"
        ;;
    esac
  done
}

usage() {
  echo "usage: ./run.sh list|all|<probe>" >&2
  exit 2
}

[[ -f ../bitstream/config.env ]] || {
  echo "ERROR: missing ../bitstream/config.env" >&2
  exit 1
}
set -a; source ../bitstream/config.env; set +a
: "${VM:?config.env must set VM}"
VM_OOC="${VM_OOC:-penzai-ooc}"

REQUEST="${1:-all}"
[[ $# -le 1 ]] || usage
if [[ "$REQUEST" == list ]]; then
  awk -F '\t' 'NR > 1 { printf "%-16s top=%-24s period=%sns\n", $1, $2, $3 }' probes.tsv
  exit 0
fi

PROBES=()
if [[ "$REQUEST" == all ]]; then
  while IFS=$'\t' read -r name _; do
    [[ "$name" == name || -z "$name" ]] || PROBES+=("$name")
  done < probes.tsv
else
  PROBES+=("$REQUEST")
fi

GIT_COMMIT="$(git -C ../.. rev-parse --short=12 HEAD)"
if [[ -n "$(git -C ../.. status --porcelain -- fpga/rtl fpga/ooc)" ]]; then
  GIT_DIRTY=1
else
  GIT_DIRTY=0
fi
REGISTRY_SHA256="$(shasum -a 256 probes.tsv | awk '{print $1}')"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${GIT_COMMIT}"
LOCAL_RUN="out/runs/$RUN_ID"
REMOTE_RUN="$VM_OOC/$RUN_ID"
mkdir -p "$LOCAL_RUN"
cp probes.tsv "$LOCAL_RUN/probes.tsv"
printf 'key\tvalue\nrun_id\t%s\ngit_commit\t%s\ngit_dirty\t%s\nregistry_sha256\t%s\nrequested\t%s\n' \
  "$RUN_ID" "$GIT_COMMIT" "$GIT_DIRTY" "$REGISTRY_SHA256" "$REQUEST" \
  > "$LOCAL_RUN/manifest.tsv"
printf 'probe\tstatus\texit_code\ttop\tperiod_ns\tdsp\tlut\tcarry8\tff\twns_ns\tfmax_mhz\n' \
  > "$LOCAL_RUN/summary.tsv"

lookup_probe() {
  local requested="$1"
  PROBE_TOP=""
  PROBE_PERIOD=""
  PROBE_SOURCES=""
  while IFS=$'\t' read -r name top period sources; do
    if [[ "$name" == "$requested" ]]; then
      PROBE_TOP="$top"
      PROBE_PERIOD="$period"
      PROBE_SOURCES="$sources"
      return 0
    fi
  done < probes.tsv
  return 1
}

metric_value() {
  local file="$1" key="$2"
  awk -F '\t' -v key="$key" '$1 == key { print $2; exit }' "$file"
}

overall_status=0
for probe in "${PROBES[@]}"; do
  lookup_probe "$probe" || { echo "ERROR: unknown OOC probe '$probe'" >&2; exit 2; }
  read -r -a source_files <<< "$PROBE_SOURCES"
  remote_sources=()
  basenames=""
  for source_file in "${source_files[@]}"; do
    [[ -f "$source_file" ]] || { echo "ERROR: $probe source missing: $source_file" >&2; exit 1; }
    base="${source_file##*/}"
    if [[ " $basenames " == *" $base "* ]]; then
      echo "ERROR: $probe has duplicate flattened basename: $base" >&2
      exit 1
    fi
    basenames+=" $base"
    remote_sources+=("$base")
  done

  remote_dir="$REMOTE_RUN/$probe"
  remote_run_win="$VM_OOC\\$RUN_ID"
  remote_dir_win="$remote_run_win\\$probe"
  local_dir="$LOCAL_RUN/$probe"
  mkdir -p "$local_dir"
  SOURCE_SHA256="$({ shasum -a 256 ooc.bat ooc_synth.tcl "${source_files[@]}"; } \
    | shasum -a 256 | awk '{print $1}')"
  printf 'key\tvalue\nprobe\t%s\ntop\t%s\nperiod_ns\t%s\nsource_sha256\t%s\nsources\t%s\n' \
    "$probe" "$PROBE_TOP" "$PROBE_PERIOD" "$SOURCE_SHA256" "$PROBE_SOURCES" \
    > "$local_dir/manifest.tsv"
  echo "== OOC $probe: top=$PROBE_TOP period=${PROBE_PERIOD}ns =="
  ssh "$VM" "if not exist $VM_OOC mkdir $VM_OOC"
  ssh "$VM" "if not exist $remote_run_win mkdir $remote_run_win"
  ssh "$VM" "if not exist $remote_dir_win mkdir $remote_dir_win"
  scp ooc.bat ooc_synth.tcl "${source_files[@]}" "$VM:$remote_dir/"

  set +e
  ssh "$VM" "cd $remote_dir && ooc.bat $PROBE_TOP $PROBE_PERIOD result ${remote_sources[*]}" \
    > "$local_dir/driver.log" 2>&1
  vivado_status=$?
  set -e
  scp "$VM:$remote_dir/result_*" "$local_dir/" 2>/dev/null || true
  scp "$VM:$remote_dir/vivado.log" "$local_dir/" 2>/dev/null || true
  normalize_text_dir "$local_dir"
  metrics="$local_dir/result_metrics.tsv"
  probe_status=$vivado_status
  collection_status=complete
  if (( vivado_status == 0 )) && [[ ! -f "$metrics" ]]; then
    probe_status=1
    collection_status=missing_metrics
  fi
  printf 'key\tvalue\nvivado_exit_code\t%s\nprobe_exit_code\t%s\ncollection_status\t%s\n' \
    "$vivado_status" "$probe_status" "$collection_status" \
    > "$local_dir/driver_status.tsv"
  if (( probe_status == 0 )); then
    probe_result=PASS
  else
    probe_result=FAIL
  fi

  if [[ -f "$metrics" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$probe" "$probe_result" "$probe_status" \
      "$(metric_value "$metrics" top)" "$(metric_value "$metrics" period_ns)" \
      "$(metric_value "$metrics" dsp)" "$(metric_value "$metrics" lut)" \
      "$(metric_value "$metrics" carry8)" "$(metric_value "$metrics" ff)" \
      "$(metric_value "$metrics" wns_ns)" "$(metric_value "$metrics" fmax_mhz)" \
      >> "$LOCAL_RUN/summary.tsv"
  else
    printf '%s\t%s\t%s\t%s\t%s\t-\t-\t-\t-\t-\t-\n' \
      "$probe" "$probe_result" "$probe_status" "$PROBE_TOP" "$PROBE_PERIOD" \
      >> "$LOCAL_RUN/summary.tsv"
  fi
  if (( probe_status != 0 )); then overall_status=1; fi
done

ln -sfn "$RUN_ID" out/runs/latest
echo "OOC artifacts: $LOCAL_RUN"
column -t -s $'\t' "$LOCAL_RUN/summary.tsv" 2>/dev/null || cat "$LOCAL_RUN/summary.tsv"
exit "$overall_status"
