#!/usr/bin/env bash

# Load the closed production RTL manifest into two global arrays. Callers may
# filter PENZAI_RTL_SOURCES for a smaller elaboration root, but may not add RTL
# outside this manifest.
penzai_load_production_rtl() {
  local repo="${1:?repository root is required}"
  local manifest="$repo/fpga/build/sources.f"
  local relative path basename
  local seen_paths=$'\n'
  local seen_basenames=$'\n'

  PENZAI_RTL_SOURCES=()
  PENZAI_RTL_HEADERS=()

  [[ -f "$manifest" ]] || {
    echo "ERROR: missing production RTL manifest: $manifest" >&2
    return 1
  }

  while IFS= read -r relative || [[ -n "$relative" ]]; do
    relative="${relative%%#*}"
    relative="${relative#"${relative%%[![:space:]]*}"}"
    relative="${relative%"${relative##*[![:space:]]}"}"
    [[ -n "$relative" ]] || continue

    case "/$relative/" in
      */../*|*/./*)
        echo "ERROR: non-canonical production RTL path: $relative" >&2
        return 1
        ;;
    esac
    if [[ "$relative" == /* || "$relative" =~ [[:space:]] ]]; then
      echo "ERROR: invalid production RTL path: $relative" >&2
      return 1
    fi

    if [[ "$seen_paths" == *$'\n'"$relative"$'\n'* ]]; then
      echo "ERROR: duplicate production RTL path: $relative" >&2
      return 1
    fi
    seen_paths+="$relative"$'\n'

    basename="${relative##*/}"
    if [[ "$seen_basenames" == *$'\n'"$basename"$'\n'* ]]; then
      echo "ERROR: duplicate production RTL basename: $basename" >&2
      return 1
    fi
    seen_basenames+="$basename"$'\n'

    path="$repo/fpga/rtl/$relative"
    [[ -f "$path" ]] || {
      echo "ERROR: production RTL manifest entry does not exist: $path" >&2
      return 1
    }

    case "$relative" in
      *.v|*.sv) PENZAI_RTL_SOURCES+=("$path") ;;
      *.vh) PENZAI_RTL_HEADERS+=("$path") ;;
      *)
        echo "ERROR: unsupported production RTL extension: $relative" >&2
        return 1
        ;;
    esac
  done < "$manifest"

  ((${#PENZAI_RTL_SOURCES[@]} > 0)) || {
    echo "ERROR: production RTL manifest contains no source modules" >&2
    return 1
  }
}
