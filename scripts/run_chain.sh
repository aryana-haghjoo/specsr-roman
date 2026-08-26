#!/usr/bin/env bash
#
# run_chain.sh — train the full published chain, in order.
#
#   ./scripts/run_chain.sh                 # all three stages
#   ./scripts/run_chain.sh zhead sr2       # resume from a stage
#
# Each stage consumes the previous one's best checkpoint, so they cannot be
# run in parallel. Long runs belong in a detached session:
#
#   screen -dmS chain ./scripts/run_chain.sh
#
# Set NOTIFY to a wrapper command (e.g. a start/finish mailer) to be told when
# each stage begins and ends; it is a no-op if unset.

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${SPECSR_ROMAN_RUNS:-runs}"
NOTIFY="${NOTIFY:-}"
LOGS="${LOGS:-logs}"
mkdir -p "$LOGS"

SR1_CKPT="$RUNS/sr1/sr1_ou2024_v6_best.pth"
ZHEAD_CKPT="$RUNS/zhead/zhead_ou2024_roman_med3_noisy_best.pth"

run() {
  local label="$1"; shift
  echo "=== $label  $(date) ==="
  if [[ -n "$NOTIFY" ]]; then
    $NOTIFY "$label" "$@" 2>&1 | tee "$LOGS/$label.log"
  else
    "$@" 2>&1 | tee "$LOGS/$label.log"
  fi
}

stages=("${@:-sr1 zhead sr2}")
# shellcheck disable=SC2206  # deliberate word splitting of the default list
stages=(${stages[@]})

for stage in "${stages[@]}"; do
  case "$stage" in
    sr1)
      run sr1_ou2024_v6 specsr-roman train sr1 --config configs/sr1.yaml \
          --out-dir "$RUNS/sr1"
      ;;
    zhead)
      [[ -f "$SR1_CKPT" ]] || { echo "missing $SR1_CKPT — run the sr1 stage first" >&2; exit 1; }
      run zhead_ou2024_roman_med3_noisy specsr-roman train zhead \
          --config configs/zhead.yaml --sr1-ckpt "$SR1_CKPT" \
          --out-dir "$RUNS/zhead"
      ;;
    sr2)
      [[ -f "$ZHEAD_CKPT" ]] || { echo "missing $ZHEAD_CKPT — run the zhead stage first" >&2; exit 1; }
      run sr2_ou2024_v5_romanonly specsr-roman train sr2 --config configs/sr2.yaml \
          --sr1-ckpt "$SR1_CKPT" --zhead-ckpt "$ZHEAD_CKPT" \
          --out-dir "$RUNS/sr2"
      ;;
    *)
      echo "unknown stage: $stage (expected sr1, zhead or sr2)" >&2; exit 2
      ;;
  esac
done

echo "=== chain complete  $(date) ==="
