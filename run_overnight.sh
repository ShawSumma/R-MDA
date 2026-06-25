#!/usr/bin/env bash
# Overnight training with auto-restart. If train crashes/segfaults/NaNs out, it is
# relaunched and resumes from the latest checkpoint. A VRAM watchdog kills the
# process if GPU memory runs away (protects the shared display GPU).
cd "$(dirname "$0")"
LOG=models/train.log
# append across restarts so the full loss history persists (don't truncate)

# VRAM watchdog: kill train if VRAM exceeds 20 GiB
(
  while true; do
    u=$(($(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1) / 1024 / 1024))
    if [ "$u" -gt 20000 ]; then
      echo "[watchdog] VRAM ${u} MiB > 20000 -- killing train" >> "$LOG"
      pkill -f build/train
    fi
    sleep 3
  done
) &
WD=$!
trap "kill $WD 2>/dev/null" EXIT

restart=0
while true; do
  echo "=== launch (restart #$restart) $(date) ===" >> "$LOG"
  ./build/train >> "$LOG" 2>&1
  code=$?
  echo "=== train exited code $code $(date) ===" >> "$LOG"
  # exit 0 means it hit max_steps (done); anything else = crash -> restart
  if [ "$code" -eq 0 ]; then
    echo "=== finished cleanly ===" >> "$LOG"
    break
  fi
  restart=$((restart + 1))
  if [ "$restart" -gt 200 ]; then
    echo "=== too many restarts, giving up ===" >> "$LOG"
    break
  fi
  sleep 2
done
