#!/bin/bash
# Run ON THE VICTIM. Copies the model files and the monitor from the Windows shared folder
# into ~/ids, but only the files whose checksum changed (SHA256SUMS written by publish_models.py).
# Usage:  bash sync_from_windows.sh
# Needs the shared folder mounted:  sudo mount -t vboxsf -o uid=$(id -u),gid=$(id -g) F_DRIVE /media/sf_F_DRIVE

SRC=/media/sf_F_DRIVE/all_10_csv_manual/live_detection/victim
DST=$HOME/ids
SUMS="$SRC/models/SHA256SUMS"

if [ ! -f "$SUMS" ]; then
  echo "cannot find $SUMS - is the share mounted? (see the usage line above)"; exit 1
fi
mkdir -p "$DST/models" || exit 1

changed=0
while read -r want name; do
  [ -z "$name" ] && continue
  have=$(sha256sum "$DST/models/$name" 2>/dev/null | awk '{print $1}')
  if [ "$want" = "$have" ]; then
    echo "ok       $name"
  else
    cp "$SRC/models/$name" "$DST/models/$name" && echo "UPDATED  $name" && changed=1
  fi
done < "$SUMS"

# the monitor itself
want=$(sha256sum "$SRC/live_monitor.py" | awk '{print $1}')
have=$(sha256sum "$DST/live_monitor.py" 2>/dev/null | awk '{print $1}')
if [ "$want" = "$have" ]; then
  echo "ok       live_monitor.py"
else
  cp "$SRC/live_monitor.py" "$DST/live_monitor.py" && echo "UPDATED  live_monitor.py" && changed=1
fi

if [ "$changed" = "1" ]; then
  echo "NOTE: restart live_monitor.py so it loads the new files (it reads models only at start)."
else
  echo "nothing changed."
fi
