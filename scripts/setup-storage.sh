#!/usr/bin/env bash
# One-time root setup (run with sudo):
#   1. Mount the NTFS Storage drive at boot via /etc/fstab.
#   2. Keep Docker's data on Storage, inside an ext4 image file
#      (Docker can't use NTFS directly), mounted at /mnt/docker-data.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo"; exit 1; }

UUID=7A40A25040A21345
STORAGE="${STORAGE:-/media/$USER/Storage}"
IMG=$STORAGE/docker-data.img
DATA=/mnt/docker-data
SIZE=16G

findmnt -rn "$STORAGE" >/dev/null || { echo "Storage is not mounted at $STORAGE"; exit 1; }
cp -p /etc/fstab "/etc/fstab.bak-$(date +%Y%m%d%H%M%S)"

echo "== fstab: Storage drive"
if ! grep -q "UUID=$UUID" /etc/fstab; then
  echo "UUID=$UUID $STORAGE ntfs3 uid=1000,gid=1000,iocharset=utf8,prealloc,nofail,x-systemd.device-timeout=15s,x-gvfs-show 0 0" >> /etc/fstab
fi

echo "== Docker data image ($SIZE) on Storage"
if [ ! -s "$IMG" ]; then
  # ntfs3 doesn't support fallocate; write zeros so the space is really allocated.
  dd if=/dev/zero of="$IMG" bs=16M count=$(( ${SIZE%G} * 64 )) status=progress conv=fsync
  mkfs.ext4 -F -q -L docker-data "$IMG"
fi
mkdir -p "$DATA"
if ! grep -q " $DATA " /etc/fstab; then
  echo "$IMG $DATA ext4 loop,nofail,x-systemd.requires-mounts-for=$STORAGE 0 0" >> /etc/fstab
fi
findmnt --verify --tab-file /etc/fstab
systemctl daemon-reload
findmnt -rn "$DATA" >/dev/null || mount "$DATA"

echo "== Point Docker at $DATA"
systemctl stop docker.service docker.socket
cat > /etc/docker/daemon.json <<EOF
{
  "data-root": "$DATA",
  "features": { "containerd-snapshotter": false }
}
EOF
mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/storage.conf <<EOF
[Unit]
RequiresMountsFor=$DATA
EOF
systemctl daemon-reload
systemctl start docker.socket docker.service

docker info --format 'Docker root: {{.DockerRootDir}} ({{.Driver}})'
df -h "$DATA" | tail -1
echo "Done."
