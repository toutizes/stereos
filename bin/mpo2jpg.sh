#!/bin/bash -eux
# Create parallel stereo pairs from MPO files.
#
# With no parameters run on all images from the Fuji SD Card.
#
# Requires exiftool-5.20 and ImageMagick
# $ sudo port install p5.20-image-exiftool ImageMagick
tmp="/tmp/$$"

DESTDIR="$HOME/Pictures/Stereos/$(date '+%Y')/$(date '+%Y-%m-%d')"
mkdir -p "$DESTDIR"

mpo2jpg() {
    left="${tmp}-L.jpg"
    rm -f "$left"
    exiftool-5.20 -q -trailer:all= "$1" -o "$left"
    right="${tmp}-R.jpg"
    exiftool-5.20 "$1" -mpimage2 -b > "$right"
    bn=$(basename "$1" .MPO)
    bn="$DESTDIR/$bn"
    montage -mode Concatenate "$left" "$right" "$bn.jpg"
    exiftool-5.20 -overwrite_original -q -tagsFromFile "$left" --MPImage2 "$bn.jpg"
}

case $# in
    0) for img in /Volumes/MATT\ W3/DCIM/*_FUJI/*.MPO; do mpo2jpg "$img"; done;;
    *) for img in "$@"; do mpo2jpg "$img"; done;;
esac
