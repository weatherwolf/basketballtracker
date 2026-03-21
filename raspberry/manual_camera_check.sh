#!/bin/sh
set -e

mkdir -p videos
rpicam-vid -t 600 -o videos/manual_camera_check.mp4 --rotation 180 --mode 1536:864:120 --framerate 120
