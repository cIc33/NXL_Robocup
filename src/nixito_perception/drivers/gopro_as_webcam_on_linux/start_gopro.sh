#!/bin/bash

sudo gopro webcam -a -n -r "480" &
GOPRO_PID=$!

echo "Esperando video de GoPro..."

while ! ls /dev/video42 2>/dev/null; do
    sleep 0.5
done

echo "GoPro lista!"

trap "kill $GOPRO_PID" EXIT

wait $GOPRO_PID
