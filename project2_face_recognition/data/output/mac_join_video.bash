gls -v *.jpg > input.txt
sed -i'' -e 's/^/file /g' input.txt
ffmpeg -r 24 -f concat -i input.txt -c:v libx264 output.mp4
