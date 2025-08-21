#!/bin/bash

INPUT_DIR="./test_images"
OUTPUT_DIR="output_images"

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.jpg; do
  filename=$(basename "$img" .jpg)
  output_chunked="${OUTPUT_DIR}/${filename}_chunked.jpg"
  output_original="${OUTPUT_DIR}/${filename}_default.jpg"

  echo "Processing $img -> $output_chunked (chunked)"


  dimensions=$(magick identify -format "%wx%h" "$img")


  time_output=$(/usr/bin/time -v python3 slicposterizer.py \
    "$img" "$output_chunked" --quality 85 --chunk-size 128 --strict-size 1920 --low-memory \
    2>&1 >/dev/null)

  max_mem_kb=$(echo "$time_output" | grep "Maximum resident set size" | awk '{print $6}')
  max_mem_mb=$(awk "BEGIN {printf \"%.2f\", $max_mem_kb/1024}")


  elapsed_sec=$(echo "$time_output" | grep "Elapsed (wall clock) time" | awk '{print $8}')

  echo "$img (chunked) : size=${dimensions} : Max mem=${max_mem_mb} MB : time=${elapsed_sec}" >> mem.log
  echo "Peak memory usage (chunked): ${max_mem_mb} MB, elapsed time: ${elapsed_sec}"

  echo "Processing $img -> $output_original (default)"
  time_output=$(/usr/bin/time -v python3 slicposterizer.py \
    "$img" "$output_original" \
    2>&1 >/dev/null)

  max_mem_kb=$(echo "$time_output" | grep "Maximum resident set size" | awk '{print $6}')
  max_mem_mb=$(awk "BEGIN {printf \"%.2f\", $max_mem_kb/1024}")

  elapsed_sec=$(echo "$time_output" | grep "Elapsed (wall clock) time" | awk '{print $8}')

  echo "$img (original) : size=${dimensions} : Max mem=${max_mem_mb} MB : time=${elapsed_sec}" >> mem.log
  echo "Peak memory usage (original): ${max_mem_mb} MB, elapsed time: ${elapsed_sec}"
done
