#!/bin/bash

# Script to run benchmarks on different image sizes

echo "Generating test images..."
./target/generate_test_images 256 test_small.data
./target/generate_test_images 1024 test_medium.data
./target/generate_test_images 4096 test_large.data

echo ""
echo "Running benchmarks..."
echo ""

echo "=== Small image (256x256) ==="
./target/benchmark test_small.data
echo ""

echo "=== Medium image (1024x1024) ==="
./target/benchmark test_medium.data
echo ""

echo "=== Large image (4096x4096) ==="
./target/benchmark test_large.data

rm test_small.data test_medium.data test_large.data
