#!/bin/bash

# Script to run benchmarks on different image sizes

echo "Generating test images..."
./target/generate_test_images 32 test_small.data
./target/generate_test_images 512 test_medium.data
./target/generate_test_images 2048 test_large.data

echo ""
echo "Running benchmarks..."
echo ""

echo "=== Small image (32x32) ==="
./target/benchmark test_small.data
echo ""

echo "=== Medium image (512x512) ==="
./target/benchmark test_medium.data
echo ""

echo "=== Large image (2048x2048) ==="
./target/benchmark test_large.data

rm test_small.data test_medium.data test_large.data
