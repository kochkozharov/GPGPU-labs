#!/bin/bash

echo "Generating test images and class samples..."
./target/generate_test_images 256 test_small.data test_small_classes.txt
./target/generate_test_images 1024 test_medium.data test_medium_classes.txt
./target/generate_test_images 4096 test_large.data test_large_classes.txt

echo ""
echo "Running benchmarks..."
echo ""

echo "=== Small image (256x256) ==="
./target/benchmark test_small.data test_small_classes.txt
echo ""

echo "=== Medium image (1024x1024) ==="
./target/benchmark test_medium.data test_medium_classes.txt
echo ""

echo "=== Large image (4096x4096) ==="
./target/benchmark test_large.data test_large_classes.txt

rm test_small.data test_small_classes.txt test_medium.data test_medium_classes.txt test_large.data test_large_classes.txt

