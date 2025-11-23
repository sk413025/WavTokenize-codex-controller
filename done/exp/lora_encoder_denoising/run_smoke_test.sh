#!/bin/bash

# Smoke Test Runner for LoRA Encoder Denoising
# This runs a quick 2-5 minute validation test

set -e  # Exit on error

echo "================================="
echo "  LoRA Encoder Denoising"
echo "  Smoke Test Runner"
echo "================================="
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Run smoke test
echo "Running smoke test..."
python smoke_test.py

echo ""
echo "================================="
echo "Smoke test completed!"
echo "================================="
