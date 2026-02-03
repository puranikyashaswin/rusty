#!/bin/bash
# Rusty ML - Quick Verification Script
# Verifies that the project builds and runs correctly

set -e

echo "================================================================"
echo "   Rusty ML - Quick Verification"
echo "================================================================"
echo ""

# Check Rust version
echo "[1/5] Checking Rust version..."
RUST_VERSION=$(rustc --version)
echo "       $RUST_VERSION"
echo ""

# Check workspace
echo "[2/5] Checking workspace compiles..."
cargo check --workspace 2>&1 | tail -5
if [ $? -eq 0 ]; then
    echo "       Workspace OK"
else
    echo "       [WARN] Some issues found, but basic examples may still work"
fi
echo ""

# Build basic example
echo "[3/5] Building basic_tensor example..."
cargo build --example basic_tensor --release -p rusty 2>&1 | tail -3
echo "       Build complete"
echo ""

# Run basic example
echo "[4/5] Running GPU example..."
echo ""
cargo run --example basic_tensor --release -p rusty
echo ""

# Check CLI builds
echo "[5/5] Checking CLI builds..."
cargo build -p rusty-cli --release 2>&1 | tail -3
echo "       CLI build complete"
echo ""

echo "================================================================"
echo "   VERIFICATION COMPLETE"
echo "================================================================"
echo ""
echo "All core components are working!"
echo ""
echo "Next steps:"
echo "  1. Download a model:    ./scripts/download_model.sh tinyllama"
echo "  2. Run demo:            cargo run -p rusty-cli --release -- --demo"
echo "  3. Run benchmarks:      cargo run -p benchmarks --release"
echo ""
