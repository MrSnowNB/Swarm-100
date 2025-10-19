#!/bin/bash
# Build script for swarm-core with pybind11 bindings

set -e  # Exit on any error

# Ensure we're in the swarm-core directory
cd "$(dirname "$0")"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated. Please activate swarm_testing_env first."
    echo "Run: source ../swarm_testing_env/bin/activate"
    exit 1
fi

echo "Building swarm_core module in virtual environment: $VIRTUAL_ENV"
echo "Python interpreter: $(which python3)"
echo "pybind11 available: $(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || echo "Not found")"

# Clean previous build
echo "Cleaning previous build..."
rm -rf build

# Create build directory and build
echo "Configuring with CMake..."
mkdir build && cd build
cmake ..

echo "Building with make..."
make -j$(nproc)

# Test the built module
echo "Testing module import..."
cd ..
python3 -c "
import sys
sys.path.insert(0, 'build')
try:
    import swarm_core
    print('✓ Module loads successfully')
    print(f'Module version: {getattr(swarm_core, \"__version__\", \"N/A\")}')
    print(f'Available classes: {[name for name in dir(swarm_core) if not name.startswith(\"_\")]}')
except ImportError as e:
    print(f'✗ Module failed to load: {e}')
    import sys
    sys.exit(1)
"

echo ""
echo "Build completed successfully!"
echo "To use the module in Python:"
echo "  import sys; sys.path.insert(0, 'swarm-core/build'); import swarm_core"
echo ""
echo "Or copy swarm_core.cpython-312-x86_64-linux-gnu.so to your Python package directory."
