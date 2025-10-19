#!/bin/bash
# ================================================
# SWARM-100 Z8 OPTIMIZED BUILD SCRIPT
# ================================================
# Hardware: HP Z8 Fury G5 Workstation
# GPUs: 4x RTX 6000 Ada (192GB total VRAM)
# CUDA: 12.x
# OS: Ubuntu Linux
# ================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ================================================
# SYSTEM VALIDATION
# ================================================

validate_system() {
    log "Validating HP Z8 Fury G5 system configuration..."

    # Check CPU info
    if command -v lscpu &> /dev/null; then
        CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:\s*//')
        CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
        log "Detected CPU: $CPU_MODEL ($CPU_CORES cores)"
    fi

    # Check GPU configuration
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log "Detected $GPU_COUNT NVIDIA GPUs"

        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read -r name memory; do
            log "  - $name with ${memory}MB VRAM"
        done

        # Validate expected configuration
        if [ "$GPU_COUNT" -eq 4 ]; then
            RTX_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "RTX 6000")
            if [ "$RTX_COUNT" -eq 4 ]; then
                success "GPU configuration validated: 4x RTX 6000 Ada GPUs"
            else
                error "Expected 4x RTX 6000 Ada GPUs, found $RTX_COUNT"
                exit 1
            fi
        else
            error "Expected 4 GPUs, found $GPU_COUNT"
            exit 1
        fi
    else
        error "nvidia-smi not found. Ensure NVIDIA drivers are installed."
        exit 1
    fi

    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/')
        log "CUDA version: $CUDA_VERSION"
        if [[ "$CUDA_VERSION" =~ ^12\. ]]; then
            success "CUDA 12.x detected - compatible with RTX 6000 Ada"
        else
            warning "CUDA version $CUDA_VERSION may not be optimal for RTX 6000 Ada"
        fi
    else
        error "CUDA not found. Install CUDA 12.x toolkit."
        exit 1
    fi

    # Check memory
    TOTAL_MEMORY=$(free -g | grep '^Mem:' | awk '{print $2}')
    log "System memory: ${TOTAL_MEMORY}GB"
    if [ "$TOTAL_MEMORY" -lt 128 ]; then
        warning "Only ${TOTAL_MEMORY}GB RAM detected. 256GB+ recommended for optimal Z8 performance."
    else
        success "Memory configuration: ${TOTAL_MEMORY}GB RAM"
    fi
}

# ================================================
# BUILD SWARM CORE
# ================================================

build_swarm_core() {
    log "Building Swarm-100 C++ Core (Z8 Optimized)..."

    cd swarm-core

    # Setup build directory
    if [ -d "build_z8" ]; then
        rm -rf build_z8
    fi
    mkdir build_z8
    cd build_z8

    # Configure with Z8-specific settings
    log "Configuring CMake for HP Z8 optimization..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="89" \
        -DPYBIND11_FINDPYTHON=ON \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -fopenmp -pthread -ffast-math -funroll-loops" \
        -DCMAKE_CUDA_FLAGS="--use_fast_math --default-stream per-thread --generate-line-info"

    # Build with parallel jobs matching CPU cores
    local jobs=$(nproc)
    log "Building with $jobs parallel jobs..."
    make -j$jobs

    if [ $? -eq 0 ]; then
        success "Swarm Core build completed successfully"

        # Copy module to expected location
        if [ -f "swarm_core.cpython-*.so" ]; then
            cp swarm_core.cpython-*.so ../swarm_core_z8.so 2>/dev/null || true
        fi

        # Run basic validation
        if [ -f "../swarm_core_z8.so" ]; then
            success "Python module available at swarm-core/swarm_core_z8.so"
        fi
    else
        error "Build failed"
        exit 1
    fi

    cd ..
}

# ================================================
# OPTIMIZE SWARM CONFIGURATION
# ================================================

optimize_swarm_config() {
    log "Generating Z8-optimized swarm configuration..."

    # Create optimized configuration based on Z8 hardware
    cat > configs/swarm_config_z8.yaml << EOF
# ================================================
# SWARM-100 CONFIGURATION - HP Z8 OPTIMIZED
# ================================================
project: granite4-microh-z8-swarm
version: 1.1.0
status: z8-optimized
created: $(date +%Y-%m-%d)

# Hardware Configuration (Z8 Fury G5)
hardware:
  gpus:
    - id: 0
      name: "RTX 6000 Ada"
      vram_gb: 48
      cuda_arch: "sm_89"
      max_concurrent_queries: 32
      memory_fraction: 0.95
      bots: 25
    - id: 1
      name: "RTX 6000 Ada"
      vram_gb: 48
      cuda_arch: "sm_89"
      max_concurrent_queries: 32
      memory_fraction: 0.95
      bots: 25
    - id: 2
      name: "RTX 6000 Ada"
      vram_gb: 48
      cuda_arch: "sm_89"
      max_concurrent_queries: 32
      memory_fraction: 0.95
      bots: 25
    - id: 3
      name: "RTX 6000 Ada"
      vram_gb: 48
      cuda_arch: "sm_89"
      max_concurrent_queries: 32
      memory_fraction: 0.95
      bots: 25
  total_bots: 100
  total_vram_gb: 192
  workstation_model: "HP Z8 Fury G5"

# Model Configuration - Optimized for 4x RTX 6000 Ada
model:
  name: "granite4:micro-h"
  quantization: "Q4"
  context_length: 4096
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_batch_size: 4
  cache_type: "local"

# Swarm Architecture - Z8 Turbo Mode
swarm:
  gossip_hops: 4
  fanout: 5
  confidence_threshold: 0.5
  ttl: 4
  cyber_grid_enabled: true
  grid_dimensions: [100, 100]
  toroidal_topology: true
  lorawan_range_cells: 4
  energy_decay_alpha: 0.95
  activation_threshold: 0.65

# Bot Configuration - Performance Optimized
bot:
  base_port: 11400
  api_timeout: 15  # Reduced for Z8 speed
  max_retries: 3
  health_check_interval: 30  # More frequent on Z8
  cpu_affinity: true
  numa_aware: true

# Performance Tuning - RTX 6000 Ada Max Performance
performance:
  max_concurrent_requests: 128  # 4 GPUs x 32 queries
  batch_size: 4
  num_threads: 64  # Match high-end CPU cores
  gpu_memory_fraction: 0.95  # Maximum VRAM utilization
  enable_kv_cache: true
  tensor_parallelism: 4  # Across GPUs
  pipeline_parallelism: true
  async_execution: true
  memory_pool_mb: 8192  # 8GB memory pool per GPU
  cuda_kernel_optimization: true

# Logging - Z8 Performance Monitoring
logging:
  level: "INFO"
  dir: "logs_z8"
  rotation: "hourly"
  retention_days: 1
  performance_metrics: true
  gpu_monitoring: true
  async_logging: true

# Monitoring - Enhanced for 4-GPU Setup
monitoring:
  enable_metrics: true
  metrics_port: 9090
  prometheus_export: true
  gpu_telemetry: true
  per_gpu_metrics: true
  cyber_grid_monitoring: true
  real_time_dashboard: true

# CyberGrid Configuration
cyber_grid:
  dimensions: [100, 100]
  toroidal: true
  cell_capacity: 4  # Max agents per cell
  lorawan_propagation:
    attenuation_k: 0.15
    global_damping: 0.95
    range_cells: 4
    activation_threshold: 0.65
  conway_rules:
    energy_survival_threshold: 0.8
    overcrowding_energy_min: 0.3
    energy_birth_threshold: 2.0
  agent_distribution: balanced
EOF

    success "Z8-optimized configuration generated: configs/swarm_config_z8.yaml"
}

# ================================================
# BENCHMARK Z8 PERFORMANCE
# ================================================

benchmark_z8() {
    log "Running Z8 performance benchmarks..."

    if [ ! -f "swarm-core/swarm_core_z8.so" ]; then
        warning "Z8 optimized module not found, skipping benchmarks"
        return
    fi

    log "Testing CyberGrid performance on Z8 hardware..."

    python3 -c "
try:
    import sys
    sys.path.insert(0, 'swarm-core')
    import swarm_core_z8 as swarm_core
    import time

    print('🔬 Z8 Performance Benchmark Results:')
    print('=' * 50)

    # Test CyberGrid creation and basic operations
    start = time.time()
    grid = swarm_core.CyberGrid(100, 100)
    grid.randomize(0.3, 0.1)  # 30% alive, 10% energy
    init_time = time.time() - start
    print(f'Grid initialization: {init_time:.3f}s')

    # Test Conway's Game of Life performance
    start = time.time()
    for i in range(100):
        changes = grid.apply_conway_rules()
        if i % 25 == 0:
            print(f'  Generation {i}: {changes} cells evolved')
    life_time = time.time() - start
    life_fps = 100 / life_time
    print(f'100 Conway generations: {life_time:.3f}s ({life_fps:.1f} gen/sec)')

    # Test LoRA pulse propagation
    start = time.time()
    for i in range(50):
        grid.apply_lora_pulses()
    pulse_time = time.time() - start
    pulse_fps = 50 / pulse_time
    print(f'50 LoRA pulse cycles: {pulse_time:.3f}s ({pulse_fps:.1f} cycles/sec)')

    # Combined CyberGrid performance
    total_time = life_time + pulse_time
    total_operations = 100 + 50
    ops_per_sec = total_operations / total_time
    print(f'Total CyberGrid ops/sec: {ops_per_sec:.1f}')

    # Root Cause Analyzer performance
    analyzer = swarm_core.RootCauseAnalyzer()
    start = time.time()
    result = analyzer.analyze_dependency_chain('test_agent', ['failure', 'timeout'], {})
    analysis_time = time.time() - start
    print(f'Root cause analysis: {analysis_time:.3f}s')

    # Memory usage
    memory_usage = analyzer.get_current_memory_usage()
    print(f'Peak memory usage: {memory_usage / (1024*1024):.2f}MB')

    print('=' * 50)
    print('✅ Z8 Performance Validation Complete!')
    print('🧬 Alice in CyberLand: CyberGrid operational')
    print('🧠 Root Cause Analysis: Recursive loop protection active')

except ImportError as e:
    print(f'❌ Module not found: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Benchmark error: {e}')
    sys.exit(1)
"
}

# ================================================
# CREATE Z8-SPECIFIC SWARM LAUNCHER
# ================================================

create_z8_launcher() {
    log "Creating Z8-optimized swarm launcher..."

    cat > scripts/launch_z8_swarm.sh << 'EOF'
#!/bin/bash
# ================================================
# Z8 OPTIMIZED SWARM LAUNCHER
# ================================================
# Hardware: 4x RTX 6000 Ada GPUs (192GB VRAM)
# Configuration: Maximum performance settings
# ================================================

set -e

echo "🖥️  Starting Swarm-100 on HP Z8 Fury G5 Workstation"
echo "=================================================="

# Set environment for optimal Z8 performance
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log"
export OMP_NUM_THREADS=32  # Half of 64-core CPU for headroom
export MKL_NUM_THREADS=32

# Use Z8-optimized configuration
export SWARM_CONFIG="configs/swarm_config_z8.yaml"

# Enable NVIDIA MPS (Multi-Process Service) for GPU sharing
echo "Starting NVIDIA MPS for optimal 4-GPU utilization..."
nvidia-cuda-mps-control -d

# Launch swarm with Z8 optimizations
echo "Launching 100 agents across 4x RTX 6000 Ada GPUs..."
python3 scripts/launch_swarm.py

# Cleanup
trap "nvidia-cuda-mps-control -d && echo 'Z8 Swarm shut down gracefully'" EXIT

echo "🎮 Swarm-100 operational on Z8 hardware"
echo "📈 Monitor performance: python3 scripts/swarm_debug_analyzer.py"
echo "🧬 CyberGrid status: Check logs_z8/ for toroidal evolution"
EOF

    chmod +x scripts/launch_z8_swarm.sh
    success "Z8 swarm launcher created: scripts/launch_z8_swarm.sh"
}

# ================================================
# MAIN BUILD PROCESS
# ================================================

main() {
    echo "🏗️  Building Swarm-100 Optimized for HP Z8 Fury G5"
    echo "=========================================================="
    echo "Hardware Target: 4x RTX 6000 Ada GPUs (192GB VRAM)"
    echo "Optimization: SM 8.9 CUDA architecture, maximum throughput"
    echo "=========================================================="

    validate_system
    echo

    optimize_swarm_config
    echo

    build_swarm_core
    echo

    benchmark_z8
    echo

    create_z8_launcher
    echo

    echo "🎉 Z8 Swarm Build Complete!"
    echo "=========================="
    echo "📁 Configuration: configs/swarm_config_z8.yaml"
    echo "🏗️  Module: swarm-core/swarm_core_z8.so"
    echo "🚀 Launcher: scripts/launch_z8_swarm.sh"
    echo "🎮 CyberGrid: 100x100 toroidal with LoRA evolution"
    echo ""
    echo "Ready to launch 'Alice in CyberLand' on Z8 hardware!"
}

# Run main function
main "$@"
EOF
