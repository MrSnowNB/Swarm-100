# Swarm-100 Changelog

All notable changes to Swarm-100 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v1.0.0.html).

## [0.2.0-beta] - 2025-10-19

### Added
- **Statistical Replication Framework**: Comprehensive confidence bounds and significance testing implemented
- **Serialization Resilience Fix**: Robust NumPy object handling with JSON double-conversion for YAML compatibility
- **Parameter Cross-Validation**: Alpha (diffusion) × Sigma (noise) parameter sweep framework
- **CI Automation**: GitHub Actions workflow for linting, unit tests, replication validation, and artifact upload
- **Repository Metadata**: CITATION.cff and MIT LICENSE added
- **Publication Materials**: Aggregated results CSV, statistical validation appendix, and figure templates
- **Testing Suite Expansion**: Unit tests for serialization and statistical computation functions

### Fixed
- **NumPy YAML Serialization**: Replaced round-trip conversion with recursive JSON double-conversion to neutralize numpy.object tags (~25% processing time reduction)
- **Replication Reliability**: Full 5/5 successful replications with automated statistical analysis

### Changed
- **File Organization**: Reorganized repository structure with dedicated data/, logs/archive/, and configs/ directories
- **Statistical Reporting**: Enhanced with Levene's test for variance comparison and bootstrapped 95% CI for effect sizes
- **Documentation**: Updated to reflect statistical enhancements and publication readiness

### Security
- Hardware-locked timing prevents unintended acceleration effects
- Bounded toroidal grid ensures safety bounds on emergent behaviors

## [0.1.0-alpha] - 2025-09-15

### Added
- **Hardware-Locked Timing System**: 120 Hz steady_clock with sub-millisecond drift correction
- **Toroidal Cellular Automata**: Conway's rules coupled with LoRA energy propagation
- **Phase-Locked Communication**: Anisotropic energy diffusion on quad-pulse cycles
- **PyBind11 Integration**: C++↔Python interop for AI system coupling
- **Gemma3 Zombie Agent System**: AI-driven emergent behaviors with swarm monitoring
- **WebSocket Real-Time Dashboard**: Live swarm state broadcasting and visualization
- **Zombie Supervisor**: Automated failure detection and recovery mechanisms
- **Multi-GPU Support**: Distributed grid simulation across RTX 6000 Ada workstations

### Performance
- **Scalability Target**: 500x500 toroidal grid maintained at 120+ Hz
- **Latency**: <8ms end-to-end pulse cycle propagation

---

**Legend:**
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** in case of vulnerabilities

For more detailed information about each release, see the [release notes](https://github.com/MrSnowNB/Swarm-100/releases).
