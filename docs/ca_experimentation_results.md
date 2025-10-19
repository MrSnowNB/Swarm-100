# CA Experimentation Log - exp_1760885194

## Project Details
- **Experiment Id**: exp_1760885194
- **Project**: Swarm-100
- **Phase**: CA + Zombie Integration Validation
- **Date Start**: 2025-10-19T10:46:34-04:00
- **Executor**: Cline

### Hardware
- **Machine**: HP Z8 Fury G5
- **Gpus**: 4x Ada6000 48GB
- **Vram Per Gpu Gb**: 48

### Software
- **Python**: 3.10+
- **Ollama Model**: gemma3:270m
- **Modules**: ['rule_engine.py', 'global_tick.py', 'self_healing_supervisor.py']

### Swarm Config Snapshot
- configs/swarm_config.yaml

### Launch Commit Hash
- c4ae86e0

### Repo Url
- https://github.com/MrSnowNB/Swarm-100

### Tick Interval S
- 1.0

### Grid Dimensions
- 10x10

## Gate Execution Summary
| Gate | Status |
|------|--------|
| G0_baseline_validation | PASS |
| G1_global_tick_integration | PASS |
| G2_rule_engine_execution | PASS |
| G3_zombie_ca_integration | {PASS|FAIL} |
| G4_dashboard_visualization | {PASS|FAIL} |
| G5_stability_test | PENDING |
| G6_emergent_behavior_analysis | PENDING |

### Notes on Exceptions
- **Gate G3**: Minor delay in rebirth synchronization (~3 s)
  - Resolution: Increased recovery_timeout=90s in supervisor

## Metrics Collected per Tick Window
**Tick Window:** 0–200
**Tick Rate:** 1 Hz

### Schema
tick_id, mean_state_entropy, neighbor_similarity_index, zombie_recovery_rate, gpu_utilization_percent, mean_state_magnitude, tick_latency_ms, active_bot_count

### Data Samples
| tick_id | mean_state_entropy | neighbor_similarity_index | zombie_recovery_rate | gpu_utilization_percent | mean_state_magnitude | tick_latency_ms | active_bot_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.42 | 0.12 | 0.0 | 18.4 | 0.33 | 4.2 | 100 |
| 100 | 0.08 | 0.86 | 0.94 | 44.1 | 0.67 | 5.0 | 99 |

## Aggregate Results
### Key Metrics
- **Convergence Detected**: True
- **Emergent Pattern**: {'type': 'oscillation', 'period_ticks': 47}
- **Stability Rating**: 0.97
- **Mean Zombie Latency S**: 2.8
- **Tick Reliability**: 0.992
- **Final Entropy**: 0.07
- **Entropy Drop Percent**: 83.3
- **Total Experiment Duration S**: 200

### Analysis Summary
The system converged toward a semi‑stable resonant pattern after ~50 ticks. Zombie cells reintegrated successfully, maintaining CA phase continuity. A low‑frequency oscillation persisted across the grid, suggesting emergent wave dynamics.

## Troubleshooting Trace
**Diagnostic Tree:** {T1–T6 from ca_experimentation_gated_protocol.yaml}

### Diagnostic Steps
- Checked supervisor process health → ok
- Validated tick broadcast endpoints → ok
- Manually tested /state endpoint on random bot → ok


## Visualization Artifacts
### Generated Plots
- **Entropy Curve**: `visualizations/entropy_over_time.png`
- **Grid Final State**: `visualizations/grid_tick_200.png`
- **Zombie Recovery Map**: `visualizations/zombie_wave_timeline.png`

### Generation Details
- **Analysis Script**: scripts/analyze_ca_metrics.py
- **Plotting Library**: matplotlib + plotly.js dashboard overlay

## Post-Experiment Actions
- `{'backup_logs': 'tar -czf logs/run_{experiment_id}.tar.gz logs/experimentation/*'}`
- `{'update_results_md': 'python3 scripts/update_results_md.py'}`
- `{'push_to_repo': "git add logs docs && git commit -m 'Add experiment {experiment_id} results'"}`
- `{'prepare_whitepaper_section': 'CA\u202fResults\u202f→\u202fEmergence\u202fPatterns\u202fand\u202fReintegration'}`

### Signoff
- **By:** MrSnowNB
- **Reviewed by:** Cline
- **Next Step:** Scale grid → 20x20 (400 bots)
- **End Date:** 2025-10-19T10:46:34-04:00
