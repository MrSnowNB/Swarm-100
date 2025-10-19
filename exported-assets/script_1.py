
# Generate all configuration files for the swarm

# 1. Swarm Configuration YAML
swarm_config = """---
project: granite4-microh-swarm
version: 1.0.0
status: active
created: 2025-10-18

# Hardware Configuration
hardware:
  gpus:
    - id: 0
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
    - id: 1
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
    - id: 2
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
    - id: 3
      name: "RTX 6000 Ada"
      vram_gb: 48
      bots: 25
  total_bots: 100

# Model Configuration
model:
  name: "granite4:micro-h"
  quantization: "Q4"
  context_length: 4096  # Reduced for max concurrency
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  
# Swarm Architecture
swarm:
  gossip_hops: 4
  fanout: 5
  confidence_threshold: 0.5
  ttl: 4
  
# Bot Configuration
bot:
  base_port: 11400
  api_timeout: 30
  max_retries: 3
  health_check_interval: 60
  
# Performance Tuning
performance:
  max_concurrent_requests: 10
  batch_size: 1
  num_threads: 4
  gpu_memory_fraction: 0.95
  enable_kv_cache: true
  
# Logging
logging:
  level: "INFO"
  dir: "logs"
  rotation: "daily"
  retention_days: 7
  
# Monitoring
monitoring:
  enable_metrics: true
  metrics_port: 9090
  prometheus_export: false
"""

# 2. Bot Template Configuration
bot_template = """---
bot_id: "{{BOT_ID}}"
gpu_id: {{GPU_ID}}
port: {{PORT}}
status: initialized
created: {{TIMESTAMP}}

# Model Runtime
model:
  name: "granite4:micro-h"
  quant: "Q4"
  context_length: 4096
  
# Local Memory
memory:
  vectors: []
  capacity: 1000
  eviction_policy: "FIFO"
  
# Gossip Network
network:
  entry_nodes: {{ENTRY_NODES}}
  neighbors: []
  gossip_interval: 1.0
  
# Performance
performance:
  avg_latency_ms: 0
  total_requests: 0
  errors: 0
  uptime_seconds: 0
"""

print("Configuration files generated:")
print("\n1. swarm_config.yaml")
print("-" * 60)
print(swarm_config)
print("\n2. bot_template.yaml")
print("-" * 60)
print(bot_template)
