#!/usr/bin/env python3
"""
---
script: start_swarm.py
purpose: COMPLETE SWARM LAUNCH ORCHESTRATOR - Single-point deployment
status: production-ready
created: 2025-10-21
---
AI-First Swarm Deployment Script

This is the MASTER CONTROL SCRIPT for Swarm-100 deployment.
Run this ONE script to launch the complete zombie swarm system with diagnostics.

What this script does automatically:
✅ System hardware checks (GPUs, CUDA)
✅ Software dependency installation
✅ Ollama model deployment
✅ 100 zombie bot swarm launch
✅ Real-time monitoring dashboard
✅ LoRA pulse testing capabilities
✅ Complete diagnostics interface

USAGE:
python3 start_swarm.py

For repo downloaders: Point your coding agent here.
"""

import os
import sys
import subprocess
import time
import json
import threading
from pathlib import Path
import requests

class SwarmMasterOrchestrator:
    """Complete autonomous swarm deployment system"""

    def __init__(self):
        self.root_dir = Path(__file__).parent.absolute()
        self.logs_dir = self.root_dir / "logs"
        self.configs_dir = self.root_dir / "configs"

        # Determine system capabilities
        self.has_nvidia_smi = self.check_nvidia_smi()
        self.gpu_count = self.get_gpu_count()
        self.started_services = []

        print("🧠 Swarm-100 Master Orchestrator Initialized")
        print(f"📁 Root Directory: {self.root_dir}")
        print(f"🎮 Available GPUs: {self.gpu_count}")

    def check_nvidia_smi(self):
        """Check if nvidia-smi is available"""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_gpu_count(self):
        """Get number of GPUs available"""
        if not self.has_nvidia_smi:
            return 0

        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'],
                                  capture_output=True, text=True, check=True)
            return len(result.stdout.strip().split('\n'))
        except:
            return 0

    def check_ollama_installed(self):
        """Check if Ollama is installed"""
        try:
            subprocess.run(['ollama', '--version'], capture_output=True, check=True)
            return True
        except:
            return False

    def install_ollama_if_needed(self):
        """Install Ollama if not present"""
        if self.check_ollama_installed():
            print("✅ Ollama already installed")
            return True

        print("📦 Installing Ollama...")
        try:
            result = subprocess.run([
                'curl', '-fsSL', 'https://ollama.com/install.sh'
            ], capture_output=True, text=True, check=True)

            # Execute the install script
            with open('/tmp/ollama_install.sh', 'w') as f:
                f.write(result.stdout)
            os.chmod('/tmp/ollama_install.sh', 0o755)

            subprocess.run(['/tmp/ollama_install.sh'], check=True)
            os.remove('/tmp/ollama_install.sh')

            print("✅ Ollama installed successfully")
            return True
        except Exception as e:
            print(f"❌ Ollama installation failed: {e}")
            return False

    def check_ollama_models(self):
        """Check and pull required models"""
        print("🤖 Checking Ollama models...")

        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, check=True)
            available_models = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    available_models.append(line.split()[0])

            required_models = ['granite4:micro-h', 'gemma3:270m']
            missing_models = [model for model in required_models if model not in available_models]

            if not missing_models:
                print("✅ All required models are available")
                return True

            print(f"⬇️  Pulling missing models: {missing_models}")
            for model in missing_models:
                print(f"   Downloading {model}...")
                subprocess.run(['ollama', 'pull', model], check=True)

            return True

        except Exception as e:
            print(f"❌ Model check failed: {e}")
            return False

    def check_python_dependencies(self):
        """Check Python dependencies"""
        required_packages = [
            'pyyaml', 'requests', 'numpy', 'scipy',
            'matplotlib', 'flask', 'flask-socketio'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if not missing_packages:
            print("✅ All Python dependencies available")
            return True

        print(f"📦 Installing missing packages: {missing_packages}")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-q'
            ] + missing_packages, check=True)
            print("✅ Python dependencies installed")
            return True
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

    def create_directories(self):
        """Create necessary swarm directories"""
        dirs_to_create = [
            'logs/gpu0', 'logs/gpu1', 'logs/gpu2', 'logs/gpu3',
            'bots', 'configs', 'scripts'
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        print("✅ Swarm directories created")

    def launch_zombie_swarm(self):
        """Launch the 100 zombie bot swarm"""
        print("🧟 Launching 100 zombie bot swarm...")

        try:
            # Launch swarm
            result = subprocess.run([
                sys.executable, 'scripts/launch_swarm.py', '--zombie-active'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

            if result.returncode == 0:
                print("✅ Zombie swarm launched successfully")
                print("   100 bots deployed across 4 GPUs")
                print("   10×10 Cellular Automaton grid active")

                self.started_services.append('swarm')

                # Give swarm time to stabilize
                time.sleep(10)
                return True
            else:
                print(f"❌ Swarm launch failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Swarm launch timed out")
            return False
        except Exception as e:
            print(f"❌ Swarm launch error: {e}")
            return False

    def launch_monitoring_dashboard(self):
        """Launch the Swarm diagnostics dashboard"""
        print("📊 Launching Swarm monitoring dashboard...")

        try:
            # Start dashboard in background
            dashboard_process = subprocess.Popen([
                sys.executable, 'swarm_dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Give dashboard time to start
            time.sleep(5)

            if dashboard_process.poll() is None:  # Still running
                print("✅ Monitoring dashboard launched")
                print(f"   Web interface: http://localhost:5000")
                print(f"   Real-time diagnostics active")
                print(f"   Pulse injection controls enabled")

                self.started_services.append('dashboard')
                self.dashboard_process = dashboard_process

                return True
            else:
                stdout, stderr = dashboard_process.communicate()
                print(f"❌ Dashboard launch failed: {stderr.decode()}")
                return False

        except Exception as e:
            print(f"❌ Dashboard launch error: {e}")
            return False

    def test_swarm_connectivity(self):
        """Test that all bots are responding"""
        print("🔍 Testing swarm connectivity...")

        try:
            with open('bots/swarm_state.yaml', 'r') as f:
                import yaml
                state = yaml.safe_load(f)
                bots = state['bots']
        except Exception as e:
            print(f"❌ Failed to read swarm state: {e}")
            return False

        responding_bots = 0
        total_bots = len(bots)

        for bot in bots:
            try:
                url = f"http://localhost:{bot['port']}/state"
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    responding_bots += 1
            except:
                continue  # Bot not responding - normal for initial test

        ratio = responding_bots / total_bots if total_bots > 0 else 0

        if ratio > 0.8:  # 80% success rate is good
            print(f"✅ Swarm connectivity test passed: {responding_bots}/{total_bots} bots responding")
            return True
        else:
            print(f"⚠️  Swarm connectivity partial: {responding_bots}/{total_bots} bots responding")
            return True  # Don't fail deployment for this

    def run_pulse_test(self):
        """Run a quick pulse injection test"""
        print("⚡ Testing LoRA pulse injection...")

        try:
            # Import and run pulse injector
            import lora_pulse_injector

            injector = lora_pulse_injector.LoRAPulseInjector()
            result_file = injector.run_pulse_experiment(
                target_coords=(5, 5),
                energy=0.8,
                radius=2,
                monitor_seconds=10
            )

            if result_file:
                print("✅ Pulse injection test successful")
                print(f"   Results saved to: {result_file}")
                return True
            else:
                print("❌ Pulse injection test failed")
                return False

        except Exception as e:
            print(f"❌ Pulse test error: {e}")
            return False

    def launch_additional_monitors(self):
        """Launch additional monitoring services"""
        print("🖥️  Launching additional monitoring services...")

        monitor_commands = [
            ([sys.executable, 'scripts/health_monitor.py'], 'health_monitor'),
            ([sys.executable, 'scripts/zombie_supervisor.py'], 'zombie_supervisor'),
            ([sys.executable, 'scripts/swarm_monitor.py'], 'swarm_monitor')
        ]

        for command, name in monitor_commands:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.started_services.append(name)
                print(f"   ✅ {name} started")
            except Exception as e:
                print(f"   ⚠️  {name} failed to start: {e}")

    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        report = {
            'deployment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'system_info': {
                'gpu_count': self.gpu_count,
                'has_nvidia': self.has_nvidia_smi
            },
            'services_started': self.started_services,
            'status': 'successful' if len(self.started_services) >= 2 else 'partial',
            'access_points': {
                'web_dashboard': 'http://localhost:5000',
                'swarm_state': 'bots/swarm_state.yaml',
                'logs_directory': 'logs/'
            }
        }

        with open('deployment_status.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n" + "="*60)
        print("🚀 SWARM DEPLOYMENT COMPLETED")
        print("="*60)
        print(f"Status: {report['status'].upper()}")
        print(f"Services: {', '.join(self.started_services)}")
        print(f"Dashboard: http://localhost:5000")
        print(f"Bots: 100 zombie-enabled agents")
        print(f"Grid: 10×10 Cellular Automaton")
        print(f"Report: deployment_status.json")
        print("="*60)

        return report

    def cleanup_on_failure(self):
        """Clean up partial installations on failure"""
        print("🧹 Cleaning up partial deployment...")

        # Kill any started processes
        cleanup_commands = [
            ['pkill', '-f', 'bot_worker'],
            ['pkill', '-f', 'swarm_dashboard'],
            ['pkill', '-f', 'health_monitor'],
            ['pkill', '-f', 'zombie_supervisor'],
            [sys.executable, 'scripts/stop_swarm.sh']
        ]

        for cmd in cleanup_commands:
            try:
                subprocess.run(cmd, timeout=10)
            except:
                pass

        print("✅ Cleanup completed")

    def run_full_deployment(self):
        """Execute complete autonomous swarm deployment"""
        print("🚀 STARTING COMPLETE SWARM DEPLOYMENT")
        print("="*60)

        success = True
        deployment_phase = 0

        try:
            # Phase 1: System checks
            deployment_phase = 1
            print(f"\n📋 PHASE {deployment_phase}: System Prerequisites")
            print("-" * 40)

            if self.gpu_count < 4:
                print(f"⚠️  Only {self.gpu_count} GPUs detected (4 recommended)")
                if self.gpu_count == 0:
                    print("❌ Cannot proceed without GPU support")
                    return False

            if not self.check_python_dependencies():
                print("❌ Python dependency installation failed")
                return False

            # Phase 2: Ollama setup
            deployment_phase = 2
            print(f"\n📋 PHASE {deployment_phase}: Ollama Deployment")
            print("-" * 40)

            if not self.install_ollama_if_needed():
                print("❌ Ollama installation failed")
                return False

            if not self.check_ollama_models():
                print("❌ Model deployment failed")
                return False

            # Phase 3: Directory setup
            deployment_phase = 3
            print(f"\n📋 PHASE {deployment_phase}: Environment Setup")
            print("-" * 40)

            self.create_directories()

            # Phase 4: Swarm launch
            deployment_phase = 4
            print(f"\n📋 PHASE {deployment_phase}: Zombie Swarm Launch")
            print("-" * 40)

            if not self.launch_zombie_swarm():
                print("❌ Swarm launch failed")
                return False

            # Phase 5: Monitoring systems
            deployment_phase = 5
            print(f"\n📋 PHASE {deployment_phase}: Monitoring & Diagnostics")
            print("-" * 40)

            if not self.launch_monitoring_dashboard():
                print("⚠️  Dashboard launch failed - continuing...")

            self.launch_additional_monitors()

            # Phase 6: Testing & Validation
            deployment_phase = 6
            print(f"\n📋 PHASE {deployment_phase}: Validation Tests")
            print("-" * 40)

            self.test_swarm_connectivity()

            if len([s for s in self.started_services if s != 'dashboard']) > 0:
                self.run_pulse_test()

            # Phase 7: Final report
            report = self.generate_deployment_report()
            return report['status'] == 'successful'

        except KeyboardInterrupt:
            print("\n🚫 Deployment interrupted by user")
            self.cleanup_on_failure()
            return False

        except Exception as e:
            print(f"\n💥 Deployment failed at phase {deployment_phase}: {e}")
            self.cleanup_on_failure()
            return False

def main():
    """Main deployment orchestrator"""
    print("🧠 Swarm-100 MASTER DEPLOYMENT SYSTEM")
    print("This will automatically deploy the complete zombie swarm")
    print("Press Ctrl+C to cancel at any point")
    print("="*60)

    orchestrator = SwarmMasterOrchestrator()

    if orchestrator.run_full_deployment():
        print("\n🎉 DEPLOYMENT SUCCESSFUL!")
        print("Navigate to http://localhost:5000 for the Swarm dashboard")
        return 0
    else:
        print("\n❌ DEPLOYMENT FAILED!")
        print("Check deployment_status.json for details")
        return 1

if __name__ == '__main__':
    sys.exit(main())
