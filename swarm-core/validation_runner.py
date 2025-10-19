#!/usr/bin/env python3
"""
Swarm-100 C++ Core Gated Validation Runner
Executes validation workflow with 100% success requirement per gate
"""

import yaml
import subprocess
import sys
import os
import datetime
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

class GatedValidationRunner:
    workflow: Dict[str, Any]

    def __init__(self, workflow_file: str):
        with open(workflow_file, 'r') as f:
            raw_workflow = yaml.safe_load(f)

        if not raw_workflow:
            raise ValueError("Invalid or empty YAML file")
        if not isinstance(raw_workflow, dict):
            raise ValueError("Workflow file must contain a dictionary at root level")

        self.workflow = raw_workflow

        self.results = {
            'workflow_name': self.workflow['name'],
            'start_time': datetime.datetime.now().isoformat(),
            'gates_passed': 0,
            'gates_failed': 0,
            'gates_total': len(self.workflow['gates']),
            'failed_gates': [],
            'gate_results': {},
            'overall_success': False
        }

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def run_command(self, command: str, cwd: Optional[str] = None) -> dict:
        """Run a shell command and return results"""
        try:
            self.log(f"Executing: {command}")
            if cwd is None:
                cwd = os.getcwd()
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )
            success = result.returncode == 0
            return {
                'success': success,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'exit_code': -1,
                'stdout': '',
                'stderr': 'Command timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e)
            }

    def run_python_script(self, script: str, cwd: Optional[str] = None) -> dict:
        """Run a Python script inline"""
        try:
            # Create a temporary file with the script
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_file = f.name

            # Run the script
            command = f"{sys.executable} {script_file}"
            result = subprocess.run(
                command,
                shell=True,
                cwd=os.path.abspath(cwd) if cwd else os.getcwd(),
                capture_output=True,
                text=True,
                timeout=60
            )

            # Clean up
            os.unlink(script_file)

            success = result.returncode == 0
            return {
                'success': success,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except Exception as e:
            return {
                'success': False,
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e)
            }

    def check_file_exists(self, file_path: str) -> dict:
        """Check if a file exists"""
        full_path = Path(file_path)
        exists = full_path.exists()
        return {
            'success': exists,
            'file_path': str(full_path),
            'exists': exists
        }

    def execute_gate(self, gate_name: str, gate_config: dict) -> dict:
        """Execute a validation gate"""
        self.log(f"🔒 Executing Gate: {gate_config['name']} - {gate_config['description']}")

        gate_result = {
            'name': gate_config['name'],
            'steps_executed': 0,
            'steps_passed': 0,
            'steps_failed': 0,
            'step_results': [],
            'success': False,
            'duration_seconds': 0
        }

        gate_start_time = datetime.datetime.now()

        for step in gate_config['steps']:
            gate_result['steps_executed'] += 1
            step_name = step['name']
            step_type = step['type']

            self.log(f"  📋 Running step: {step_name}")

            # Execute step based on type
            if step_type == 'command':
                result = self.run_command(step['command'])
            elif step_type == 'python_script':
                result = self.run_python_script(step['script'])
            elif step_type == 'file_check':
                result = self.check_file_exists(step['file_path'])
            else:
                result = {'success': False, 'error': f'Unknown step type: {step_type}'}

            step_result = {
                'name': step_name,
                'type': step_type,
                'success': result['success'],
                'duration_seconds': 0
            }

            gate_result['step_results'].append(step_result)

            if result['success']:
                gate_result['steps_passed'] += 1
                self.log(f"  ✅ Step passed: {step_name}")
                if 'stdout' in result and result['stdout'].strip():
                    self.log(f"  📝 Output: {result['stdout'].strip()}")
            else:
                gate_result['steps_failed'] += 1
                self.log(f"  ❌ Step failed: {step_name}")
                if 'stderr' in result and result['stderr'].strip():
                    self.log(f"  📝 Error: {result['stderr'].strip()}")
                if 'stdout' in result and result['stdout'].strip():
                    self.log(f"  📝 Output: {result['stdout'].strip()}")

                # Check if we should fail the entire gate
                if step.get('fail_on_error', True):
                    gate_result['success'] = False
                    gate_result['duration_seconds'] = (datetime.datetime.now() - gate_start_time).total_seconds()
                    self.log(f"❌ GATE FAILED: {gate_config['name']} - Step '{step_name}' failed")
                    return gate_result

        # All steps in gate passed
        success_rate = gate_result['steps_passed'] / gate_result['steps_executed'] * 100
        required_rate = gate_config.get('required_success_rate', 100)

        if success_rate >= required_rate:
            gate_result['success'] = True
            gate_result['duration_seconds'] = (datetime.datetime.now() - gate_start_time).total_seconds()
            self.log(f"✅ GATE PASSED: {gate_config['name']} ({success_rate:.1f}% / {required_rate}%)")
        else:
            gate_result['success'] = False
            gate_result['duration_seconds'] = (datetime.datetime.now() - gate_start_time).total_seconds()
            self.log(f"❌ GATE FAILED: {gate_config['name']} ({success_rate:.1f}% / {required_rate}%)")

        return gate_result

    def run_validation(self) -> bool:
        """Run the complete validation workflow"""
        self.log(f"🚀 Starting Gated Validation: {self.workflow['name']}")  # type: ignore
        self.log(f"📋 {self.workflow['description']}")  # type: ignore

        gate_order = list(self.workflow['gates'].keys())

        for gate_name in gate_order:
            gate_config = self.workflow['gates'][gate_name]

            # Check failure policy
            if self.workflow.get('failure_policy') == 'stop_all_gates' and self.results['gates_failed'] > 0:
                self.log("⏹️  Stopping validation - previous gate failure detected")
                break

            # Execute the gate
            gate_result = self.execute_gate(gate_name, gate_config)
            self.results['gate_results'][gate_name] = gate_result

            if gate_result['success']:
                self.results['gates_passed'] += 1
            else:
                self.results['gates_failed'] += 1
                self.results['failed_gates'].append(gate_name)

        # Final results
        self.results['end_time'] = datetime.datetime.now().isoformat()
        self.results['overall_success'] = self.results['gates_failed'] == 0

        if self.results['overall_success']:
            self.log(f"🎉 VALIDATION SUCCESSFUL: All {self.results['gates_passed']} gates passed!")
            return True
        else:
            self.log(f"💥 VALIDATION FAILED: {self.results['gates_failed']} gates failed")
            self.log(f"Failed gates: {', '.join(self.results['failed_gates'])}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run gated validation workflow')
    parser.add_argument('--workflow', default='validation_workflow.yaml', help='Workflow YAML file')
    parser.add_argument('--results', default='validation_results.yaml', help='Results output file')

    args = parser.parse_args()

    try:
        runner = GatedValidationRunner(args.workflow)
        success = runner.run_validation()

        # Write results
        runner.results['command_line'] = sys.argv
        runner.results['working_directory'] = os.getcwd()

        with open(args.results, 'w') as f:
            yaml.dump(runner.results, f, default_flow_style=False, sort_keys=False)

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"Fatal error during validation: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
