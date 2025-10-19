#!/usr/bin/env python3
"""
update_results_md.py - Convert YAML experiment logs to Markdown summaries
Usage: python3 scripts/update_results_md.py [input.yaml] [output.md]
If no args provided, uses ca_experimentation_log_template.yaml and outputs to docs/ca_experimentation_results.md
"""

import yaml
import sys
import datetime
from pathlib import Path
import subprocess
import re
from typing import Any, Dict, List, cast

def get_git_commit():
    """Get current git commit hash"""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:8]
    except subprocess.CalledProcessError:
        return "unknown"

def generate_experiment_id():
    """Generate unique experiment ID"""
    timestamp = int(datetime.datetime.now().timestamp())
    return f"exp_{timestamp}"

def current_timestamp():
    """Current timestamp in ISO format"""
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S-04:00')

def process_placeholders(text):
    """Replace template placeholders with actual values"""
    replacements = {
        '{AUTO_GENERATED}': generate_experiment_id(),
        '{YYYY-MM-DDTHH:MM:SS-04:00}': current_timestamp(),
        '{git_commit_hash}': get_git_commit()
    }

    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    return text

def format_environment(env_dict):
    """Format environment dictionary to Markdown"""
    md = []
    for section, details in env_dict.items():
        md.append(f"### {section.title().replace('_', ' ')}")
        if isinstance(details, dict):
            for k, v in details.items():
                md.append(f"- **{k.title().replace('_', ' ')}**: {v}")
        elif isinstance(details, list):
            md.append(f"- {repr(details)}")
        else:
            md.append(f"- {details}")
        md.append("")
    return md

def format_gates(gates_dict, notes_list):
    """Format gate execution summary"""
    md = []
    md.append("| Gate | Status |")
    md.append("|------|--------|")
    for gate, status in gates_dict.items():
        md.append(f"| {gate} | {status} |")
    md.append("")

    if notes_list:
        md.append("### Notes on Exceptions")
        for note in notes_list:
            if note.get('issue'):
                md.append(f"- **Gate {note['gate']}**: {note['issue']}")
                if 'resolution' in note:
                    md.append(f"  - Resolution: {note['resolution']}")
        md.append("")
    return md

def format_metrics_windows(metrics_dict):
    """Format metrics section"""
    md = []
    md.append(f"**Tick Window:** {metrics_dict['tick_window']}")
    md.append(f"**Tick Rate:** {metrics_dict['tick_rate_hz']} Hz")
    md.append("")

    md.append("### Schema")
    md.append(", ".join(metrics_dict['metrics_schema']))
    md.append("")

    if metrics_dict['data_samples']:
        md.append("### Data Samples")
        headers = list(metrics_dict['data_samples'][0].keys())
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for sample in metrics_dict['data_samples']:
            row = [str(sample[h]) for h in headers]
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    return md

def format_aggregates(agg_dict, interp_dict):
    """Format aggregate results"""
    md = []
    md.append("### Key Metrics")
    for k, v in agg_dict.items():
        if isinstance(v, dict):
            md.append(f"- **{k.title().replace('_', ' ')}**: {v}")
        else:
            md.append(f"- **{k.title().replace('_', ' ')}**: {v}")
    md.append("")

    md.append("### Analysis Summary")
    if interp_dict.get('summary'):
        summary = interp_dict['summary'].replace('>', '')  # Remove YAML fold indicator
        md.append(summary.strip())
    if interp_dict.get('anomaly_notes'):
        md.append(f"**Anomaly Notes:** {interp_dict['anomaly_notes']}")
    md.append("")

    return md

def format_troubleshooting(trace_dict):
    """Format troubleshooting trace"""
    md = []
    if trace_dict.get('triggered_tree'):
        md.append(f"**Diagnostic Tree:** {trace_dict['triggered_tree']}")
        md.append("")
    if trace_dict.get('diagnostic_steps_taken'):
        md.append("### Diagnostic Steps")
        for step in trace_dict['diagnostic_steps_taken']:
            md.append(f"- {step}")
        md.append("")
    if trace_dict.get('root_cause_summary'):
        md.append(f"**Root Cause:** {trace_dict['root_cause_summary']}")
    if trace_dict.get('resolution_summary'):
        md.append(f"**Resolution:** {trace_dict['resolution_summary']}")
    md.append("")
    return md

def format_visualizations(viz_dict):
    """Format visualization artifacts"""
    md = []
    md.append("### Generated Plots")
    if viz_dict.get('visualizations'):
        for name, path in viz_dict['visualizations'].items():
            md.append(f"- **{name.title().replace('_', ' ')}**: `{path}`")
    md.append("")

    md.append("### Generation Details")
    if viz_dict.get('generated_with'):
        for k, v in viz_dict['generated_with'].items():
            md.append(f"- **{k.title().replace('_', ' ')}**: {v}")
    md.append("")
    return md

def format_actions(actions_dict):
    """Format post-experiment actions"""
    md = []
    for action in actions_dict.get('actions', []):
        md.append(f"- `{action}`")
    md.append("")

    if actions_dict.get('signoff'):
        md.append("### Signoff")
        signoff = actions_dict['signoff']
        md.append(f"- **By:** {signoff.get('by', 'N/A')}")
        md.append(f"- **Reviewed by:** {signoff.get('reviewed_by', 'N/A')}")
        md.append(f"- **Next Step:** {signoff.get('next_step', 'N/A')}")
        md.append(f"- **End Date:** {signoff.get('date_end', 'N/A')}")
    md.append("")
    return md

def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "logs/experimentation/ca_experimentation_log_template.yaml"

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = "docs/ca_experimentation_results.md"

    # Ensure input exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)

    # Read and process template
    with open(input_file, 'r', encoding='utf-8') as f:
        template_content = f.read()

    template_content = process_placeholders(template_content)

    # Parse YAML documents
    try:
        yaml_docs = cast(List[Dict[str, Any]], list(yaml.safe_load_all(template_content)))
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        sys.exit(1)

    if len(yaml_docs) < 6:
        print(f"Warning: Expected at least 6 YAML documents, got {len(yaml_docs)}")

    # Generate Markdown
    md_content = []

    # Title and metadata
    meta = yaml_docs[0] if yaml_docs else {}
    exp_id = meta.get('experiment_id', 'unknown')
    md_content.append(f"# CA Experimentation Log - {exp_id}")
    md_content.append("")
    md_content.append("## Project Details")
    for k, v in meta.items():
        if k == 'environment':
            continue
        md_content.append(f"- **{k.title().replace('_', ' ')}**: {v}")
    md_content.append("")
    md_content.extend(format_environment(meta.get('environment', {})))

    # Gate summary
    if len(yaml_docs) > 1:
        gates_doc = yaml_docs[1]
        md_content.append("## Gate Execution Summary")
        md_content.extend(format_gates(gates_doc.get('gates_executed', {}), gates_doc.get('notes_on_exceptions', [])))

    # Metrics windows
    if len(yaml_docs) > 2:
        metrics_doc = yaml_docs[2]
        md_content.append("## Metrics Collected per Tick Window")
        md_content.extend(format_metrics_windows(metrics_doc))

    # Aggregate results
    if len(yaml_docs) > 3:
        agg_doc = yaml_docs[3]
        md_content.append("## Aggregate Results")
        md_content.extend(format_aggregates(agg_doc.get('aggregates', {}), agg_doc.get('interpretation', {})))

    # Troubleshooting
    if len(yaml_docs) > 4:
        trace_doc = yaml_docs[4]
        if trace_doc.get('triggered_tree') or trace_doc.get('diagnostic_steps_taken'):
            md_content.append("## Troubleshooting Trace")
            md_content.extend(format_troubleshooting(trace_doc))

    # Visualizations
    if len(yaml_docs) > 5:
        viz_doc = yaml_docs[5]
        md_content.append("## Visualization Artifacts")
        md_content.extend(format_visualizations(viz_doc))

    # Actions
    if len(yaml_docs) > 6:
        actions_doc = yaml_docs[6]
        md_content.append("## Post-Experiment Actions")
        md_content.extend(format_actions(actions_doc))

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))

    print(f"✅ Markdown summary generated: {output_file}")
    print(f"Experiment ID: {exp_id}")

if __name__ == "__main__":
    main()
