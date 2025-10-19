#!/usr/bin/env python3
"""
---
script: analyze_ca_metrics.py
purpose: Analyze cellular automata dynamics from logged metrics
status: development
created: 2025-10-19
---
"""

import pandas as pd  # type: ignore
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns  # type: ignore[import]
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

SCIPY_AVAILABLE = False

class CAMetricsAnalyzer:
    """Analyze CA evolution metrics for convergence, oscillation, or chaos"""

    def __init__(self, metrics_csv: str, output_dir: str = "logs/experimentation/results"):
        self.metrics_file = Path(metrics_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure plotting style
        if SEABORN_AVAILABLE and MATPLOTLIB_AVAILABLE:
            sns.set.style("darkgrid")  # type: ignore
            plt.style.use('seaborn-v0_8')  # type: ignore

        self.logger = logging.getLogger('CAMetricsAnalyzer')

    def load_metrics(self) -> pd.DataFrame:
        """Load CA metrics from CSV"""
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")

        df = pd.read_csv(self.metrics_file)
        self.logger.info(f"Loaded {len(df)} metric records")

        # Ensure tick column exists
        if 'tick' not in df.columns:
            df['tick'] = range(len(df))

        return df

    def calculate_state_entropy(self, state_vectors: List[np.ndarray]) -> float:
        """Calculate entropy of bot state distributions"""
        if not state_vectors:
            return 0.0

        # Flatten all states into one distribution
        all_states = np.concatenate([v.flatten() for v in state_vectors])

        # Calculate histogram entropy
        hist, _ = np.histogram(all_states, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist))
        return entropy

    def detect_convergence(self, entropy_series: pd.Series, window: int = 50) -> bool:
        """Detect if system is converging to stable state"""
        if len(entropy_series) < window * 2:
            return False

        # Check if entropy is decreasing and stabilizing
        recent_change = entropy_series.pct_change().tail(window).abs()
        mean_recent_change = recent_change.mean()

        # Convergence if change rate < 1% over last window
        return mean_recent_change < 0.01

    def detect_oscillation(self, entropy_series: pd.Series, min_period: int = 10, max_period: int = 100):
        """Detect periodic oscillations in system"""
        if len(entropy_series) < max_period * 2:
            return None

        try:
            from scipy import signal  # type: ignore
            # Autocorrelation to find periodic peaks
            autocorr = np.correlate(entropy_series, entropy_series, mode='full')
            autocorr = autocorr[autocorr.size // 2:]  # Second half only

            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr, distance=min_period, height=0.1)  # type: ignore

            if len(peaks) > 0:
                # Most prominent period
                best_peak = peaks[np.argmax(autocorr[peaks])]
                if min_period <= best_peak <= max_period:
                    return int(best_peak)

        except ImportError:
            self.logger.info("SciPy not available, skipping oscillation detection")
        except Exception as e:
            self.logger.warning(f"Oscillation detection error: {e}")

        return None

    def analyze_zombie_patterns(self, zombie_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze zombie rebirth patterns"""
        if not zombie_events:
            return {'no_zombie_events': True}

        rebirth_times = [e.get('timestamp', 0) for e in zombie_events if e.get('type') == 'reborn']
        failure_times = [e.get('timestamp', 0) for e in zombie_events if e.get('type') == 'failed']

        # Calculate recovery rate
        total_events = len(zombie_events)
        recovery_rate = len(rebirth_times) / max(total_events, 1)

        # Analyze temporal clustering
        if rebirth_times:
            rebirth_intervals = np.diff(sorted(rebirth_times))
            avg_interval = np.mean(rebirth_intervals) if len(rebirth_intervals) > 0 else 0
            std_interval = np.std(rebirth_intervals) if len(rebirth_intervals) > 1 else 0
        else:
            avg_interval = 0
            std_interval = 0

        return {
            'recovery_rate': recovery_rate,
            'total_events': total_events,
            'avg_rebirth_interval': avg_interval,
            'std_rebirth_interval': std_interval,
            'cluster_coefficient': std_interval / max(avg_interval, 0.001)  # <1 = clustered, >1 = uniform
        }

    def generate_analysis_report(self, metrics_df: pd.DataFrame, zombie_events: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive CA evolution analysis"""
        self.logger.info("Generating CA evolution analysis report...")

        report = {
            'summary': {
                'total_ticks': len(metrics_df),
                'total_bots': metrics_df.get('alive_bots', pd.Series([40])).iloc[0],
                'avg_state_entropy': metrics_df.get('state_entropy', pd.Series()).mean(),
                'final_entropy': metrics_df.get('state_entropy', pd.Series()).iloc[-1] if len(metrics_df) > 0 else 0
            }
        }

        # Convergence analysis
        if 'state_entropy' in metrics_df.columns:
            entropy_converged = self.detect_convergence(metrics_df['state_entropy'])
            oscillation_period = self.detect_oscillation(metrics_df['state_entropy'])

            report['convergence'] = {
                'detected': entropy_converged,
                'convergence_entropy_threshold': 0.05,
                'final_entropy': metrics_df['state_entropy'].iloc[-1],
                'entropy_reduction': metrics_df['state_entropy'].pct_change().tail(100).mean()
            }

            osc_dict = {
                'detected': oscillation_period is not None,
                'period_ticks': oscillation_period,
                'confidence': 'Medium' if oscillation_period else 'None'
            }
            report['oscillation'] = osc_dict
        else:
            report['oscillation'] = {
                'detected': False,
                'period_ticks': None,
                'confidence': 'No data'
            }

        # Zombie pattern analysis
        zombie_analysis = self.analyze_zombie_patterns(zombie_events)
        report['zombie_behavior'] = zombie_analysis

        # Emergence classification
        emergence_type: str = 'unknown'
        convergence_detected = report['convergence'].get('detected', False)
        oscillation_detected = report['oscillation'].get('detected', False)

        if convergence_detected:
            emergence_type = 'convergent'
        elif oscillation_detected:
            period = report['oscillation'].get('period_ticks')
            emergence_type = f'oscillatory_period_{period}' if period else 'oscillatory_unknown'
        else:
            final_entropy = report['summary']['final_entropy']
            if isinstance(final_entropy, (int, float)):
                if final_entropy > 2.0:
                    emergence_type = 'chaotic_high_entropy'
                elif final_entropy < 0.5:
                    emergence_type = 'stable_low_entropy'
                else:
                    emergence_type = 'transitioning_medium_entropy'

        report['emergence_classification'] = {'type': emergence_type}

        # Success criteria evaluation
        convergence_entropy = report['convergence'].get('final_entropy', 1.0)
        success_convergence = isinstance(convergence_entropy, (int, float)) and convergence_entropy < 0.05
        success_emergence = emergence_type != 'unknown'

        report['success_criteria'] = {
            'convergence_achieved': success_convergence,
            'emergent_behavior_detected': success_emergence,
            'overall_success': success_convergence or success_emergence,
            'metrics_completeness': len(metrics_df) > 100  # Sufficient data collected
        }

        return report

    def create_visualizations(self, metrics_df: pd.DataFrame, report: Dict[str, Any]):
        """Generate analysis visualizations"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping visualizations")
            return

        self.logger.info("Creating analysis visualizations...")

        plt.figure(figsize=(15, 10))  # type: ignore
        plt.suptitle('Cellular Automata Evolution Analysis', fontsize=16)  # type: ignore

        # Create subplots manually
        plt.subplot(2, 2, 1)  # type: ignore

        # Entropy evolution
        if 'state_entropy' in metrics_df.columns:
            plt.plot(metrics_df['tick'], metrics_df['state_entropy'], 'b-', linewidth=2, alpha=0.7)  # type: ignore
            plt.title('State Entropy Evolution')  # type: ignore
            plt.xlabel('Tick')  # type: ignore
            plt.ylabel('Entropy')  # type: ignore
            plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Convergence Threshold')  # type: ignore
            plt.legend()  # type: ignore
            plt.grid(True, alpha=0.3)  # type: ignore

        plt.subplot(2, 2, 2)  # type: ignore
        # Similarity index
        if 'neighbor_similarity' in metrics_df.columns:
            plt.plot(metrics_df['tick'], metrics_df['neighbor_similarity'], 'g-', linewidth=2, alpha=0.7)  # type: ignore
            plt.title('Neighbor Similarity Evolution')  # type: ignore
            plt.xlabel('Tick')  # type: ignore
            plt.ylabel('Similarity Index')  # type: ignore
            plt.grid(True, alpha=0.3)  # type: ignore

        plt.subplot(2, 2, 3)  # type: ignore
        # Wave propagation
        if 'zombie_wave_speed' in metrics_df.columns:
            plt.plot(metrics_df['tick'], metrics_df['zombie_wave_speed'], 'r-', linewidth=2, alpha=0.7)  # type: ignore
            plt.title('Zombie Wave Propagation')  # type: ignore
            plt.xlabel('Tick')  # type: ignore
            plt.ylabel('Wave Speed (cells/tick)')  # type: ignore
            plt.grid(True, alpha=0.3)  # type: ignore

        plt.subplot(2, 2, 4)  # type: ignore
        # State variance
        if 'state_variance' in metrics_df.columns:
            plt.plot(metrics_df['tick'], metrics_df['state_variance'], 'purple', linewidth=2, alpha=0.7)  # type: ignore
            plt.title('State Vector Variance')  # type: ignore
            plt.xlabel('Tick')  # type: ignore
            plt.ylabel('Variance')  # type: ignore
            plt.grid(True, alpha=0.3)  # type: ignore

        plt.tight_layout()  # type: ignore
        plt.savefig(self.output_dir / 'ca_evolution_analysis.png', dpi=150, bbox_inches='tight')  # type: ignore
        plt.close()  # type: ignore

        # Emergence classification pie chart
        emergence_classification = report.get('emergence_classification', {}).get('type', '')
        emergence_types = {
            'convergent': 1 if emergence_classification == 'convergent' else 0,
            'oscillatory': 1 if 'oscillatory' in emergence_classification else 0,
            'chaotic': 1 if 'chaotic' in emergence_classification else 0,
            'stable': 1 if 'stable' in emergence_classification else 0,
            'transitioning': 1 if 'transitioning' in emergence_classification else 0
        }

        # Filter non-zero
        emergence_types = {k: v for k, v in emergence_types.items() if v > 0}

        if emergence_types:
            plt.figure(figsize=(8, 8))  # type: ignore
            plt.pie(list(emergence_types.values()), labels=list(emergence_types.keys()), autopct='%1.1f%%', startangle=90)  # type: ignore
            plt.title('Emergence Pattern Classification', fontsize=14)  # type: ignore
            plt.axis('equal')  # type: ignore

            plt.savefig(self.output_dir / 'emergence_patterns.png', dpi=150, bbox_inches='tight')  # type: ignore
            plt.close()  # type: ignore

        self.logger.info(f"Visualizations saved to {self.output_dir}")

    def run_analysis(self):
        """Main analysis pipeline"""
        try:
            self.logger.info("Starting CA metrics analysis...")

            # Load data
            metrics_df = self.load_metrics()

            # Try loading zombie events (optional)
            zombie_events = []
            zombie_yaml = Path('logs/experimentation/zombie_recovery_stats.yaml')
            if zombie_yaml.exists():
                with open(zombie_yaml, 'r') as f:
                    zombie_data = yaml.safe_load(f)
                    if isinstance(zombie_data, dict):
                        zombie_events = zombie_data.get('events', [])

            # Generate analysis report
            report = self.generate_analysis_report(metrics_df, zombie_events)

            # Create visualizations
            self.create_visualizations(metrics_df, report)

            # Save report
            report_file = self.output_dir / 'ca_analysis_report.yaml'
            with open(report_file, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)

            print(f"✅ CA Analysis Complete")
            print(f"Report saved to: {report_file}")
            print(f"Visualizations saved to: {self.output_dir}")
            print(f"Emergence Classification: {report['emergence_classification']['type']}")
            print(f"Success Criteria Met: {report['success_criteria']['overall_success']}")

            return report

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Analyze CA evolution metrics')
    parser.add_argument('--metrics', default='logs/experimentation/ca_metrics.csv',
                       help='Path to CA metrics CSV')
    parser.add_argument('--output', default='logs/experimentation/results',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - CAnalysis - %(levelname)s - %(message)s')

    analyzer = CAMetricsAnalyzer(args.metrics, args.output)
    analyzer.run_analysis()

if __name__ == '__main__':
    main()
