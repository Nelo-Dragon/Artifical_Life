#!/usr/bin/env python3
"""
Analyze and compare pipeline training results across multiple runs.
Parses curriculum_report.txt files to track convergence, identify
bottlenecks (stage-to-stage fitness drops), and compare hyperparameter
sensitivity.
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    fitness: Optional[int] = None
    ranking: Optional[int] = None
    generations: Optional[int] = None
    silence: Optional[int] = None  # For memory stage only
    
    def __repr__(self) -> str:
        s = f"{self.stage_name}: fitness={self.fitness} ranking={self.ranking}"
        if self.generations:
            s += f" gen={self.generations}"
        if self.silence is not None:
            s += f" silence={self.silence}"
        return s


@dataclass
class TrainingRun:
    """Complete training run result."""
    run_id: str
    grid_size: str  # e.g. "4x8"
    task: str  # e.g. "identity", "xor"
    stages: Dict[str, StageResult]
    joint_before: Optional[StageResult] = None
    joint_after: Optional[StageResult] = None
    
    def stage_order(self) -> List[str]:
        """Return stages in order they appear in pipeline."""
        return ['memory', 'input', 'output', 'cortex', 'joint']
    
    def __repr__(self) -> str:
        return f"Run({self.run_id}, {self.grid_size}, {self.task})"


def parse_report(report_file: Path) -> Optional[TrainingRun]:
    """Parse a curriculum_report.txt file into a TrainingRun object."""
    if not report_file.exists():
        print(f"  WARNING: Report not found: {report_file}")
        return None
    
    text = report_file.read_text()
    
    # Extract grid size from report header
    grid_match = re.search(r'XS=(\d+).*YS=(\d+)', text)
    grid_size = f"{grid_match.group(1)}x{grid_match.group(2)}" if grid_match else "unknown"
    
    # Parse stage results
    # Format: "Stage N (name) ...: fitness=X ranking=Y generations=Z"
    stages = {}
    stage_pattern = r'Stage \d+ \((\w+)\).*?: fitness=(\d+) ranking=(\d+) generations=(\d+)'
    silence_pattern = r'Stage 1 \(memory\).*silence=(\d+)'
    
    for match in re.finditer(stage_pattern, text):
        stage_name = match.group(1)
        result = StageResult(
            stage_name=stage_name,
            fitness=int(match.group(2)),
            ranking=int(match.group(3)),
            generations=int(match.group(4)),
        )
        stages[stage_name] = result
    
    # Extract memory silence gap if present
    silence_match = re.search(silence_pattern, text)
    if silence_match and 'memory' in stages:
        stages['memory'].silence = int(silence_match.group(1))
    
    # Parse joint before/after
    joint_before = None
    joint_after = None
    joint_pattern = r'Stage 5 \(joint\) (before|after).*?: fitness=(\d+) ranking=(\d+)'
    for match in re.finditer(joint_pattern, text):
        phase = match.group(1)
        result = StageResult(
            stage_name=f"joint_{phase}",
            fitness=int(match.group(2)),
            ranking=int(match.group(3)),
        )
        # Extract generations if present (only after phase has it)
        gen_pattern = rf"joint\) {phase}.*?generations=(\d+)"
        gen_match = re.search(gen_pattern, text)
        if gen_match:
            result.generations = int(gen_match.group(1))
        
        if phase == "before":
            joint_before = result
        else:
            joint_after = result
    
    run = TrainingRun(
        run_id=report_file.parent.name,
        grid_size=grid_size,
        task="unknown",
        stages=stages,
        joint_before=joint_before,
        joint_after=joint_after,
    )
    return run


def find_bottlenecks(run: TrainingRun) -> List[tuple[str, str, float]]:
    """
    Identify stage-to-stage fitness drops (bottlenecks).
    Returns list of (from_stage, to_stage, fitness_drop).
    """
    bottlenecks = []
    order = ['memory', 'input', 'output', 'cortex']
    
    for i in range(len(order) - 1):
        from_stage = order[i]
        to_stage = order[i + 1]
        
        if from_stage not in run.stages or to_stage not in run.stages:
            continue
        
        from_fitness = run.stages[from_stage].fitness
        to_fitness = run.stages[to_stage].fitness
        
        if from_fitness is not None and to_fitness is not None:
            drop = from_fitness - to_fitness
            if drop > 0:
                bottlenecks.append((from_stage, to_stage, drop))
    
    # Check composition bottleneck (isolated vs. joint)
    if 'cortex' in run.stages and run.joint_before:
        cortex_fitness = run.stages['cortex'].fitness
        joint_before = run.joint_before.fitness
        if cortex_fitness is not None and joint_before is not None:
            drop = cortex_fitness - joint_before
            if drop > 0:
                bottlenecks.append(("cortex→joint", "composition", drop))
    
    return sorted(bottlenecks, key=lambda x: x[2], reverse=True)


def compare_runs(runs: List[TrainingRun]) -> None:
    """Print side-by-side comparison of multiple runs."""
    if not runs:
        print("No runs to compare.")
        return
    
    print(f"\n{'='*80}")
    print(f"  Comparison of {len(runs)} training runs")
    print(f"{'='*80}\n")
    
    # Table header
    stage_names = ['Memory', 'Input', 'Output', 'Cortex', 'Joint Before', 'Joint After']
    print(f"{'Run ID':<20} {'Grid':<10} {'Task':<12}", end='')
    for stage in stage_names:
        print(f" {stage:<12}", end='')
    print()
    print("-" * 130)
    
    # Rows
    for run in runs:
        print(f"{run.run_id:<20} {run.grid_size:<10} {run.task:<12}", end='')
        
        for stage_name in ['memory', 'input', 'output', 'cortex']:
            if stage_name in run.stages:
                f = run.stages[stage_name].fitness
                print(f" {f:<12}" if f is not None else f" {'N/A':<12}", end='')
            else:
                print(f" {'N/A':<12}", end='')
        
        if run.joint_before:
            print(f" {run.joint_before.fitness:<12}", end='')
        else:
            print(f" {'N/A':<12}", end='')
        
        if run.joint_after:
            improvement = ""
            if run.joint_before and run.joint_after.fitness is not None:
                delta = run.joint_after.fitness - run.joint_before.fitness
                improvement = f"({delta:+d})"
            print(f" {run.joint_after.fitness} {improvement:<6}", end='')
        else:
            print(f" {'N/A':<12}", end='')
        
        print()
    
    print("\n")


def analyze_bottlenecks(runs: List[TrainingRun]) -> None:
    """Identify and report bottlenecks across runs."""
    print(f"\n{'='*80}")
    print(f"  Bottleneck Analysis")
    print(f"{'='*80}\n")
    
    for run in runs:
        bottlenecks = find_bottlenecks(run)
        if bottlenecks:
            print(f"{run.run_id} ({run.grid_size}, {run.task}):")
            for from_stage, to_stage, drop in bottlenecks:
                print(f"  {from_stage:12} → {to_stage:12}  fitness drop: {drop:.1f}")
        else:
            print(f"{run.run_id}: No bottlenecks detected")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze pipeline training results across multiple runs."
    )
    parser.add_argument('directories', nargs='+', type=Path,
                        help='Directories containing curriculum_report.txt files')
    parser.add_argument('--compare', action='store_true',
                        help='Show side-by-side comparison table')
    parser.add_argument('--bottlenecks', action='store_true',
                        help='Identify stage-to-stage fitness drops')
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')
    
    args = parser.parse_args()
    
    # Collect runs
    runs = []
    for directory in args.directories:
        if not directory.is_dir():
            print(f"WARNING: Not a directory: {directory}")
            continue
        
        report_file = directory / "curriculum_report.txt"
        run = parse_report(report_file)
        if run:
            # Extract task from directory name if possible
            if 'xor' in str(directory).lower():
                run.task = "xor"
            elif 'and' in str(directory).lower():
                run.task = "and"
            elif 'identity' in str(directory).lower():
                run.task = "identity"
            
            runs.append(run)
            print(f"Loaded: {run}")
    
    if not runs:
        print("No reports found.")
        return
    
    # Run analyses
    if args.compare or args.all:
        compare_runs(runs)
    
    if args.bottlenecks or args.all:
        analyze_bottlenecks(runs)
    
    if not (args.compare or args.bottlenecks or args.all):
        # Default: show summary
        print(f"\nLoaded {len(runs)} training runs:\n")
        for run in runs:
            print(f"  {run}")
            for stage in ['memory', 'input', 'output', 'cortex']:
                if stage in run.stages:
                    print(f"    {run.stages[stage]}")
            if run.joint_before and run.joint_after:
                print(f"    {run.joint_before}")
                print(f"    {run.joint_after} (improvement: "
                      f"{run.joint_after.fitness - run.joint_before.fitness:+d})")
            print()
        
        print("Use --compare, --bottlenecks, or --all for detailed analysis.")


if __name__ == "__main__":
    main()
