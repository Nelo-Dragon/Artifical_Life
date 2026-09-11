#!/usr/bin/env python3
"""
Orchestrate pipeline training runs end-to-end:
1. Generate training patterns (memory keys, I/O tasks).
2. Build curriculum trainer with specified grid parameters.
3. Run curriculum stages sequentially (memory → input → output → cortex → joint).
4. Parse results and report convergence.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for a full pipeline training run."""
    grid_xs: int = 4
    grid_ys: int = 8
    stages: str = "all"  # "memory", "input", "output", "cortex", "joint", "all"
    
    # Generation / trial parameters
    generations: int = 100
    trials: int = 24
    pattern_count: int = 5
    
    # Memory-specific
    initial_silence: int = 2
    max_silence: int = 16
    
    # Joint fine-tuning
    joint_generations: int = 10
    joint_trials: int = 6
    joint_perturb: int = 3
    
    # GA mutation rates
    mask_rate: float = 0.01
    sens_rate: float = 0.01
    
    # Robustness (jitter)
    jitter_density: float = 0.0
    jitter_drop: float = 0.0
    
    # I/O
    seed: int = 0xA17F2026
    genome_dir: Path = Path("genomes")
    log_every: int = 10
    
    # Optional: preset task ('identity', 'xor', 'and', etc.)
    task: str = "identity"


def run_command(cmd: list[str], description: str = None) -> int:
    """Run a shell command and return exit code."""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def generate_patterns(config: TrainingConfig) -> None:
    """Generate training patterns for pipeline stages."""
    if not config.genome_dir.exists():
        config.genome_dir.mkdir(parents=True, exist_ok=True)
    
    # Memory keys: small bit width (just hold arbitrary patterns)
    mem_keys_file = config.genome_dir / "memory_keys.txt"
    cmd = [
        "python3", "src/pipeline_patterns.py", "memory",
        "--width", str(config.grid_xs),
        "--keys", str(config.pattern_count),
        "--seed", str(config.seed),
        "--output", str(mem_keys_file),
    ]
    if config.jitter_density > 0 or config.jitter_drop > 0:
        cmd.extend([
            "--jitter-density", str(config.jitter_density),
            "--jitter-drop", str(config.jitter_drop),
        ])
    run_command(cmd, f"Generating memory keys ({config.grid_xs} bits, "
                      f"{config.pattern_count} patterns)")
    
    # I/O tasks: identity or logic-based
    io_tasks_file = config.genome_dir / "io_tasks.txt"
    if config.task == "identity":
        cmd = [
            "python3", "src/pipeline_patterns.py", "identity",
            "--width", str(config.grid_xs),
            "--tasks", str(config.pattern_count),
            "--seed", str(config.seed),
            "--output", str(io_tasks_file),
        ]
    else:
        # Assume task is a bitwise logic operation
        cmd = [
            "python3", "src/pipeline_patterns.py", "logic",
            "--width", str(config.grid_xs * 2),  # Two operands
            "--operation", config.task,
            "--tasks", str(config.pattern_count),
            "--seed", str(config.seed),
            "--output", str(io_tasks_file),
        ]
    if config.jitter_density > 0 or config.jitter_drop > 0:
        cmd.extend([
            "--jitter-density", str(config.jitter_density),
            "--jitter-drop", str(config.jitter_drop),
        ])
    run_command(cmd, f"Generating I/O tasks ({config.task}, "
                      f"{config.pattern_count} patterns)")
    
    print(f"\nPatterns written to {config.genome_dir}/")
    return io_tasks_file


def build_curriculum(config: TrainingConfig) -> None:
    """Build curriculum trainer binary."""
    cmd = [
        "make", "curriculum-build",
        f"CURR_XS={config.grid_xs}",
        f"CURR_YS={config.grid_ys}",
    ]
    code = run_command(cmd, f"Building curriculum trainer "
                            f"({config.grid_xs}x{config.grid_ys})")
    if code != 0:
        print(f"ERROR: Build failed with code {code}")
        sys.exit(1)


def run_curriculum(config: TrainingConfig) -> None:
    """Run curriculum training for all stages."""
    cmd = [
        "./obj_curriculum/curriculum_train",
        f"--stage={config.stages}",
        f"--generations={config.generations}",
        f"--trials={config.trials}",
        f"--pattern-count={config.pattern_count}",
        f"--initial-silence={config.initial_silence}",
        f"--max-silence={config.max_silence}",
        f"--joint-generations={config.joint_generations}",
        f"--joint-trials={config.joint_trials}",
        f"--joint-perturb={config.joint_perturb}",
        f"--mask-rate={config.mask_rate}",
        f"--sens-rate={config.sens_rate}",
        f"--seed={config.seed}",
        f"--genome-dir={config.genome_dir}",
        f"--log-every={config.log_every}",
    ]
    
    code = run_command(cmd, f"Running curriculum training "
                            f"(stages: {config.stages})")
    if code != 0:
        print(f"ERROR: Training failed with code {code}")
        sys.exit(1)


def report_results(config: TrainingConfig) -> None:
    """Print training results."""
    report_file = config.genome_dir / "curriculum_report.txt"
    if not report_file.exists():
        print(f"\nNo report file found at {report_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"  Training Report")
    print(f"{'='*60}\n")
    with open(report_file) as f:
        print(f.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automate full pipeline (curriculum) training runs."
    )
    parser.add_argument('--xs', type=int, default=4,
                        help='Grid X dimension (default 4)')
    parser.add_argument('--ys', type=int, default=8,
                        help='Grid Y dimension (default 8)')
    parser.add_argument('--stages', type=str, default='all',
                        choices=['memory', 'input', 'output', 'cortex', 'joint', 'all'],
                        help='Stages to run (default: all)')
    
    parser.add_argument('--generations', type=int, default=100,
                        help='GA generations per stage (default 100)')
    parser.add_argument('--trials', type=int, default=24,
                        help='Candidates per generation (default 24)')
    parser.add_argument('--patterns', type=int, default=5,
                        help='Training patterns per stage (default 5)')
    
    parser.add_argument('--initial-silence', type=int, default=2,
                        help='Initial silence gap for memory (default 2)')
    parser.add_argument('--max-silence', type=int, default=16,
                        help='Max silence gap (default 16)')
    
    parser.add_argument('--joint-gen', type=int, default=10,
                        help='Joint fine-tune generations (default 10)')
    parser.add_argument('--joint-trials', type=int, default=6,
                        help='Joint fine-tune candidates (default 6)')
    parser.add_argument('--joint-perturb', type=int, default=3,
                        help='Neurons perturbed per joint candidate (default 3)')
    
    parser.add_argument('--mask-rate', type=float, default=0.01,
                        help='Mask bit-flip rate (default 0.01)')
    parser.add_argument('--sens-rate', type=float, default=0.01,
                        help='Sensitivity bit-flip rate (default 0.01)')
    
    parser.add_argument('--jitter-density', type=float, default=0.0,
                        help='Pattern bit-flip rate (default 0.0)')
    parser.add_argument('--jitter-drop', type=float, default=0.0,
                        help='Pattern drop rate (default 0.0)')
    
    parser.add_argument('--task', type=str, default='identity',
                        choices=['identity', 'and', 'or', 'xor', 'nand', 'nor'],
                        help='I/O task type (default: identity)')
    
    parser.add_argument('--seed', type=int, default=0xA17F2026,
                        help='RNG seed (default 0xA17F2026)')
    parser.add_argument('--genome-dir', type=Path, default=Path('genomes'),
                        help='Output directory for genomes/reports (default: genomes/)')
    parser.add_argument('--log-every', type=int, default=10,
                        help='Log frequency in generations (default 10)')
    
    parser.add_argument('--skip-pattern-gen', action='store_true',
                        help='Skip pattern generation (use existing)')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip build (use existing binary)')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        grid_xs=args.xs,
        grid_ys=args.ys,
        stages=args.stages,
        generations=args.generations,
        trials=args.trials,
        pattern_count=args.patterns,
        initial_silence=args.initial_silence,
        max_silence=args.max_silence,
        joint_generations=args.joint_gen,
        joint_trials=args.joint_trials,
        joint_perturb=args.joint_perturb,
        mask_rate=args.mask_rate,
        sens_rate=args.sens_rate,
        jitter_density=args.jitter_density,
        jitter_drop=args.jitter_drop,
        task=args.task,
        seed=args.seed,
        genome_dir=args.genome_dir,
        log_every=args.log_every,
    )
    
    print(f"\n{'='*60}")
    print(f"  Pipeline Training Orchestrator")
    print(f"{'='*60}")
    print(f"  Grid: {config.grid_xs}x{config.grid_ys}")
    print(f"  Stages: {config.stages}")
    print(f"  Generations: {config.generations}")
    print(f"  Trials: {config.trials}")
    print(f"  Patterns: {config.pattern_count}")
    print(f"  Task: {config.task}")
    print(f"  Output: {config.genome_dir}/")
    print(f"{'='*60}\n")
    
    # Generate patterns
    if not args.skip_pattern_gen:
        generate_patterns(config)
    
    # Build
    if not args.skip_build:
        build_curriculum(config)
    
    # Run
    run_curriculum(config)
    
    # Report
    report_results(config)
    
    print(f"\n{'='*60}")
    print(f"  Pipeline training complete!")
    print(f"  Genomes and report in: {config.genome_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
