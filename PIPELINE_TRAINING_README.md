# Pipeline Training Automation

Three Python scripts automate the curriculum-based pipeline training workflow, modeled after `src/train_help.py`.

## 1. `pipeline_patterns.py` — Pattern Generation

Generates training data for different pipeline stages.

### Usage

```bash
# Generate memory retention keys (bit patterns to store/recall)
python3 src/pipeline_patterns.py memory \
  --width 4 \
  --keys 10 \
  --repeats 2 \
  --seed 0xDEADBEEF \
  --output genomes/memory_keys.txt

# Generate identity I/O tasks (input = output)
python3 src/pipeline_patterns.py identity \
  --width 4 \
  --tasks 20 \
  --repeats 1 \
  --seed 0xDEADBEEF \
  --output genomes/identity_tasks.txt

# Generate XOR logic tasks (8-bit input, split into two 4-bit operands)
python3 src/pipeline_patterns.py logic \
  --width 8 \
  --operation xor \
  --tasks 20 \
  --repeats 1 \
  --seed 0xDEADBEEF \
  --output genomes/xor_tasks.txt

# Add robustness via jittered/dropped bits
python3 src/pipeline_patterns.py logic \
  --width 8 \
  --operation and \
  --tasks 20 \
  --jitter-density 0.05 \
  --jitter-drop 0.02
```

### Modes

- **`memory`** — Random bit patterns for memory retention (write, silence, probe cycle)
  - `--width` — bits per pattern (default: 4)
  - `--keys` — distinct patterns (default: 10)
  - `--repeats` — times to repeat each key (default: 1)

- **`identity`** — I/O pairs where input equals output
  - `--width` — bits per pattern (default: 4)
  - `--tasks` — I/O pairs (default: 10)

- **`logic`** — Two-input bitwise logic (8-bit = two 4-bit operands)
  - `--operation` — and, or, xor, nand, nor (default: xor)
  - `--width` — bits total (must be even, default: 8)
  - `--tasks` — I/O pairs (default: 20)

### All Modes

- `--repeats` — repeat each pattern N times (default: 1)
- `--seed` — RNG seed (default: 0xDEADBEEF)
- `--output` — output file (default: based on mode)
- `--jitter-density` — bit-flip probability per bit (default: 0.0)
- `--jitter-drop` — probability entire cycle is zeroed (default: 0.0)

---

## 2. `pipeline_train_runner.py` — Full Orchestration

Automates the complete pipeline training workflow: pattern generation, build, training, reporting.

### Quick Start

```bash
# Run full curriculum training on default 4x8 grid, all stages, identity task
python3 src/pipeline_train_runner.py

# Custom grid and task
python3 src/pipeline_train_runner.py \
  --xs 8 --ys 8 \
  --task xor \
  --generations 200 \
  --trials 32 \
  --patterns 10

# Skip pattern generation (use existing genomes/memory_keys.txt, genomes/io_tasks.txt)
python3 src/pipeline_train_runner.py --skip-pattern-gen

# Skip build (use existing obj_curriculum/curriculum_train)
python3 src/pipeline_train_runner.py --skip-build
```

### Options

**Grid & Stages**
- `--xs`, `--ys` — grid dimensions (default: 4×8)
- `--stages` — memory | input | output | cortex | joint | all (default: all)

**Training**
- `--generations` — GA generations per stage (default: 100)
- `--trials` — candidates per generation (default: 24)
- `--patterns` — training patterns per stage (default: 5)

**Memory Stage (Stage 1)**
- `--initial-silence` — starting silence gap (default: 2)
- `--max-silence` — silence increases until this cap (default: 16)

**Joint Fine-Tune (Stage 5)**
- `--joint-gen` — generations (default: 10)
- `--joint-trials` — candidates per generation (default: 6)
- `--joint-perturb` — neurons mutated per candidate (default: 3)

**Mutation**
- `--mask-rate` — mask bit-flip probability (default: 0.01)
- `--sens-rate` — sensitivity bit-flip probability (default: 0.01)

**Robustness**
- `--jitter-density` — pattern bit-flip probability (default: 0.0)
- `--jitter-drop` — pattern dropout probability (default: 0.0)

**I/O & Output**
- `--task` — identity | and | or | xor | nand | nor (default: identity)
- `--seed` — RNG seed (default: 0xA17F2026)
- `--genome-dir` — output directory (default: genomes/)
- `--log-every` — logging frequency in generations (default: 10)

**Skipping Steps**
- `--skip-pattern-gen` — use existing patterns (don't regenerate)
- `--skip-build` — use existing binary (don't rebuild)

---

## 3. `pipeline_analysis.py` — Result Analysis

Parse and compare `curriculum_report.txt` files across multiple runs.

### Quick Start

```bash
# Summarize one run
python3 src/pipeline_analysis.py genomes/

# Side-by-side comparison of multiple runs
python3 src/pipeline_analysis.py \
  genomes_run1/ genomes_run2/ genomes_run3/ \
  --compare

# Identify bottlenecks (stage-to-stage fitness drops)
python3 src/pipeline_analysis.py \
  genomes_run1/ genomes_run2/ \
  --bottlenecks

# Full analysis
python3 src/pipeline_analysis.py genomes_*/ --all
```

### Options

- `directories` — one or more directories with `curriculum_report.txt`
- `--compare` — show side-by-side fitness table
- `--bottlenecks` — identify stage-to-stage fitness drops
- `--all` — run all analyses

### Output

**Default (no flags):** Summary of each run's fitness progression.

**`--compare`:** Table showing fitness across all stages for each run, making it easy to spot:
- Which stages are bottlenecks
- How compositions compare to isolated training
- Which grid/task combinations work best

**`--bottlenecks`:** Lists all fitness drops > 0 between consecutive stages, e.g.:
```
input → output:  fitness drop 2.0
cortex → joint:  fitness drop 1.0
```

This identifies where composition suffers most — a signal to revisit that stage's training (e.g., increase generations, adjust jitter).

---

## Workflow Example

Run a full pipeline experiment with three task conditions:

```bash
# Identity task
python3 src/pipeline_train_runner.py \
  --task identity \
  --xs 4 --ys 8 \
  --genome-dir genomes_identity_4x8 \
  --generations 150 --trials 32

# XOR task
python3 src/pipeline_train_runner.py \
  --task xor \
  --xs 4 --ys 8 \
  --genome-dir genomes_xor_4x8 \
  --generations 150 --trials 32

# AND task with jitter
python3 src/pipeline_train_runner.py \
  --task and \
  --xs 4 --ys 8 \
  --genome-dir genomes_and_4x8_jitter \
  --jitter-density 0.05 \
  --generations 150 --trials 32

# Compare all three
python3 src/pipeline_analysis.py \
  genomes_identity_4x8 \
  genomes_xor_4x8 \
  genomes_and_4x8_jitter \
  --all
```

---

## Integration with Makefile

All scripts live in `src/` and can be called directly. The runner optionally calls `make curriculum-build` with grid parameters.

```bash
# Manual: generate patterns, build, train
python3 src/pipeline_patterns.py memory --keys 5 --output genomes/keys.txt
make curriculum-build CURR_XS=4 CURR_YS=8
./obj_curriculum/curriculum_train --stage all

# Automated: everything in one command
python3 src/pipeline_train_runner.py --xs 4 --ys 8 --task xor
```

---

## Notes

- **Memory stage curriculum:** automatically increases silence gap until reaching `--max-silence` as fitness improves.
- **Robustness via jitter:** small `--jitter-density` and `--jitter-drop` values stress the system and can prevent overfitting to clean patterns.
- **Joint fine-tuning:** uses a gentler perturbation strategy (mutate only ~3 neurons) rather than full bit-flip to refine frozen stages.
- **Report parsing:** `pipeline_analysis.py` extracts fitness/generation/ranking from the report and compares across runs; bottleneck detection highlights where composition fails.
