# Staged Curriculum: Memory → Input → Output → Cortex → Joint Fine-Tune

This document describes the staged-curriculum trainer added by this change:
`src/curriculum_train.cpp`, `src/verilog/pipeline.v`, and the `make
curriculum` / `make curriculum-build` targets. It also records the current
state of the pieces it builds on, since the task asked for that orientation
before any changes were made.

## 1. Orientation: what already existed

| Piece | File | State found |
|---|---|---|
| `chunk` module | [chunk.v](/workspaces/Artifical_Life/src/verilog/chunk.v) | One `XS`x`YS` grid of `neuron` instances, cardinal-neighbor pulse routing, one external input row (`in[XS]`, 4 bits/column), one output row (`out[XS]`, 1 bit/column = south-firing bit of the last row). |
| `neuron` module | [neuron.v](/workspaces/Artifical_Life/src/verilog/neuron.v) | 4-bit mask genome, 2-bit sensitivity genome (thresholds `1/3/5/7`), 3-bit accumulator, one-cycle refractory period. No state decay: an unfired accumulator holds its value indefinitely until it next receives input. |
| `c_link` / `c_link_bank` | [chunk.v](/workspaces/Artifical_Life/src/verilog/chunk.v) (bottom) | **Confirmed by reading the code, not assumed:** `c_link` is `destination_pulse = enable ? source_pulse : 0` — a single `enable`-gated identity pass, **directional** (one source bus to one destination bus), carrying whatever width (`WIDTH`) it's parameterized with. `c_link_bank` is just an array of independent `c_link`s. **It only routes whatever bits are handed to it; it does not itself carry "multi-bit values" beyond what's wired in.** Because a chunk's `out` port is only 1 bit/column (the south-firing bit) while its `in` port accepts 4 bits/column, an inter-chunk link built from one chunk's `out` into another's `in` is, in practice, a 1-bit-per-column pulse bus, zero-extended at the receiving end — not a genuinely multi-bit value channel. There is no system-level wrapper that instantiates multiple chunks and a link table; the README explicitly says this is still needed. `pipeline.v` (added here) is that wrapper. |
| CPU simulator/trainer | [sim_server.cpp](/workspaces/Artifical_Life/src/sim_server.cpp), [trainer_server.cpp](/workspaces/Artifical_Life/src/trainer_server.cpp), [main.cpp](/workspaces/Artifical_Life/src/main.cpp) | `sim_server.cpp`'s `evolve()`/`evaluateGenome()` **already implements** the three fixes from `FITNESS_FIXES.md`: warm-up exclusion (`warmupCycles = YS - 1`), a 0.6·min + 0.4·avg blended score, and an accumulator-distance tiebreaker. This is the pattern `curriculum_train.cpp` generalizes to multiple chunks/stages. `trainer_server.cpp` is a separate (1+λ)/elitist population trainer with a CUDA backend option; it does not yet have the warm-up fix and works on one chunk against `wanted_outputs.txt`. |
| CUDA fitness/trainer | [cuda_fitness.h](/workspaces/Artifical_Life/src/cuda_fitness.h), `cuda_fitness.cu`, `cuda_train.cu` | `CudaGenome` has no `C_LINK` genes (confirms the README's note: "the current genome format has no C_LINK genes"). Single-chunk only. |
| Fitness scoring | `FITNESS.md`, `FITNESS_FIXES.md` | Graded per-bit scoring (`fitness = XS - falsePositives - misses`), worst-case-over-window aggregation, later revised (in `sim_server.cpp`) to blended min/avg + tiebreaker. This is the scheme extended here for every new fitness function. |

**Answers to the flagged ambiguous points, from reading the code (not assumed):**

- C_LINK capacity/capabilities: single directional enable-gated pulse bus,
  effectively 1 bit/column end-to-end given the existing `chunk` port
  shapes; `c_link_bank` supports an arbitrary fixed number of such links
  (`LINKS` parameter), each independently enabled.
- One cortex chunk vs a split read/compute pair: **one chunk**, see the
  rationale in [Stage 4](#stage-4-cortex-chunk) below.
- Silence-phase starting duration: **2 cycles**, doubling up to a
  configurable cap (default 16) each time the memory chunk reaches 90% of
  max fitness at the current gap (see [Stage 1](#stage-1-memory-chunk)).

## 2. Assumption: shared grid size across all four chunks

`pipeline.v` and `curriculum_train.cpp` assume all four chunks (memory,
input, cortex, output) use the same `XS`x`YS` grid, set at compile time via
`CURR_XS`/`CURR_YS` (default `4`x`8`, matching the browser/dashboard
defaults). This keeps a single Verilated `Vchunk` model reusable across all
four isolated-training stages (stages 1-4 never need more than one chunk
"role" instantiated at a time) and a single `Vpipeline` model for the joint
stage. Per-role grid sizes (e.g. a larger memory chunk than input chunk)
would require compiling a distinct Verilated model per role and are not
implemented; this is called out here as a scope limit rather than left
implicit.

## 3. Stage 1: memory chunk

Implemented in `evaluateMemory*` in `curriculum_train.cpp`.

- **Write phase** (`writeCycles`, default 4): the chunk's input row is held
  at a training pattern ("key") every cycle.
- **Silence phase** (`silenceCycles`, starts at 2, doubles up to
  `maxSilence` — default 16 — each time the previous gap reaches 90% of
  `XS` ranking fitness): input row held at 0. This is the actual retention
  capability under test.
- **Probe phase** (`probeCycles`, default `max(YS+4, 12)`): input row is
  held at a constant, information-free stimulus (all columns = 1) so that
  whichever locations still hold more residual accumulator charge from the
  write phase are the first to cross threshold and fire. The chunk's warm-up
  window (`YS - 1` cycles) is excluded from scoring within the probe phase
  itself, per the FITNESS_FIXES.md diagnosis.
- **Fitness**: reconstruction is graded per-bit (false positives / misses,
  as in `FITNESS.md`) against the original key, aggregated across the
  scored probe cycles with the existing 0.6·min + 0.4·avg blend plus the
  accumulator-distance tiebreaker, then averaged (with a worst-case floor)
  across a small fixed set of training keys so the genome doesn't overfit
  one specific bit pattern.
- Silence duration is a runtime CLI parameter (`--initial-silence`,
  `--max-silence`), and the trainer automatically curricula through
  increasing gaps once the current gap is solved well enough, carrying the
  genome forward rather than restarting.

## 4. Stage 2: input chunk

Implemented in `evaluateInputChunk*`.

The memory chunk's genome from stage 1 is frozen and loaded unchanged. The
input chunk being trained is driven with the *same* keys used in stage 1;
its own output is fed, cycle-by-cycle, directly into the (frozen) memory
chunk's input row for the whole write phase — i.e. the exact same
write/silence/probe harness as stage 1, just with the input chunk sitting in
front of the memory chunk instead of a hand-fed pattern. The fitness target
is still the original key (not some new, independently-designed target),
and it is scored through the frozen memory chunk's **real** simulated
response, not an idealized stand-in — this satisfies the "derive the input
chunk's training signal from the memory chunk's actual, trained behavior"
constraint. A separate, smaller jitter probability is applied to the
input→memory link itself (in addition to jitter on the input chunk's own
driving key) to model an imperfect connection.

## 5. Stage 3: output chunk

Implemented in `evaluateOutputChunk*`. Same graded per-bit scheme as
`FITNESS.md`, aggregated with the min/avg blend and tiebreaker exactly like
`sim_server.cpp`'s `evaluateGenome`. The excluded warm-up window is:

```
totalWarmup = (YS - 1)                      // this chunk's own settle time
            + cumulativeUpstreamHops*(YS-1) // input, memory, cortex (3 hops)
```

even though the output chunk is trained standing alone here (its eventual
upstream neighbors don't exist yet as far as this stage is concerned). This
directly implements the task's requirement that warm-up exclusion be
"scaled to that chunk's position in the pipeline (cumulative latency from
all preceding hops, not just its own internal grid size)" rather than
waiting until stage 5 to discover the problem.

Training signal: the `input`/`output` pairs from `wanted_outputs.txt` if its
bit width matches `CURR_XS` (it doesn't at the default `4`x`8` build, since
the shipped file is 16-bit), otherwise a small built-in identity-style
fallback task (documented in code) so the stage still has a well-defined
target at the curriculum's default grid size.

## 6. Stage 4: cortex chunk

Implemented in `evaluateCortex*`.

**Decision: one cortex chunk, not a split read/compute pair.** Both frozen
memory and frozen output chunks are wired around the single candidate
cortex chunk: memory is written with a task's `input`, probed, its output
fed into the cortex candidate every probe cycle, and the cortex candidate's
output is fed every cycle into the frozen output chunk, whose output is
scored against the task's `output` — again with per-bit grading, min/avg
blend, tiebreaker, and warm-up excluded for `3*(YS-1)` cycles (memory +
cortex + output settle time in series).

**Tradeoff considered and rejected:** splitting into a "read" chunk (copy
memory's output faithfully) and a "compute" chunk (map that copy to the
wanted output) would need a hand-designed intermediate target for the read
chunk — e.g., "reproduce memory's output verbatim" — to keep its fitness
well-defined in isolation. That is exactly the kind of hand-designed
intermediate signal the task asks to avoid at the input/memory boundary,
and there's no equivalent naturally-supervised signal available at the
memory/cortex boundary (unlike input/memory, where the original key is a
legitimate, already-necessary target). A single cortex chunk keeps one
unambiguous, end-task-grounded fitness function, at the cost of asking one
chunk's genome to do more work. If the single cortex chunk plateaus well
below what the output chunk can express on its own, that is the signal to
revisit this decision and try the split (see the boundary-drop check in
[Validation](#8-validation)).

## 7. Stage 5: joint fine-tune via `pipeline.v` / `C_LINK`

`src/verilog/pipeline.v` instantiates all four chunks and wires them with
three `c_link` connections (input→memory, memory→cortex, cortex→output),
matching the README's description of what a system-level wrapper still
needed to add. The harness (`curriculum_train.cpp`) directly drives the
link `enable` signals and a couple of phase-control ports on `pipeline`
(there is no learned gating of link timing here — the harness plays that
role, same as a real controller would):

1. Write phase: `link_in_to_mem_en = 1`, the input chunk is driven with the
   task's `input` pattern (with jitter).
2. Silence phase: link disabled, no drive.
3. Probe/read/compute/emit phase: `probe_en = 1` (feeds a constant probe
   straight into memory, bypassing the input chunk), `link_mem_to_cortex_en
   = 1`, `link_cortex_to_out_en = 1` — all three downstream hops run
   concurrently every cycle, and the output chunk's output is scored
   against the task's wanted output with a `3*(YS-1)`-cycle warm-up
   exclusion.

The four frozen genomes from stages 1-4 seed a short, low-mutation-rate
fine-tune: `--joint-generations` (default 10) generations of
`--joint-trials` (default 6) candidates, each produced by touching only
`--joint-perturb` (default 3) individual neurons per genome per candidate
(a small-perturbation mutation mode, not the full per-gene coin-flip sweep
used in stages 1-4) — this is deliberately gentler than isolated-stage
training, per the "assume frozen, independently-trained chunks do *not*
compose perfectly" instruction. Before/after fitness on the full pipeline is
printed and written to the report.

## 8. Validation

Run with:

```sh
make curriculum-build CURR_XS=4 CURR_YS=8
./obj_curriculum/curriculum_train --stage=all
```

Useful flags: `--stage=memory|input|output|cortex|joint|all`,
`--generations`, `--trials`, `--pattern-count`, `--initial-silence`,
`--max-silence`, `--joint-generations`, `--joint-trials`, `--joint-perturb`,
`--mask-rate`, `--sens-rate`, `--jitter-density`, `--jitter-drop`, `--seed`,
`--genome-dir` (defaults to `genomes/`, gitignored), `--log-every`.

Per the task instructions, this change was validated by building and by
short, `timeout`-bounded smoke runs rather than a full training run left
running in the background:

```sh
timeout 90 ./obj_curriculum/curriculum_train \
  --generations=3 --trials=4 --pattern-count=2 \
  --initial-silence=2 --max-silence=4 \
  --joint-generations=2 --joint-trials=3 --log-every=1 \
  --genome-dir=/tmp/curriculum_smoke
```

This completed all five stages end-to-end without error, at `XS=4, YS=8`,
producing a report of the form:

```
Curriculum training report (XS=4, YS=8)
Stage 1 (memory) silence=2: fitness=3 ranking=3 generations=3
Stage 2 (input): fitness=3 ranking=3 generations=3
Stage 3 (output): fitness=3 ranking=3 generations=3
Stage 4 (cortex): fitness=3 ranking=3 generations=3
Stage 5 (joint) before: fitness=3 ranking=3
Stage 5 (joint) after: fitness=3 ranking=3 generations=2
```

A separate longer (still `timeout`-bounded, not backgrounded) smoke run of
stage 1 alone for 60 generations showed the tiebreaker term improving
(`2` → `1`) while the integer fitness held at its 3-generation-budget value,
confirming the hill-climb loop is exerting real selection pressure rather
than being stuck — 3-4 generations is simply too small a budget to expect
`XS`-perfect reconstruction. **These smoke runs are not full training
results**; they only demonstrate the pipeline executes correctly
end-to-end. A real training pass should use much larger
`--generations`/`--trials` budgets (e.g. hundreds of generations, as in
`FITNESS.md`'s 24-trial/32-cycle single-chunk baseline) and should be run
by the user outside of this change, since this task was scoped to "get it
ready," not to produce final converged genomes.

For each stage, report fitness achieved / generations / grid parameters by
reading `genomes/curriculum_report.txt` after a real run. For the joint
stage, compare the "before" fine-tune fitness (frozen, independently
trained chunks composed for the first time through real `c_link` wiring)
against the "after" fitness; a large gap between a stage's isolated
training fitness (in the earlier report lines) and its contribution to the
joint "before" score is the signal described in the task to revisit that
chunk boundary (most likely: the cortex chunk, since it has the least
protection from distribution shift, or the input↔memory boundary, since
memory is trained before input in this order and only sees replayed
patterns, not the input chunk's actual imperfect encoding, until stage 2).

## 9. Known limitations / suggested follow-ups

- All four chunks share one grid size (see [Assumption](#2-assumption-shared-grid-size-across-all-four-chunks)).
- The probe stimulus, phase lengths, and link enables are driven by the
  C++/Verilog harness, not learned or encoded in any chunk's genome; a more
  ambitious version would give the input chunk itself a learned "probe
  cue" role instead of a harness-provided constant.
- Cortex is a single chunk (see [Stage 4](#6-stage-4-cortex-chunk) tradeoff).
- Stage evaluation uses a small, fixed pattern set (`--pattern-count`,
  default 3) for tractable smoke-testing; wider training should raise this.
