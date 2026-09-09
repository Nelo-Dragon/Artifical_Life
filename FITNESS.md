# Fitness System

This project scores one simulated chunk against a wanted output pattern. The fitness system is used by the CPU observatory's **Evolve** action.

## Output Comparison

The chunk has `XS` output bits. Each output bit is compared with the corresponding wanted bit.

For each bit:

| Actual output | Wanted output | Result |
|---|---|---|
| `0` | `0` | Correct |
| `1` | `1` | Correct |
| `1` | `0` | False positive |
| `0` | `1` | Miss |

A `1` in a wanted `0` position is therefore penalized. It is counted as a false positive.

The implementation calculates:

```text
false positives = output & ~wanted
misses          = wanted & ~output
fitness         = XS - false positives - misses
```

The maximum fitness is `XS`, and the minimum fitness is `0`.

## Time-Based Scoring

A genome is not judged only by the output on its final clock cycle.

Each candidate is:

1. Loaded into the chunk.
2. Reset so every neuron loads its mask and sensitivity genes.
3. Simulated for `32` clock cycles.
4. Scored after every clock cycle.
5. Assigned the **worst fitness** it reached during the run.

In other words:

```text
candidate fitness = minimum fitness across all 32 cycles
```

This prevents a genome from receiving a good score merely because its final output happens to match. A candidate must avoid unwanted output and missed output throughout the full evaluation window.

## Evolution

When **Evolve** is clicked:

1. The current genome is evaluated as the baseline candidate.
2. `24` mutated candidates are generated.
3. Each candidate mutates masks and sensitivities independently.
4. Each candidate is evaluated for 32 clock cycles.
5. The candidate with the highest worst-case fitness is kept.
6. The winning genome is loaded into the chunk.
7. The winning genome is simulated again for 32 cycles so the displayed output matches its reported score.
8. The generation counter increases by one.

The mutation rate is a probability applied independently to each gene field:

- Mask genes may flip one of their four direction bits.
- Sensitivity genes may flip one of their two bits.

The default mutation rate is `4%` and can be changed with the UI slider.

## Scores Shown in the UI

The observatory reports two related scores:

- **Best fitness**: the score returned by the most recent evolution run. This is the winner's worst score over its 32-cycle evaluation.
- **Current fitness**: the score of the output currently visible in the simulator.

It also reports the current output's error components:

- **False positives**: output bits that are `1` when wanted bits are `0`.
- **Misses**: wanted bits that are `1` when output bits are `0`.

The relationship is:

```text
current fitness = XS - false positives - misses
```

## Example

For `XS = 4`:

```text
actual:  1 0 0 1
wanted:  0 0 0 1
```

There is one false positive at the first bit and no misses:

```text
false positives = 1
misses          = 0
fitness         = 4 - 1 - 0 = 3
```

If a later clock cycle has two false positives, that cycle scores `2`, and the candidate's final fitness for the full run becomes the lowest score seen across all cycles.

## Current Constants

The CPU simulator currently uses:

```text
Evolution trials: 24 mutated candidates
Evaluation cycles: 32 clock cycles per candidate
Default mutation rate: 0.04 (4%)
Default grid: XS = 4, YS = 8
```

These values are defined in `src/sim_server.cpp` and the simulator defaults are defined in `Makefile`.
