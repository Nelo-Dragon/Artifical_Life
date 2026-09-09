# Fitness System: Proposed Fixes

This document proposes changes to the fitness scoring described in `FITNESS.md`,
aimed at slow convergence in the CPU and CUDA trainers. No source files were
available at the time of writing, so changes are described at the algorithm
level rather than as a patch.

## Diagnosis

Two properties of the current scoring combine to produce a weak, noisy
selection signal:

1. **Unscored propagation delay.** Pulses take roughly `YS - 1` clock cycles
   to travel from the input row to the output row. Scoring begins at cycle 1,
   so every candidate — regardless of genome quality — registers misses
   during this warm-up window purely from signal travel time, not from
   genome quality.
2. **Worst-cycle aggregation on a coarse scale.** `candidate fitness = min`
   across all 32 cycles, where each cycle's fitness is an integer in
   `[0, XS]`. Combining a 5-value scale (`XS = 4`) with a strict minimum
   collapses most of the 24 mutated candidates to the same floor score,
   most of them tied because of the warm-up problem above rather than any
   real behavioral difference. With most candidates tied, generation after
   generation, evolution selects close to randomly.

## Fix 1: Exclude propagation warm-up from scoring

```text
warmup = YS - 1                      // or measured empirically
eval_cycles = cycles[warmup+1 .. N]  // N = 32 by default
candidate_fitness = min(fitness(c) for c in eval_cycles)
```

This is the highest-leverage, lowest-risk change: it removes an unavoidable,
uninformative floor from every candidate's score without altering what
"correct sustained output" means.

## Fix 2: Soften the worst-case aggregation

A single stray cycle should not erase 20+ good cycles. Replace the strict
minimum with a blend of minimum and average, or the mean of the worst
quartile of cycles:

```text
candidate_fitness = 0.6 * min(eval_cycles) + 0.4 * average(eval_cycles)
```

```text
// alternative: mean of the worst 25% of post-warmup cycles
worst_quartile = sort(eval_cycles)[: len(eval_cycles) // 4]
candidate_fitness = average(worst_quartile)
```

Either keeps the "must sustain correct output" property while giving
candidates a distinguishable score instead of many ties at the floor.

## Fix 3: Add a continuous tiebreaker below the integer score

`fitness = XS - false_positives - misses` only has `XS + 1` possible values
per cycle, so ties will remain common even after Fixes 1–2. Add a secondary,
finer-grained ranking used only to break ties on the primary score — for
example, the sum of each output-row neuron's accumulator distance to its
firing threshold, so a neuron that is "close but not yet firing" ranks above
one that is "nowhere close," even though both currently show `miss = 1`.

```text
tiebreaker = sum(
    threshold(n) - accumulator(n) if wanted(n) == 1 else accumulator(n)
    for n in output_row_neurons
)
```

Lower `tiebreaker` is better; used only when primary `candidate_fitness`
scores are equal.

## Fix 4: Mutation bias

Once Fixes 1–3 stop the score from being swamped by warm-up and tie
collapse, the mutation operator itself matters more:

- Weight sensitivity mutations somewhat higher than mask mutations.
  Sensitivity changes shift the firing threshold by one step
  (`1 → 3 → 5 → 7`), a smaller perturbation than flipping a mask direction
  bit, which is more likely to move a near-correct genome incrementally
  rather than displace it randomly.
- Decay the mutation rate as best-fitness plateaus, rather than holding it
  fixed at the default 4% throughout a run.

## Fix 5: Scale candidate count on the CUDA trainer

The (1+N) hill-climb (current genome + N mutated candidates, keep the best)
is trivially parallel across candidates. 24 candidates is a reasonable
default for the CPU/browser path; the CUDA trainer can evaluate hundreds to
thousands of candidates per generation for the same generation count, which
improves convergence *per generation*, not just wall-clock time — this
compounds with Fixes 1–3 rather than replacing them.

## Suggested implementation order

| Priority | Fix | Effort | Expected impact |
|---|---|---|---|
| 1 | Warm-up exclusion | Low | High |
| 2 | Soft worst-case aggregation | Low | High |
| 3 | Continuous tiebreaker | Medium | Medium |
| 4 | Mutation bias / decay | Low–Medium | Medium |
| 5 | Larger CUDA candidate pool | Medium | Medium–High (compounding) |

Fixes 1 and 2 address the specific cause of ties and floor-scoring described
above and are the recommended starting point before tuning mutation
behavior or scaling candidate count.
