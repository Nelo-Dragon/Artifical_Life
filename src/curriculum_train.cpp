// Staged curriculum trainer for the multi-chunk pipeline described in
// CURRICULUM.md.
//
// Trains, in order:
//   1. a memory ("hippocampus") chunk, alone, with a write/silence/probe
//      reconstruction fitness function and a silence-length curriculum;
//   2. an input chunk, frozen-memory-in-the-loop, supervised by the memory
//      chunk's own real write/silence/probe test (not a hand-designed
//      target encoding);
//   3. an output chunk, with the existing graded per-bit + warm-up-exclusion
//      fitness style, scaled to the pipeline's eventual cumulative latency;
//   4. a cortex chunk that reads the frozen memory chunk and drives the
//      frozen output chunk;
//   5. a short, low-mutation joint fine-tune of all four genomes wired
//      together through `pipeline.v` (which uses the existing `c_link`
//      primitive for every inter-chunk connection).
//
// Every per-chunk fitness function that scores a downstream comparison
// excludes a warm-up window sized to that chunk's cumulative position in
// the pipeline, per the diagnosis in FITNESS_FIXES.md.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "Vchunk.h"
#include "Vpipeline.h"
#include "verilated.h"

namespace {

#ifndef CURR_XS
#define CURR_XS 4
#endif
#ifndef CURR_YS
#define CURR_YS 8
#endif
constexpr int XS = CURR_XS;
constexpr int YS = CURR_YS;
constexpr int N = XS * YS;

constexpr double MIN_FITNESS_WEIGHT = 0.6;
constexpr double AVERAGE_FITNESS_WEIGHT = 0.4;

// ---------------------------------------------------------------------
// Genome + generic simulation helpers, shared by every stage.
// ---------------------------------------------------------------------

struct Genome {
    std::array<std::uint8_t, N> masks{};
    std::array<std::uint8_t, N> sensitivities{};
};

struct JitterConfig {
    double densityProb = 0.0;  // probability any given column's bit is flipped
    double dropCycleProb = 0.0;  // probability the whole row is zeroed this cycle
};

struct Score {
    int falsePositives;
    int misses;
    int fitness;
};

struct Evaluation {
    int fitness = 0;
    double rankingFitness = 0.0;
    double tiebreaker = 0.0;
};

Genome randomGenome(std::mt19937& rng) {
    Genome genome;
    std::uniform_int_distribution<int> maskDist(0, 15);
    std::uniform_int_distribution<int> sensDist(0, 3);
    for (int index = 0; index < N; ++index) {
        genome.masks[index] = static_cast<std::uint8_t>(maskDist(rng));
        genome.sensitivities[index] = static_cast<std::uint8_t>(sensDist(rng));
    }
    return genome;
}

// perturbCount > 0 selects "small perturbation" mode (used by the joint
// fine-tune step): only a handful of neurons are touched per candidate,
// rather than sweeping every gene with an independent coin flip.
void mutateGenome(Genome& genome, double maskRate, double sensRate,
                  int perturbCount, std::mt19937& rng) {
    std::uniform_int_distribution<int> maskBit(0, 3);
    std::uniform_int_distribution<int> sensBit(0, 1);
    if (perturbCount > 0) {
        std::uniform_int_distribution<int> pick(0, N - 1);
        std::bernoulli_distribution touchMask(0.5);
        for (int step = 0; step < perturbCount; ++step) {
            const int index = pick(rng);
            if (touchMask(rng))
                genome.masks[index] ^= static_cast<std::uint8_t>(1u << maskBit(rng));
            else
                genome.sensitivities[index] ^= static_cast<std::uint8_t>(1u << sensBit(rng));
        }
        return;
    }
    std::bernoulli_distribution mutateMask(maskRate);
    std::bernoulli_distribution mutateSensitivity(std::min(1.0, sensRate));
    for (int index = 0; index < N; ++index) {
        if (mutateMask(rng))
            genome.masks[index] ^= static_cast<std::uint8_t>(1u << maskBit(rng));
        if (mutateSensitivity(rng))
            genome.sensitivities[index] ^= static_cast<std::uint8_t>(1u << sensBit(rng));
    }
}

void saveGenome(const std::string& path, const Genome& genome) {
    std::ofstream file(path);
    for (int index = 0; index < N; ++index)
        file << static_cast<int>(genome.masks[index]) << (index + 1 < N ? ' ' : '\n');
    for (int index = 0; index < N; ++index)
        file << static_cast<int>(genome.sensitivities[index]) << (index + 1 < N ? ' ' : '\n');
}

bool loadGenomeFile(const std::string& path, Genome& genome) {
    std::ifstream file(path);
    if (!file)
        return false;
    for (int index = 0; index < N; ++index) {
        int value = 0;
        if (!(file >> value)) return false;
        genome.masks[index] = static_cast<std::uint8_t>(value);
    }
    for (int index = 0; index < N; ++index) {
        int value = 0;
        if (!(file >> value)) return false;
        genome.sensitivities[index] = static_cast<std::uint8_t>(value);
    }
    return true;
}

template <typename Top>
void resetTop(Top& top) {
    top.clk = 0;
    top.rst = 1;
    top.eval();
    top.clk = 1;
    top.eval();
    top.clk = 0;
    top.eval();
    top.rst = 0;
    top.eval();
}

template <typename Top>
void clockOnce(Top& top) {
    top.clk = 1;
    top.eval();
    top.clk = 0;
    top.eval();
}

void loadGenome(Vchunk& top, const Genome& genome) {
    for (int index = 0; index < N; ++index) {
        top.mask_msk[index] = genome.masks[index];
        top.sens_msk[index] = genome.sensitivities[index];
    }
}

// Drives a XS-wide row (any of Vchunk::in, Vpipeline::in_chunk_drive, or
// Vpipeline::mem_probe -- all are CData[XS] arrays) from a packed bit value,
// with optional density/timing jitter so isolated stages are not overfit to
// a perfectly clean upstream signal (constraint 6).
template <typename Row>
void driveRow(Row& row, std::uint64_t bits, const JitterConfig& jitter, std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (jitter.dropCycleProb > 0.0 && uni(rng) < jitter.dropCycleProb) {
        for (int x = 0; x < XS; ++x)
            row[x] = 0;
        return;
    }
    for (int x = 0; x < XS; ++x) {
        bool bit = ((bits >> x) & 1u) != 0;
        if (jitter.densityProb > 0.0 && uni(rng) < jitter.densityProb)
            bit = !bit;
        row[x] = bit ? 1 : 0;
    }
}

template <typename Value>
std::uint64_t bitsFromValue(Value value) {
    const std::uint64_t mask = (XS >= 64) ? ~0ull : ((std::uint64_t{1} << XS) - 1);
    return static_cast<std::uint64_t>(value) & mask;
}

Score scoreOutput(std::uint64_t output, std::uint64_t wanted) {
    const std::uint64_t falsePositiveBits = output & ~wanted;
    const std::uint64_t missedBits = wanted & ~output;
    const int falsePositives = __builtin_popcountll(falsePositiveBits);
    const int misses = __builtin_popcountll(missedBits);
    return {falsePositives, misses, XS - falsePositives - misses};
}

// tail{Thresh,Accu} point at the first element of the *last row* (index
// (YS-1)*XS) of a chunk's thresh_msk/accu_msk arrays.
double computeTiebreaker(const std::uint8_t* tailThresh, const std::uint8_t* tailAccu,
                        std::uint64_t wanted) {
    double tiebreaker = 0.0;
    for (int x = 0; x < XS; ++x) {
        const bool isWanted = ((wanted >> x) & 1u) != 0;
        tiebreaker += isWanted ? static_cast<double>(tailThresh[x]) - tailAccu[x] : tailAccu[x];
    }
    return tiebreaker;
}

struct ScoreAccumulator {
    int minFitness = XS;
    double fitnessSum = 0.0;
    double tiebreakerSum = 0.0;
    int count = 0;

    void add(const Score& score, double tiebreaker) {
        minFitness = std::min(minFitness, score.fitness);
        fitnessSum += score.fitness;
        tiebreakerSum += tiebreaker;
        ++count;
    }

    Evaluation finalize() const {
        Evaluation evaluation;
        if (count == 0) {
            evaluation.fitness = 0;
            evaluation.rankingFitness = 0.0;
            evaluation.tiebreaker = 1e9;
            return evaluation;
        }
        const double average = fitnessSum / count;
        evaluation.rankingFitness = MIN_FITNESS_WEIGHT * minFitness + AVERAGE_FITNESS_WEIGHT * average;
        evaluation.fitness = static_cast<int>(evaluation.rankingFitness);
        evaluation.tiebreaker = tiebreakerSum / count;
        return evaluation;
    }
};

Evaluation aggregateAcrossPatterns(const std::vector<Evaluation>& perPattern) {
    Evaluation aggregate;
    if (perPattern.empty())
        return aggregate;
    int minFitness = perPattern.front().fitness;
    double rankingSum = 0.0;
    double tiebreakerSum = 0.0;
    for (const Evaluation& evaluation : perPattern) {
        minFitness = std::min(minFitness, evaluation.fitness);
        rankingSum += evaluation.rankingFitness;
        tiebreakerSum += evaluation.tiebreaker;
    }
    aggregate.fitness = minFitness;
    aggregate.rankingFitness = rankingSum / perPattern.size();
    aggregate.tiebreaker = tiebreakerSum / perPattern.size();
    return aggregate;
}

bool isBetter(const Evaluation& candidate, const Evaluation& best) {
    if (candidate.fitness != best.fitness)
        return candidate.fitness > best.fitness;
    if (candidate.rankingFitness != best.rankingFitness)
        return candidate.rankingFitness > best.rankingFitness;
    return candidate.tiebreaker < best.tiebreaker;
}

struct StageResult {
    Genome genome;
    Evaluation evaluation;
    int generations;
};

// One shared (1+trials) hill-climb loop used by every stage. `eval` scores
// one candidate genome; higher-level stage code supplies the fitness logic.
template <typename EvalFn>
StageResult trainStage(const std::string& label, Genome initial, int generations,
                       double maskRate, double sensRate, int trials, int perturbCount,
                       double targetRankingFitness, int logEvery, std::mt19937& rng,
                       EvalFn eval) {
    Genome best = initial;
    Evaluation bestEvaluation = eval(best);
    int completedGenerations = 0;
    for (int generation = 0; generation < generations; ++generation) {
        completedGenerations = generation + 1;
        for (int trial = 0; trial < trials; ++trial) {
            Genome candidate = best;
            mutateGenome(candidate, maskRate, sensRate, perturbCount, rng);
            const Evaluation candidateEvaluation = eval(candidate);
            if (isBetter(candidateEvaluation, bestEvaluation)) {
                best = candidate;
                bestEvaluation = candidateEvaluation;
            }
        }
        if (logEvery > 0 && (generation % logEvery == 0 || generation == generations - 1)) {
            std::cout << "[" << label << "] generation " << generation
                      << " fitness=" << bestEvaluation.fitness
                      << " ranking=" << bestEvaluation.rankingFitness
                      << " tiebreaker=" << bestEvaluation.tiebreaker << "\n";
        }
        if (bestEvaluation.rankingFitness >= targetRankingFitness)
            break;
    }
    return {best, bestEvaluation, completedGenerations};
}

// ---------------------------------------------------------------------
// Training pattern generation. Kept deterministic (seeded) so runs are
// reproducible; not read from wanted_outputs.txt unless its bit width
// matches CURR_XS (documented assumption: the default 4x8 curriculum grid
// does not match the shipped 16-bit wanted_outputs.txt, so it falls back to
// a small built-in pattern set).
// ---------------------------------------------------------------------

struct TaskPattern {
    std::uint64_t input = 0;
    std::uint64_t output = 0;
};

std::vector<std::uint64_t> builtinKeyPatterns(int count) {
    std::vector<std::uint64_t> patterns;
    for (int bit = 0; bit < XS && static_cast<int>(patterns.size()) < count; ++bit)
        patterns.push_back(std::uint64_t{1} << bit);
    if (static_cast<int>(patterns.size()) < count)
        patterns.push_back((XS >= 64) ? ~0ull : ((std::uint64_t{1} << XS) - 1));
    if (static_cast<int>(patterns.size()) < count) {
        std::uint64_t alternating = 0;
        for (int x = 0; x < XS; x += 2) alternating |= (std::uint64_t{1} << x);
        patterns.push_back(alternating);
    }
    patterns.resize(std::min<std::size_t>(patterns.size(), count));
    return patterns;
}

bool binaryToBits(const std::string& text, std::uint64_t& value) {
    std::string trimmed;
    for (char character : text)
        if (character != ' ' && character != '\t' && character != '\r') trimmed += character;
    if (static_cast<int>(trimmed.size()) != XS)
        return false;
    value = 0;
    for (int index = 0; index < XS; ++index) {
        if (trimmed[index] != '0' && trimmed[index] != '1') return false;
        if (trimmed[index] == '1') value |= std::uint64_t{1} << index;
    }
    return true;
}

std::vector<TaskPattern> loadTaskPatterns(int count) {
    std::vector<TaskPattern> patterns;
    std::ifstream file("wanted_outputs.txt");
    if (file) {
        std::string line;
        while (std::getline(file, line) && static_cast<int>(patterns.size()) < count) {
            if (line.empty()) continue;
            const std::size_t separator = line.find(':');
            if (separator == std::string::npos) continue;
            TaskPattern pattern;
            if (binaryToBits(line.substr(0, separator), pattern.input) &&
                binaryToBits(line.substr(separator + 1), pattern.output)) {
                patterns.push_back(pattern);
            }
        }
    }
    if (!patterns.empty())
        return patterns;
    // Fallback: synthesize a small input/output task from the key patterns
    // (identity-like task: output mirrors input) so the output/cortex stages
    // still have something well-defined to train against at CURR_XS widths
    // that do not match the shipped wanted_outputs.txt.
    for (std::uint64_t key : builtinKeyPatterns(count))
        patterns.push_back({key, key});
    return patterns;
}

// ---------------------------------------------------------------------
// Stage 1: memory chunk (write / silence / probe reconstruction fitness).
// ---------------------------------------------------------------------

struct MemoryParams {
    int writeCycles = 4;
    int silenceCycles = 2;
    int probeCycles = std::max(YS + 4, 12);
    std::uint64_t probeStimulus = (XS >= 64) ? ~0ull : ((std::uint64_t{1} << XS) - 1);
    JitterConfig jitter;
};

Evaluation evaluateMemorySingle(Vchunk& mem, const Genome& genome, std::uint64_t keyPattern,
                                const MemoryParams& params, std::mt19937& rng) {
    loadGenome(mem, genome);
    resetTop(mem);
    for (int cycle = 0; cycle < params.writeCycles; ++cycle) {
        driveRow(mem.in, keyPattern, params.jitter, rng);
        clockOnce(mem);
    }
    for (int cycle = 0; cycle < params.silenceCycles; ++cycle) {
        driveRow(mem.in, 0, JitterConfig{}, rng);
        clockOnce(mem);
    }
    const int warmup = std::min(YS - 1, std::max(0, params.probeCycles - 1));
    ScoreAccumulator accumulator;
    for (int cycle = 0; cycle < params.probeCycles; ++cycle) {
        driveRow(mem.in, params.probeStimulus, JitterConfig{}, rng);
        clockOnce(mem);
        if (cycle < warmup) continue;
        const std::uint64_t output = bitsFromValue(mem.out);
        const Score score = scoreOutput(output, keyPattern);
        const double tiebreaker = computeTiebreaker(&mem.thresh_msk[(YS - 1) * XS],
                                                     &mem.accu_msk[(YS - 1) * XS], keyPattern);
        accumulator.add(score, tiebreaker);
    }
    return accumulator.finalize();
}

Evaluation evaluateMemory(Vchunk& mem, const Genome& genome, const std::vector<std::uint64_t>& keys,
                          const MemoryParams& params, std::mt19937& rng) {
    std::vector<Evaluation> perKey;
    perKey.reserve(keys.size());
    for (std::uint64_t key : keys)
        perKey.push_back(evaluateMemorySingle(mem, genome, key, params, rng));
    return aggregateAcrossPatterns(perKey);
}

// ---------------------------------------------------------------------
// Stage 2: input chunk, supervised by the frozen memory chunk's real
// write/silence/probe response (not an independently designed target).
// ---------------------------------------------------------------------

Evaluation evaluateInputChunkSingle(Vchunk& inTop, Vchunk& memTop, const Genome& inGenome,
                                    const Genome& frozenMemGenome, std::uint64_t keyPattern,
                                    const MemoryParams& params, const JitterConfig& linkJitter,
                                    std::mt19937& rng) {
    loadGenome(inTop, inGenome);
    loadGenome(memTop, frozenMemGenome);
    resetTop(inTop);
    resetTop(memTop);
    for (int cycle = 0; cycle < params.writeCycles; ++cycle) {
        driveRow(inTop.in, keyPattern, params.jitter, rng);
        clockOnce(inTop);
        const std::uint64_t encoded = bitsFromValue(inTop.out);
        driveRow(memTop.in, encoded, linkJitter, rng);
        clockOnce(memTop);
    }
    for (int cycle = 0; cycle < params.silenceCycles; ++cycle) {
        driveRow(inTop.in, 0, JitterConfig{}, rng);
        clockOnce(inTop);
        driveRow(memTop.in, 0, JitterConfig{}, rng);
        clockOnce(memTop);
    }
    const int warmup = std::min(YS - 1, std::max(0, params.probeCycles - 1));
    ScoreAccumulator accumulator;
    for (int cycle = 0; cycle < params.probeCycles; ++cycle) {
        driveRow(memTop.in, params.probeStimulus, JitterConfig{}, rng);
        clockOnce(memTop);
        if (cycle < warmup) continue;
        const std::uint64_t output = bitsFromValue(memTop.out);
        // Supervision target is the original key handed to the input chunk,
        // scored through the memory chunk's real (frozen) dynamics -- this
        // is the memory chunk's actual encoding, per the task constraints.
        const Score score = scoreOutput(output, keyPattern);
        const double tiebreaker = computeTiebreaker(&memTop.thresh_msk[(YS - 1) * XS],
                                                     &memTop.accu_msk[(YS - 1) * XS], keyPattern);
        accumulator.add(score, tiebreaker);
    }
    return accumulator.finalize();
}

Evaluation evaluateInputChunk(Vchunk& inTop, Vchunk& memTop, const Genome& inGenome,
                              const Genome& frozenMemGenome, const std::vector<std::uint64_t>& keys,
                              const MemoryParams& params, const JitterConfig& linkJitter,
                              std::mt19937& rng) {
    std::vector<Evaluation> perKey;
    perKey.reserve(keys.size());
    for (std::uint64_t key : keys)
        perKey.push_back(evaluateInputChunkSingle(inTop, memTop, inGenome, frozenMemGenome, key,
                                                   params, linkJitter, rng));
    return aggregateAcrossPatterns(perKey);
}

// ---------------------------------------------------------------------
// Stage 3: output chunk. Same graded per-bit + warm-up-exclusion scheme as
// FITNESS.md/FITNESS_FIXES.md, but the excluded warm-up window is scaled to
// the cumulative latency of the chunks that will eventually sit upstream of
// it (input -> memory -> cortex), even though it is trained standalone here.
// ---------------------------------------------------------------------

struct OutputParams {
    int cumulativeUpstreamHops = 3;  // input, memory, cortex
    int evalCycles = std::max(YS + 8, 16);
    JitterConfig jitter;
};

Evaluation evaluateOutputChunkSingle(Vchunk& outTop, const Genome& genome, const TaskPattern& pattern,
                                    const OutputParams& params, std::mt19937& rng) {
    loadGenome(outTop, genome);
    resetTop(outTop);
    const int ownWarmup = YS - 1;
    const int upstreamWarmup = params.cumulativeUpstreamHops * (YS - 1);
    const int totalWarmup = std::min(params.evalCycles - 1, std::max(0, ownWarmup + upstreamWarmup));
    ScoreAccumulator accumulator;
    for (int cycle = 0; cycle < params.evalCycles; ++cycle) {
        driveRow(outTop.in, pattern.input, params.jitter, rng);
        clockOnce(outTop);
        if (cycle < totalWarmup) continue;
        const std::uint64_t output = bitsFromValue(outTop.out);
        const Score score = scoreOutput(output, pattern.output);
        const double tiebreaker = computeTiebreaker(&outTop.thresh_msk[(YS - 1) * XS],
                                                     &outTop.accu_msk[(YS - 1) * XS], pattern.output);
        accumulator.add(score, tiebreaker);
    }
    return accumulator.finalize();
}

Evaluation evaluateOutputChunk(Vchunk& outTop, const Genome& genome,
                               const std::vector<TaskPattern>& patterns,
                               const OutputParams& params, std::mt19937& rng) {
    std::vector<Evaluation> perPattern;
    perPattern.reserve(patterns.size());
    for (const TaskPattern& pattern : patterns)
        perPattern.push_back(evaluateOutputChunkSingle(outTop, genome, pattern, params, rng));
    return aggregateAcrossPatterns(perPattern);
}

// ---------------------------------------------------------------------
// Stage 4: cortex chunk. Reads the frozen memory chunk, computes a wanted
// output, and is scored by feeding that output into the frozen output chunk.
//
// Decision: a single cortex chunk, not a split read/compute pair. A split
// would need a hand-designed intermediate target for the "read" chunk (e.g.
// "reproduce memory's output verbatim") to keep its fitness well-defined,
// which adds exactly the kind of hand-designed intermediate signal the task
// asks to avoid at the input/memory boundary, and there is no equivalent
// natural supervision signal for a read-only intermediate here. One cortex
// chunk keeps a single, unambiguous, end-task-grounded fitness function at
// the cost of a larger search space for that one chunk.
// ---------------------------------------------------------------------

struct CortexParams {
    int writeCycles = 4;
    int silenceCycles = 2;
    int probeCycles = std::max(YS + 4, 12);
    std::uint64_t probeStimulus = (XS >= 64) ? ~0ull : ((std::uint64_t{1} << XS) - 1);
    JitterConfig jitter;
    JitterConfig linkJitter;
};

Evaluation evaluateCortexSingle(Vchunk& memTop, Vchunk& cortexTop, Vchunk& outTop,
                                const Genome& frozenMem, const Genome& cortexGenome,
                                const Genome& frozenOut, const TaskPattern& pattern,
                                const CortexParams& params, std::mt19937& rng) {
    loadGenome(memTop, frozenMem);
    loadGenome(cortexTop, cortexGenome);
    loadGenome(outTop, frozenOut);
    resetTop(memTop);
    resetTop(cortexTop);
    resetTop(outTop);

    for (int cycle = 0; cycle < params.writeCycles; ++cycle) {
        driveRow(memTop.in, pattern.input, params.jitter, rng);
        clockOnce(memTop);
    }
    for (int cycle = 0; cycle < params.silenceCycles; ++cycle) {
        driveRow(memTop.in, 0, JitterConfig{}, rng);
        clockOnce(memTop);
    }

    // Cumulative warm-up: memory's own settle time plus cortex's own settle
    // time plus output's own settle time, since all three now sit in series
    // during the probe/compute/emit window.
    const int totalWarmup = std::min(params.probeCycles - 1, 3 * (YS - 1));
    ScoreAccumulator accumulator;
    for (int cycle = 0; cycle < params.probeCycles; ++cycle) {
        driveRow(memTop.in, params.probeStimulus, JitterConfig{}, rng);
        clockOnce(memTop);
        const std::uint64_t memOut = bitsFromValue(memTop.out);
        driveRow(cortexTop.in, memOut, params.linkJitter, rng);
        clockOnce(cortexTop);
        const std::uint64_t cortexOut = bitsFromValue(cortexTop.out);
        driveRow(outTop.in, cortexOut, params.linkJitter, rng);
        clockOnce(outTop);

        if (cycle < totalWarmup) continue;
        const std::uint64_t finalOut = bitsFromValue(outTop.out);
        const Score score = scoreOutput(finalOut, pattern.output);
        const double tiebreaker = computeTiebreaker(&outTop.thresh_msk[(YS - 1) * XS],
                                                     &outTop.accu_msk[(YS - 1) * XS], pattern.output);
        accumulator.add(score, tiebreaker);
    }
    return accumulator.finalize();
}

Evaluation evaluateCortex(Vchunk& memTop, Vchunk& cortexTop, Vchunk& outTop, const Genome& frozenMem,
                         const Genome& cortexGenome, const Genome& frozenOut,
                         const std::vector<TaskPattern>& patterns, const CortexParams& params,
                         std::mt19937& rng) {
    std::vector<Evaluation> perPattern;
    perPattern.reserve(patterns.size());
    for (const TaskPattern& pattern : patterns)
        perPattern.push_back(evaluateCortexSingle(memTop, cortexTop, outTop, frozenMem, cortexGenome,
                                                   frozenOut, pattern, params, rng));
    return aggregateAcrossPatterns(perPattern);
}

// ---------------------------------------------------------------------
// Stage 5: joint fine-tune of all four genomes, wired through pipeline.v
// (i.e. through c_link), with a low mutation rate / small perturbations.
// ---------------------------------------------------------------------

struct FourGenomes {
    Genome inChunk;
    Genome memChunk;
    Genome cortexChunk;
    Genome outChunk;
};

template <typename Arr>
void loadPipelineGenome(Arr& maskArr, Arr& sensArr, const Genome& genome) {
    for (int index = 0; index < N; ++index) {
        maskArr[index] = genome.masks[index];
        sensArr[index] = genome.sensitivities[index];
    }
}

void loadPipelineGenomes(Vpipeline& pipe, const FourGenomes& genomes) {
    loadPipelineGenome(pipe.in_mask, pipe.in_sens, genomes.inChunk);
    loadPipelineGenome(pipe.mem_mask, pipe.mem_sens, genomes.memChunk);
    loadPipelineGenome(pipe.cortex_mask, pipe.cortex_sens, genomes.cortexChunk);
    loadPipelineGenome(pipe.out_mask, pipe.out_sens, genomes.outChunk);
}

struct PipelineParams {
    int writeCycles = 4;
    int silenceCycles = 2;
    int probeCycles = std::max(YS + 4, 12);
    JitterConfig jitter;
};

Evaluation evaluatePipelineSingle(Vpipeline& pipe, const FourGenomes& genomes,
                                 const TaskPattern& pattern, const PipelineParams& params,
                                 std::mt19937& rng) {
    loadPipelineGenomes(pipe, genomes);
    resetTop(pipe);

    pipe.probe_en = 0;
    pipe.link_mem_to_cortex_en = 0;
    pipe.link_cortex_to_out_en = 0;

    pipe.link_in_to_mem_en = 1;
    for (int cycle = 0; cycle < params.writeCycles; ++cycle) {
        driveRow(pipe.in_chunk_drive, pattern.input, params.jitter, rng);
        for (int x = 0; x < XS; ++x) pipe.mem_probe[x] = 0;
        clockOnce(pipe);
    }

    pipe.link_in_to_mem_en = 0;
    for (int cycle = 0; cycle < params.silenceCycles; ++cycle) {
        driveRow(pipe.in_chunk_drive, 0, JitterConfig{}, rng);
        clockOnce(pipe);
    }

    pipe.probe_en = 1;
    pipe.link_mem_to_cortex_en = 1;
    pipe.link_cortex_to_out_en = 1;
    for (int x = 0; x < XS; ++x)
        pipe.mem_probe[x] = 1;

    const int totalWarmup = std::min(params.probeCycles - 1, 3 * (YS - 1));
    ScoreAccumulator accumulator;
    for (int cycle = 0; cycle < params.probeCycles; ++cycle) {
        clockOnce(pipe);
        if (cycle < totalWarmup) continue;
        const std::uint64_t output = bitsFromValue(pipe.final_out);
        const Score score = scoreOutput(output, pattern.output);
        std::array<std::uint8_t, XS> tailThresh{};
        std::array<std::uint8_t, XS> tailAccu{};
        for (int x = 0; x < XS; ++x) {
            tailThresh[x] = static_cast<std::uint8_t>((pipe.out_thresh_tail >> (x * 3)) & 0x7u);
            tailAccu[x] = static_cast<std::uint8_t>((pipe.out_accu_tail >> (x * 3)) & 0x7u);
        }
        const double tiebreaker = computeTiebreaker(tailThresh.data(), tailAccu.data(), pattern.output);
        accumulator.add(score, tiebreaker);
    }
    return accumulator.finalize();
}

Evaluation evaluatePipeline(Vpipeline& pipe, const FourGenomes& genomes,
                           const std::vector<TaskPattern>& patterns, const PipelineParams& params,
                           std::mt19937& rng) {
    std::vector<Evaluation> perPattern;
    perPattern.reserve(patterns.size());
    for (const TaskPattern& pattern : patterns)
        perPattern.push_back(evaluatePipelineSingle(pipe, genomes, pattern, params, rng));
    return aggregateAcrossPatterns(perPattern);
}

// ---------------------------------------------------------------------
// CLI configuration.
// ---------------------------------------------------------------------

struct Config {
    std::string stage = "all";
    std::string genomeDir = "genomes";
    int generations = 40;
    int trials = 12;
    int patternCount = 3;
    int initialSilence = 2;
    int maxSilence = 16;
    int jointGenerations = 10;
    int jointTrials = 6;
    int jointPerturb = 3;
    double maskRate = 0.04;
    double sensRate = 0.06;
    double jitterDensity = 0.05;
    double jitterDrop = 0.05;
    unsigned seed = 12345u;
    int logEvery = 10;
};

Config parseArgs(int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        std::string arg = argv[index];
        if (arg.rfind("--", 0) != 0) continue;
        arg = arg.substr(2);
        const std::size_t equalsPos = arg.find('=');
        if (equalsPos == std::string::npos) continue;
        const std::string key = arg.substr(0, equalsPos);
        const std::string value = arg.substr(equalsPos + 1);
        if (key == "stage") config.stage = value;
        else if (key == "genome-dir") config.genomeDir = value;
        else if (key == "generations") config.generations = std::stoi(value);
        else if (key == "trials") config.trials = std::stoi(value);
        else if (key == "pattern-count") config.patternCount = std::stoi(value);
        else if (key == "initial-silence") config.initialSilence = std::stoi(value);
        else if (key == "max-silence") config.maxSilence = std::stoi(value);
        else if (key == "joint-generations") config.jointGenerations = std::stoi(value);
        else if (key == "joint-trials") config.jointTrials = std::stoi(value);
        else if (key == "joint-perturb") config.jointPerturb = std::stoi(value);
        else if (key == "mask-rate") config.maskRate = std::stod(value);
        else if (key == "sens-rate") config.sensRate = std::stod(value);
        else if (key == "jitter-density") config.jitterDensity = std::stod(value);
        else if (key == "jitter-drop") config.jitterDrop = std::stod(value);
        else if (key == "seed") config.seed = static_cast<unsigned>(std::stoul(value));
        else if (key == "log-every") config.logEvery = std::stoi(value);
    }
    return config;
}

void ensureDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const Config config = parseArgs(argc, argv);
    ensureDirectory(config.genomeDir);
    std::mt19937 rng(config.seed);

    VerilatedContext context;
    Vchunk memTop{&context};
    Vchunk inTop{&context};
    Vchunk cortexTop{&context};
    Vchunk outTop{&context};
    Vpipeline pipeTop{&context};

    const std::string memPath = config.genomeDir + "/memory.genome";
    const std::string inPath = config.genomeDir + "/input.genome";
    const std::string outPath = config.genomeDir + "/output.genome";
    const std::string cortexPath = config.genomeDir + "/cortex.genome";
    const std::string reportPath = config.genomeDir + "/curriculum_report.txt";

    std::ostringstream report;
    report << "Curriculum training report (XS=" << XS << ", YS=" << YS << ")\n";

    const std::vector<std::uint64_t> keyPatterns = builtinKeyPatterns(config.patternCount);
    const std::vector<TaskPattern> taskPatterns = loadTaskPatterns(config.patternCount);
    const JitterConfig jitter{config.jitterDensity, config.jitterDrop};

    const bool runAll = config.stage == "all";

    // ---------------- Stage 1: memory ----------------
    Genome memGenome;
    if (runAll || config.stage == "memory") {
        std::cout << "=== Stage 1: memory chunk (write/silence/probe) ===\n";
        if (!loadGenomeFile(memPath, memGenome))
            memGenome = randomGenome(rng);

        MemoryParams memParams;
        memParams.jitter = jitter;
        int silence = config.initialSilence;
        int totalGenerations = 0;
        double lastRankingFitness = 0.0;
        while (true) {
            memParams.silenceCycles = silence;
            const StageResult result = trainStage(
                "memory silence=" + std::to_string(silence), memGenome, config.generations,
                config.maskRate, config.sensRate, config.trials, 0,
                0.9 * XS, config.logEvery, rng,
                [&](const Genome& genome) { return evaluateMemory(memTop, genome, keyPatterns, memParams, rng); });
            memGenome = result.genome;
            lastRankingFitness = result.evaluation.rankingFitness;
            totalGenerations += result.generations;
            report << "Stage 1 (memory) silence=" << silence << ": fitness="
                   << result.evaluation.fitness << " ranking=" << result.evaluation.rankingFitness
                   << " generations=" << result.generations << "\n";
            if (lastRankingFitness < 0.9 * XS || silence >= config.maxSilence)
                break;
            silence = std::min(config.maxSilence, silence * 2);
        }
        saveGenome(memPath, memGenome);
        std::cout << "Memory chunk saved to " << memPath << " (silence reached=" << silence
                  << ", generations=" << totalGenerations << ")\n";
    } else {
        loadGenomeFile(memPath, memGenome);
    }

    // ---------------- Stage 2: input ----------------
    Genome inGenome;
    if (runAll || config.stage == "input") {
        std::cout << "=== Stage 2: input chunk (frozen memory in the loop) ===\n";
        if (!loadGenomeFile(inPath, inGenome))
            inGenome = randomGenome(rng);
        MemoryParams memParams;  // reuse memory's write/silence/probe cycle counts
        memParams.jitter = jitter;
        memParams.silenceCycles = config.maxSilence;  // train against the hardest retention gap reached
        const JitterConfig linkJitter{config.jitterDensity * 0.5, config.jitterDrop * 0.5};
        const StageResult result = trainStage(
            "input", inGenome, config.generations, config.maskRate, config.sensRate, config.trials, 0,
            0.9 * XS, config.logEvery, rng,
            [&](const Genome& genome) {
                return evaluateInputChunk(inTop, memTop, genome, memGenome, keyPatterns, memParams,
                                          linkJitter, rng);
            });
        inGenome = result.genome;
        report << "Stage 2 (input): fitness=" << result.evaluation.fitness
               << " ranking=" << result.evaluation.rankingFitness
               << " generations=" << result.generations << "\n";
        saveGenome(inPath, inGenome);
        std::cout << "Input chunk saved to " << inPath << "\n";
    } else {
        loadGenomeFile(inPath, inGenome);
    }

    // ---------------- Stage 3: output ----------------
    Genome outGenome;
    if (runAll || config.stage == "output") {
        std::cout << "=== Stage 3: output chunk (graded, warm-up-scaled) ===\n";
        if (!loadGenomeFile(outPath, outGenome))
            outGenome = randomGenome(rng);
        OutputParams outputParams;
        outputParams.jitter = jitter;
        const StageResult result = trainStage(
            "output", outGenome, config.generations, config.maskRate, config.sensRate, config.trials, 0,
            0.9 * XS, config.logEvery, rng,
            [&](const Genome& genome) {
                return evaluateOutputChunk(outTop, genome, taskPatterns, outputParams, rng);
            });
        outGenome = result.genome;
        report << "Stage 3 (output): fitness=" << result.evaluation.fitness
               << " ranking=" << result.evaluation.rankingFitness
               << " generations=" << result.generations << "\n";
        saveGenome(outPath, outGenome);
        std::cout << "Output chunk saved to " << outPath << "\n";
    } else {
        loadGenomeFile(outPath, outGenome);
    }

    // ---------------- Stage 4: cortex ----------------
    Genome cortexGenome;
    if (runAll || config.stage == "cortex") {
        std::cout << "=== Stage 4: cortex chunk (reads frozen memory, drives frozen output) ===\n";
        if (!loadGenomeFile(cortexPath, cortexGenome))
            cortexGenome = randomGenome(rng);
        CortexParams cortexParams;
        cortexParams.jitter = jitter;
        cortexParams.linkJitter = JitterConfig{config.jitterDensity * 0.5, config.jitterDrop * 0.5};
        const StageResult result = trainStage(
            "cortex", cortexGenome, config.generations, config.maskRate, config.sensRate, config.trials,
            0, 0.9 * XS, config.logEvery, rng,
            [&](const Genome& genome) {
                return evaluateCortex(memTop, cortexTop, outTop, memGenome, genome, outGenome,
                                      taskPatterns, cortexParams, rng);
            });
        cortexGenome = result.genome;
        report << "Stage 4 (cortex): fitness=" << result.evaluation.fitness
               << " ranking=" << result.evaluation.rankingFitness
               << " generations=" << result.generations << "\n";
        saveGenome(cortexPath, cortexGenome);
        std::cout << "Cortex chunk saved to " << cortexPath << "\n";
    } else {
        loadGenomeFile(cortexPath, cortexGenome);
    }

    // ---------------- Stage 5: joint fine-tune via C_LINK ----------------
    if (runAll || config.stage == "joint") {
        std::cout << "=== Stage 5: joint fine-tune (wired through pipeline.v / c_link) ===\n";
        FourGenomes genomes{inGenome, memGenome, cortexGenome, outGenome};
        PipelineParams pipelineParams;
        pipelineParams.jitter = jitter;

        const Evaluation before = evaluatePipeline(pipeTop, genomes, taskPatterns, pipelineParams, rng);
        std::cout << "Joint pipeline fitness before fine-tune: fitness=" << before.fitness
                  << " ranking=" << before.rankingFitness << "\n";

        auto packGenome = [](const FourGenomes& g) {
            // Flatten into a single Genome-like container is unnecessary; the
            // hill-climb helper below is specialized inline for four genomes.
            return g;
        };
        (void)packGenome;

        FourGenomes best = genomes;
        Evaluation bestEvaluation = before;
        for (int generation = 0; generation < config.jointGenerations; ++generation) {
            for (int trial = 0; trial < config.jointTrials; ++trial) {
                FourGenomes candidate = best;
                mutateGenome(candidate.inChunk, 0, 0, config.jointPerturb, rng);
                mutateGenome(candidate.memChunk, 0, 0, config.jointPerturb, rng);
                mutateGenome(candidate.cortexChunk, 0, 0, config.jointPerturb, rng);
                mutateGenome(candidate.outChunk, 0, 0, config.jointPerturb, rng);
                const Evaluation candidateEvaluation =
                    evaluatePipeline(pipeTop, candidate, taskPatterns, pipelineParams, rng);
                if (isBetter(candidateEvaluation, bestEvaluation)) {
                    best = candidate;
                    bestEvaluation = candidateEvaluation;
                }
            }
            std::cout << "[joint] generation " << generation << " fitness=" << bestEvaluation.fitness
                      << " ranking=" << bestEvaluation.rankingFitness << "\n";
        }

        std::cout << "Joint pipeline fitness after fine-tune: fitness=" << bestEvaluation.fitness
                  << " ranking=" << bestEvaluation.rankingFitness << "\n";
        report << "Stage 5 (joint) before: fitness=" << before.fitness
               << " ranking=" << before.rankingFitness << "\n";
        report << "Stage 5 (joint) after: fitness=" << bestEvaluation.fitness
               << " ranking=" << bestEvaluation.rankingFitness
               << " generations=" << config.jointGenerations << "\n";

        saveGenome(inPath, best.inChunk);
        saveGenome(memPath, best.memChunk);
        saveGenome(cortexPath, best.cortexChunk);
        saveGenome(outPath, best.outChunk);
    }

    std::ofstream reportFile(reportPath);
    reportFile << report.str();
    std::cout << "\n" << report.str();
    std::cout << "Report written to " << reportPath << "\n";
    return 0;
}
