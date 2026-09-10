#include "cuda_fitness.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <limits>

#ifndef CUDA_POPULATION_SIZE
#define CUDA_POPULATION_SIZE 65536
#endif
#ifndef CUDA_MUTATION_RATE
#define CUDA_MUTATION_RATE 0.04
#endif
#ifndef CUDA_ELITE_PERCENT
#define CUDA_ELITE_PERCENT 20
#endif
constexpr std::size_t POPULATION_SIZE = CUDA_POPULATION_SIZE;
constexpr std::size_t ELITE_COUNT = std::max<std::size_t>(
    1, POPULATION_SIZE * CUDA_ELITE_PERCENT / 100);
constexpr double MUTATION_RATE = CUDA_MUTATION_RATE;
constexpr std::size_t PLATEAU_GENERATIONS = 150;
bool verbose = false;

struct Target {
    std::uint64_t input;
    std::uint64_t output;
    std::string inputBits;
    std::string outputBits;
};

struct RankedGenome {
    CudaGenome genome{};
    int fitness = 0;
    float rankingFitness = 0.0f;
    float tiebreaker = 0.0f;
    std::uint64_t output = 0;
};

std::string stripSpaces(const std::string& text) {
    std::string result;
    for (char character : text) {
        if (character != ' ' && character != '\t' && character != '\r')
            result += character;
    }
    return result;
}

std::string gridBits(std::uint64_t value) {
    std::string result;
    result.reserve(CUDA_XS);
    for (int index = 0; index < CUDA_XS; ++index)
        result += ((value >> index) & 1) ? '1' : '0';
    return result;
}

bool parseBits(const std::string& text, std::uint64_t& value) {
    const std::string bits = stripSpaces(text);
    if (bits.size() != CUDA_XS)
        return false;
    for (char bit : bits) {
        if (bit != '0' && bit != '1')
            return false;
    }
    value = 0;
    for (int index = 0; index < CUDA_XS; ++index)
        if (bits[index] == '1')
            value |= std::uint64_t{1} << index;
    return true;
}

bool loadTargets(const std::string& path, std::vector<Target>& targets) {
    std::ifstream file(path);
    if (!file)
        return false;

    std::string line;
    std::uint64_t previousInput = 0;
    bool haveInput = false;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (line.empty())
            continue;

        const std::size_t separator = line.find(':');
        const std::string inputText = separator == std::string::npos
            ? ""
            : line.substr(0, separator);
        const std::string outputText = separator == std::string::npos
            ? line
            : line.substr(separator + 1);
        Target target{0, 0, "", stripSpaces(outputText)};

        if (separator != std::string::npos) {
            if (!parseBits(inputText, target.input)) {
                std::cerr << "Invalid input on line " << lineNumber
                          << ": expected " << CUDA_XS << " bits\n";
                return false;
            }
            previousInput = target.input;
            haveInput = true;
            target.inputBits = stripSpaces(inputText);
        } else if (haveInput) {
            target.input = previousInput;
            target.inputBits = gridBits(previousInput);
        } else {
            return false;
        }

        if (!parseBits(outputText, target.output)) {
            std::cerr << "Invalid output on line " << lineNumber
                      << ": expected " << CUDA_XS << " bits\n";
            return false;
        }
        targets.push_back(target);
    }
    return !targets.empty();
}

CudaGenome randomGenome(std::mt19937& random) {
    CudaGenome genome{};
    std::uniform_int_distribution<int> mask(0, 15);
    std::uniform_int_distribution<int> sensitivity(0, 3);
    for (int index = 0; index < CUDA_NEURON_COUNT; ++index) {
        genome.masks[index] = static_cast<std::uint8_t>(mask(random));
        genome.sensitivities[index] = static_cast<std::uint8_t>(sensitivity(random));
    }
    for (auto& input : genome.inputs)
        input = 15;
    return genome;
}

CudaGenome directSouthGenome() {
    CudaGenome genome{};
    for (int index = 0; index < CUDA_NEURON_COUNT; ++index) {
        genome.masks[index] = 0b0100;
        genome.sensitivities[index] = 0;
    }
    for (auto& input : genome.inputs)
        input = 15;
    return genome;
}

void mutate(CudaGenome& genome, std::mt19937& random) {
    std::bernoulli_distribution mutateMask(MUTATION_RATE);
    std::bernoulli_distribution mutateSensitivity(std::min(1.0, MUTATION_RATE * 1.5));
    std::uniform_int_distribution<int> maskBit(0, 3);
    std::uniform_int_distribution<int> sensitivityBit(0, 1);
    for (auto& value : genome.masks) {
        if (mutateMask(random))
            value ^= static_cast<std::uint8_t>(1u << maskBit(random));
    }
    for (auto& value : genome.sensitivities) {
        if (mutateSensitivity(random))
            value ^= static_cast<std::uint8_t>(1u << sensitivityBit(random));
    }
}

void setInput(CudaGenome& genome, std::uint64_t input) {
    for (int index = 0; index < CUDA_XS; ++index)
        genome.inputs[index] = ((input >> index) & 1) ? 15 : 0;
}

CudaGenome crossover(const CudaGenome& first, const CudaGenome& second,
                     std::mt19937& random) {
    CudaGenome child{};
    std::bernoulli_distribution chooseFirst(0.5);
    for (int index = 0; index < CUDA_NEURON_COUNT; ++index) {
        child.masks[index] = chooseFirst(random) ? first.masks[index] : second.masks[index];
        child.sensitivities[index] = chooseFirst(random)
            ? first.sensitivities[index]
            : second.sensitivities[index];
    }
    for (int index = 0; index < CUDA_XS; ++index)
        child.inputs[index] = first.inputs[index];
    return child;
}

std::vector<RankedGenome> rankPopulation(const std::vector<CudaGenome>& population,
                                         std::uint64_t wantedOutput) {
    std::vector<CudaEvaluation> evaluations;
    evaluatePopulationCuda(population, wantedOutput, evaluations);

    std::vector<RankedGenome> ranked(population.size());
    for (std::size_t index = 0; index < population.size(); ++index)
        ranked[index] = {population[index], evaluations[index].fitness,
                         evaluations[index].rankingFitness,
                         evaluations[index].tiebreaker, evaluations[index].output};
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedGenome& first, const RankedGenome& second) {
                  if (first.fitness != second.fitness)
                      return first.fitness > second.fitness;
                  if (first.rankingFitness != second.rankingFitness)
                      return first.rankingFitness > second.rankingFitness;
                  return first.tiebreaker < second.tiebreaker;
              });
    return ranked;
}

std::vector<CudaGenome> evolve(std::vector<CudaGenome> population,
                               std::uint64_t input,
                               std::uint64_t wantedOutput,
                               std::mt19937& random,
                               std::size_t targetIndex) {
    std::uniform_int_distribution<std::size_t> parent(0, ELITE_COUNT - 1);
    std::size_t generation = 0;
    std::size_t stagnantGenerations = 0;
    float bestRankingFitness = -1.0f;
    while (true) {
        const auto ranked = rankPopulation(population, wantedOutput);
        const std::uint64_t error = ranked.front().output ^ wantedOutput;
        std::cout << (targetIndex + 1) << " : " << generation
                  << " | " << gridBits(error)
                  << " : " << gridBits(input) << " : " << gridBits(wantedOutput) << "\n";
        if (ranked.front().fitness == CUDA_XS) {
            population.clear();
            for (const auto& item : ranked)
                population.push_back(item.genome);
            return population;
        }

        if (ranked.front().rankingFitness > bestRankingFitness) {
            bestRankingFitness = ranked.front().rankingFitness;
            stagnantGenerations = 0;
        } else {
            ++stagnantGenerations;
        }

        if (stagnantGenerations >= PLATEAU_GENERATIONS) {
            population.clear();
            population.push_back(ranked.front().genome);
            while (population.size() < POPULATION_SIZE) {
                CudaGenome genome = randomGenome(random);
                setInput(genome, input);
                population.push_back(genome);
            }
            stagnantGenerations = 0;
            continue;
        }

        population.clear();
        for (std::size_t index = 0; index < ELITE_COUNT; ++index)
            population.push_back(ranked[index].genome);
        while (population.size() < POPULATION_SIZE) {
            const auto& firstParent = ranked[parent(random)].genome;
            const auto& secondParent = ranked[parent(random)].genome;
            CudaGenome child = crossover(firstParent, secondParent, random);
            mutate(child, random);
            population.push_back(child);
        }
        ++generation;
    }
}

std::vector<CudaGenome> evolveCuda(CudaTrainingContext* context,
                                   std::uint64_t input,
                                   std::uint64_t wantedOutput,
                                   std::size_t targetIndex,
                                   std::size_t maxGenerations) {
    std::size_t generation = 0;
    std::size_t stagnantGenerations = 0;
    float bestRankingFitness = -1.0f;
    while (true) {
        const CudaEvaluation best = stepCudaTraining(context, wantedOutput);
        const std::uint64_t error = best.output ^ wantedOutput;
        std::cout << (targetIndex + 1) << " : " << generation
                  << " | " << gridBits(error)
                  << " : " << gridBits(input)
                  << " : " << gridBits(wantedOutput) << "\n";
        if (verbose) {
            CudaGenome bestGenome{};
            CudaTrainingDiagnostics diagnostics{};
            downloadCudaBestGenome(context, bestGenome);
            downloadCudaTrainingDiagnostics(context, wantedOutput, bestGenome,
                                            diagnostics);
            std::cerr << "[debug] target=" << targetIndex + 1
                      << " generation=" << generation
                      << " fitness=" << best.fitness << "/" << CUDA_XS
                      << " ranking=" << best.rankingFitness
                      << " tiebreaker=" << best.tiebreaker
                      << " output=" << gridBits(best.output)
                      << " input=" << gridBits(input)
                      << " wanted=" << gridBits(wantedOutput)
                      << " error=" << gridBits(error)
                      << " exact-output=" << diagnostics.exactWantedOutputCount
                      << " nonzero-output=" << diagnostics.nonzeroOutputCount
                      << " unique-outputs=" << diagnostics.uniqueOutputCount
                      << " exact-best=" << diagnostics.exactBestGenomeCount
                      << " unique=" << diagnostics.uniqueGenomeHashCount
                      << " avg-mask-bits=" << diagnostics.averageMaskBits
                      << " avg-sensitivity=" << diagnostics.averageSensitivity
                      << " fitness-counts=";
            for (int fitness = CUDA_XS; fitness >= 0; --fitness)
                if (diagnostics.fitnessCounts[fitness] != 0)
                    std::cerr << fitness << ":"
                              << diagnostics.fitnessCounts[fitness] << ",";
            std::cerr << " output-ones=";
            for (int bit = 0; bit < CUDA_XS; ++bit)
                std::cerr << diagnostics.outputBitCounts[bit]
                          << (bit + 1 == CUDA_XS ? "" : ",");
            std::cerr << "\n";
        }
        if (best.fitness == CUDA_XS) {
            std::vector<CudaGenome> population;
            downloadCudaPopulation(context, population);
            return population;
        }
        if (generation + 1 >= maxGenerations) {
            std::cerr << "Reached generation limit " << maxGenerations
                      << " for target " << targetIndex + 1 << "\n";
            std::vector<CudaGenome> population;
            downloadCudaPopulation(context, population);
            return population;
        }
        if (best.rankingFitness > bestRankingFitness) {
            bestRankingFitness = best.rankingFitness;
            stagnantGenerations = 0;
        } else {
            ++stagnantGenerations;
        }
        if (stagnantGenerations >= PLATEAU_GENERATIONS) {
            CudaGenome bestGenome{};
            downloadCudaBestGenome(context, bestGenome);
            reseedCudaTraining(context, bestGenome, input);
            stagnantGenerations = 0;
        }
        ++generation;
    }
}

int main(int argc, char** argv) {
    std::string targetPath = "wanted_outputs.txt";
    std::size_t maxGenerations = std::numeric_limits<std::size_t>::max();
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "-v" || option == "--verbose") {
            verbose = true;
        } else if (option == "--max-generations" && argument + 1 < argc) {
            maxGenerations = std::stoull(argv[++argument]);
        } else if (option.rfind("--max-generations=", 0) == 0) {
            maxGenerations = std::stoull(option.substr(17));
        } else if (!option.empty() && option[0] != '-') {
            targetPath = option;
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [-v|--verbose] [--max-generations N] [target-file]\n";
            return 1;
        }
    }
    std::vector<Target> targets;
    if (!loadTargets(targetPath, targets)) {
        std::cerr << "Could not load targets from " << targetPath << "\n";
        return 1;
    }

    std::mt19937 random(0xA17F2026u);
    std::vector<CudaGenome> population(POPULATION_SIZE);
    for (auto& genome : population)
        genome = randomGenome(random);
    // Preserve a working propagation scaffold so arithmetic evolution starts
    // with signal-carrying genomes rather than an all-zero-output population.
    population[0] = directSouthGenome();
    CudaTrainingContext* context = createCudaTrainingContext(population);

    for (std::size_t index = 0; index < targets.size(); ++index) {
        setCudaTrainingInput(context, targets[index].input);
        std::cout << "Target " << index + 1 << "/" << targets.size()
                  << " | input: " << targets[index].inputBits
                  << " | wanted output: " << targets[index].outputBits << "\n";
        population = evolveCuda(context, targets[index].input,
                                 targets[index].output, index, maxGenerations);
    }

    destroyCudaTrainingContext(context);
    std::cout << "Completed CUDA training for " << targets.size() << " targets.\n";
    return 0;
}
