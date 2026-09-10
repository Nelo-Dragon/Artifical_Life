#include "cuda_fitness.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef CUDA_POPULATION_SIZE
#define CUDA_POPULATION_SIZE 32768
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

bool parseBits(const std::string& text, std::uint64_t& value) {
    const std::string bits = stripSpaces(text);
    if (bits.size() != CUDA_XS)
        return false;
    for (char bit : bits) {
        if (bit != '0' && bit != '1')
            return false;
    }
    value = std::bitset<CUDA_XS>(bits).to_ullong();
    return true;
}

std::uint64_t reverseBits(std::uint64_t value) {
    std::uint64_t reversed = 0;
    for (int index = 0; index < CUDA_XS; ++index)
        reversed |= ((value >> index) & 1) << (CUDA_XS - 1 - index);
    return reversed;
}

std::string gridBits(std::uint64_t value) {
    std::string result;
    result.reserve(CUDA_XS);
    for (int index = 0; index < CUDA_XS; ++index)
        result += ((value >> index) & 1) ? '1' : '0';
    return result;
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
            target.inputBits = std::bitset<CUDA_XS>(previousInput).to_string();
        } else {
            return false;
        }

        if (!parseBits(outputText, target.output)) {
            std::cerr << "Invalid output on line " << lineNumber
                      << ": expected " << CUDA_XS << " bits\n";
            return false;
        }
        // Map the file's leftmost output bit to physical grid column 0,
        // matching the input mapping and the CPU trainer.
        target.output = reverseBits(target.output);
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
        genome.inputs[index] = ((input >> (CUDA_XS - 1 - index)) & 1) ? 15 : 0;
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
                  << " | " << gridBits(error) << "\n";
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

int main(int argc, char** argv) {
    const std::string targetPath = argc > 1 ? argv[1] : "wanted_outputs.txt";
    std::vector<Target> targets;
    if (!loadTargets(targetPath, targets)) {
        std::cerr << "Could not load targets from " << targetPath << "\n";
        return 1;
    }

    std::mt19937 random(0xA17F2026u);
    std::vector<CudaGenome> population(POPULATION_SIZE);
    for (auto& genome : population)
        genome = randomGenome(random);

    for (std::size_t index = 0; index < targets.size(); ++index) {
        for (auto& genome : population)
            setInput(genome, targets[index].input);
        std::cout << "Target " << index + 1 << "/" << targets.size()
                  << " | input: " << targets[index].inputBits
                  << " | wanted output: " << targets[index].outputBits << "\n";
        population = evolve(std::move(population), targets[index].input,
                            targets[index].output, random, index);
    }

    std::cout << "Completed CUDA training for " << targets.size() << " targets.\n";
    return 0;
}
