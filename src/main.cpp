#include <array>
#include <algorithm>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <atomic>
#include <vector>

#include "Vchunk.h"
#include "verilated.h"

constexpr int XS = 16;
constexpr int YS = 16;
constexpr int NEURON_COUNT = XS * YS;
constexpr std::size_t POPULATION_SIZE = 256;
constexpr std::size_t ELITE_COUNT = POPULATION_SIZE / 5;
constexpr int SIMULATION_CYCLES = 32;
constexpr double MUTATION_RATE = 0.04;

struct ChunkGenome {
    std::array<std::uint8_t, NEURON_COUNT> masks{};
    std::array<std::uint8_t, NEURON_COUNT> sensitivities{};
    std::array<std::uint8_t, XS> inputs{};
};

struct EvaluatedGenome {
    ChunkGenome genome;
    int fitness = 0;
    std::uint64_t output = 0;
};

struct WantedPattern {
    std::uint64_t input;
    std::uint64_t output;
    std::string inputBits;
    std::string outputBits;
};

std::string binaryPart(const std::string& text) {
    std::string result;
    for (char character : text) {
        if (character != ' ' && character != '\t' && character != '\r')
            result += character;
    }
    return result;
}

bool parseBits(const std::string& text, std::uint64_t& value) {
    const std::string bits = binaryPart(text);
    if (bits.size() != XS)
        return false;
    for (char bit : bits) {
        if (bit != '0' && bit != '1')
            return false;
    }

    value = std::bitset<XS>(bits).to_ullong();
    return true;
}

bool loadWantedPatterns(const std::string& path,
                        std::vector<WantedPattern>& patterns) {
    std::ifstream file(path);
    if (!file)
        return false;

    std::string line;
    std::uint64_t previousInput = 0;
    bool havePreviousInput = false;
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
        WantedPattern pattern{0, 0, "", binaryPart(outputText)};

        if (separator != std::string::npos) {
            pattern.inputBits = binaryPart(inputText);
            if (!parseBits(inputText, pattern.input)) {
                std::cerr << "Invalid input pattern on line " << lineNumber
                          << ": expected exactly " << XS << " binary bits\n";
                return false;
            }
            previousInput = pattern.input;
            havePreviousInput = true;
        } else if (havePreviousInput) {
            pattern.input = previousInput;
            pattern.inputBits = std::bitset<XS>(previousInput).to_string();
        } else {
            std::cerr << "Missing input pattern on line " << lineNumber << "\n";
            return false;
        }

        if (!parseBits(outputText, pattern.output)) {
            std::cerr << "Invalid wanted pattern on line " << lineNumber
                      << ": expected exactly " << XS << " binary bits\n";
            return false;
        }
        if (pattern.inputBits.empty())
            pattern.inputBits = std::bitset<XS>(pattern.input).to_string();
        patterns.push_back(pattern);
    }
    return !patterns.empty();
}

void applyInputPattern(std::vector<ChunkGenome>& pool, std::uint64_t input) {
    for (auto& genome : pool) {
        for (int index = 0; index < XS; ++index)
            genome.inputs[index] = (input >> (XS - 1 - index) & 1) ? 15 : 0;
    }
}

std::array<std::uint8_t, XS> randomInputs(std::mt19937& random) {
    std::array<std::uint8_t, XS> inputs{};
    std::uniform_int_distribution<int> pulse(0, 15);

    for (auto& input : inputs)
        input = static_cast<std::uint8_t>(pulse(random));

    return inputs;
}

ChunkGenome randomGenome(std::mt19937& random) {
    ChunkGenome genome;
    std::uniform_int_distribution<int> mask(0, 15);
    std::uniform_int_distribution<int> sensitivity(0, 3);

    for (std::size_t index = 0; index < genome.masks.size(); ++index) {
        genome.masks[index] = static_cast<std::uint8_t>(mask(random));
        genome.sensitivities[index] = static_cast<std::uint8_t>(sensitivity(random));
    }
    genome.inputs = randomInputs(random);
    return genome;
}

void mutateGenome(ChunkGenome& genome, std::mt19937& random) {
    std::bernoulli_distribution mutate(MUTATION_RATE);
    std::uniform_int_distribution<int> maskBit(0, 3);
    std::uniform_int_distribution<int> sensitivityBit(0, 1);
    for (auto& value : genome.masks) {
        if (mutate(random))
            value ^= static_cast<std::uint8_t>(1u << maskBit(random));
    }
    for (auto& value : genome.sensitivities) {
        if (mutate(random))
            value ^= static_cast<std::uint8_t>(1u << sensitivityBit(random));
    }
}

ChunkGenome crossover(const ChunkGenome& first, const ChunkGenome& second,
                      std::mt19937& random) {
    ChunkGenome child;
    std::bernoulli_distribution chooseFirst(0.5);

    for (std::size_t index = 0; index < child.masks.size(); ++index) {
        child.masks[index] = chooseFirst(random) ? first.masks[index] : second.masks[index];
        child.sensitivities[index] = chooseFirst(random)
            ? first.sensitivities[index]
            : second.sensitivities[index];
    }
    child.inputs = first.inputs;
    return child;
}

EvaluatedGenome evaluateGenome(const ChunkGenome& genome,
                               std::uint64_t wantedOutput) {
    auto top = std::make_unique<Vchunk>();
    top->clk = 0;
    top->rst = 1;

    for (int index = 0; index < NEURON_COUNT; ++index) {
        top->mask_msk[index] = genome.masks[index];
        top->sens_msk[index] = genome.sensitivities[index];
    }
    for (int index = 0; index < XS; ++index)
        top->in[index] = genome.inputs[index];

    top->eval();
    top->clk = 1;
    top->eval();
    top->clk = 0;
    top->eval();
    top->rst = 0;

    EvaluatedGenome evaluated{genome};
    for (int cycle = 0; cycle < SIMULATION_CYCLES; ++cycle) {
        top->clk = 1;
        top->eval();
        top->clk = 0;
        top->eval();

        const std::uint64_t output = top->out;
        const int fitness = static_cast<int>(__builtin_popcountll(output ^ wantedOutput));
        if (cycle == 0 || fitness < evaluated.fitness) {
            evaluated.fitness = fitness;
            evaluated.output = output;
        }
    }
    evaluated.fitness = XS - evaluated.fitness;
    return evaluated;
}

std::vector<EvaluatedGenome> evolve(std::vector<ChunkGenome> pool,
                                    std::uint64_t wantedOutput,
                                    std::mt19937& random) {
    std::vector<EvaluatedGenome> evaluated;
    std::uniform_int_distribution<std::size_t> parent(0, ELITE_COUNT - 1);

    std::size_t generation = 0;
    while (true) {
        evaluated.clear();
        evaluated.resize(pool.size());

        const unsigned workerCount = std::max(
            1u,
            std::min<unsigned>(
                static_cast<unsigned>(pool.size()),
                std::thread::hardware_concurrency()));
        std::atomic<std::size_t> nextGenome{0};
        std::vector<std::thread> workers;
        workers.reserve(workerCount);

        for (unsigned worker = 0; worker < workerCount; ++worker) {
            workers.emplace_back([&]() {
                while (true) {
                    const std::size_t index = nextGenome.fetch_add(1);
                    if (index >= pool.size())
                        return;
                    evaluated[index] = evaluateGenome(pool[index], wantedOutput);
                }
            });
        }
        for (auto& worker : workers)
            worker.join();

        std::sort(evaluated.begin(), evaluated.end(),
                  [](const EvaluatedGenome& first, const EvaluatedGenome& second) {
                      return first.fitness > second.fitness;
                  });

        std::cout << "Generation " << generation
                  << " | best fitness: " << evaluated.front().fitness
                  << "/" << XS << "\n";

        if (evaluated.front().fitness == XS)
            break;

        pool.clear();
        for (std::size_t index = 0; index < ELITE_COUNT; ++index)
            pool.push_back(evaluated[index].genome);

        while (pool.size() < POPULATION_SIZE) {
            const auto& first = evaluated[parent(random)].genome;
            const auto& second = evaluated[parent(random)].genome;
            ChunkGenome child = crossover(first, second, random);
            mutateGenome(child, random);
            pool.push_back(child);
        }
        ++generation;
    }
    return evaluated;
}

std::vector<ChunkGenome> genomesFromEvaluated(
    const std::vector<EvaluatedGenome>& evaluated) {
    std::vector<ChunkGenome> genomes;
    genomes.reserve(evaluated.size());
    for (const auto& item : evaluated)
        genomes.push_back(item.genome);
    return genomes;
}

std::vector<ChunkGenome> createGenePool(std::mt19937& random) {
    std::vector<ChunkGenome> pool;
    pool.reserve(POPULATION_SIZE);
    ChunkGenome seed;
    seed.masks.fill(0b0100);
    seed.sensitivities.fill(0);
    seed.inputs.fill(15);
    pool.push_back(seed);
    pool.push_back(randomGenome(random));

    std::uniform_int_distribution<std::size_t> parent(0, POPULATION_SIZE - 1);
    while (pool.size() < POPULATION_SIZE) {
        ChunkGenome child = pool[parent(random) % pool.size()];
        mutateGenome(child, random);
        pool.push_back(child);
    }
    return pool;
}

bool saveGenePool(const std::vector<EvaluatedGenome>& pool, const std::string& path) {
    std::ofstream file(path);
    if (!file)
        return false;

    file << "XS " << XS << "\nYS " << YS << "\nPOPULATION " << pool.size() << "\n";
    for (std::size_t genomeIndex = 0; genomeIndex < pool.size(); ++genomeIndex) {
           const auto& evaluated = pool[genomeIndex];
           const auto& genome = evaluated.genome;
           file << "GENOME " << genomeIndex
               << "\nFITNESS " << evaluated.fitness
               << "\nBEST_OUTPUT " << evaluated.output
               << "\nMASKS ";
        for (auto value : genome.masks)
            file << static_cast<unsigned>(value) << ' ';
        file << "\nSENSITIVITIES ";
        for (auto value : genome.sensitivities)
            file << static_cast<unsigned>(value) << ' ';
        file << "\nINPUTS ";
        for (auto value : genome.inputs)
            file << static_cast<unsigned>(value) << ' ';
        file << "\n";
    }
    return true;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const std::string patternPath = argc > 1 ? argv[1] : "wanted_outputs.txt";
    std::vector<WantedPattern> patterns;
    if (!loadWantedPatterns(patternPath, patterns)) {
        std::cerr << "Could not load wanted patterns from " << patternPath << "\n";
        return 1;
    }

    std::mt19937 random(0xA17F2026u);
    auto pool = createGenePool(random);

    for (std::size_t patternIndex = 0; patternIndex < patterns.size(); ++patternIndex) {
        const auto& pattern = patterns[patternIndex];
        applyInputPattern(pool, pattern.input);
        std::cout << "Target " << (patternIndex + 1) << "/" << patterns.size()
                  << " | input: " << pattern.inputBits
                  << " | wanted output: " << pattern.outputBits << "\n";

        const auto evaluated = evolve(pool, pattern.output, random);
        if (evaluated.empty()) {
            std::cerr << "Target " << (patternIndex + 1)
                      << " produced an empty evaluation pool\n";
            return 1;
        }

        if (!saveGenePool(evaluated, "gene_pool.txt")) {
            std::cerr << "Could not save gene_pool.txt\n";
            return 1;
        }
        pool = genomesFromEvaluated(evaluated);
        std::cout << "Reached target " << (patternIndex + 1)
                  << " with fitness " << evaluated.front().fitness
                  << "/" << XS << "\n";
    }

    std::cout << "Completed " << patterns.size()
              << " wanted outputs. Saved gene_pool.txt\n";

    return 0;
}