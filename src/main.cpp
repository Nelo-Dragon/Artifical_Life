#include <array>
#include <algorithm>
#include <arpa/inet.h>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

#include "native_chunk.h"

#ifndef MAIN_XS
#define MAIN_XS 8
#endif
#ifndef MAIN_YS
#define MAIN_YS 8
#endif
constexpr int XS = MAIN_XS;
constexpr int YS = MAIN_YS;
constexpr int NEURON_COUNT = XS * YS;
constexpr std::size_t POPULATION_SIZE = 256;
constexpr std::size_t ELITE_COUNT = POPULATION_SIZE / 5;
constexpr int SIMULATION_CYCLES = 32;
constexpr double MUTATION_RATE = 0.04;
constexpr double MIN_FITNESS_WEIGHT = 0.6;
constexpr double AVERAGE_FITNESS_WEIGHT = 0.4;
constexpr unsigned MAX_EVALUATION_WORKERS = 16;

enum class EvolutionStrategy {
    GeneticAlgorithm,
    MuPlusLambda
};

struct EvolutionConfig {
    EvolutionStrategy strategy = EvolutionStrategy::GeneticAlgorithm;
    std::size_t mu = 32;
    std::size_t lambda = POPULATION_SIZE - 32;
    int mutationGenes = 2;
};

struct ChunkGenome {
    std::array<std::uint8_t, NEURON_COUNT> masks{};
    std::array<std::uint8_t, NEURON_COUNT> sensitivities{};
    std::array<std::uint8_t, XS> inputs{};
};

struct EvaluatedGenome {
    ChunkGenome genome;
    int fitness = 0;
    double rankingFitness = 0.0;
    double tiebreaker = 0.0;
    std::uint64_t output = 0;
    std::array<std::uint8_t, NEURON_COUNT> fire{};
    std::array<std::uint8_t, NEURON_COUNT> accu{};
    std::array<std::uint8_t, NEURON_COUNT> thresh{};
};

struct WantedPattern {
    std::uint64_t input;
    std::uint64_t output;
    std::string inputBits;
    std::string outputBits;
};

struct ProgressSnapshot {
    EvaluatedGenome best;
    std::uint64_t wanted = 0;
    std::uint64_t input = 0;
    std::size_t generation = 0;
    std::size_t target = 0;
    std::size_t targetCount = 0;
    bool finished = false;
};

std::mutex progressMutex;
ProgressSnapshot progress;
std::atomic<bool> progressServerRunning{true};

std::string progressJson() {
    std::lock_guard<std::mutex> lock(progressMutex);
    std::ostringstream json;
    json << "{\"xs\":" << XS << ",\"ys\":" << YS
         << ",\"generation\":" << progress.generation
         << ",\"target\":" << progress.target
         << ",\"target_count\":" << progress.targetCount
         << ",\"fitness\":" << progress.best.fitness
         << ",\"ranking_fitness\":" << progress.best.rankingFitness
         << ",\"output\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << ((progress.best.output >> index) & 1);
    }
    json << "],\"wanted\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << ((progress.wanted >> index) & 1);
    }
    json << "],\"input\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << ((progress.input >> index) & 1);
    }
    json << "],\"mask\":[";
    for (int index = 0; index < NEURON_COUNT; ++index) {
        if (index) json << ',';
        json << static_cast<unsigned>(progress.best.genome.masks[index]);
    }
    json << "],\"sens\":[";
    for (int index = 0; index < NEURON_COUNT; ++index) {
        if (index) json << ',';
        json << static_cast<unsigned>(progress.best.genome.sensitivities[index]);
    }
    json << "],\"fire\":[";
    for (int index = 0; index < NEURON_COUNT; ++index) {
        if (index) json << ',';
        json << static_cast<unsigned>(progress.best.fire[index]);
    }
    json << "],\"accu\":[";
    for (int index = 0; index < NEURON_COUNT; ++index) {
        if (index) json << ',';
        json << static_cast<unsigned>(progress.best.accu[index]);
    }
    json << "],\"thresh\":[";
    for (int index = 0; index < NEURON_COUNT; ++index) {
        if (index) json << ',';
        json << static_cast<unsigned>(progress.best.thresh[index]);
    }
    json << "],\"finished\":" << (progress.finished ? "true" : "false") << '}';
    return json.str();
}

void sendProgressResponse(int client, const std::string& type,
                          const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 200 OK\r\nContent-Type: " << type
             << "\r\nContent-Length: " << body.size()
             << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
             << body;
    const std::string data = response.str();
    send(client, data.data(), data.size(), 0);
}

void runProgressServer() {
    const int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) return;
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(8082);
    if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        listen(server, 8) < 0) {
        close(server);
        return;
    }
    std::cout << "Main trainer dashboard: http://localhost:8082\n";
    while (progressServerRunning) {
        const int client = accept(server, nullptr, nullptr);
        if (client < 0) continue;
        char request[2048];
        const ssize_t received = recv(client, request, sizeof(request) - 1, 0);
        if (received > 0) {
            request[received] = '\0';
            const std::string line(request, received);
            if (line.rfind("GET /api/state ", 0) == 0 ||
                line.rfind("POST /api/generation ", 0) == 0) {
                sendProgressResponse(client, "application/json", progressJson());
            } else {
                std::ifstream page("web/trainer.html", std::ios::binary);
                std::ostringstream contents;
                contents << page.rdbuf();
                sendProgressResponse(client, "text/html; charset=utf-8", contents.str());
            }
        }
        close(client);
    }
    close(server);
}

void publishProgress(const EvaluatedGenome& best, std::uint64_t input,
                     std::uint64_t wanted, std::size_t generation,
                     std::size_t target, std::size_t targetCount) {
    std::lock_guard<std::mutex> lock(progressMutex);
    progress.best = best;
    progress.input = input;
    progress.wanted = wanted;
    progress.generation = generation;
    progress.target = target;
    progress.targetCount = targetCount;
}

std::string binaryPart(const std::string& text) {
    std::string result;
    for (char character : text) {
        if (character != ' ' && character != '\t' && character != '\r')
            result += character;
    }
    return result;
}

std::string gridBits(std::uint64_t value) {
    std::string result;
    result.reserve(XS);
    for (int index = 0; index < XS; ++index)
        result += ((value >> index) & 1) ? '1' : '0';
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

    value = 0;
    for (int index = 0; index < XS; ++index)
        if (bits[index] == '1')
            value |= std::uint64_t{1} << index;
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
            pattern.inputBits = gridBits(previousInput);
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
            pattern.inputBits = gridBits(pattern.input);
        patterns.push_back(pattern);
    }
    return !patterns.empty();
}

void applyInputPattern(std::vector<ChunkGenome>& pool, std::uint64_t input) {
    for (auto& genome : pool) {
        for (int index = 0; index < XS; ++index)
            genome.inputs[index] = (input >> index & 1) ? 15 : 0;
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

void mutateDiscrete(ChunkGenome& genome, int mutationGenes,
                    std::mt19937& random) {
    std::uniform_int_distribution<int> neuron(0, NEURON_COUNT - 1);
    std::uniform_int_distribution<int> mask(0, 15);
    std::uniform_int_distribution<int> sensitivity(0, 3);
    std::bernoulli_distribution mutateMask(0.5);
    for (int mutation = 0; mutation < mutationGenes; ++mutation) {
        const int index = neuron(random);
        if (mutateMask(random))
            genome.masks[index] = static_cast<std::uint8_t>(mask(random));
        else
            genome.sensitivities[index] =
                static_cast<std::uint8_t>(sensitivity(random));
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
    NativeChunk<XS, YS> top;
    top.load(genome.masks, genome.sensitivities);
    top.reset();

    EvaluatedGenome evaluated{genome};
    const int warmupCycles = YS - 1;
    int bestFitness = -1;
    int bestError = XS + 1;
    double bestTiebreaker = 0.0;
    int scoredCycles = 0;
    double fitnessTotal = 0.0;
    double tiebreakerTotal = 0.0;
    for (int cycle = 0; cycle < SIMULATION_CYCLES; ++cycle) {
        top.step(genome.inputs);

        const std::uint64_t output = top.output();
        if (cycle < warmupCycles)
            continue;

        const int error = static_cast<int>(__builtin_popcountll(output ^ wantedOutput));
        const int fitness = XS - error;
        double tiebreaker = 0.0;
        for (int x = 0; x < XS; ++x) {
            const int index = (YS - 1) * XS + x;
            const int threshold = top.thresholdAt(index);
            const int accumulator = top.accumulator()[index];
            const bool wanted = ((wantedOutput >> x) & 1) != 0;
            tiebreaker += wanted ? threshold - accumulator : accumulator;
        }
        if (fitness > bestFitness ||
            (fitness == bestFitness && tiebreaker < bestTiebreaker)) {
            bestFitness = fitness;
            bestError = error;
            bestTiebreaker = tiebreaker;
            evaluated.output = output;
            for (int index = 0; index < NEURON_COUNT; ++index) {
                evaluated.fire[index] = top.fired()[index];
                evaluated.accu[index] = top.accumulator()[index];
                evaluated.thresh[index] = top.thresholdAt(index);
            }
        }
        fitnessTotal += fitness;
        tiebreakerTotal += tiebreaker;
        ++scoredCycles;
    }
    const double averageFitness = fitnessTotal / scoredCycles;
    evaluated.rankingFitness = MIN_FITNESS_WEIGHT * bestFitness
        + AVERAGE_FITNESS_WEIGHT * averageFitness;
    evaluated.fitness = bestFitness;
    evaluated.tiebreaker = bestTiebreaker;
    return evaluated;
}

std::vector<EvaluatedGenome> evolve(std::vector<ChunkGenome> pool,
                                    std::uint64_t wantedOutput,
                                    std::uint64_t inputPattern,
                                    std::mt19937& random,
                                    std::size_t targetIndex,
                                    std::size_t targetCount,
                                    const EvolutionConfig& config) {
    std::vector<EvaluatedGenome> evaluated;

    std::size_t generation = 0;
    while (true) {
        evaluated.clear();
        evaluated.resize(pool.size());

        const unsigned workerCount = std::max(
            1u,
            std::min<unsigned>(
                static_cast<unsigned>(pool.size()),
                std::min<unsigned>(MAX_EVALUATION_WORKERS,
                                   std::thread::hardware_concurrency())));
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
                      if (first.fitness != second.fitness)
                          return first.fitness > second.fitness;
                      if (first.rankingFitness != second.rankingFitness)
                          return first.rankingFitness > second.rankingFitness;
                      return first.tiebreaker < second.tiebreaker;
                  });

        std::cout << "Generation " << generation
                  << " | best fitness: " << evaluated.front().fitness
                  << "/" << XS << "\n";
        publishProgress(evaluated.front(), inputPattern, wantedOutput,
                generation, targetIndex, targetCount);

        if (evaluated.front().fitness == XS)
            break;

        const std::size_t eliteCount = config.strategy == EvolutionStrategy::MuPlusLambda
            ? config.mu
            : ELITE_COUNT;
        std::uniform_int_distribution<std::size_t> parent(0, eliteCount - 1);
        std::vector<ChunkGenome> nextPool;
        const std::size_t offspringCount = config.strategy == EvolutionStrategy::MuPlusLambda
            ? config.lambda
            : POPULATION_SIZE - eliteCount;
        nextPool.reserve(eliteCount + offspringCount);
        for (std::size_t index = 0; index < eliteCount; ++index)
            nextPool.push_back(evaluated[index].genome);

        while (nextPool.size() < eliteCount + offspringCount) {
            ChunkGenome child;
            if (config.strategy == EvolutionStrategy::MuPlusLambda) {
                child = evaluated[parent(random)].genome;
                mutateDiscrete(child, config.mutationGenes, random);
            } else {
                const auto& first = evaluated[parent(random)].genome;
                const auto& second = evaluated[parent(random)].genome;
                child = crossover(first, second, random);
                mutateGenome(child, random);
            }
            nextPool.push_back(child);
        }
        pool = std::move(nextPool);
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
    EvolutionConfig evolutionConfig;
    std::string patternPath = "wanted_outputs.txt";
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option.rfind("--strategy=", 0) == 0) {
            const std::string value = option.substr(11);
            if (value == "es")
                evolutionConfig.strategy = EvolutionStrategy::MuPlusLambda;
            else if (value != "ga") {
                std::cerr << "Unknown strategy: " << value << "\n";
                return 1;
            }
        } else if (option.rfind("--mu=", 0) == 0) {
            evolutionConfig.mu = std::stoul(option.substr(5));
        } else if (option.rfind("--lambda=", 0) == 0) {
            evolutionConfig.lambda = std::stoul(option.substr(9));
        } else if (option.rfind("--mutation-genes=", 0) == 0) {
            evolutionConfig.mutationGenes = std::stoi(option.substr(17));
        } else if (option.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << option << "\n";
            return 1;
        } else {
            patternPath = option;
        }
    }
    if (evolutionConfig.mu == 0 || evolutionConfig.lambda == 0 ||
        evolutionConfig.mu + evolutionConfig.lambda > POPULATION_SIZE ||
        evolutionConfig.mutationGenes < 1) {
        std::cerr << "Invalid evolution configuration\n";
        return 1;
    }

    std::vector<WantedPattern> patterns;
    if (!loadWantedPatterns(patternPath, patterns)) {
        std::cerr << "Could not load wanted patterns from " << patternPath << "\n";
        return 1;
    }

    std::mt19937 random(0xA17F2026u);
    auto pool = createGenePool(random);
    std::thread progressThread(runProgressServer);

    for (std::size_t patternIndex = 0; patternIndex < patterns.size(); ++patternIndex) {
        const auto& pattern = patterns[patternIndex];
        // Each target can require a different spatial route. Re-seed the
        // population so a previous target's specialized genes cannot block
        // discovery of the next target's route.
        if (patternIndex > 0)
            pool = createGenePool(random);
        applyInputPattern(pool, pattern.input);
        std::cout << "Target " << (patternIndex + 1) << "/" << patterns.size()
                  << " | input: " << pattern.inputBits
                  << " | wanted output: " << pattern.outputBits << "\n";

        const auto evaluated = evolve(pool, pattern.output, pattern.input, random,
                          patternIndex, patterns.size(), evolutionConfig);
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

    {
        std::lock_guard<std::mutex> lock(progressMutex);
        progress.finished = true;
    }
    progressServerRunning = false;
    progressThread.detach();

    return 0;
}