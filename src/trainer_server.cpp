#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_CUDA
#include "cuda_fitness.h"
#else
#include "Vchunk.h"
#include "verilated.h"
#endif

namespace {
#ifndef USE_CUDA
#ifndef SIM_XS
#define SIM_XS 4
#endif
#ifndef SIM_YS
#define SIM_YS 8
#endif
constexpr int XS = SIM_XS;
constexpr int YS = SIM_YS;
constexpr int N = XS * YS;
constexpr int CYCLES = 32;
#else
constexpr int XS = CUDA_XS;
constexpr int YS = CUDA_YS;
constexpr int N = CUDA_NEURON_COUNT;
#endif
constexpr int PORT = 8081;
#ifndef CUDA_POPULATION_SIZE
#define CUDA_POPULATION_SIZE 32768
#endif
#ifndef CUDA_MUTATION_RATE
#define CUDA_MUTATION_RATE 0.04
#endif
#ifndef CUDA_ELITE_PERCENT
#define CUDA_ELITE_PERCENT 20
#endif
#ifdef USE_CUDA
constexpr int POPULATION_SIZE = CUDA_POPULATION_SIZE;
#else
constexpr int POPULATION_SIZE = 256;
#endif
#ifdef USE_CUDA
constexpr int ELITE_COUNT = std::max(1, POPULATION_SIZE * CUDA_ELITE_PERCENT / 100);
constexpr double MUTATION_RATE = CUDA_MUTATION_RATE;
#else
constexpr int ELITE_COUNT = POPULATION_SIZE / 5;
constexpr double MUTATION_RATE = 0.04;
#endif

std::string fileText(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

void reply(int client, const char* status, const char* type, const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status << "\r\nContent-Type: " << type
             << "\r\nContent-Length: " << body.size()
             << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n" << body;
    const std::string data = response.str();
    send(client, data.data(), data.size(), 0);
}

std::uint64_t bits(const std::array<std::uint8_t, XS>& values) {
    std::uint64_t result = 0;
    for (int index = 0; index < XS; ++index)
        result |= static_cast<std::uint64_t>(values[index] != 0) << index;
    return result;
}

int query(const std::string& request, const std::string& key, int fallback) {
    const std::string marker = key + "=";
    const std::size_t start = request.find(marker);
    if (start == std::string::npos) return fallback;
    try { return std::stoi(request.substr(start + marker.size())); }
    catch (...) { return fallback; }
}

std::string arrayJson(const char* name, const std::uint8_t* values, int count) {
    std::ostringstream json;
    json << '"' << name << "\":[";
    for (int index = 0; index < count; ++index) {
        if (index) json << ',';
        json << static_cast<unsigned>(values[index]);
    }
    json << ']';
    return json.str();
}

struct WantedPattern {
    std::uint64_t input = 0;
    std::uint64_t output = 0;
};

std::string binaryPart(const std::string& text) {
    std::string result;
    for (char character : text) {
        if (character != ' ' && character != '\t' && character != '\r')
            result += character;
    }
    return result;
}

bool parsePatternBits(const std::string& text, std::uint64_t& value) {
    const std::string binary = binaryPart(text);
    if (static_cast<int>(binary.size()) != XS)
        return false;
    for (char bit : binary) {
        if (bit != '0' && bit != '1')
            return false;
    }
    value = 0;
    for (int index = 0; index < XS; ++index)
        if (binary[index] == '1')
            value |= std::uint64_t{1} << (XS - 1 - index);
    return true;
}

std::uint64_t reversePatternBits(std::uint64_t value) {
    std::uint64_t reversed = 0;
    for (int index = 0; index < XS; ++index)
        reversed |= ((value >> index) & 1) << (XS - 1 - index);
    return reversed;
}

bool loadWantedPatterns(const std::string& path, std::vector<WantedPattern>& patterns) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Could not open wanted-pattern file " << path << "\n";
        return false;
    }

    std::string line;
    std::uint64_t previousInput = 0;
    bool havePreviousInput = false;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (line.empty())
            continue;
        const std::size_t separator = line.find(':');
        const std::string inputText = separator == std::string::npos ? "" : line.substr(0, separator);
        const std::string outputText = separator == std::string::npos ? line : line.substr(separator + 1);

        WantedPattern pattern{};
        if (separator != std::string::npos) {
            if (!parsePatternBits(inputText, pattern.input)) {
                std::cerr << "Invalid input pattern on line " << lineNumber
                          << ": expected exactly " << XS << " binary bits\n";
                return false;
            }
            previousInput = pattern.input;
            havePreviousInput = true;
        } else if (havePreviousInput) {
            pattern.input = previousInput;
        } else {
            std::cerr << "Missing input pattern on line " << lineNumber << "\n";
            return false;
        }

        std::uint64_t output = 0;
        if (!parsePatternBits(outputText, output)) {
            std::cerr << "Invalid wanted pattern on line " << lineNumber
                      << ": expected exactly " << XS << " binary bits\n";
            return false;
        }
        pattern.output = reversePatternBits(output);
        patterns.push_back(pattern);
    }
    return !patterns.empty();
}

#ifdef USE_CUDA
struct RankedGenome {
    CudaGenome genome{};
    int fitness = 0;
    float rankingFitness = 0.0f;
    float tiebreaker = 0.0f;
    std::uint64_t output = 0;
    std::array<std::uint8_t, N> fire{};
    std::array<std::uint8_t, N> accu{};
    std::array<std::uint8_t, N> thresh{};
};

std::string stateJson(const RankedGenome& best, const std::array<std::uint8_t, XS>& input,
                     const std::array<std::uint8_t, XS>& wanted, int generation,
                     std::size_t targetIndex, std::size_t targetCount) {
    std::ostringstream json;
    json << "{\"xs\":" << XS << ",\"ys\":" << YS
         << ",\"generation\":" << generation << ",\"fitness\":" << best.fitness
         << ",\"ranking_fitness\":" << best.rankingFitness
         << ",\"target\":" << targetIndex << ",\"target_count\":" << targetCount
         << ",\"finished\":" << (best.fitness == XS ? "true" : "false")
         << ",\"output\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << ((best.output >> index) & 1);
    }
    json << "]," << arrayJson("mask", best.genome.masks, N) << ','
         << arrayJson("sens", best.genome.sensitivities, N) << ','
         << arrayJson("fire", best.fire.data(), N) << ','
         << arrayJson("accu", best.accu.data(), N) << ','
         << arrayJson("thresh", best.thresh.data(), N) << ','
         << arrayJson("input", input.data(), XS) << ','
         << arrayJson("wanted", wanted.data(), XS) << '}';
    return json.str();
}

CudaGenome randomGenome(std::mt19937& random) {
    CudaGenome genome{};
    std::uniform_int_distribution<int> mask(0, 15);
    std::uniform_int_distribution<int> sensitivity(0, 3);
    for (int index = 0; index < CUDA_NEURON_COUNT; ++index) {
        genome.masks[index] = static_cast<std::uint8_t>(mask(random));
        genome.sensitivities[index] = static_cast<std::uint8_t>(sensitivity(random));
    }
    for (auto& value : genome.inputs)
        value = 15;
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
    for (std::size_t index = 0; index < population.size(); ++index) {
        ranked[index] = {population[index], evaluations[index].fitness,
                        evaluations[index].rankingFitness,
                        evaluations[index].tiebreaker, evaluations[index].output, {}};
        std::copy(std::begin(evaluations[index].fire), std::end(evaluations[index].fire), ranked[index].fire.begin());
        std::copy(std::begin(evaluations[index].accu), std::end(evaluations[index].accu), ranked[index].accu.begin());
        std::copy(std::begin(evaluations[index].thresh), std::end(evaluations[index].thresh), ranked[index].thresh.begin());
    }
    std::sort(ranked.begin(), ranked.end(), [](const RankedGenome& first, const RankedGenome& second) {
        if (first.fitness != second.fitness)
            return first.fitness > second.fitness;
        if (first.rankingFitness != second.rankingFitness)
            return first.rankingFitness > second.rankingFitness;
        return first.tiebreaker < second.tiebreaker;
    });
    return ranked;
}

class Trainer {
public:
    Trainer() : random(0xA17F2026u) {
        seedPopulation();
        if (loadWantedPatterns("wanted_outputs.txt", patterns)) {
            usingPatterns = true;
            applyTarget();
        } else {
            input.fill(15);
            wanted.fill(0);
        }
        advance();
    }

    void advance() {
        const std::uint64_t target = bits(wanted);
        auto ranked = rankPopulation(population, target);
        best = ranked.front();
        population.clear();
        for (std::size_t index = 0; index < ELITE_COUNT; ++index)
            population.push_back(ranked[index].genome);
        std::uniform_int_distribution<std::size_t> parent(0, ELITE_COUNT - 1);
        while (population.size() < POPULATION_SIZE) {
            const auto& firstParent = ranked[parent(random)].genome;
            const auto& secondParent = ranked[parent(random)].genome;
            CudaGenome child = crossover(firstParent, secondParent, random);
            mutate(child, random);
            population.push_back(child);
        }
        ++generation;

        if (usingPatterns && best.fitness == XS) {
            // Move on to the next wanted pattern, re-seeding the population
            // so a previous target's specialized genes cannot block
            // discovery of the next target's route (mirrors main.cpp).
            targetIndex = (targetIndex + 1) % patterns.size();
            applyTarget();
            seedPopulation();
            generation = 0;
        }
    }

    void reset() {
        generation = 0;
        population.assign(POPULATION_SIZE, best.genome);
        advance();
    }

    void setInput(int index, int value) {
        input[index] = static_cast<std::uint8_t>(value);
        applyInputToPopulation();
    }

    void setWanted(int index, int value) {
        wanted[index] = static_cast<std::uint8_t>(value);
    }

    const std::array<std::uint8_t, XS>& inputRef() const { return input; }
    const std::array<std::uint8_t, XS>& wantedRef() const { return wanted; }
    const RankedGenome& current() const { return best; }
    int generationNumber() const { return generation; }
    std::size_t targetNumber() const { return targetIndex; }
    std::size_t targetTotal() const { return usingPatterns ? patterns.size() : 1; }

private:
    void applyInputToPopulation() {
        for (auto& genome : population)
            for (int index = 0; index < XS; ++index)
                genome.inputs[index] = input[index];
    }

    void applyTarget() {
        const auto& pattern = patterns[targetIndex];
        for (int index = 0; index < XS; ++index)
            input[index] = ((pattern.input >> (XS - 1 - index)) & 1) ? 15 : 0;
        for (int index = 0; index < XS; ++index)
            wanted[index] = static_cast<std::uint8_t>((pattern.output >> index) & 1);
        applyInputToPopulation();
    }

    void seedPopulation() {
        population.resize(POPULATION_SIZE);
        for (auto& genome : population)
            genome = randomGenome(random);
        applyInputToPopulation();
    }

    std::mt19937 random;
    std::vector<CudaGenome> population;
    RankedGenome best{};
    std::array<std::uint8_t, XS> input{};
    std::array<std::uint8_t, XS> wanted{};
    int generation = 0;
    std::vector<WantedPattern> patterns;
    std::size_t targetIndex = 0;
    bool usingPatterns = false;
};
#else
struct Genome {
    std::array<std::uint8_t, N> mask{};
    std::array<std::uint8_t, N> sens{};
};
struct Result {
    Genome genome;
    int fitness = 0;
    int falsePositives = 0;
    int misses = 0;
    std::uint64_t output = 0;
    std::array<std::uint8_t, N> fire{};
    std::array<std::uint8_t, N> accu{};
    std::array<std::uint8_t, N> thresh{};
};

std::string stateJson(const Result& best, const std::array<std::uint8_t, XS>& input,
                     const std::array<std::uint8_t, XS>& wanted, int generation,
                     std::size_t targetIndex, std::size_t targetCount) {
    std::ostringstream json;
    json << "{\"xs\":" << XS << ",\"ys\":" << YS
         << ",\"generation\":" << generation << ",\"fitness\":" << best.fitness
         << ",\"false_positives\":" << best.falsePositives
         << ",\"misses\":" << best.misses
         << ",\"target\":" << targetIndex << ",\"target_count\":" << targetCount
         << ",\"finished\":" << (best.fitness == XS ? "true" : "false")
         << ",\"output\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << ((best.output >> index) & 1);
    }
    json << "]," << arrayJson("mask", best.genome.mask.data(), N) << ','
         << arrayJson("sens", best.genome.sens.data(), N) << ','
         << arrayJson("fire", best.fire.data(), N) << ','
         << arrayJson("accu", best.accu.data(), N) << ','
         << arrayJson("thresh", best.thresh.data(), N) << ','
         << arrayJson("input", input.data(), XS) << ','
         << arrayJson("wanted", wanted.data(), XS) << '}';
    return json.str();
}

Result evaluate(const Genome& genome, const std::array<std::uint8_t, XS>& input,
               std::uint64_t wanted) {
    auto top = std::make_unique<Vchunk>();
    top->clk = 0;
    top->rst = 1;
    for (int index = 0; index < N; ++index) {
        top->mask_msk[index] = genome.mask[index];
        top->sens_msk[index] = genome.sens[index];
    }
    for (int index = 0; index < XS; ++index) top->in[index] = input[index];
    top->eval();
    top->clk = 1; top->eval();
    top->clk = 0; top->eval();
    top->rst = 0;

    Result result{genome};
    int worstFitness = XS;
    int worstFalse = 0;
    int worstMisses = 0;
    for (int cycle = 0; cycle < CYCLES; ++cycle) {
        top->clk = 1; top->eval();
        top->clk = 0; top->eval();
        const std::uint64_t output = top->out;
        const std::uint64_t falseBits = output & ~wanted;
        const std::uint64_t missedBits = wanted & ~output;
        const int falseCount = __builtin_popcountll(falseBits);
        const int missCount = __builtin_popcountll(missedBits);
        const int fitness = XS - falseCount - missCount;
        if (fitness < worstFitness) {
            worstFitness = fitness;
            worstFalse = falseCount;
            worstMisses = missCount;
            result.output = output;
            for (int index = 0; index < N; ++index) {
                result.fire[index] = top->fire_msk[index];
                result.accu[index] = top->accu_msk[index];
                result.thresh[index] = top->thresh_msk[index];
            }
        }
    }
    result.fitness = worstFitness;
    result.falsePositives = worstFalse;
    result.misses = worstMisses;
    return result;
}

Genome randomGenome(std::mt19937& random) {
    Genome genome;
    std::uniform_int_distribution<int> mask(0, 15);
    std::uniform_int_distribution<int> sens(0, 3);
    for (int index = 0; index < N; ++index) {
        genome.mask[index] = mask(random);
        genome.sens[index] = sens(random);
    }
    return genome;
}

void mutate(Genome& genome, std::mt19937& random) {
    std::bernoulli_distribution maskMutation(MUTATION_RATE);
    std::bernoulli_distribution sensMutation(MUTATION_RATE * 1.5);
    std::uniform_int_distribution<int> maskBit(0, 3);
    std::uniform_int_distribution<int> sensBit(0, 1);
    for (int index = 0; index < N; ++index) {
        if (maskMutation(random)) genome.mask[index] ^= 1u << maskBit(random);
        if (sensMutation(random)) genome.sens[index] ^= 1u << sensBit(random);
    }
}

class Trainer {
public:
    Trainer() : random(0xA17F2026u), best{randomGenome(random)} {
        population.resize(POPULATION_SIZE, best.genome);
        std::generate(population.begin() + 1, population.end(), [&] { return randomGenome(random); });
        if (loadWantedPatterns("wanted_outputs.txt", patterns)) {
            usingPatterns = true;
            applyTarget();
        } else {
            input.fill(15);
            wanted.fill(0);
        }
        advance();
    }

    void advance() {
        const std::uint64_t target = bits(wanted);
        std::vector<Result> ranked;
        ranked.reserve(population.size());
        for (const Genome& genome : population) ranked.push_back(evaluate(genome, input, target));
        std::sort(ranked.begin(), ranked.end(), [](const Result& first, const Result& second) {
            return first.fitness > second.fitness;
        });
        best = ranked.front();
        population.clear();
        for (int index = 0; index < ELITE_COUNT; ++index) population.push_back(ranked[index].genome);
        std::uniform_int_distribution<int> parent(0, ELITE_COUNT - 1);
        while (static_cast<int>(population.size()) < POPULATION_SIZE) {
            Genome child = ranked[parent(random)].genome;
            mutate(child, random);
            population.push_back(child);
        }
        ++generation;

        if (usingPatterns && best.fitness == XS) {
            // Move on to the next wanted pattern, re-seeding the population
            // so a previous target's specialized genes cannot block
            // discovery of the next target's route (mirrors main.cpp).
            targetIndex = (targetIndex + 1) % patterns.size();
            applyTarget();
            seedPopulation();
            generation = 0;
        }
    }

    void reset() { generation = 0; population.assign(POPULATION_SIZE, best.genome); advance(); }

    void setInput(int index, int value) { input[index] = static_cast<std::uint8_t>(value); }
    void setWanted(int index, int value) { wanted[index] = static_cast<std::uint8_t>(value); }

    const std::array<std::uint8_t, XS>& inputRef() const { return input; }
    const std::array<std::uint8_t, XS>& wantedRef() const { return wanted; }
    const Result& current() const { return best; }
    int generationNumber() const { return generation; }
    std::size_t targetNumber() const { return targetIndex; }
    std::size_t targetTotal() const { return usingPatterns ? patterns.size() : 1; }

private:
    void applyTarget() {
        const auto& pattern = patterns[targetIndex];
        for (int index = 0; index < XS; ++index)
            input[index] = ((pattern.input >> (XS - 1 - index)) & 1) ? 15 : 0;
        for (int index = 0; index < XS; ++index)
            wanted[index] = static_cast<std::uint8_t>((pattern.output >> index) & 1);
    }

    void seedPopulation() {
        population.resize(POPULATION_SIZE);
        for (auto& genome : population)
            genome = randomGenome(random);
    }

    std::mt19937 random;
    std::vector<Genome> population;
    Result best;
    std::array<std::uint8_t, XS> input{};
    std::array<std::uint8_t, XS> wanted{};
    int generation = 0;
    std::vector<WantedPattern> patterns;
    std::size_t targetIndex = 0;
    bool usingPatterns = false;
};
#endif

void handle(int client, Trainer& trainer, std::mutex& trainerMutex,
            std::string& latestState, std::mutex& stateMutex) {
    char buffer[4096];
    const ssize_t received = recv(client, buffer, sizeof(buffer) - 1, 0);
    if (received <= 0) return;
    buffer[received] = '\0';
    const std::string request(buffer, received);
    const std::string line = request.substr(0, request.find("\r\n"));
    const bool get = line.rfind("GET ", 0) == 0;
    const bool post = line.rfind("POST ", 0) == 0;
    if (get && (line.find("GET / ") == 0 || line.find("GET /trainer.html ") == 0)) {
        const std::string page = fileText("web/trainer.html");
        reply(client, page.empty() ? "404 Not Found" : "200 OK", "text/html; charset=utf-8", page);
        return;
    }
    if (get && line.find("GET /api/state ") == 0) {
        std::lock_guard<std::mutex> lock(stateMutex);
        reply(client, "200 OK", "application/json", latestState);
        return;
    }

    std::lock_guard<std::mutex> lock(trainerMutex);
    auto snapshot = [&]() {
        return stateJson(trainer.current(), trainer.inputRef(), trainer.wantedRef(),
                          trainer.generationNumber(), trainer.targetNumber(), trainer.targetTotal());
    };
    auto publishSnapshot = [&]() {
        const std::string response = snapshot();
        std::lock_guard<std::mutex> stateLock(stateMutex);
        latestState = response;
        return response;
    };
    if (post && line.find("POST /api/generation ") == 0) {
        trainer.advance();
        reply(client, "200 OK", "application/json", publishSnapshot());
    } else if (post && line.find("POST /api/reset ") == 0) {
        trainer.reset();
        reply(client, "200 OK", "application/json", publishSnapshot());
    } else if (post && line.find("POST /api/input") == 0) {
        const int index = query(line, "index", -1);
        const int value = query(line, "value", 0);
        if (index < 0 || index >= XS || value < 0 || value > 15) reply(client, "400 Bad Request", "text/plain", "invalid input");
        else { trainer.setInput(index, value); reply(client, "200 OK", "application/json", publishSnapshot()); }
    } else if (post && line.find("POST /api/wanted") == 0) {
        const int index = query(line, "index", -1);
        const int value = query(line, "value", 0);
        if (index < 0 || index >= XS || value < 0 || value > 1) reply(client, "400 Bad Request", "text/plain", "invalid wanted bit");
        else { trainer.setWanted(index, value); reply(client, "200 OK", "application/json", publishSnapshot()); }
    } else reply(client, "404 Not Found", "text/plain", "not found");
}
}

int main(int argc, char** argv) {
#ifdef USE_CUDA
    Trainer trainer;
#else
    Verilated::commandArgs(argc, argv);
    VerilatedContext context;
    Trainer trainer;
#endif

    // Continuously advance generations in the background, mirroring the
    // always-training dashboard in main.cpp, so the browser sees live
    // progress without needing to trigger POST /api/generation itself.
    std::mutex trainerMutex;
    std::mutex stateMutex;
    std::string latestState = stateJson(trainer.current(), trainer.inputRef(),
                                        trainer.wantedRef(), trainer.generationNumber(),
                                        trainer.targetNumber(), trainer.targetTotal());
    std::atomic<bool> keepTraining{true};
    std::thread trainerThread([&]() {
        while (keepTraining) {
            {
                std::lock_guard<std::mutex> lock(trainerMutex);
                trainer.advance();
                const std::string snapshot = stateJson(
                    trainer.current(), trainer.inputRef(), trainer.wantedRef(),
                    trainer.generationNumber(), trainer.targetNumber(), trainer.targetTotal());
                std::lock_guard<std::mutex> stateLock(stateMutex);
                latestState = snapshot;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    const int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) return 1;
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(PORT);
    if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 || listen(server, 8) < 0) return 1;
    std::cout << "Trainer dashboard: http://localhost:" << PORT << "\n";
#ifdef USE_CUDA
    while (true) {
        const int client = accept(server, nullptr, nullptr);
        if (client >= 0) { handle(client, trainer, trainerMutex, latestState, stateMutex); close(client); }
    }
#else
    while (!context.gotFinish()) {
        const int client = accept(server, nullptr, nullptr);
        if (client >= 0) { handle(client, trainer, trainerMutex, latestState, stateMutex); close(client); }
    }
#endif
    keepTraining = false;
    trainerThread.join();
    close(server);
    return 0;
}
