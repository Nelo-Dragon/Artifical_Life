#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "Vchunk.h"
#include "verilated.h"

namespace {
#ifndef SIM_XS
#define SIM_XS 64
#endif
#ifndef SIM_YS
#define SIM_YS 32
#endif
constexpr int XS = SIM_XS;
constexpr int YS = SIM_YS;
constexpr int N = XS * YS;
constexpr int PORT = 8080;
constexpr int EVOLUTION_TRIALS = 24;
constexpr int EVOLUTION_CYCLES = 32;
constexpr double MIN_FITNESS_WEIGHT = 0.6;
constexpr double AVERAGE_FITNESS_WEIGHT = 0.4;

struct Genome {
    std::array<std::uint8_t, N> masks{};
    std::array<std::uint8_t, N> sensitivities{};
};

std::string readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return {};
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

void sendResponse(int client, const std::string& status,
                  const std::string& contentType, const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status << "\r\n"
             << "Content-Type: " << contentType << "\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Cache-Control: no-store\r\n"
             << "Connection: close\r\n\r\n"
             << body;
    const std::string data = response.str();
    send(client, data.data(), data.size(), 0);
}

std::string numberArray(const char* name, const std::uint8_t* values, int count) {
    std::ostringstream json;
    json << '"' << name << "\":[";
    for (int index = 0; index < count; ++index) {
        if (index != 0)
            json << ',';
        json << static_cast<unsigned>(values[index]);
    }
    json << ']';
    return json.str();
}

std::string bitArray(const char* name, std::uint64_t value) {
    std::ostringstream json;
    json << '"' << name << "\":[";
    for (int index = 0; index < XS; ++index) {
        if (index != 0)
            json << ',';
        json << ((value >> index) & 1);
    }
    json << ']';
    return json.str();
}

std::uint64_t wantedValue(const std::array<std::uint8_t, XS>& wanted);

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

Score scoreOutput(std::uint64_t output, std::uint64_t wanted) {
    const std::uint64_t falsePositiveBits = output & ~wanted;
    const std::uint64_t missedBits = wanted & ~output;
    const int falsePositives = __builtin_popcountll(falsePositiveBits);
    const int misses = __builtin_popcountll(missedBits);
    return {falsePositives, misses, XS - falsePositives - misses};
}

std::string stateJson(const Vchunk& top, std::uint64_t step,
                      const std::array<std::uint8_t, XS>& wanted,
                      double mutationRate, std::uint64_t generation,
                      int fitness) {
    const Score currentScore = scoreOutput(static_cast<std::uint64_t>(top.out), wantedValue(wanted));
    std::ostringstream json;
    json << "{\"xs\":" << XS << ",\"ys\":" << YS
        << ",\"step\":" << step << ",\"mutation_rate\":" << mutationRate
         << ",\"generation\":" << generation << ",\"fitness\":" << fitness
         << ",\"current_fitness\":" << currentScore.fitness
         << ",\"false_positives\":" << currentScore.falsePositives
         << ",\"misses\":" << currentScore.misses << ',';
    json << numberArray("mask", top.mask_msk, N) << ',';
    json << numberArray("sens", top.sens_msk, N) << ',';
    json << numberArray("fire", top.fire_msk, N) << ',';
    json << numberArray("accu", top.accu_msk, N) << ',';
    json << numberArray("thresh", top.thresh_msk, N) << ',';
    json << numberArray("input", top.in, XS) << ',';
    json << bitArray("output", top.out) << ',';
    json << numberArray("wanted", wanted.data(), XS) << '}';
    return json.str();
}

int queryValue(const std::string& request, const std::string& key, int fallback) {
    const std::string marker = key + "=";
    const std::size_t start = request.find(marker);
    if (start == std::string::npos)
        return fallback;
    const std::size_t valueStart = start + marker.size();
    return std::max(0, std::stoi(request.substr(valueStart)));
}

double queryRate(const std::string& request, double fallback) {
    const std::string marker = "value=";
    const std::size_t start = request.find(marker);
    if (start == std::string::npos)
        return fallback;
    try {
        return std::clamp(std::stod(request.substr(start + marker.size())), 0.0, 1.0);
    } catch (...) {
        return fallback;
    }
}

void reset(Vchunk& top) {
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

void clockOnce(Vchunk& top) {
    top.clk = 1;
    top.eval();
    top.clk = 0;
    top.eval();
}

Genome readGenome(const Vchunk& top) {
    Genome genome;
    for (int index = 0; index < N; ++index) {
        genome.masks[index] = top.mask_msk[index];
        genome.sensitivities[index] = top.sens_msk[index];
    }
    return genome;
}

void loadGenome(Vchunk& top, const Genome& genome) {
    for (int index = 0; index < N; ++index) {
        top.mask_msk[index] = genome.masks[index];
        top.sens_msk[index] = genome.sensitivities[index];
    }
}

std::uint64_t wantedValue(const std::array<std::uint8_t, XS>& wanted) {
    std::uint64_t value = 0;
    for (int index = 0; index < XS; ++index)
        value |= static_cast<std::uint64_t>(wanted[index] != 0) << index;
    return value;
}

Evaluation evaluateGenome(Vchunk& top, const Genome& genome,
                          std::uint64_t wanted) {
    loadGenome(top, genome);
    reset(top);
    const int warmupCycles = YS - 1;
    int minimumFitness = XS;
    int minimumError = XS;
    int scoredCycles = 0;
    double fitnessTotal = 0.0;
    double tiebreakerTotal = 0.0;
    for (int cycle = 0; cycle < EVOLUTION_CYCLES; ++cycle) {
        clockOnce(top);
        if (cycle < warmupCycles)
            continue;

        const Score score = scoreOutput(static_cast<std::uint64_t>(top.out), wanted);
        minimumFitness = std::min(minimumFitness, score.fitness);
        minimumError = std::min(minimumError, score.falsePositives + score.misses);
        double tiebreaker = 0.0;
        for (int x = 0; x < XS; ++x) {
            const int index = (YS - 1) * XS + x;
            const int threshold = top.thresh_msk[index];
            const int accumulator = top.accu_msk[index];
            const bool isWanted = ((wanted >> x) & 1) != 0;
            tiebreaker += isWanted ? threshold - accumulator : accumulator;
        }
        fitnessTotal += score.fitness;
        tiebreakerTotal += tiebreaker;
        ++scoredCycles;
    }
    Evaluation evaluation;
    const double averageFitness = fitnessTotal / scoredCycles;
    evaluation.rankingFitness = MIN_FITNESS_WEIGHT * minimumFitness
        + AVERAGE_FITNESS_WEIGHT * averageFitness;
    evaluation.fitness = static_cast<int>(evaluation.rankingFitness);
    evaluation.tiebreaker = tiebreakerTotal / scoredCycles;
    return evaluation;
}

bool isBetter(const Evaluation& candidate, const Evaluation& best) {
    if (candidate.fitness != best.fitness)
        return candidate.fitness > best.fitness;
    if (candidate.rankingFitness != best.rankingFitness)
        return candidate.rankingFitness > best.rankingFitness;
    return candidate.tiebreaker < best.tiebreaker;
}

int evolve(Vchunk& top, const std::array<std::uint8_t, XS>& wanted,
           double mutationRate, std::mt19937& random) {
    const std::uint64_t target = wantedValue(wanted);
    const Genome base = readGenome(top);
    Genome best = base;
    Evaluation bestEvaluation = evaluateGenome(top, best, target);
    std::bernoulli_distribution mutateMask(mutationRate);
    std::bernoulli_distribution mutateSensitivity(std::min(1.0, mutationRate * 1.5));
    std::uniform_int_distribution<int> maskBit(0, 3);
    std::uniform_int_distribution<int> sensitivityBit(0, 1);

    for (int trial = 0; trial < EVOLUTION_TRIALS; ++trial) {
        Genome candidate = base;
        for (int index = 0; index < N; ++index) {
            if (mutateMask(random))
                candidate.masks[index] ^= static_cast<std::uint8_t>(1u << maskBit(random));
            if (mutateSensitivity(random))
                candidate.sensitivities[index] ^= static_cast<std::uint8_t>(1u << sensitivityBit(random));
        }
        const Evaluation candidateEvaluation = evaluateGenome(top, candidate, target);
        if (isBetter(candidateEvaluation, bestEvaluation)) {
            best = candidate;
            bestEvaluation = candidateEvaluation;
        }
    }
    loadGenome(top, best);
    reset(top);
    return evaluateGenome(top, best, target).fitness;
}

void handleClient(int client, Vchunk& top, std::uint64_t& step,
                  std::array<std::uint8_t, XS>& wanted, double& mutationRate,
                  std::uint64_t& generation, int& fitness,
                  std::mt19937& random) {
    std::string request;
    char buffer[4096];
    const ssize_t received = recv(client, buffer, sizeof(buffer) - 1, 0);
    if (received <= 0)
        return;
    buffer[received] = '\0';
    request.assign(buffer, received);

    const std::size_t lineEnd = request.find("\r\n");
    const std::string requestLine = request.substr(0, lineEnd);
    const bool isGet = requestLine.rfind("GET ", 0) == 0;
    const bool isPost = requestLine.rfind("POST ", 0) == 0;

    if (isGet && (requestLine.find("GET / ") == 0 || requestLine.find("GET /sim.html ") == 0)) {
        const std::string page = readFile("web/sim.html");
        if (page.empty())
            sendResponse(client, "500 Internal Server Error", "text/plain", "web/sim.html not found");
        else
            sendResponse(client, "200 OK", "text/html; charset=utf-8", page);
    } else if (isGet && requestLine.find("GET /api/state ") == 0) {
        sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
    } else if (isPost && requestLine.find("POST /api/step ") == 0) {
        clockOnce(top);
        ++step;
        sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
    } else if (isPost && requestLine.find("POST /api/reset ") == 0) {
        reset(top);
        step = 0;
        sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
    } else if (isPost && requestLine.find("POST /api/input") == 0) {
        const int index = queryValue(requestLine, "index", -1);
        const int value = queryValue(requestLine, "value", 0);
        if (index < 0 || index >= XS || value < 0 || value > 15) {
            sendResponse(client, "400 Bad Request", "text/plain", "index must be 0..XS-1 and value must be 0..15");
        } else {
            top.in[index] = static_cast<std::uint8_t>(value);
            top.eval();
            sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
        }
    } else if (isPost && requestLine.find("POST /api/wanted") == 0) {
        const int index = queryValue(requestLine, "index", -1);
        const int value = queryValue(requestLine, "value", 0);
        if (index < 0 || index >= XS || value < 0 || value > 1) {
            sendResponse(client, "400 Bad Request", "text/plain", "index must be 0..XS-1 and value must be 0 or 1");
        } else {
            wanted[index] = static_cast<std::uint8_t>(value);
            sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
        }
    } else if (isPost && requestLine.find("POST /api/gene") == 0) {
        const int index = queryValue(requestLine, "index", -1);
        const int mask = queryValue(requestLine, "mask", -1);
        const int sensitivity = queryValue(requestLine, "sens", -1);
        if (index < 0 || index >= N || mask < 0 || mask > 15 || sensitivity < 0 || sensitivity > 3) {
            sendResponse(client, "400 Bad Request", "text/plain", "index, mask, and sens are out of range");
        } else {
            top.mask_msk[index] = static_cast<std::uint8_t>(mask);
            top.sens_msk[index] = static_cast<std::uint8_t>(sensitivity);
            reset(top);
            step = 0;
            sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
        }
    } else if (isPost && requestLine.find("POST /api/mutation") == 0) {
        mutationRate = queryRate(requestLine, mutationRate);
        sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
    } else if (isPost && requestLine.find("POST /api/evolve ") == 0) {
        fitness = evolve(top, wanted, mutationRate, random);
        ++generation;
        step = 0;
        sendResponse(client, "200 OK", "application/json", stateJson(top, step, wanted, mutationRate, generation, fitness));
    } else {
        sendResponse(client, "404 Not Found", "text/plain", "not found");
    }
}
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    VerilatedContext context;
    Vchunk top{&context};

    for (int index = 0; index < N; ++index) {
        top.mask_msk[index] = 0b0100;
        top.sens_msk[index] = 0;
    }
    for (int index = 0; index < XS; ++index)
        top.in[index] = 0;

    reset(top);

    const int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) {
        std::cerr << "Could not create server socket\n";
        return 1;
    }
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(PORT);
    if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        listen(server, 8) < 0) {
        std::cerr << "Could not listen on port " << PORT << '\n';
        close(server);
        return 1;
    }

    std::cout << "Chunk simulator: http://localhost:" << PORT << "\n";
    std::uint64_t step = 0;
    std::array<std::uint8_t, XS> wanted{};
    double mutationRate = 0.04;
    std::uint64_t generation = 0;
    int fitness = 0;
    std::mt19937 random(0xA17F2026u);
    while (!context.gotFinish()) {
        const int client = accept(server, nullptr, nullptr);
        if (client < 0)
            continue;
        handleClient(client, top, step, wanted, mutationRate, generation, fitness, random);
        close(client);
    }

    close(server);
    return 0;
}
