#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Vchunk.h"
#include "verilated.h"

namespace {
#ifndef SIM_XS
#define SIM_XS 4
#endif
#ifndef SIM_YS
#define SIM_YS 8
#endif
constexpr int XS = SIM_XS;
constexpr int YS = SIM_YS;
constexpr int N = XS * YS;
constexpr int PORT = 8081;
constexpr int POPULATION_SIZE = 256;
constexpr int ELITE_COUNT = POPULATION_SIZE / 5;
constexpr int CYCLES = 32;
constexpr double MUTATION_RATE = 0.04;

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
};

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

std::string stateJson(const Result& best, const std::array<std::uint8_t, XS>& input,
                      const std::array<std::uint8_t, XS>& wanted, int generation) {
    std::ostringstream json;
    json << "{\"xs\":" << XS << ",\"ys\":" << YS
         << ",\"generation\":" << generation << ",\"fitness\":" << best.fitness
         << ",\"false_positives\":" << best.falsePositives
         << ",\"misses\":" << best.misses << ",\"output\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << ((best.output >> index) & 1);
    }
    json << "]," << arrayJson("mask", best.genome.mask.data(), N) << ','
         << arrayJson("sens", best.genome.sens.data(), N) << ','
         << arrayJson("input", input.data(), XS) << ','
         << arrayJson("wanted", wanted.data(), XS) << '}';
    return json.str();
}

class Trainer {
public:
    Trainer() : random(0xA17F2026u), best{randomGenome(random)} {
        population.resize(POPULATION_SIZE, best.genome);
        std::generate(population.begin() + 1, population.end(), [&] { return randomGenome(random); });
        input.fill(15);
        wanted.fill(0);
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
    }

    void reset() { generation = 0; population.assign(POPULATION_SIZE, best.genome); advance(); }
    std::array<std::uint8_t, XS>& inputRef() { return input; }
    std::array<std::uint8_t, XS>& wantedRef() { return wanted; }
    const Result& current() const { return best; }
    int generationNumber() const { return generation; }
private:
    std::mt19937 random;
    std::vector<Genome> population;
    Result best;
    std::array<std::uint8_t, XS> input{};
    std::array<std::uint8_t, XS> wanted{};
    int generation = 0;
};

void handle(int client, Trainer& trainer) {
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
    } else if (get && line.find("GET /api/state ") == 0) {
        reply(client, "200 OK", "application/json", stateJson(trainer.current(), trainer.inputRef(), trainer.wantedRef(), trainer.generationNumber()));
    } else if (post && line.find("POST /api/generation ") == 0) {
        trainer.advance();
        reply(client, "200 OK", "application/json", stateJson(trainer.current(), trainer.inputRef(), trainer.wantedRef(), trainer.generationNumber()));
    } else if (post && line.find("POST /api/reset ") == 0) {
        trainer.reset();
        reply(client, "200 OK", "application/json", stateJson(trainer.current(), trainer.inputRef(), trainer.wantedRef(), trainer.generationNumber()));
    } else if (post && line.find("POST /api/input") == 0) {
        const int index = query(line, "index", -1);
        const int value = query(line, "value", 0);
        if (index < 0 || index >= XS || value < 0 || value > 15) reply(client, "400 Bad Request", "text/plain", "invalid input");
        else { trainer.inputRef()[index] = value; reply(client, "200 OK", "application/json", stateJson(trainer.current(), trainer.inputRef(), trainer.wantedRef(), trainer.generationNumber())); }
    } else if (post && line.find("POST /api/wanted") == 0) {
        const int index = query(line, "index", -1);
        const int value = query(line, "value", 0);
        if (index < 0 || index >= XS || value < 0 || value > 1) reply(client, "400 Bad Request", "text/plain", "invalid wanted bit");
        else { trainer.wantedRef()[index] = value; reply(client, "200 OK", "application/json", stateJson(trainer.current(), trainer.inputRef(), trainer.wantedRef(), trainer.generationNumber())); }
    } else reply(client, "404 Not Found", "text/plain", "not found");
}
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    VerilatedContext context;
    Trainer trainer;
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
    while (!context.gotFinish()) {
        const int client = accept(server, nullptr, nullptr);
        if (client >= 0) { handle(client, trainer); close(client); }
    }
    close(server);
    return 0;
}
