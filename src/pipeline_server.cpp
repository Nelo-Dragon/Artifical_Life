// HTTP server exposing the multi-chunk `pipeline` module (src/verilog/pipeline.v)
// for the "multi-chunk / C_LINK" section of the observatory demo. Mirrors the
// write/silence/probe phase harness used by curriculum_train.cpp, but driven
// interactively from the browser instead of automatically by an offline
// trainer, and reports real Verilated pipeline state for every value shown.
//
// Genomes are loaded once at startup from --genome-dir (default "genomes/"),
// the same directory/format curriculum_train.cpp writes
// (genomes/{input,memory,cortex,output}.genome). Any chunk whose genome file
// is missing falls back to the same default "south-pass" genome sim_server.cpp
// starts with (mask=0b0100, sensitivity=0), and is reported as untrained in
// /api/topology so the frontend can label it rather than fake trained output.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "Vpipeline.h"
#include "verilated.h"

namespace {
#ifndef PIPE_XS
#define PIPE_XS 4
#endif
#ifndef PIPE_YS
#define PIPE_YS 8
#endif
constexpr int XS = PIPE_XS;
constexpr int YS = PIPE_YS;
constexpr int N = XS * YS;
constexpr int PORT = 8082;

struct Genome {
    std::array<std::uint8_t, N> masks{};
    std::array<std::uint8_t, N> sensitivities{};
};

struct ChunkSource {
    Genome genome;
    bool trained = false;
    std::string source;
};

void sendResponse(int client, const std::string& status, const std::string& contentType,
                  const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status << "\r\n"
             << "Content-Type: " << contentType << "\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Access-Control-Allow-Origin: *\r\n"
             << "Cache-Control: no-store\r\n"
             << "Connection: close\r\n\r\n"
             << body;
    const std::string data = response.str();
    send(client, data.data(), data.size(), 0);
}

Genome defaultGenome() {
    Genome genome;
    genome.masks.fill(0b0100);
    genome.sensitivities.fill(0);
    return genome;
}

ChunkSource loadChunkSource(const std::string& genomeDir, const std::string& fileName) {
    ChunkSource result;
    const std::string path = genomeDir + "/" + fileName;
    std::ifstream file(path);
    if (!file) {
        result.genome = defaultGenome();
        result.trained = false;
        result.source = "default fallback (no " + path + " on disk yet)";
        return result;
    }
    Genome genome;
    bool ok = true;
    for (int index = 0; index < N && ok; ++index) {
        int value = 0;
        if (!(file >> value)) { ok = false; break; }
        genome.masks[index] = static_cast<std::uint8_t>(value);
    }
    for (int index = 0; index < N && ok; ++index) {
        int value = 0;
        if (!(file >> value)) { ok = false; break; }
        genome.sensitivities[index] = static_cast<std::uint8_t>(value);
    }
    if (!ok) {
        result.genome = defaultGenome();
        result.trained = false;
        result.source = "default fallback (" + path + " unreadable/wrong size)";
        return result;
    }
    result.genome = genome;
    result.trained = true;
    result.source = path;
    return result;
}

template <typename Arr>
void applyGenome(Arr& maskPort, Arr& sensPort, const Genome& genome) {
    for (int index = 0; index < N; ++index) {
        maskPort[index] = genome.masks[index];
        sensPort[index] = genome.sensitivities[index];
    }
}

void resetTop(Vpipeline& pipe) {
    pipe.clk = 0;
    pipe.rst = 1;
    pipe.eval();
    pipe.clk = 1;
    pipe.eval();
    pipe.clk = 0;
    pipe.eval();
    pipe.rst = 0;
    pipe.eval();
}

void clockOnce(Vpipeline& pipe) {
    pipe.clk = 1;
    pipe.eval();
    pipe.clk = 0;
    pipe.eval();
}

std::string bitsJson(const char* name, std::uint32_t value) {
    std::ostringstream json;
    json << '"' << name << "\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << ((value >> index) & 1);
    }
    json << ']';
    return json.str();
}

std::string numberArray(const char* name, const std::array<std::uint8_t, XS>& values) {
    std::ostringstream json;
    json << '"' << name << "\":[";
    for (int index = 0; index < XS; ++index) {
        if (index) json << ',';
        json << static_cast<unsigned>(values[index]);
    }
    json << ']';
    return json.str();
}

int queryValue(const std::string& request, const std::string& key, int fallback) {
    const std::string marker = key + "=";
    const std::size_t start = request.find(marker);
    if (start == std::string::npos) return fallback;
    try { return std::stoi(request.substr(start + marker.size())); }
    catch (...) { return fallback; }
}

std::string queryString(const std::string& request, const std::string& key, const std::string& fallback) {
    const std::string marker = key + "=";
    const std::size_t start = request.find(marker);
    if (start == std::string::npos) return fallback;
    std::size_t valueStart = start + marker.size();
    std::size_t valueEnd = request.find_first_of(" &\r\n", valueStart);
    return request.substr(valueStart, valueEnd - valueStart);
}

struct PipelineState {
    Vpipeline* pipe = nullptr;
    std::uint64_t step = 0;
    std::string phase = "idle";
    std::array<std::uint8_t, XS> drive{};

    void applyPhase() {
        pipe->link_in_to_mem_en = 0;
        pipe->link_mem_to_cortex_en = 0;
        pipe->link_cortex_to_out_en = 0;
        pipe->probe_en = 0;
        for (int x = 0; x < XS; ++x) {
            pipe->in_chunk_drive[x] = 0;
            pipe->mem_probe[x] = 0;
        }
        if (phase == "write") {
            pipe->link_in_to_mem_en = 1;
            for (int x = 0; x < XS; ++x)
                pipe->in_chunk_drive[x] = drive[x];
        } else if (phase == "silence") {
            // Every link disabled, no drive: this is the retention interval
            // under test for the memory chunk.
        } else if (phase == "probe") {
            pipe->probe_en = 1;
            pipe->link_mem_to_cortex_en = 1;
            pipe->link_cortex_to_out_en = 1;
            for (int x = 0; x < XS; ++x)
                pipe->mem_probe[x] = 1;
        }
    }

    void tick() {
        applyPhase();
        clockOnce(*pipe);
        ++step;
    }
};

std::string stateJson(const PipelineState& state) {
    Vpipeline& pipe = *state.pipe;
    std::ostringstream json;
    json << "{\"xs\":" << XS << ",\"ys\":" << YS
         << ",\"step\":" << state.step
         << ",\"phase\":\"" << state.phase << '"'
         << ",\"link_in_to_mem\":" << (pipe.link_in_to_mem_en ? "true" : "false")
         << ",\"link_mem_to_cortex\":" << (pipe.link_mem_to_cortex_en ? "true" : "false")
         << ",\"link_cortex_to_out\":" << (pipe.link_cortex_to_out_en ? "true" : "false")
         << ",\"probe_en\":" << (pipe.probe_en ? "true" : "false") << ','
         << numberArray("drive", state.drive) << ','
         << bitsJson("input_out", static_cast<std::uint32_t>(pipe.in_chunk_out)) << ','
         << bitsJson("memory_out", static_cast<std::uint32_t>(pipe.mem_out)) << ','
         << bitsJson("cortex_out", static_cast<std::uint32_t>(pipe.cortex_out)) << ','
         << bitsJson("output_out", static_cast<std::uint32_t>(pipe.final_out))
         << '}';
    return json.str();
}

std::string topologyJson(const ChunkSource& in, const ChunkSource& mem,
                         const ChunkSource& cortex, const ChunkSource& out) {
    auto node = [](const char* id, const char* label, const ChunkSource& source) {
        std::ostringstream json;
        json << "{\"id\":\"" << id << "\",\"label\":\"" << label << "\",\"xs\":" << XS
             << ",\"ys\":" << YS << ",\"trained\":" << (source.trained ? "true" : "false")
             << ",\"source\":\"" << source.source << "\"}";
        return json.str();
    };
    std::ostringstream json;
    json << "{\"nodes\":[" << node("input", "Input chunk", in) << ','
         << node("memory", "Memory chunk", mem) << ','
         << node("cortex", "Cortex chunk", cortex) << ','
         << node("output", "Output chunk", out) << "],"
         << "\"edges\":["
            "{\"from\":\"input\",\"to\":\"memory\",\"enable_signal\":\"link_in_to_mem_en\",\"width_bits\":" << XS << "},"
            "{\"from\":\"memory\",\"to\":\"cortex\",\"enable_signal\":\"link_mem_to_cortex_en\",\"width_bits\":" << XS << "},"
            "{\"from\":\"cortex\",\"to\":\"output\",\"enable_signal\":\"link_cortex_to_out_en\",\"width_bits\":" << XS << "}"
            "],"
         << "\"note\":\"Each C_LINK is a single enable-gated pulse bus carrying only the 1-bit south-firing output per column, zero-extended back up to the receiving chunk's 4-bit input row. See CURRICULUM.md section 1.\"}";
    return json.str();
}

void handleClient(int client, PipelineState& state, const ChunkSource& in,
                  const ChunkSource& mem, const ChunkSource& cortex, const ChunkSource& out) {
    char buffer[4096];
    const ssize_t received = recv(client, buffer, sizeof(buffer) - 1, 0);
    if (received <= 0) return;
    buffer[received] = '\0';
    const std::string request(buffer, received);
    const std::string line = request.substr(0, request.find("\r\n"));
    const bool isGet = line.rfind("GET ", 0) == 0;
    const bool isPost = line.rfind("POST ", 0) == 0;
    const bool isOptions = line.rfind("OPTIONS ", 0) == 0;

    if (isOptions) {
        sendResponse(client, "204 No Content", "text/plain", "");
    } else if (isGet && line.find("GET /api/topology") == 0) {
        sendResponse(client, "200 OK", "application/json", topologyJson(in, mem, cortex, out));
    } else if (isGet && line.find("GET /api/state") == 0) {
        sendResponse(client, "200 OK", "application/json", stateJson(state));
    } else if (isPost && line.find("POST /api/reset") == 0) {
        resetTop(*state.pipe);
        state.step = 0;
        state.phase = "idle";
        state.drive.fill(0);
        sendResponse(client, "200 OK", "application/json", stateJson(state));
    } else if (isPost && line.find("POST /api/step") == 0) {
        state.tick();
        sendResponse(client, "200 OK", "application/json", stateJson(state));
    } else if (isPost && line.find("POST /api/phase") == 0) {
        const std::string value = queryString(line, "value", state.phase);
        if (value == "idle" || value == "write" || value == "silence" || value == "probe")
            state.phase = value;
        sendResponse(client, "200 OK", "application/json", stateJson(state));
    } else if (isPost && line.find("POST /api/drive") == 0) {
        const int index = queryValue(line, "index", -1);
        const int value = queryValue(line, "value", 0);
        if (index < 0 || index >= XS || value < 0 || value > 15) {
            sendResponse(client, "400 Bad Request", "text/plain", "index must be 0..XS-1 and value must be 0..15");
        } else {
            state.drive[index] = static_cast<std::uint8_t>(value);
            sendResponse(client, "200 OK", "application/json", stateJson(state));
        }
    } else {
        sendResponse(client, "404 Not Found", "text/plain", "not found");
    }
}
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    VerilatedContext context;
    Vpipeline pipe{&context};

    std::string genomeDir = "genomes";
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const std::string marker = "--genome-dir=";
        if (arg.rfind(marker, 0) == 0)
            genomeDir = arg.substr(marker.size());
    }

    const ChunkSource inSource = loadChunkSource(genomeDir, "input.genome");
    const ChunkSource memSource = loadChunkSource(genomeDir, "memory.genome");
    const ChunkSource cortexSource = loadChunkSource(genomeDir, "cortex.genome");
    const ChunkSource outSource = loadChunkSource(genomeDir, "output.genome");

    applyGenome(pipe.in_mask, pipe.in_sens, inSource.genome);
    applyGenome(pipe.mem_mask, pipe.mem_sens, memSource.genome);
    applyGenome(pipe.cortex_mask, pipe.cortex_sens, cortexSource.genome);
    applyGenome(pipe.out_mask, pipe.out_sens, outSource.genome);

    PipelineState state;
    state.pipe = &pipe;
    resetTop(pipe);

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

    std::cout << "Pipeline (multi-chunk) observatory API: http://localhost:" << PORT << "\n";
    std::cout << "  input: " << inSource.source << (inSource.trained ? " [trained]" : " [untrained]") << "\n";
    std::cout << "  memory: " << memSource.source << (memSource.trained ? " [trained]" : " [untrained]") << "\n";
    std::cout << "  cortex: " << cortexSource.source << (cortexSource.trained ? " [trained]" : " [untrained]") << "\n";
    std::cout << "  output: " << outSource.source << (outSource.trained ? " [trained]" : " [untrained]") << "\n";

    while (!context.gotFinish()) {
        const int client = accept(server, nullptr, nullptr);
        if (client < 0) continue;
        handleClient(client, state, inSource, memSource, cortexSource, outSource);
        close(client);
    }

    close(server);
    return 0;
}
