#include <cuda_runtime.h>

#include "cuda_fitness.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int SIMULATION_CYCLES = 64;

__device__ int threshold(std::uint8_t sensitivity) {
    return sensitivity == 0 ? 1 : sensitivity == 1 ? 3 : sensitivity == 2 ? 5 : 7;
}

__global__ void evaluatePopulation(const CudaGenome* genomes,
                                   std::uint64_t wantedOutput,
                                   int populationSize,
                                   int* fitness,
                                   std::uint64_t* bestOutput) {
    const int genomeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIndex >= populationSize)
        return;

    const CudaGenome& genome = genomes[genomeIndex];
    std::uint8_t accumulator[CUDA_NEURON_COUNT] = {};
    bool refractory[CUDA_NEURON_COUNT] = {};
    std::uint8_t fired[CUDA_NEURON_COUNT] = {};
    std::uint8_t nextInput[CUDA_NEURON_COUNT];
    int bestFitness = -1;
    std::uint64_t best = 0;

    for (int cycle = 0; cycle < SIMULATION_CYCLES; ++cycle) {
        for (int index = 0; index < CUDA_NEURON_COUNT; ++index) {
            const int x = index % CUDA_XS;
            const int y = index / CUDA_XS;
            int pulses = index < CUDA_XS ? genome.inputs[index] : 0;

            if (y > 0)
                pulses += (fired[index - CUDA_XS] & 0b0100) != 0;
            if (y < CUDA_YS - 1)
                pulses += (fired[index + CUDA_XS] & 0b0001) != 0;
            if (x > 0)
                pulses += (fired[index - 1] & 0b0010) != 0;
            if (x < CUDA_XS - 1)
                pulses += (fired[index + 1] & 0b1000) != 0;

            nextInput[index] = static_cast<std::uint8_t>(pulses);
        }

        std::uint64_t output = 0;
        for (int index = 0; index < CUDA_NEURON_COUNT; ++index) {
            if (refractory[index]) {
                refractory[index] = false;
                fired[index] = 0;
                continue;
            }

            const int sum = accumulator[index] + nextInput[index];
            if (sum >= threshold(genome.sensitivities[index])) {
                fired[index] = genome.masks[index];
                accumulator[index] = 0;
                refractory[index] = true;
            } else {
                fired[index] = 0;
                accumulator[index] = static_cast<std::uint8_t>(sum & 0b111);
            }

            if (index >= (CUDA_YS - 1) * CUDA_XS && (fired[index] & 0b0100) != 0)
                output |= std::uint64_t{1} << (index - (CUDA_YS - 1) * CUDA_XS);
        }

        const int currentFitness = CUDA_XS - (__popc(static_cast<unsigned int>(
            output ^ wantedOutput)) + __popc(static_cast<unsigned int>(
            (output ^ wantedOutput) >> 32)));
        if (currentFitness > bestFitness) {
            bestFitness = currentFitness;
            best = output;
        }
    }

    fitness[genomeIndex] = bestFitness;
    bestOutput[genomeIndex] = best;
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

// Evaluates one population generation. Breeding and file I/O remain on the CPU.
void evaluatePopulationCuda(const std::vector<CudaGenome>& hostGenomes,
                            std::uint64_t wantedOutput,
                            std::vector<int>& hostFitness,
                            std::vector<std::uint64_t>& hostBestOutput) {
    const std::size_t populationSize = hostGenomes.size();
    hostFitness.resize(populationSize);
    hostBestOutput.resize(populationSize);

    CudaGenome* deviceGenomes = nullptr;
    int* deviceFitness = nullptr;
    std::uint64_t* deviceBestOutput = nullptr;
    checkCuda(cudaMalloc(&deviceGenomes, populationSize * sizeof(CudaGenome)), "cudaMalloc genomes");
    checkCuda(cudaMalloc(&deviceFitness, populationSize * sizeof(int)), "cudaMalloc fitness");
    checkCuda(cudaMalloc(&deviceBestOutput, populationSize * sizeof(std::uint64_t)), "cudaMalloc outputs");

    checkCuda(cudaMemcpy(deviceGenomes, hostGenomes.data(),
                         populationSize * sizeof(CudaGenome), cudaMemcpyHostToDevice),
              "cudaMemcpy genomes");

    constexpr int threadsPerBlock = 128;
    const int blocks = static_cast<int>((populationSize + threadsPerBlock - 1) / threadsPerBlock);
    evaluatePopulation<<<blocks, threadsPerBlock>>>(
        deviceGenomes, wantedOutput, static_cast<int>(populationSize),
        deviceFitness, deviceBestOutput);
    checkCuda(cudaGetLastError(), "evaluatePopulation launch");
    checkCuda(cudaDeviceSynchronize(), "evaluatePopulation synchronize");

    checkCuda(cudaMemcpy(hostFitness.data(), deviceFitness,
                         populationSize * sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy fitness");
    checkCuda(cudaMemcpy(hostBestOutput.data(), deviceBestOutput,
                         populationSize * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy outputs");

    cudaFree(deviceBestOutput);
    cudaFree(deviceFitness);
    cudaFree(deviceGenomes);
}
