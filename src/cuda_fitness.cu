#include <cuda_runtime.h>

#include "cuda_fitness.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int SIMULATION_CYCLES = 64;
constexpr float MIN_FITNESS_WEIGHT = 0.6f;
constexpr float AVERAGE_FITNESS_WEIGHT = 0.4f;

__device__ int threshold(std::uint8_t sensitivity) {
    return sensitivity == 0 ? 1 : sensitivity == 1 ? 3 : sensitivity == 2 ? 5 : 7;
}

__global__ void evaluatePopulation(const CudaGenome* genomes,
                                   std::uint64_t wantedOutput,
                                   int populationSize,
                                   CudaEvaluation* evaluations) {
    const int genomeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (genomeIndex >= populationSize)
        return;

    const CudaGenome& genome = genomes[genomeIndex];
    std::uint8_t accumulator[CUDA_NEURON_COUNT] = {};
    bool refractory[CUDA_NEURON_COUNT] = {};
    std::uint8_t fired[CUDA_NEURON_COUNT] = {};
    std::uint8_t nextInput[CUDA_NEURON_COUNT];
    const int warmupCycles = CUDA_YS - 1;
    int minimumFitness = CUDA_XS;
    int minimumError = CUDA_XS;
    int scoredCycles = 0;
    float fitnessTotal = 0.0f;
    float tiebreakerTotal = 0.0f;
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

        if (cycle < warmupCycles)
            continue;

        const int error = __popc(static_cast<unsigned int>(
            output ^ wantedOutput)) + __popc(static_cast<unsigned int>(
            (output ^ wantedOutput) >> 32));
        const int currentFitness = CUDA_XS - error;
        float tiebreaker = 0.0f;
        for (int x = 0; x < CUDA_XS; ++x) {
            const int index = (CUDA_YS - 1) * CUDA_XS + x;
            const int thresholdValue = threshold(genome.sensitivities[index]);
            const int accumulatorValue = accumulator[index];
            const bool wanted = ((wantedOutput >> x) & 1) != 0;
            tiebreaker += wanted ? thresholdValue - accumulatorValue : accumulatorValue;
        }
        if (currentFitness < minimumFitness)
            minimumFitness = currentFitness;
        if (error < minimumError) {
            minimumError = error;
            best = output;
        }
        fitnessTotal += currentFitness;
        tiebreakerTotal += tiebreaker;
        ++scoredCycles;
    }

    const float averageFitness = fitnessTotal / scoredCycles;
    evaluations[genomeIndex].fitness = static_cast<int>(
        MIN_FITNESS_WEIGHT * minimumFitness + AVERAGE_FITNESS_WEIGHT * averageFitness);
    evaluations[genomeIndex].rankingFitness = MIN_FITNESS_WEIGHT * minimumFitness
        + AVERAGE_FITNESS_WEIGHT * averageFitness;
    evaluations[genomeIndex].tiebreaker = tiebreakerTotal / scoredCycles;
    evaluations[genomeIndex].output = best;
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
                            std::vector<CudaEvaluation>& hostEvaluations) {
    const std::size_t populationSize = hostGenomes.size();
    hostEvaluations.resize(populationSize);

    CudaGenome* deviceGenomes = nullptr;
    CudaEvaluation* deviceEvaluations = nullptr;
    checkCuda(cudaMalloc(&deviceGenomes, populationSize * sizeof(CudaGenome)), "cudaMalloc genomes");
    checkCuda(cudaMalloc(&deviceEvaluations, populationSize * sizeof(CudaEvaluation)), "cudaMalloc evaluations");

    checkCuda(cudaMemcpy(deviceGenomes, hostGenomes.data(),
                         populationSize * sizeof(CudaGenome), cudaMemcpyHostToDevice),
              "cudaMemcpy genomes");

    constexpr int threadsPerBlock = 128;
    const int blocks = static_cast<int>((populationSize + threadsPerBlock - 1) / threadsPerBlock);
    evaluatePopulation<<<blocks, threadsPerBlock>>>(
        deviceGenomes, wantedOutput, static_cast<int>(populationSize),
        deviceEvaluations);
    checkCuda(cudaGetLastError(), "evaluatePopulation launch");
    checkCuda(cudaDeviceSynchronize(), "evaluatePopulation synchronize");

    checkCuda(cudaMemcpy(hostEvaluations.data(), deviceEvaluations,
                         populationSize * sizeof(CudaEvaluation), cudaMemcpyDeviceToHost),
              "cudaMemcpy evaluations");

    cudaFree(deviceEvaluations);
    cudaFree(deviceGenomes);
}
