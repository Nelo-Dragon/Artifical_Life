#pragma once

#include <cstdint>
#include <vector>

#ifndef TRAIN_XS
#define TRAIN_XS 4
#endif

#ifndef TRAIN_YS
#define TRAIN_YS 4
#endif

constexpr int CUDA_XS = TRAIN_XS;
constexpr int CUDA_YS = TRAIN_YS;
constexpr int CUDA_NEURON_COUNT = CUDA_XS * CUDA_YS;

struct CudaGenome {
    std::uint8_t masks[CUDA_NEURON_COUNT];
    std::uint8_t sensitivities[CUDA_NEURON_COUNT];
    std::uint8_t inputs[CUDA_XS];
};

struct CudaEvaluation {
    int fitness = 0;
    float rankingFitness = 0.0f;
    float tiebreaker = 0.0f;
    std::uint64_t output = 0;
};

void evaluatePopulationCuda(const std::vector<CudaGenome>& hostGenomes,
                            std::uint64_t wantedOutput,
                            std::vector<CudaEvaluation>& hostEvaluations);
