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
    std::uint8_t fire[CUDA_NEURON_COUNT]{};
    std::uint8_t accu[CUDA_NEURON_COUNT]{};
    std::uint8_t thresh[CUDA_NEURON_COUNT]{};
};

struct CudaTrainingDiagnostics {
    int populationSize = 0;
    int fitnessCounts[CUDA_XS + 1]{};
    int exactBestGenomeCount = 0;
    int uniqueGenomeHashCount = 0;
    int uniqueOutputCount = 0;
    int nonzeroOutputCount = 0;
    int exactWantedOutputCount = 0;
    int outputBitCounts[CUDA_XS]{};
    int directionMaskCounts[4]{};
    int activeFireNeuronCount = 0;
    double averageMaskBits = 0.0;
    double averageSensitivity = 0.0;
};

void evaluatePopulationCuda(const std::vector<CudaGenome>& hostGenomes,
                            std::uint64_t wantedOutput,
                            std::vector<CudaEvaluation>& hostEvaluations);

struct CudaTrainingContext;

CudaTrainingContext* createCudaTrainingContext(
    const std::vector<CudaGenome>& initialPopulation);
void destroyCudaTrainingContext(CudaTrainingContext* context);
void setCudaTrainingInput(CudaTrainingContext* context, std::uint64_t input);
CudaEvaluation stepCudaTraining(CudaTrainingContext* context,
                                std::uint64_t wantedOutput);
void downloadCudaPopulation(CudaTrainingContext* context,
                            std::vector<CudaGenome>& population);
void downloadCudaBestGenome(CudaTrainingContext* context, CudaGenome& genome);
void downloadCudaTrainingDiagnostics(CudaTrainingContext* context,
                                     std::uint64_t wantedOutput,
                                     const CudaGenome& bestGenome,
                                     CudaTrainingDiagnostics& diagnostics);
void reseedCudaTraining(CudaTrainingContext* context,
                        const CudaGenome& best,
                        std::uint64_t input);
