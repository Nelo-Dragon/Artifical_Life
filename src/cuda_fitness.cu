#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "cuda_fitness.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace {

#ifndef CUDA_SIMULATION_CYCLES
#define CUDA_SIMULATION_CYCLES 32
#endif
constexpr int SIMULATION_CYCLES = CUDA_SIMULATION_CYCLES;
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
    bool southLatch[CUDA_NEURON_COUNT] = {};
    std::uint8_t fired[CUDA_NEURON_COUNT] = {};
    std::uint8_t nextInput[CUDA_NEURON_COUNT];
    const int warmupCycles = CUDA_YS - 1;
    int bestFitness = -1;
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
            } else {
                const int sum = accumulator[index] + nextInput[index];
                if (sum >= threshold(genome.sensitivities[index])) {
                    fired[index] = genome.masks[index];
                    southLatch[index] = (genome.masks[index] & 0b0100) != 0;
                    accumulator[index] = 0;
                    refractory[index] = true;
                } else {
                    fired[index] = 0;
                    accumulator[index] = static_cast<std::uint8_t>(sum & 0b111);
                }
            }

            if (index >= (CUDA_YS - 1) * CUDA_XS && southLatch[index])
                output |= std::uint64_t{1} << (index - (CUDA_YS - 1) * CUDA_XS);
        }

        if (cycle < warmupCycles)
            continue;

        const int currentFitness = CUDA_XS - __popc(static_cast<unsigned int>(
            output ^ wantedOutput)) - __popc(static_cast<unsigned int>(
            (output ^ wantedOutput) >> 32));
        float tiebreaker = 0.0f;
        for (int x = 0; x < CUDA_XS; ++x) {
            const int index = (CUDA_YS - 1) * CUDA_XS + x;
            const int thresholdValue = threshold(genome.sensitivities[index]);
            const int accumulatorValue = accumulator[index];
            const bool wanted = ((wantedOutput >> x) & 1) != 0;
            tiebreaker += wanted ? thresholdValue - accumulatorValue : accumulatorValue;
        }
        if (currentFitness > bestFitness) {
            bestFitness = currentFitness;
            best = output;
        }
        fitnessTotal += currentFitness;
        tiebreakerTotal += tiebreaker;
        ++scoredCycles;
    }

    const float averageFitness = fitnessTotal / scoredCycles;
    evaluations[genomeIndex].fitness = static_cast<int>(
        bestFitness);
    evaluations[genomeIndex].rankingFitness = MIN_FITNESS_WEIGHT * bestFitness
        + AVERAGE_FITNESS_WEIGHT * averageFitness;
    evaluations[genomeIndex].tiebreaker = tiebreakerTotal / scoredCycles;
    evaluations[genomeIndex].output = best;
    for (int index = 0; index < CUDA_NEURON_COUNT; ++index) {
        evaluations[genomeIndex].fire[index] = fired[index];
        evaluations[genomeIndex].accu[index] = accumulator[index];
        evaluations[genomeIndex].thresh[index] = threshold(genome.sensitivities[index]);
    }
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

    static CudaGenome* deviceGenomes = nullptr;
    static CudaEvaluation* deviceEvaluations = nullptr;
    static std::size_t deviceCapacity = 0;
    if (populationSize > deviceCapacity) {
        if (deviceEvaluations != nullptr)
            checkCuda(cudaFree(deviceEvaluations), "cudaFree evaluations");
        if (deviceGenomes != nullptr)
            checkCuda(cudaFree(deviceGenomes), "cudaFree genomes");
        checkCuda(cudaMalloc(&deviceGenomes, populationSize * sizeof(CudaGenome)),
                  "cudaMalloc genomes");
        checkCuda(cudaMalloc(&deviceEvaluations, populationSize * sizeof(CudaEvaluation)),
                  "cudaMalloc evaluations");
        deviceCapacity = populationSize;
    }

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
}

    namespace {

    #ifndef CUDA_POPULATION_SIZE
    #define CUDA_POPULATION_SIZE 65536
    #endif
    #ifndef CUDA_ELITE_PERCENT
    #define CUDA_ELITE_PERCENT 20
    #endif
    #ifndef CUDA_MUTATION_RATE
    #define CUDA_MUTATION_RATE 0.04
    #endif

    constexpr int TRAIN_THREADS = 256;

    __global__ void initializeRandomStates(curandStatePhilox4_32_10_t* states,
                                           unsigned long long seed,
                                           int count) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count)
            curand_init(seed, index, 0, &states[index]);
    }

    __global__ void setInputKernel(CudaGenome* genomes, int count, std::uint64_t input) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count)
            return;
        for (int x = 0; x < CUDA_XS; ++x)
            genomes[index].inputs[x] = ((input >> x) & 1) ? 15 : 0;
    }

    __global__ void copyElitesKernel(const CudaGenome* current, CudaGenome* next,
                                     const int* order, int eliteCount) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= eliteCount)
            return;
        next[index] = current[order[index]];
    }

    __global__ void breedKernel(const CudaGenome* current, CudaGenome* next,
                                const int* order,
                                curandStatePhilox4_32_10_t* states,
                                int populationSize, int eliteCount) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < eliteCount || index >= populationSize)
            return;

        curandStatePhilox4_32_10_t state = states[index];
        const int firstIndex = order[curand(&state) % eliteCount];
        const int secondIndex = order[curand(&state) % eliteCount];
        const CudaGenome& first = current[firstIndex];
        const CudaGenome& second = current[secondIndex];
        CudaGenome child{};

        for (int neuron = 0; neuron < CUDA_NEURON_COUNT; ++neuron) {
            child.masks[neuron] =
                (curand(&state) & 1) ? first.masks[neuron] : second.masks[neuron];
            child.sensitivities[neuron] =
                (curand(&state) & 1) ? first.sensitivities[neuron]
                                     : second.sensitivities[neuron];
            if (curand_uniform(&state) < CUDA_MUTATION_RATE)
                child.masks[neuron] ^= static_cast<std::uint8_t>(
                    1u << (curand(&state) % 4));
            if (curand_uniform(&state) < CUDA_MUTATION_RATE * 1.5)
                child.sensitivities[neuron] ^= static_cast<std::uint8_t>(
                    1u << (curand(&state) % 2));
        }
        for (int x = 0; x < CUDA_XS; ++x)
            child.inputs[x] = first.inputs[x];
        next[index] = child;
        states[index] = state;
    }

    __global__ void reseedKernel(CudaGenome* current, CudaGenome* next,
                                 curandStatePhilox4_32_10_t* states,
                                 CudaGenome best, std::uint64_t input,
                                 int populationSize) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= populationSize)
            return;
        if (index == 0) {
            current[index] = best;
            next[index] = best;
            return;
        }
        curandStatePhilox4_32_10_t state = states[index];
        CudaGenome genome{};
        for (int neuron = 0; neuron < CUDA_NEURON_COUNT; ++neuron) {
            genome.masks[neuron] = static_cast<std::uint8_t>(curand(&state) % 16);
            genome.sensitivities[neuron] =
                static_cast<std::uint8_t>(curand(&state) % 4);
        }
        for (int x = 0; x < CUDA_XS; ++x)
            genome.inputs[x] = ((input >> x) & 1) ? 15 : 0;
        current[index] = genome;
        next[index] = genome;
        states[index] = state;
    }

    struct EvaluationOrder {
        const CudaEvaluation* evaluations;
        __host__ __device__ bool operator()(int first, int second) const {
            const CudaEvaluation& a = evaluations[first];
            const CudaEvaluation& b = evaluations[second];
            if (a.fitness != b.fitness)
                return a.fitness > b.fitness;
            if (a.rankingFitness != b.rankingFitness)
                return a.rankingFitness > b.rankingFitness;
            if (a.tiebreaker != b.tiebreaker)
                return a.tiebreaker < b.tiebreaker;
            return first < second;
        }
    };

    } // namespace

    struct CudaTrainingContext {
        CudaGenome* current = nullptr;
        CudaGenome* next = nullptr;
        CudaGenome* evaluatedPopulation = nullptr;
        // Caches the best genome found by the most recent stepCudaTraining()
        // call. Captured before current/next are swapped and before breeding
        // overwrites next, so it stays valid regardless of swap timing —
        // context->order is only meaningful against the population that was
        // just evaluated, and current/next get swapped every step, so
        // re-deriving "best" from order + current after the fact (as
        // downloadCudaBestGenome used to) reads the wrong generation.
        CudaGenome* bestGenome = nullptr;
        CudaEvaluation* evaluations = nullptr;
        int* order = nullptr;
        curandStatePhilox4_32_10_t* randomStates = nullptr;
        int populationSize = 0;
        int eliteCount = 0;
    };

    CudaTrainingContext* createCudaTrainingContext(
        const std::vector<CudaGenome>& initialPopulation) {
        auto* context = new CudaTrainingContext;
        context->populationSize = static_cast<int>(initialPopulation.size());
        context->eliteCount = std::max(1, context->populationSize * CUDA_ELITE_PERCENT / 100);
        checkCuda(cudaMalloc(&context->current,
                             initialPopulation.size() * sizeof(CudaGenome)),
                  "cudaMalloc current population");
        checkCuda(cudaMalloc(&context->next,
                             initialPopulation.size() * sizeof(CudaGenome)),
                  "cudaMalloc next population");
        checkCuda(cudaMalloc(&context->evaluations,
                             initialPopulation.size() * sizeof(CudaEvaluation)),
                  "cudaMalloc evaluations");
        checkCuda(cudaMalloc(&context->order,
                             initialPopulation.size() * sizeof(int)),
                  "cudaMalloc evaluation order");
        checkCuda(cudaMalloc(&context->randomStates,
                             initialPopulation.size() *
                                 sizeof(curandStatePhilox4_32_10_t)),
                  "cudaMalloc random states");
        checkCuda(cudaMalloc(&context->bestGenome, sizeof(CudaGenome)),
                  "cudaMalloc best genome");
        checkCuda(cudaMemcpy(context->current, initialPopulation.data(),
                             initialPopulation.size() * sizeof(CudaGenome),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy initial population");
        // Seed with a real genome so downloadCudaBestGenome() never returns
        // uninitialized memory if it's somehow called before the first step.
        checkCuda(cudaMemcpy(context->bestGenome, initialPopulation.data(),
                             sizeof(CudaGenome), cudaMemcpyHostToDevice),
                  "cudaMemcpy seed best genome");
        const int blocks = (context->populationSize + TRAIN_THREADS - 1) / TRAIN_THREADS;
        initializeRandomStates<<<blocks, TRAIN_THREADS>>>(
            context->randomStates, 0xA17F2026ULL, context->populationSize);
        checkCuda(cudaGetLastError(), "initialize random states launch");
        checkCuda(cudaDeviceSynchronize(), "initialize random states synchronize");
        return context;
    }

    void destroyCudaTrainingContext(CudaTrainingContext* context) {
        if (context == nullptr)
            return;
        cudaFree(context->randomStates);
        cudaFree(context->bestGenome);
        cudaFree(context->order);
        cudaFree(context->evaluations);
        cudaFree(context->next);
        cudaFree(context->current);
        delete context;
    }

    void setCudaTrainingInput(CudaTrainingContext* context, std::uint64_t input) {
        const int blocks = (context->populationSize + TRAIN_THREADS - 1) / TRAIN_THREADS;
        setInputKernel<<<blocks, TRAIN_THREADS>>>(
            context->current, context->populationSize, input);
        setInputKernel<<<blocks, TRAIN_THREADS>>>(
            context->next, context->populationSize, input);
        checkCuda(cudaGetLastError(), "set training input launch");
        checkCuda(cudaDeviceSynchronize(), "set training input synchronize");
    }

    CudaEvaluation stepCudaTraining(CudaTrainingContext* context,
                                    std::uint64_t wantedOutput) {
        const int blocks = (context->populationSize + TRAIN_THREADS - 1) / TRAIN_THREADS;
        evaluatePopulation<<<blocks, TRAIN_THREADS>>>(
            context->current, wantedOutput, context->populationSize,
            context->evaluations);
        context->evaluatedPopulation = context->current;
        checkCuda(cudaGetLastError(), "resident evaluation launch");
        thrust::device_ptr<int> order(context->order);
        thrust::sequence(order, order + context->populationSize);
        thrust::sort(order, order + context->populationSize,
                     EvaluationOrder{context->evaluations});
        int bestIndex = 0;
        checkCuda(cudaMemcpy(&bestIndex, context->order, sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy best index");
        CudaEvaluation best{};
        checkCuda(cudaMemcpy(&best, context->evaluations + bestIndex,
                             sizeof(CudaEvaluation), cudaMemcpyDeviceToHost),
                  "cudaMemcpy best evaluation");
        // Cache the best genome now, while bestIndex is still valid against
        // context->current. Breeding + the swap below make current/order
        // mismatched afterward, so this must happen first.
        checkCuda(cudaMemcpy(context->bestGenome, context->current + bestIndex,
                             sizeof(CudaGenome), cudaMemcpyDeviceToDevice),
                  "cudaMemcpy cache best genome");
        if (best.fitness == CUDA_XS)
            return best;

        copyElitesKernel<<<(context->eliteCount + TRAIN_THREADS - 1) / TRAIN_THREADS,
                           TRAIN_THREADS>>>(context->current, context->next,
                                            context->order, context->eliteCount);
        breedKernel<<<blocks, TRAIN_THREADS>>>(
            context->current, context->next, context->order, context->randomStates,
            context->populationSize, context->eliteCount);
        checkCuda(cudaGetLastError(), "resident breeding launch");
        checkCuda(cudaDeviceSynchronize(), "resident breeding synchronize");
        std::swap(context->current, context->next);
        context->evaluatedPopulation = context->next;
        return best;
    }


    void downloadCudaPopulation(CudaTrainingContext* context,
                                std::vector<CudaGenome>& population) {
        population.resize(context->populationSize);
        checkCuda(cudaMemcpy(population.data(), context->current,
                             population.size() * sizeof(CudaGenome),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy population");
    }

    void downloadCudaBestGenome(CudaTrainingContext* context, CudaGenome& genome) {
        // context->bestGenome is captured inside stepCudaTraining(), before
        // breeding/swap run, so it's always the true best genome from the
        // most recently evaluated generation — unlike re-deriving it from
        // context->order + context->current here, which would read from
        // whichever population current happens to point at *now*, not the
        // one order was computed against.
        checkCuda(cudaMemcpy(&genome, context->bestGenome,
                             sizeof(CudaGenome), cudaMemcpyDeviceToHost),
                  "cudaMemcpy best genome");
    }

    void downloadCudaTrainingDiagnostics(CudaTrainingContext* context,
                                        std::uint64_t wantedOutput,
                                        const CudaGenome& bestGenome,
                                        CudaTrainingDiagnostics& diagnostics) {
        std::vector<CudaGenome> population(context->populationSize);
        std::vector<CudaEvaluation> evaluations(context->populationSize);
        checkCuda(cudaMemcpy(population.data(), context->evaluatedPopulation,
                            population.size() * sizeof(CudaGenome),
                            cudaMemcpyDeviceToHost),
                  "cudaMemcpy diagnostic population");
        checkCuda(cudaMemcpy(evaluations.data(), context->evaluations,
                            evaluations.size() * sizeof(CudaEvaluation),
                            cudaMemcpyDeviceToHost),
                  "cudaMemcpy diagnostic evaluations");

        diagnostics = {};
        diagnostics.populationSize = context->populationSize;
        std::unordered_set<std::uint64_t> genomeHashes;
        std::unordered_set<std::uint64_t> outputs;
        genomeHashes.reserve(population.size());
        outputs.reserve(population.size());
        double totalMaskBits = 0.0;
        double totalSensitivity = 0.0;
        for (std::size_t index = 0; index < population.size(); ++index) {
            const CudaGenome& genome = population[index];
            const CudaEvaluation& evaluation = evaluations[index];
            if (evaluation.fitness >= 0 && evaluation.fitness <= CUDA_XS)
                ++diagnostics.fitnessCounts[evaluation.fitness];
            if (evaluation.output == wantedOutput)
                ++diagnostics.exactWantedOutputCount;
            if (evaluation.output != 0)
                ++diagnostics.nonzeroOutputCount;
            outputs.insert(evaluation.output);
            if (std::memcmp(&genome, &bestGenome, sizeof(CudaGenome)) == 0)
                ++diagnostics.exactBestGenomeCount;

            std::uint64_t hash = 1469598103934665603ULL;
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&genome);
            for (std::size_t byte = 0; byte < sizeof(CudaGenome); ++byte) {
                hash ^= bytes[byte];
                hash *= 1099511628211ULL;
            }
            genomeHashes.insert(hash);
            for (int neuron = 0; neuron < CUDA_NEURON_COUNT; ++neuron) {
                totalMaskBits += __builtin_popcount(genome.masks[neuron]);
                totalSensitivity += genome.sensitivities[neuron];
                for (int direction = 0; direction < 4; ++direction)
                    diagnostics.directionMaskCounts[direction] +=
                        (genome.masks[neuron] >> direction) & 1;
                diagnostics.activeFireNeuronCount +=
                    evaluation.fire[neuron] != 0;
            }
            for (int bit = 0; bit < CUDA_XS; ++bit)
                diagnostics.outputBitCounts[bit] +=
                   static_cast<int>((evaluation.output >> bit) & 1);
        }
        diagnostics.uniqueGenomeHashCount =
            static_cast<int>(genomeHashes.size());
        diagnostics.uniqueOutputCount = static_cast<int>(outputs.size());
        const double genomeCount = static_cast<double>(population.size());
        diagnostics.averageMaskBits =
            totalMaskBits / (genomeCount * CUDA_NEURON_COUNT);
        diagnostics.averageSensitivity =
            totalSensitivity / (genomeCount * CUDA_NEURON_COUNT);
    }

    void reseedCudaTraining(CudaTrainingContext* context,
                            const CudaGenome& best,
                            std::uint64_t input) {
        const int blocks = (context->populationSize + TRAIN_THREADS - 1) / TRAIN_THREADS;
        reseedKernel<<<blocks, TRAIN_THREADS>>>(
            context->current, context->next, context->randomStates, best,
            input, context->populationSize);
        checkCuda(cudaGetLastError(), "reseed launch");
        checkCuda(cudaDeviceSynchronize(), "reseed synchronize");
    }
