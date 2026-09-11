#include <array>
#include <cstdint>
#include <iostream>
#include <random>

#include "Vchunk.h"
#include "native_chunk.h"
#include "verilated.h"

#ifndef EQUIV_XS
#define EQUIV_XS 4
#endif
#ifndef EQUIV_YS
#define EQUIV_YS 8
#endif

namespace {
constexpr int XS = EQUIV_XS;
constexpr int YS = EQUIV_YS;
constexpr int N = XS * YS;

void resetReference(Vchunk& top) {
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

bool compareState(const Vchunk& reference, const NativeChunk<XS, YS>& native,
                  int caseIndex, int cycle) {
    if (static_cast<std::uint64_t>(reference.out) != native.output()) {
        std::cerr << "output mismatch in case " << caseIndex
                  << ", cycle " << cycle << "\n";
        return false;
    }
    for (int index = 0; index < N; ++index) {
        if (reference.fire_msk[index] != native.fired()[index] ||
            reference.accu_msk[index] != native.accumulator()[index] ||
            reference.thresh_msk[index] != native.thresholdAt(index)) {
            std::cerr << "state mismatch at neuron " << index
                      << " in case " << caseIndex << ", cycle " << cycle
                      << " (reference fire=" << static_cast<int>(reference.fire_msk[index])
                      << ", native fire=" << static_cast<int>(native.fired()[index])
                      << ", reference accu=" << static_cast<int>(reference.accu_msk[index])
                      << ", native accu=" << static_cast<int>(native.accumulator()[index])
                      << ", reference thresh=" << static_cast<int>(reference.thresh_msk[index])
                      << ", native thresh=" << static_cast<int>(native.thresholdAt(index))
                      << ")\n";
            return false;
        }
    }
    return true;
}
}  // namespace

int main() {
    std::mt19937 random(0xA17F2026u);
    std::uniform_int_distribution<int> mask(0, 15);
    std::uniform_int_distribution<int> sensitivity(0, 3);
    std::uniform_int_distribution<int> input(0, 15);

    for (int caseIndex = 0; caseIndex < 256; ++caseIndex) {
        std::array<std::uint8_t, N> masks{};
        std::array<std::uint8_t, N> sensitivities{};
        for (int index = 0; index < N; ++index) {
            masks[index] = static_cast<std::uint8_t>(mask(random));
            sensitivities[index] = static_cast<std::uint8_t>(sensitivity(random));
        }

        Vchunk reference;
        for (int index = 0; index < N; ++index) {
            reference.mask_msk[index] = masks[index];
            reference.sens_msk[index] = sensitivities[index];
        }
        NativeChunk<XS, YS> native;
        native.load(masks, sensitivities);
        native.reset();
        resetReference(reference);

        for (int cycle = 0; cycle < 64; ++cycle) {
            std::array<std::uint8_t, XS> row{};
            for (int x = 0; x < XS; ++x) {
                row[x] = static_cast<std::uint8_t>(input(random));
                reference.in[x] = row[x];
            }
            reference.clk = 1;
            reference.eval();
            reference.clk = 0;
            reference.eval();
            native.step(row);
            if (!compareState(reference, native, caseIndex, cycle))
                return 1;
        }
    }

    std::cout << "Native/Verilator equivalence passed: 256 cases x 64 cycles\n";
    return 0;
}
