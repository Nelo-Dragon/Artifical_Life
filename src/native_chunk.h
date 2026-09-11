#pragma once

#include <array>
#include <cstdint>

template <int XS, int YS>
class NativeChunk {
public:
    static constexpr int neuronCount = XS * YS;

    void load(const std::array<std::uint8_t, neuronCount>& masks,
              const std::array<std::uint8_t, neuronCount>& sensitivities) {
        masks_ = masks;
        sensitivities_ = sensitivities;
    }

    void reset() {
        accumulator_.fill(0);
        refractory_.fill(false);
        fired_.fill(0);
        southLatch_.fill(false);
    }

    void step(const std::array<std::uint8_t, XS>& input) {
        std::array<std::uint8_t, neuronCount> nextInput{};
        for (int index = 0; index < neuronCount; ++index) {
            int pulses = index < XS ? input[index] : 0;
            const int x = index % XS;
            const int y = index / XS;
            if (y > 0)
                pulses += (fired_[index - XS] & 0b0100) != 0;
            if (y < YS - 1)
                pulses += (fired_[index + XS] & 0b0001) != 0;
            if (x > 0)
                pulses += (fired_[index - 1] & 0b0010) != 0;
            if (x < XS - 1)
                pulses += (fired_[index + 1] & 0b1000) != 0;
            nextInput[index] = static_cast<std::uint8_t>(pulses & 0xf);
        }

        for (int index = 0; index < neuronCount; ++index) {
            if (refractory_[index]) {
                refractory_[index] = false;
                fired_[index] = 0;
                continue;
            }

            const int thresholdValue = threshold(sensitivities_[index]);
            const int sum = accumulator_[index] + nextInput[index];
            if (sum >= thresholdValue) {
                fired_[index] = masks_[index];
                southLatch_[index] = (masks_[index] & 0b0100) != 0;
                accumulator_[index] = 0;
                refractory_[index] = true;
            } else {
                fired_[index] = 0;
                accumulator_[index] = static_cast<std::uint8_t>(sum & 0b111);
            }
        }
    }

    std::uint64_t output() const {
        std::uint64_t result = 0;
        for (int x = 0; x < XS; ++x)
            result |= static_cast<std::uint64_t>(
                          southLatch_[(YS - 1) * XS + x])
                      << x;
        return result;
    }

    const auto& fired() const { return fired_; }
    const auto& accumulator() const { return accumulator_; }

    std::uint8_t thresholdAt(int index) const {
        return static_cast<std::uint8_t>(threshold(sensitivities_[index]));
    }

private:
    static int threshold(std::uint8_t sensitivity) {
        return sensitivity == 0 ? 1 : sensitivity == 1 ? 3 : sensitivity == 2 ? 5 : 7;
    }

    std::array<std::uint8_t, neuronCount> masks_{};
    std::array<std::uint8_t, neuronCount> sensitivities_{};
    std::array<std::uint8_t, neuronCount> accumulator_{};
    std::array<std::uint8_t, neuronCount> fired_{};
    std::array<bool, neuronCount> refractory_{};
    std::array<bool, neuronCount> southLatch_{};
};
