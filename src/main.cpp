#include <bitset>
#include <iostream>
#include <memory>

#include "Vchunk.h"
#include "verilated.h"

constexpr int XS = 64;
constexpr int YS = 32;
constexpr int NEURON_COUNT = XS * YS;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto top = std::make_unique<Vchunk>();

    top->clk = 0;
    top->rst = 1;
    for (int index = 0; index < NEURON_COUNT; ++index) {
        top->mask_msk[index] = 0b0100;
        top->sens_msk[index] = 0b00;
    }
    for (int index = 0; index < XS; ++index)
        top->in[index] = 0;

    top->eval();
    top->clk = 1;
    top->eval();
    top->clk = 0;
    top->eval();
    top->rst = 0;

    std::cout << "Starting chunk simulation...\n";
    for (int cycle = 0; cycle < 1000; ++cycle) {
        for (int index = 0; index < XS; ++index)
            top->in[index] = 0b1111;

        top->clk = 1;
        top->eval();
        top->clk = 0;
        top->eval();

        std::cout << "Cycle " << cycle
                  << " | output: " << std::bitset<XS>(top->out)
                  << std::endl;
    }

    return 0;
}