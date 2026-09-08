#include <iostream>
#include <memory>
#include <bitset>
// Verilator creates this header automatically based on the Verilog file name
#include "Vneuron.h"
#include "verilated.h"

int main(int argc, char** argv) {
    // Initialize Verilator internal state
    Verilated::commandArgs(argc, argv);

    // Instantiate your Verilog module as a standard C++ object
    auto top = std::make_unique<Vneuron>();

    // Initial state setup
    top->clk = 0;
    top->mask_in = 0b0001;

    std::cout << "Starting real-time evaluation loop...\n";

    // Simulate 5 clock steps directly in C++ memory
    for (int step = 0; step < 5; ++step) {
        
        // 1. Modify input signals (e.g., your C++ mutation logic)
        top->mask_in = (top->mask_in + 1) & 0x0F;

        // 2. Pulse the clock HIGH
        top->clk = 1;
        top->eval(); // Evaluates the Verilog logic inside memory

        // 3. Pulse the clock LOW
        top->clk = 0;
        top->eval();

        // 4. Read the output state back into C++ instantly
        std::cout << "Step " << step 
                  << " | C++ Input: "  << std::bitset<4>(top->mask_in)
                  << " | Verilog Output: " << std::bitset<4>(top->mask_out) 
                  << std::endl;
    }

    return 0;
}