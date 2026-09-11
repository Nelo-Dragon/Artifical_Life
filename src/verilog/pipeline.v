/* verilator lint_off DECLFILENAME */
// System-level multi-chunk wrapper for the staged curriculum described in
// CURRICULUM.md. It instantiates four independently-evolved chunk regions
// (memory / input / cortex / output) and wires them together with the
// existing `c_link` primitive rather than inventing a new interconnect.
//
// Assumption (documented in CURRICULUM.md): all four chunks share the same
// XS x YS grid size. C_LINK only carries the one south-firing bit per column
// that a chunk's `out` port exposes (not the full 4-bit per-column value a
// chunk's `in` port accepts), so every link is a 1-bit-per-column bus,
// zero-extended back up to 4 bits at the receiving chunk's `in` port.
//
// The harness (curriculum_train.cpp) drives the phase-control signals
// (probe_en / link enables / in_chunk_drive / mem_probe) directly; this
// module only defines the fixed wiring table between chunks, matching the
// README's description of what C_LINK is (an interconnect primitive) versus
// what a system wrapper must add (an actual link table and instantiation).
module pipeline #(
    parameter XS = 4,
    parameter YS = 8
) (
    input wire clk,
    input wire rst,

    // Genomes for each of the four chunks.
    input wire [3:0] in_mask   [0:(XS*YS)-1],
    input wire [1:0] in_sens   [0:(XS*YS)-1],
    input wire [3:0] mem_mask  [0:(XS*YS)-1],
    input wire [1:0] mem_sens  [0:(XS*YS)-1],
    input wire [3:0] cortex_mask [0:(XS*YS)-1],
    input wire [1:0] cortex_sens [0:(XS*YS)-1],
    input wire [3:0] out_mask  [0:(XS*YS)-1],
    input wire [1:0] out_sens  [0:(XS*YS)-1],

    // External drive for the input chunk (the task's key/query pattern).
    input wire [3:0] in_chunk_drive [0:XS-1],

    // External probe stimulus applied straight to the memory chunk's input
    // row, bypassing the input chunk, during the probe phase.
    input wire [3:0] mem_probe [0:XS-1],
    input wire probe_en,

    // Link enables: the harness turns these on/off to select which phase
    // (write / read / compute / emit) is currently active.
    input wire link_in_to_mem_en,
    input wire link_mem_to_cortex_en,
    input wire link_cortex_to_out_en,

    output wire [XS-1:0] in_chunk_out,
    output wire [XS-1:0] mem_out,
    output wire [XS-1:0] cortex_out,
    output wire [XS-1:0] final_out
);

    localparam N = XS * YS;

    wire [3:0] in_chunk_in [0:XS-1];
    wire [3:0] mem_in [0:XS-1];
    wire [3:0] cortex_in [0:XS-1];
    wire [3:0] out_in [0:XS-1];

    // Unused per-chunk diagnostic outputs (only the output chunk's are
    // exposed above, since that is what the joint fine-tune tiebreaker uses).
    wire [3:0] in_Mask_unused [0:N-1];
    wire [1:0] in_Sens_unused [0:N-1];
    wire [3:0] in_fire_unused [0:N-1];
    wire [2:0] in_accu_unused [0:N-1];
    wire [2:0] in_thresh_unused [0:N-1];

    wire [3:0] mem_Mask_unused [0:N-1];
    wire [1:0] mem_Sens_unused [0:N-1];
    wire [3:0] mem_fire_unused [0:N-1];
    wire [2:0] mem_accu_unused [0:N-1];
    wire [2:0] mem_thresh_unused [0:N-1];

    wire [3:0] cortex_Mask_unused [0:N-1];
    wire [1:0] cortex_Sens_unused [0:N-1];
    wire [3:0] cortex_fire_unused [0:N-1];
    wire [2:0] cortex_accu_unused [0:N-1];
    wire [2:0] cortex_thresh_unused [0:N-1];

    wire [3:0] out_Mask_unused [0:N-1];
    wire [1:0] out_Sens_unused [0:N-1];
    wire [3:0] out_fire_unused [0:N-1];
    wire [2:0] out_accu_unused [0:N-1];
    wire [2:0] out_thresh_unused [0:N-1];

    genvar column;
    generate
        for (column = 0; column < XS; column = column + 1) begin : gen_columns
            assign in_chunk_in[column] = in_chunk_drive[column];
        end
    endgenerate

    chunk #(.XS(XS), .YS(YS)) in_chunk_inst (
        .clk(clk), .rst(rst),
        .mask_msk(in_mask), .sens_msk(in_sens),
        .in(in_chunk_in),
        .Mask_msk(in_Mask_unused), .Sens_msk(in_Sens_unused),
        .fire_msk(in_fire_unused), .accu_msk(in_accu_unused), .thresh_msk(in_thresh_unused),
        .out(in_chunk_out)
    );

    // link 1: input chunk -> memory chunk (write phase)
    wire [XS-1:0] link_in_to_mem_dest;
    c_link #(.WIDTH(XS)) link_in_to_mem (
        .enable(link_in_to_mem_en),
        .source_pulse(in_chunk_out),
        .destination_pulse(link_in_to_mem_dest)
    );

    generate
        for (column = 0; column < XS; column = column + 1) begin : gen_mem_in
            assign mem_in[column] = probe_en
                ? mem_probe[column]
                : {3'b000, link_in_to_mem_dest[column]};
        end
    endgenerate

    chunk #(.XS(XS), .YS(YS)) mem_chunk_inst (
        .clk(clk), .rst(rst),
        .mask_msk(mem_mask), .sens_msk(mem_sens),
        .in(mem_in),
        .Mask_msk(mem_Mask_unused), .Sens_msk(mem_Sens_unused),
        .fire_msk(mem_fire_unused), .accu_msk(mem_accu_unused), .thresh_msk(mem_thresh_unused),
        .out(mem_out)
    );

    // link 2: memory chunk -> cortex chunk (read phase)
    wire [XS-1:0] link_mem_to_cortex_dest;
    c_link #(.WIDTH(XS)) link_mem_to_cortex (
        .enable(link_mem_to_cortex_en),
        .source_pulse(mem_out),
        .destination_pulse(link_mem_to_cortex_dest)
    );

    generate
        for (column = 0; column < XS; column = column + 1) begin : gen_cortex_in
            assign cortex_in[column] = {3'b000, link_mem_to_cortex_dest[column]};
        end
    endgenerate

    chunk #(.XS(XS), .YS(YS)) cortex_chunk_inst (
        .clk(clk), .rst(rst),
        .mask_msk(cortex_mask), .sens_msk(cortex_sens),
        .in(cortex_in),
        .Mask_msk(cortex_Mask_unused), .Sens_msk(cortex_Sens_unused),
        .fire_msk(cortex_fire_unused), .accu_msk(cortex_accu_unused), .thresh_msk(cortex_thresh_unused),
        .out(cortex_out)
    );

    // link 3: cortex chunk -> output chunk (emit phase)
    wire [XS-1:0] link_cortex_to_out_dest;
    c_link #(.WIDTH(XS)) link_cortex_to_out (
        .enable(link_cortex_to_out_en),
        .source_pulse(cortex_out),
        .destination_pulse(link_cortex_to_out_dest)
    );

    generate
        for (column = 0; column < XS; column = column + 1) begin : gen_out_in
            assign out_in[column] = {3'b000, link_cortex_to_out_dest[column]};
        end
    endgenerate

    chunk #(.XS(XS), .YS(YS)) out_chunk_inst (
        .clk(clk), .rst(rst),
        .mask_msk(out_mask), .sens_msk(out_sens),
        .in(out_in),
        .Mask_msk(out_Mask_unused), .Sens_msk(out_Sens_unused),
        .fire_msk(out_fire_unused), .accu_msk(out_accu_unused), .thresh_msk(out_thresh_unused),
        .out(final_out)
    );

endmodule
/* verilator lint_on DECLFILENAME */
