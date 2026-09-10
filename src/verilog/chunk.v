module chunk #(
    parameter XS = 8,
    parameter YS = 8
)(
    input wire clk,
    input wire rst,
    input wire [3:0] mask_msk [0:(XS*YS)-1],
    input wire [1:0] sens_msk [0:(XS*YS)-1],
    input wire [3:0] in [0:XS-1],
    output wire [3:0] Mask_msk [0:(XS*YS)-1],
    output wire [1:0] Sens_msk [0:(XS*YS)-1],
    output wire [3:0] fire_msk [0:(XS*YS)-1],
    output wire [2:0] accu_msk [0:(XS*YS)-1],
    output wire [2:0] thresh_msk [0:(XS*YS)-1],
    output reg [XS - 1:0] out
);
    
    localparam N = XS * YS;

    wire [3:0] neuron_out [0:N-1];
    wire neuron_south_latched [0:N-1];
    wire [3:0] neuron_in [0:N-1];
    wire [3:0] input_term [0:N-1];
    wire [3:0] north_term [0:N-1];
    wire [3:0] south_term [0:N-1];
    wire [3:0] east_term [0:N-1];
    wire [3:0] west_term [0:N-1];

    integer output_x;

    genvar i;

    generate

        for (i = 0; i < N; i = i + 1) begin : gen_inputs

            if (i < XS) begin : gen_input
                assign input_term[i] = in[i];
            end else begin : gen_no_input
                assign input_term[i] = 4'd0;
            end

            if (i >= XS) begin : gen_north
                assign north_term[i] = neuron_out[i - XS][2] ? 4'd1 : 4'd0;
            end else begin : gen_no_north
                assign north_term[i] = 4'd0;
            end
            if (i < N - XS) begin : gen_south
                assign south_term[i] = neuron_out[i + XS][0] ? 4'd1 : 4'd0;
            end else begin : gen_no_south
                assign south_term[i] = 4'd0;
            end
            if ((i % XS) > 0) begin : gen_west
                assign west_term[i] = neuron_out[i - 1][1] ? 4'd1 : 4'd0;
            end else begin : gen_no_west
                assign west_term[i] = 4'd0;
            end
            if ((i % XS) < XS - 1) begin : gen_east
                assign east_term[i] = neuron_out[i + 1][3] ? 4'd1 : 4'd0;
            end else begin : gen_no_east
                assign east_term[i] = 4'd0;
            end

            assign neuron_in[i] = input_term[i] + north_term[i] + south_term[i]
                + east_term[i] + west_term[i];
        end

        for (i = 0; i < N; i = i + 1) begin : gen_neurons

            neuron n (
                .clk(clk),
                .rst(rst),
                .in_fire(neuron_in[i]),
                .mask_in(mask_msk[i]),
                .sens_in(sens_msk[i]),
                .out_fire(neuron_out[i]),
                .Mask(Mask_msk[i]),
                .Sens(Sens_msk[i])
                ,.Accu(accu_msk[i])
                ,.Thresh(thresh_msk[i])
                ,.south_latched(neuron_south_latched[i])
            );
            assign fire_msk[i] = neuron_out[i];
        end
    endgenerate

    always @(*) begin
        for (output_x = 0; output_x < XS; output_x = output_x + 1)
            out[output_x] = neuron_south_latched[(YS - 1) * XS + output_x];
    end


endmodule

/* verilator lint_off DECLFILENAME */
// Serial byte-to-pulse adapter. A byte is emitted least-significant bit first
// as WIDTH-wide pulse values (0 or 15), matching the chunk input convention.
module in_chunk #(
    parameter WIDTH = 8
) (
    input wire clk,
    input wire rst,
    input wire [7:0] data_in,
    input wire data_valid,
    output wire data_ready,
    output reg [WIDTH-1:0] pulse,
    output reg pulse_valid
);
    reg [7:0] shift_reg;
    reg [3:0] remaining;

    assign data_ready = (remaining == 0);

    always @(posedge clk) begin
        if (rst) begin
            shift_reg <= 8'b0;
            remaining <= 0;
            pulse <= {WIDTH{1'b0}};
            pulse_valid <= 1'b0;
        end else begin
            pulse_valid <= 1'b0;
            pulse <= {WIDTH{1'b0}};
            if (remaining != 0) begin
                pulse <= shift_reg[0] ? {WIDTH{1'b1}} : {WIDTH{1'b0}};
                pulse_valid <= 1'b1;
                shift_reg <= {1'b0, shift_reg[7:1]};
                remaining <= remaining - 1'b1;
            end else if (data_valid) begin
                shift_reg <= data_in;
                remaining <= 8;
            end
        end
    end
endmodule

// Pulse-to-serial byte adapter. It samples one pulse vector per cycle and
// presents a byte after WIDTH samples, least-significant bit first.
module out_chunk #(
    parameter WIDTH = 8
) (
    input wire clk,
    input wire rst,
    input wire [WIDTH-1:0] pulse,
    input wire pulse_valid,
    output reg [7:0] data_out,
    output reg data_valid
);
    reg [7:0] shift_reg;
    reg [3:0] count;

    always @(posedge clk) begin
        if (rst) begin
            shift_reg <= 0;
            count <= 0;
            data_out <= 0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0;
            if (pulse_valid) begin
                shift_reg[count] <= |pulse;
                if (count == WIDTH - 1) begin
                    data_out[count] <= |pulse;
                    data_out <= shift_reg;
                    data_out[count] <= |pulse;
                    data_valid <= 1'b1;
                    count <= 0;
                end else begin
                    count <= count + 1'b1;
                end
            end
        end
    end
endmodule

// Explicit long-range connection between chunk regions. Keeping this as a
// separate primitive lets a network provide sparse links without changing
// local cardinal-neighbor routing.
module c_link #(
    parameter WIDTH = 8
) (
    input wire enable,
    input wire [WIDTH-1:0] source_pulse,
    output wire [WIDTH-1:0] destination_pulse
);
    assign destination_pulse = enable ? source_pulse : {WIDTH{1'b0}};
endmodule

// Bank of sparse long-range links for a multi-chunk interconnect.
module c_link_bank #(
    parameter WIDTH = 8,
    parameter LINKS = 4
) (
    input wire [WIDTH-1:0] source_pulse [0:LINKS-1],
    input wire [LINKS-1:0] enable,
    output wire [WIDTH-1:0] destination_pulse [0:LINKS-1]
);
    genvar link_index;
    generate
        for (link_index = 0; link_index < LINKS; link_index = link_index + 1) begin : gen_links
            c_link #(.WIDTH(WIDTH)) link (
                .enable(enable[link_index]),
                .source_pulse(source_pulse[link_index]),
                .destination_pulse(destination_pulse[link_index])
            );
        end
    endgenerate
endmodule

// Small synchronous byte memory region for experiments that need explicit
// write/read state around a trainable chunk.
module memory_region #(
    parameter ADDRESS_BITS = 8
) (
    input wire clk,
    input wire rst,
    input wire write_valid,
    input wire [ADDRESS_BITS-1:0] write_address,
    input wire [7:0] write_data,
    input wire read_valid,
    input wire [ADDRESS_BITS-1:0] read_address,
    output reg [7:0] read_data,
    output reg read_data_valid
);
    localparam DEPTH = 1 << ADDRESS_BITS;
    reg [7:0] memory [0:DEPTH-1];
    integer index;

    always @(posedge clk) begin
        if (rst) begin
            read_data <= 0;
            read_data_valid <= 1'b0;
            for (index = 0; index < DEPTH; index = index + 1)
                memory[index] <= 0;
        end else begin
            read_data_valid <= 1'b0;
            if (write_valid)
                memory[write_address] <= write_data;
            if (read_valid) begin
                read_data <= memory[read_address];
                read_data_valid <= 1'b1;
            end
        end
    end
endmodule

// Cortex boundary: a clocked byte stream between the evolvable chunk region
// and external memory/output regions. The internal chunk remains responsible
// for pulse processing; this module provides a stable byte handshake.
module cortex_region (
    input wire clk,
    input wire rst,
    input wire [7:0] data_in,
    input wire data_valid,
    output wire data_ready,
    output reg [7:0] data_out,
    output reg data_out_valid
);
    assign data_ready = !data_out_valid;

    always @(posedge clk) begin
        if (rst) begin
            data_out <= 0;
            data_out_valid <= 1'b0;
        end else begin
            if (data_out_valid)
                data_out_valid <= 1'b0;
            else if (data_valid) begin
                data_out <= data_in;
                data_out_valid <= 1'b1;
            end
        end
    end
endmodule

// Selects whether a cortex byte is committed to memory or emitted as the
// final output stream.
module output_region (
    input wire clk,
    input wire rst,
    input wire to_memory,
    input wire [7:0] cortex_data,
    input wire cortex_valid,
    input wire [7:0] memory_data,
    input wire memory_valid,
    output wire memory_write_valid,
    output wire [7:0] memory_write_data,
    output reg [7:0] output_data,
    output reg output_valid
);
    assign memory_write_valid = cortex_valid && to_memory;
    assign memory_write_data = cortex_data;

    always @(posedge clk) begin
        if (rst) begin
            output_data <= 0;
            output_valid <= 1'b0;
        end else begin
            output_valid <= 1'b0;
            if (!to_memory && cortex_valid) begin
                output_data <= cortex_data;
                output_valid <= 1'b1;
            end else if (to_memory && memory_valid) begin
                output_data <= memory_data;
                output_valid <= 1'b1;
            end
        end
    end
endmodule

// Minimal memory-cortex-output pipeline. The existing evolvable chunk can be
// placed between these boundaries; this wrapper supplies a deterministic
// byte-stream proof-of-concept for integration and hardware tests.
module cognitive_pipeline #(
    parameter ADDRESS_BITS = 8
) (
    input wire clk,
    input wire rst,
    input wire to_memory,
    input wire [ADDRESS_BITS-1:0] write_address,
    input wire [ADDRESS_BITS-1:0] read_address,
    input wire [7:0] input_data,
    input wire input_valid,
    output wire input_ready,
    input wire read_valid,
    output wire [7:0] output_data,
    output wire output_valid
);
    wire [7:0] cortex_data;
    wire cortex_valid;
    wire memory_write_valid;
    wire [7:0] memory_write_data;
    wire [7:0] memory_data;
    wire memory_valid;

    cortex_region cortex (
        .clk(clk),
        .rst(rst),
        .data_in(input_data),
        .data_valid(input_valid),
        .data_ready(input_ready),
        .data_out(cortex_data),
        .data_out_valid(cortex_valid)
    );
    memory_region #(.ADDRESS_BITS(ADDRESS_BITS)) memory (
        .clk(clk),
        .rst(rst),
        .write_valid(memory_write_valid),
        .write_address(write_address),
        .write_data(memory_write_data),
        .read_valid(read_valid),
        .read_address(read_address),
        .read_data(memory_data),
        .read_data_valid(memory_valid)
    );
    output_region output_router (
        .clk(clk),
        .rst(rst),
        .to_memory(to_memory),
        .cortex_data(cortex_data),
        .cortex_valid(cortex_valid),
        .memory_data(memory_data),
        .memory_valid(memory_valid),
        .memory_write_valid(memory_write_valid),
        .memory_write_data(memory_write_data),
        .output_data(output_data),
        .output_valid(output_valid)
    );
endmodule
/* verilator lint_on DECLFILENAME */
