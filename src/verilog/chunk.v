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
// The only structure beyond local cardinal routing: an explicit long-range
// pulse pointer that can be wired between independently evolved chunks.
module c_link #(
    parameter WIDTH = 8
) (
    input wire enable,
    input wire [WIDTH-1:0] source_pulse,
    output wire [WIDTH-1:0] destination_pulse
);
    assign destination_pulse = enable ? source_pulse : {WIDTH{1'b0}};
endmodule

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
/* verilator lint_on DECLFILENAME */
