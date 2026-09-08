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
    output reg [XS - 1:0] out
);
    
    localparam N = XS * YS;

    wire [3:0] neuron_out [0:N-1];
    reg [3:0] neuron_in [0:N-1];

    integer target;
    integer output_x;
    
    always_comb begin

        for (target = 0; target < N; target = target + 1) begin
            // Bit 0: north, bit 1: east, bit 2: south, bit 3: west.
            neuron_in[target] = (target < XS ? in[target] : 4'd0)
                + (target >= XS && neuron_out[target - XS][2] ? 4'd1 : 4'd0)
                + (target < N - XS && neuron_out[target + XS][0] ? 4'd1 : 4'd0)
                + (target % XS > 0 && neuron_out[target - 1][1] ? 4'd1 : 4'd0)
                + (target % XS < XS - 1 && neuron_out[target + 1][3] ? 4'd1 : 4'd0);
        end
    end

    genvar i;

    generate

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
            );
        end
    endgenerate

    always_comb begin
        for (output_x = 0; output_x < XS; output_x = output_x + 1)
            out[output_x] = neuron_out[(YS - 1) * XS + output_x][2];
    end


endmodule
