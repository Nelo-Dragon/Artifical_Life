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

    integer source;
    integer target;
    integer x;
    integer y;
    integer output_x;
    
    always @(*) begin

        for (target = 0; target < N; target = target + 1)
            neuron_in[target] = 4'b0000;

        for (x = 0; x < XS; x = x + 1)
            neuron_in[x] = in[x];

        for (source = 0; source < N; source = source + 1) begin
            x = source % XS;
            y = source / XS;

            // Bit 0: north, bit 1: east, bit 2: south, bit 3: west.
            if (neuron_out[source][0] && y > 0)
                neuron_in[source - XS] = neuron_in[source - XS] + 4'd1;
            if (neuron_out[source][1] && x < XS - 1)
                neuron_in[source + 1] = neuron_in[source + 1] + 4'd1;
            if (neuron_out[source][2] && y < YS - 1)
                neuron_in[source + XS] = neuron_in[source + XS] + 4'd1;
            if (neuron_out[source][3] && x > 0)
                neuron_in[source - 1] = neuron_in[source - 1] + 4'd1;
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

    always @(*) begin

        Mask_msk = mask_msk;
        Sens_msk = Sens_msk;
        
        for (output_x = 0; output_x < XS; output_x = output_x + 1)
            out[output_x] = neuron_out[(YS - 1) * XS + output_x][2];
    end


endmodule
