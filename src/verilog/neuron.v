module neuron (
    input wire clk,
    input wire rst,
    input wire [3:0] in_fire,
    output reg [3:0] out_fire,
    output reg [3:0] Mask,
    output reg [1:0] Sens
);

    reg [3:0] mask;
    reg [3:0] accu;
    reg [1:0] sens;
    reg refa;

    wire [3:0] thresh =
    (sens == 2'b00) ? 4'd3 :
    (sens == 2'b01) ? 4'd7 :
    (sens == 2'b10) ? 4'd11 : 4'd15;



    always @(posedge clk) begin
        
        if (rst) begin
            accu <= 4'b0000;
            refa <= 1'b0;
            mask <= Mask;
            sens <= Sens;
            out_fire <= 4'b0000;
        end else begin
            
            if (refa) begin

                refa <= 1'b0;
                out_fire <= 1'b0000;
            end
            else if ((accu + in_fire) > thresh) begin

                out_fire <= mask;
                accu <= 4'b0000;
                refa <= 1'b1;
            end
            else begin
                
                accu <= accu + in_fire;
                out_fire <= 4'b0000;
            end
        end
    end

    always @(*) begin

        Mask <= mask;
        Sens <= sens;
    end

endmodule
