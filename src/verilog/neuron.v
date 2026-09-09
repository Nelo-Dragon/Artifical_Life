module neuron (
    input wire clk,
    input wire rst,
    input wire [1:0] sens_in,
    input wire [3:0] mask_in,
    input wire [3:0] in_fire,
    output reg [3:0] out_fire,
    output reg [3:0] Mask,
    output reg [1:0] Sens
);

    reg [3:0] mask;
    reg [2:0] accu;
    reg [1:0] sens;
    reg refa;

    wire [2:0] thresh =
    (sens == 2'b00) ? 3'd1 :
    (sens == 2'b01) ? 3'd3 :
    (sens == 2'b10) ? 3'd5 : 3'd7;
    wire [4:0] accu_sum = {2'b00, accu} + {1'b0, in_fire};



    always @(posedge clk) begin
        
        if (rst) begin

            accu <= 3'b000;
            refa <= 1'b0;
            mask <= mask_in;
            sens <= sens_in;
            out_fire <= 4'b0000;
        end else begin
            
            if (refa) begin

                refa <= 1'b0;
                out_fire <= 4'b0000;
            end
            else if (accu_sum >= {2'b00, thresh}) begin

                out_fire <= mask;
                accu <= 3'b000;
                refa <= 1'b1;
            end
            else begin
                
                accu <= accu_sum[2:0];
                out_fire <= 4'b0000;
            end
        end
    end

    always @(*) begin

        Mask = mask;
        Sens = sens;
    end

endmodule
