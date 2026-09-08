module chunk #(
    parameter XS = 8,
    parameter YS = 8
)(
    input wire [XS - 1:0] xs_in,
    input wire [YS - 1:0] ys_in,
    input wire [((XS + YS) * 4) - 1:0] mask_msk,
    input wire [((XS + YS) * 2) - 1:0] sens_msk,
    input wire [XS - 1:0] in,
    output reg [((XS + YS) * 4) - 1:0] Mask_msk,
    output reg [((XS + YS) * 2) - 1:0] Sens_msk,
    output reg [XS - 1:0] out
);
    
    

endmodule