module left_shifter #(
    parameter int W_INPUT = 32,
    parameter int W_SHIFT = 8
)(
    input  logic signed [W_INPUT-1:0] in_value,
    input  logic signed [W_SHIFT-1:0] shift_amt,
    output logic signed [W_INPUT-1:0] out_value
);

    assign out_value = in_value <<< shift_amt;
    
endmodule
