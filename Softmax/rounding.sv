module fixed_rounder #(
    parameter int INT_BITS  = 6,
    parameter int FRAC_BITS = 11,
    parameter int W_OUT     = 8
)(
    input  logic signed [INT_BITS+FRAC_BITS-1:0] in_value,
    output logic signed [W_OUT-1:0]               out_value
);
    localparam int W_INPUT = INT_BITS + FRAC_BITS;

    logic signed [W_INPUT-1:0] abs_value;
    logic                      value_sign;
    logic [FRAC_BITS-1:0]      frac_part;
    logic [W_OUT-1:0]          int_part;
    logic [FRAC_BITS-1:0]      half;
    logic                      is_greater_than_half;
    logic                      is_half_and_odd;

    always_comb begin
        value_sign = in_value[W_INPUT-1];
        abs_value  = value_sign ? -in_value : in_value;
        int_part   = {{W_OUT-INT_BITS{1'b0}}, abs_value[W_INPUT-1:FRAC_BITS]};
        frac_part  = abs_value[FRAC_BITS-1:0];
        half       = {{FRAC_BITS-1{1'b0}}, 1'b1} << (FRAC_BITS - 1);
        is_greater_than_half = (frac_part > half);
        is_half_and_odd      = (frac_part == half) && int_part[0];
        if (is_greater_than_half || is_half_and_odd)
            int_part = int_part + 1'b1;
        out_value = value_sign ? -$signed(int_part) : $signed(int_part);
    end
endmodule

`timescale 1ns/1ps

module fixed_rounder_tb;

  // Parameters
  localparam INT_BITS  = 6;
  localparam FRAC_BITS = 11;
  localparam W_INPUT   = INT_BITS + FRAC_BITS;
  localparam W_OUT = 8;

  // DUT signals
  logic signed [W_INPUT-1:0] in_value;
  logic signed [W_OUT-1:0] out_value;

  // Instantiate DUT
  fixed_rounder #(
    .INT_BITS(INT_BITS),
    .FRAC_BITS(FRAC_BITS),
    .W_OUT(W_OUT)
  ) dut (
    .in_value(in_value),
    .out_value(out_value)
  );

  // Stimulus
  initial begin
    $display("Time\tin_value\tout_value");

    // ????: 6.25 (6 + 0.25)
    in_value = (6 <<< FRAC_BITS) + (1 <<< (FRAC_BITS-2)); // 6.25
    #1 $display("%0t\t%d\t%d", $time, in_value, out_value);

    // ????: 6.5 (tie case)
    in_value = (6 <<< FRAC_BITS) + (1 <<< (FRAC_BITS-1)); // 6.5
    #1 $display("%0t\t%d\t%d", $time, in_value, out_value);

    // ????: 7.5 (tie case, odd integer)
    in_value = (7 <<< FRAC_BITS) + (1 <<< (FRAC_BITS-1)); // 7.5
    #1 $display("%0t\t%d\t%d", $time, in_value, out_value);

    // ????: -3.75
    in_value = -( (3 <<< FRAC_BITS) + (3 <<< (FRAC_BITS-2)) ); // -3.75
    #1 $display("%0t\t%d\t%d", $time, in_value, out_value);

    // ????: -2.5 (tie case)
    in_value = -( (2 <<< FRAC_BITS) + (1 <<< (FRAC_BITS-1)) ); // -2.5
    #1 $display("%0t\t%d\t%d", $time, in_value, out_value);

    $finish;
  end

endmodule