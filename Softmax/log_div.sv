module log_div #(
    parameter int INT_BITS  = 6,
    parameter int FRAC_BITS = 11
)(
    input  logic [INT_BITS+FRAC_BITS-1:0] a,    // log(numerator)
    input  logic [INT_BITS+FRAC_BITS-1:0] b,    // log(denominator)
    output logic [INT_BITS+FRAC_BITS-1:0] quot  // log(a/b)
);
    localparam int          WIDTH   = INT_BITS + FRAC_BITS;
    localparam logic [WIDTH-1:0] NEG_INF = {1'b1, {WIDTH-1{1'b0}}};

    always_comb begin
        if      (a == NEG_INF) quot = NEG_INF;  // 0 / x = 0
        else if (b == NEG_INF) quot = '0;        // x / 0 = +inf ? saturate to 0 log
        else                   quot = a - b;
    end

endmodule


`timescale 1ns / 1ps

module log_div_tb;

    // Parameters
    localparam INT_BITS  = 6;
    localparam FRAC_BITS = 11;
    localparam WIDTH     = INT_BITS + FRAC_BITS;
    localparam NEG_INF      = {WIDTH{1'b1}};

    // Signals
    logic [WIDTH-1:0] a, b;
    logic [WIDTH-1:0] quot;

    // --- Functions ??????? ????? ?????? ---
    function real lns_to_real(input logic [WIDTH-1:0] val);
        if (val == NEG_INF) return 0.0;
        return 2.0 ** ($signed(val) * 1.0 / (2.0**FRAC_BITS));
    endfunction

    // DUT Instantiation
    log_div #(INT_BITS, FRAC_BITS) dut (
        .a(a), .b(b),
        .quot(quot)
    );

    initial begin
        $display("--- Starting Log Divider Test (Combinational) ---");

        // ?????? 1: ???? 4 / 2
        // log2(4) = 2, log2(2) = 1 -> ??????? log2(2) = 1
        a = (2 << FRAC_BITS); 
        b = (1 << FRAC_BITS);
        #10; // ??? ???? ??? Simulation
        $display("Test 4/2: In_A=%.1f, In_B=%.1f | Out_Quot=%.1f", lns_to_real(a), lns_to_real(b), lns_to_real(quot));

        // ?????? 2: ???? 1 / 4
        // log2(1) = 0, log2(4) = 2 -> ??????? log2(0.25) = -2
        a = (0 << FRAC_BITS);
        b = (2 << FRAC_BITS);
        #10;
        $display("Test 1/4: In_A=%.1f, In_B=%.1f | Out_Quot=%.2f", lns_to_real(a), lns_to_real(b), lns_to_real(quot));

        // ?????? 3: ???? ??? ??? ?? ??? (0 / X = 0)
        a = NEG_INF; 
        b = (3 << FRAC_BITS);
        #10;
        $display("Test 0/8: In_A=%.1f, In_B=%.1f | Out_Quot=%.1f", lns_to_real(a), lns_to_real(b), lns_to_real(quot));

        // ?????? 4: ???? ??? ??? ???? (X / X = 1)
        // log2(X) - log2(X) = 0 -> ???? 0 ?? ?????????? ???? 1 ?? ???????
        a = (5 << FRAC_BITS);
        b = (5 << FRAC_BITS);
        #10;
        $display("Test X/X: In_A=%.1f, In_B=%.1f | Out_Quot=%.1f", lns_to_real(a), lns_to_real(b), lns_to_real(quot));

        $display("--- Divider Test Finished ---");
        $finish;
    end

endmodule