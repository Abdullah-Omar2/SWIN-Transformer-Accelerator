module accum_and_div #(
    parameter int INT_BITS  = 6,
    parameter int FRAC_BITS = 11,
    parameter int N_IN      = 7,       // real inputs per location
    parameter int DATA_W    = 8        // width of each signed input
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    mode_i,
    input  logic                    valid_i,
    input  logic signed [DATA_W-1:0]    in_i [0:N_IN-1],

    output logic signed [DATA_W-1:0]    out_o   [0:N_IN-1],
    output logic                    valid_o,
    output logic                    ready
);

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------
    localparam int           WIDTH   = INT_BITS + FRAC_BITS;
    localparam logic [WIDTH-1:0] NEG_INF = {1'b1, {WIDTH-1{1'b0}}};
    localparam logic signed [DATA_W-1:0] MIN_VAL = {1'b1, {DATA_W-1{1'b0}}};
    // -------------------------------------------------------------------------
    // Step 1: combinational 8-bit -> log conversion
    // -------------------------------------------------------------------------
    logic [WIDTH-1:0] log_val [0:N_IN-1];

    always_comb begin
        for (int i = 0; i < N_IN; i++) begin
            if (in_i[i] == MIN_VAL)
                log_val[i] = NEG_INF;
            else
                log_val[i] = {in_i[i][INT_BITS-1:0], {FRAC_BITS{1'b0}}};
        end
    end

    // -------------------------------------------------------------------------
    // Accumulation register (feedback)
    // -------------------------------------------------------------------------
    logic [WIDTH-1:0] accum_reg;
    logic [WIDTH-1:0] adder_sum;
    logic             adder_valid_out;
    logic [$clog2(N_IN):0] out_cnt;    // counts output-pass valid cycles
    logic                  do_reset;

    // Build the 8-input array: log_val[0..6] + accum_reg
    logic [WIDTH-1:0] adder_in [0:N_IN-1];
    always_comb begin
        for (int i = 0; i < N_IN; i++) adder_in[i] = log_val[i];
    end
    
    logic adder_valid_in;
    assign adder_valid_in = valid_i & ~mode_i;
    
    // log_adder_n: N=8
    log_adder_n #(
        .N         (N_IN),
        .INT_BITS  (INT_BITS),
        .FRAC_BITS (FRAC_BITS)
    ) u_adder (
        .clk       (clk),
        .rst_n       (rst_n),
        .valid_in  (adder_valid_in),   // only active during accumulate pass
        .x         (adder_in),
        .valid_out (adder_valid_out),
        .sum       (adder_sum)
    );
    
    assign ready = adder_valid_out;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_reg <= NEG_INF;
            out_cnt   <= '0;
            do_reset  <= 1'b0;
        end else begin

            // Reset fires one cycle after the N_IN-th output word
            if (do_reset) begin
                accum_reg <= NEG_INF;
                do_reset  <= 1'b0;
            end

            // Count output-pass valid pulses
            if (valid_i && mode_i) begin
                if (out_cnt == N_IN - 1) begin
                    out_cnt  <= '0;
                    do_reset <= 1'b1;   // schedule reset for next cycle
                end else begin
                    	out_cnt  <= out_cnt + 1;
                end
          	 end

            // Latch accumulator result from adder
            if (adder_valid_out && !mode_i)
               accum_reg <= adder_sum;

        end
    end

    // -------------------------------------------------------------------------
    // Output pass: log_div -> fixed_rounder -> +7 -> left_shifter
    // -------------------------------------------------------------------------
    logic [WIDTH-1:0]      log_ratio  [0:N_IN-1];
    logic signed [7:0]     rounded    [0:N_IN-1];
    logic signed [7:0]     shift_amt  [0:N_IN-1];
    logic signed [7:0]     shifted    [0:N_IN-1];

    generate
        genvar i;
        for (i = 0; i < N_IN; i++) begin : gen_out

            // 2a. log division: log(x[i]) - log(accum)
            log_div #(
                .INT_BITS  (INT_BITS),
                .FRAC_BITS (FRAC_BITS)
            ) u_div (
                .a    (log_val[i]),
                .b    (accum_reg),
                .quot (log_ratio[i])
            );

            // 2b. round to 8-bit integer
            fixed_rounder #(
                .INT_BITS  (INT_BITS),
                .FRAC_BITS (FRAC_BITS),
                .W_OUT     (DATA_W)
            ) u_round (
                .in_value  (log_ratio[i]),
                .out_value (rounded[i])
            );

            // 2c. shift_amt = rounded + 7
            assign shift_amt[i] = rounded[i] + 8'd7;

            // 2d. 2^(rounded + 7): left_shifter(1, shift_amt)
            left_shifter #(
                .W_INPUT (DATA_W),
                .W_SHIFT (DATA_W)
            ) u_shift (
                .in_value  ({DATA_W{1'b0}} | 1'b1),
                .shift_amt (shift_amt[i]),
                .out_value (shifted[i])
            );

        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N_IN; i++) out_o[i] <= '0;
            valid_o <= 1'b0;
        end else begin
            valid_o <= valid_i & mode_i;
            for (int i = 0; i < N_IN; i++) begin
                if (!mode_i)
                    out_o[i] <= '0;
                else if (in_i[i] == MIN_VAL)
                    out_o[i] <= '0;
                else if (shifted[i] == MIN_VAL)
                    out_o[i] <= {1'b0, {DATA_W-1{1'b1}}};  // 127
                else
                    out_o[i] <= shifted[i];
            end
        end
    end

endmodule

`timescale 1ns / 1ps
module accum_and_div_tb;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int INT_BITS  = 6;
    localparam int FRAC_BITS = 11;
    localparam int N_IN      = 7;
    localparam int DATA_W    = 8;
    localparam int WIDTH     = INT_BITS + FRAC_BITS;
    localparam int LATENCY   = $clog2(N_IN + 1);  // log_adder_n with N=8

    localparam logic signed [DATA_W-1:0] MIN_VAL = {1'b1, {DATA_W-1{1'b0}}};

    // -------------------------------------------------------------------------
    // Test vector: 7 locations x 7 elements (signed 8-bit, already clamped)
    // -------------------------------------------------------------------------
    logic signed [DATA_W-1:0] vectors [0:N_IN-1][0:N_IN-1] = '{
        '{   0,  -1,  -7,  -3,  -4,   -6,  -1 },
        '{  -8,  -5,  -4,   0,  -1,  -2,  -3 },
        '{  -4,   0,  -1,  -2,  -3,  -4,   0 },
        '{  -1,  -2,   0,  -4,  -6,  -2,  -1 },
        '{   -4,   -4,   -4,   -4,   -4,   -4,   -4 },
        '{  -4,  -4,  -4,  -5,  -4,  -4,  -4 },
        '{   0,  -1,  -5,  -3,  -5, MIN_VAL, MIN_VAL }
    };

    // -------------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------------
    logic                        clk, rst_n;
    logic                        mode_i, valid_i;
    logic signed [DATA_W-1:0]    in_i    [0:N_IN-1];
    logic signed [DATA_W-1:0]    out_o   [0:N_IN-1];
    logic                        valid_o;
    logic                        ready;

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    accum_and_div #(
        .INT_BITS  (INT_BITS),
        .FRAC_BITS (FRAC_BITS),
        .N_IN      (N_IN),
        .DATA_W    (DATA_W)
    ) UUT (
        .clk     (clk),
        .rst_n     (rst_n),
        .mode_i  (mode_i),
        .valid_i (valid_i),
        .in_i    (in_i),
        .out_o   (out_o),
        .valid_o (valid_o),
        .ready   (ready)
    );

    // -------------------------------------------------------------------------
    // Helper: print accum_reg via hierarchical path
    // -------------------------------------------------------------------------
    function automatic real fixed_to_real(input logic [WIDTH-1:0] val);
        return $signed(val) * 1.0 / (2.0 ** FRAC_BITS);
    endfunction

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    initial begin
        rst_n     = 0;
        mode_i  = 0;
        valid_i = 0;
        for (int i = 0; i < N_IN; i++) in_i[i] = '0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        $display("\n================================================================");
        $display("  accum_and_div TB   N_IN=%0d  DATA_W=%0d  LATENCY=%0d cycles",
                 N_IN, DATA_W, LATENCY);
        $display("================================================================\n");

        // ----------------------------------------------------------------
        // Accumulate pass
        // ----------------------------------------------------------------
        $display("-- ACCUMULATE PASS (mode=0) --");
        mode_i = 1'b0;

        for (int loc = 0; loc < N_IN; loc++) begin
            for (int i = 0; i < N_IN; i++) in_i[i] = vectors[loc][i];
            valid_i = 1'b1;
            @(posedge clk); #1;
            valid_i = 1'b0;

            // Wait for adder valid_out, then one more cycle for accum_reg to latch
            @(posedge ready);
            @(posedge clk); #1;

            $display("  loc[%0d] inputs: %p", loc, vectors[loc]);
            $display("         accum_reg = %0.4f  (raw=0x%04h)",
                     fixed_to_real(UUT.accum_reg), UUT.accum_reg);
        end

        $display("\n  Final accum_reg = %0.4f\n", fixed_to_real(UUT.accum_reg));

        // ----------------------------------------------------------------
        // Output pass
        // ----------------------------------------------------------------
        $display("-- OUTPUT PASS (mode=1) --");
        mode_i = 1'b1;

        for (int loc = 0; loc < N_IN; loc++) begin
            for (int i = 0; i < N_IN; i++) in_i[i] = vectors[loc][i];
            valid_i = 1'b1;
            @(posedge clk); #1;
            valid_i = 1'b0;

            // Wait for registered output
            @(posedge clk); #1;

            $display("  loc[%0d]:", loc);
            for (int i = 0; i < N_IN; i++)
                $display("    [%0d] in=%4d  out=%4d",
                         i, vectors[loc][i], out_o[i]);
        end

        $display("\n================================================================");
        $display("  DONE");
        $display("================================================================\n");
        $finish;
    end

endmodule
