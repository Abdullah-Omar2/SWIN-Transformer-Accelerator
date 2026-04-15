module max_and_clamp #( 
    parameter int DATA_W = 8, 
    parameter int N      = 7 
)( 
    input  logic                      clk, 
    input  logic                      rst_n, 
    input  logic                      mode_i, 
    input  logic                      valid_i, 
    input  logic signed [DATA_W-1:0]  in_i [N], 
 
    output logic signed [DATA_W-1:0]  clamp_o [N], 
    output logic                      valid_o 
); 
 
    // ------------------------------------------------------------------------- 
    // Constants 
    // ------------------------------------------------------------------------- 
    localparam signed [DATA_W-1:0] MIN_VAL = {1'b1, {DATA_W-1{1'b0}}};  // 0x80 
    localparam signed [DATA_W-1:0] THRESH  = -4; 
 
    // ------------------------------------------------------------------------- 
    // Tree comparator - combinational max of 7 inputs 
    // ------------------------------------------------------------------------- 
    logic signed [DATA_W-1:0] tree_max; 
 
    tree_comparator #( 
        .DATA_W (DATA_W), 
        .N      (N) 
    ) u_tree ( 
        .in      (in_i), 
        .max_out (tree_max) 
    ); 
 
    // ------------------------------------------------------------------------- 
    // Running max register 
    // ------------------------------------------------------------------------- 
    logic signed [DATA_W-1:0] max_reg; 
    logic [$clog2(N):0]       clamp_cnt;   // counts valid clamp cycles (0..N) 
 
    logic do_reset;   // delayed reset pulse: fires the cycle after last clamp 
 
    always_ff @(posedge clk or negedge rst_n) begin 
        if (!rst_n) begin 
            max_reg   <= MIN_VAL; 
            clamp_cnt <= '0; 
            do_reset  <= 1'b0; 
        end else begin 
            // reset fires one cycle after the 7th clamp location 
            if (do_reset) begin 
                max_reg  <= MIN_VAL; 
                do_reset <= 1'b0; 
            end 
 
            if (valid_i && (mode_i == 1'b1)) begin 
                if (clamp_cnt == N - 1) begin 
                    clamp_cnt <= '0; 
                    do_reset  <= 1'b1;  // schedule reset for next cycle 
                end else begin 
                    clamp_cnt <= clamp_cnt + 1; 
                end 
            end else if (valid_i && (mode_i == 1'b0)) begin 
                clamp_cnt <= '0; 
                max_reg   <= (tree_max > max_reg) ? tree_max : max_reg; 
            end 
        end 
    end 
 
    // ------------------------------------------------------------------------- 
    // Subtract / clamp - registered output, all 7 in parallel 
    // ------------------------------------------------------------------------- 
    logic signed [DATA_W:0] diff [N];   // 9-bit to safely hold subtraction 

    // FIX: Compute difference combinationally to prevent 1-cycle skew!
    always_comb begin
        for (int i = 0; i < N; i++) begin
            diff[i] = {in_i[i][DATA_W-1], in_i[i]} - {max_reg[DATA_W-1], max_reg};
        end
    end
 
    always_ff @(posedge clk or negedge rst_n) begin 
        if (!rst_n) begin 
            for (int i = 0; i < N; i++) clamp_o[i] <= '0; 
            valid_o <= 1'b0; 
        end else begin 
            valid_o <= valid_i & mode_i; 
            for (int i = 0; i < N; i++) begin 
                clamp_o[i] <= (diff[i] < THRESH) ? MIN_VAL : diff[i][DATA_W-1:0]; 
            end 
        end 
    end 
 
endmodule

`timescale 1ns / 1ps
module max_and_clamp_tb;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int DATA_W = 8;
    localparam int N      = 7;

    localparam signed [DATA_W-1:0] MIN_VAL = {1'b1, {DATA_W-1{1'b0}}};  // 0x80 = -128
    localparam signed [DATA_W-1:0] THRESH  = -4;

    // -------------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------------
    logic                     clk, rst_n;
    logic                     mode_i, valid_i;
    logic signed [DATA_W-1:0] in_i [N];
    logic signed [DATA_W-1:0] clamp_o [N];
    logic                     valid_o;

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    int cycles = 0;
    always @(posedge clk) cycles++;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    max_and_clamp #(
        .DATA_W (DATA_W),
        .N      (N)
    ) UUT (
        .clk     (clk),
        .rst_n     (rst_n),
        .mode_i  (mode_i),
        .valid_i (valid_i),
        .in_i    (in_i),
        .clamp_o (clamp_o),
        .valid_o (valid_o)
    );

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    function automatic int max_of(input logic signed [DATA_W-1:0] v [N]);
        automatic int m = int'(MIN_VAL);
        for (int i = 0; i < N; i++)
            if ($signed(v[i]) > m) m = int'(v[i]);
        return m;
    endfunction

    function automatic logic signed [DATA_W-1:0] expected_clamp(
        input logic signed [DATA_W-1:0] val,
        input int                        mx
    );
        automatic int d = int'(val) - mx;
        return (d < int'(THRESH)) ? MIN_VAL : (d);
    endfunction

    // -------------------------------------------------------------------------
    // Task: run one full batch (compare pass then clamp pass) on `locs`
    //   locs[location][element] ? 7 locations, N elements each
    // -------------------------------------------------------------------------
    task automatic run_batch(
        input string                      desc,
        input logic signed [DATA_W-1:0]   locs [N][N]
    );
        automatic int    global_max = int'(MIN_VAL);
        automatic int    loc_max;
        automatic int    exp_clamp;
        automatic int    got_clamp;
        automatic int    errs = 0;

        $display("==================================================================");
        $display("  BATCH: %s", desc);
        $display("==================================================================");

        // ---- Compare pass ----
        $display("  -- Compare pass --");
        mode_i  = 1'b0;
        for (int loc = 0; loc < N; loc++) begin
            for (int i = 0; i < N; i++) in_i[i] = locs[loc][i];
            valid_i = 1'b1;
            @(posedge clk); #1;

            loc_max = max_of(locs[loc]);
            if (loc_max > global_max) global_max = loc_max;

            $display("    loc[%0d]: inputs = %p  loc_max = %0d  running_max = %0d",
                     loc, locs[loc], loc_max, global_max);
        end
        valid_i = 1'b0;
        @(posedge clk); #1;

        $display("  Global max = %0d", global_max);

        // ---- Clamp pass ----
        $display("  -- Clamp pass --");
        mode_i = 1'b1;
        for (int loc = 0; loc < N; loc++) begin
            for (int i = 0; i < N; i++) in_i[i] = locs[loc][i];
            valid_i = 1'b1;
            @(posedge clk); #1;
            valid_i = 1'b0;

            // valid_o fires one cycle after valid_i
            @(posedge clk); #1;

            $display("    loc[%0d]:", loc);
            for (int i = 0; i < N; i++) begin
                exp_clamp = int'(expected_clamp(locs[loc][i], global_max));
                got_clamp = int'(clamp_o[i]);
                if (got_clamp !== exp_clamp) begin
                    $display("      [%0d] in=%4d  diff=%4d  exp=%4d  got=%4d  ** MISMATCH **",
                             i, int'(locs[loc][i]),
                             int'(locs[loc][i]) - global_max,
                             exp_clamp, got_clamp);
                    errs++;
                end else begin
                    $display("      [%0d] in=%4d  diff=%4d  clamp=%4d  PASS",
                             i, int'(locs[loc][i]),
                             int'(locs[loc][i]) - global_max,
                             got_clamp);
                end
            end
        end
        valid_i = 1'b0;

        // Let auto-reset settle
        @(posedge clk); #1;

        if (errs == 0)
            $display("  >> ALL PASS\n");
        else
            $display("  >> %0d MISMATCH(ES)\n", errs);
    endtask

    // -------------------------------------------------------------------------
    // Test vectors: locs[location][element]
    // -------------------------------------------------------------------------

    // Batch 1: uniform values ? max = 10, all diffs = 0
    logic signed [DATA_W-1:0] b1 [N][N] = '{
        '{ 10, 10, 10, 10, 10, 10, 10 },
        '{ 10, 10, 10, 10, 10, 10, 10 },
        '{ 10, 10, 10, 10, 10, 10, 10 },
        '{ 10, 10, 10, 10, 10, 10, 10 },
        '{ 10, 10, 10, 10, 10, 10, 10 },
        '{ 10, 10, 10, 10, 10, 10, 10 },
        '{ 10, 10, 10, 10, 10, 10, 10 }
    };

    // Batch 2: one dominant element (max=50), rest far below (-4 boundary check)
    logic signed [DATA_W-1:0] b2 [N][N] = '{
        '{ 50,  46,  45,  44,  43,  42,  41 },  // diffs: 0,-4,-5,-6,-7,-8,-9
        '{  0,   0,   0,   0,   0,   0,   0 },
        '{  0,   0,   0,   0,   0,   0,   0 },
        '{  0,   0,   0,   0,   0,   0,   0 },
        '{  0,   0,   0,   0,   0,   0,   0 },
        '{  0,   0,   0,   0,   0,   0,   0 },
        '{  0,   0,   0,   0,   0,   0,   0 }
    };

    // Batch 3: max is split across locations
    logic signed [DATA_W-1:0] b3 [N][N] = '{
        '{ 20,  15,  10,   5,   0,  -5, -10 },
        '{ 25,  20,  15,  10,   5,   0,  -5 },
        '{ 30,  25,  20,  15,  10,   5,   0 },  // max=30 here
        '{ 28,  24,  20,  16,  12,   8,   4 },
        '{ 10,   8,   6,   4,   2,   0,  -2 },
        '{  5,   4,   3,   2,   1,   0,  -1 },
        '{  1,   0,  -1,  -2,  -3,  -4,  -5 }
    };

    // Batch 4: all MIN_VAL ? everything clamps to MIN_VAL
    logic signed [DATA_W-1:0] b4 [N][N];
    initial begin
        for (int l = 0; l < N; l++)
            for (int i = 0; i < N; i++)
                b4[l][i] = MIN_VAL;
    end

    // Batch 5: negative values, max = -1
    logic signed [DATA_W-1:0] b5 [N][N] = '{
        '{ -1,  -2,  -3,  -4,  -5,  -6,  -7 },
        '{ -2,  -3,  -4,  -5,  -6,  -7,  -8 },
        '{ -3,  -4,  -5,  -6,  -7,  -8,  -9 },
        '{ -4,  -5,  -6,  -7,  -8,  -9, -10 },
        '{ -5,  -6,  -7,  -8,  -9, -10, -11 },
        '{ -6,  -7,  -8,  -9, -10, -11, -12 },
        '{ -7,  -8,  -9, -10, -11, -12, -13 }
    };

    // -------------------------------------------------------------------------
    // Random batch generator
    // -------------------------------------------------------------------------
    task automatic run_random_batch(input int seed);
        automatic logic signed [DATA_W-1:0] locs [N][N];
        automatic string desc;
        automatic int rng = seed;

        // Simple LCG to stay portable across simulators
        for (int l = 0; l < N; l++)
            for (int i = 0; i < N; i++) begin
                rng = (rng * 1664525 + 1013904223) & 32'hFFFFFFFF;
                locs[l][i] = rng[DATA_W-1:0];   // take low byte as signed
            end

        desc = $sformatf("Random batch (seed=%0d)", seed);
        run_batch(desc, locs);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    initial begin
        rst_n     = 0;
        mode_i  = 0;
        valid_i = 0;
        for (int i = 0; i < N; i++) in_i[i] = '0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        $display("\n################################################################");
        $display("  MAX_AND_CLAMP TB   DATA_W=%0d  N=%0d  THRESH=%0d  MIN_VAL=%0d",
                 DATA_W, N, int'(THRESH), int'(MIN_VAL));
        $display("################################################################\n");

        run_batch("Uniform values (all 10)",               b1);
        run_batch("One dominant, -4 boundary check",       b2);
        run_batch("Max split across locations",            b3);
        run_batch("All MIN_VAL",                           b4);
        run_batch("All negative, max = -1",                b5);

        // Random batches
        for (int s = 0; s < 10; s++)
            run_random_batch(s * 7919 + 1);

        $display("################################################################");
        $display("  ALL BATCHES DONE");
        $display("################################################################\n");
        $finish;
    end

endmodule
