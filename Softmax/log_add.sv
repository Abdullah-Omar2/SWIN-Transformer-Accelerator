module log_add_pair #(
    parameter int INT_BITS  = 6,
    parameter int FRAC_BITS = 11
)(
    input  logic clk, rst_n,
    input  logic [INT_BITS+FRAC_BITS-1:0] a, b,
    // ROM port signals (address out this cycle; data back next cycle)
    output logic [INT_BITS+FRAC_BITS-1:0] rom_addr_o,
    input  logic [INT_BITS+FRAC_BITS-1:0] rom_data_i,
    output logic [INT_BITS+FRAC_BITS-1:0] sum
);
    localparam int   WIDTH   = INT_BITS + FRAC_BITS;
    localparam logic [WIDTH-1:0] NEG_INF = {1'b1, {WIDTH-1{1'b0}}};

    // Stage 1 ? combinational
    logic [WIDTH-1:0] st1_max;
    always_comb begin
        if ($signed(a) >= $signed(b)) begin
            st1_max    = a;
            rom_addr_o = $signed(a) - $signed(b);
        end else begin
            st1_max    = b;
            rom_addr_o = $signed(b) - $signed(a);
        end
    end

    // Stage 2 ? registered, aligned with 1-cycle ROM latency
    logic [WIDTH-1:0] st2_max;
    logic             st2_has_neginf;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st2_max        <= '0;
            st2_has_neginf <= 1'b0;
        end else begin
            st2_max        <= st1_max;
            st2_has_neginf <= (a == NEG_INF) || (b == NEG_INF);
        end
    end

    // Output ? combinational on stage-2 + ROM result
    always_comb
        sum = st2_has_neginf ? st2_max : st2_max + rom_data_i;

endmodule


module log_adder_n #(
    parameter int N         = 49,
    parameter int INT_BITS  = 6,
    parameter int FRAC_BITS = 11
)(
    input  logic clk, rst_n,
    input  logic valid_in,
    input  logic [INT_BITS+FRAC_BITS-1:0] x [0:N-1],
    output logic valid_out,
    output logic [INT_BITS+FRAC_BITS-1:0] sum
);
    localparam int WIDTH  = INT_BITS + FRAC_BITS;
    localparam int LEVELS = $clog2(N);

    // -------------------------------------------------------------------------
    //  Valid shift register ? LEVELS deep, matches pipeline latency exactly
    // -------------------------------------------------------------------------
    logic [LEVELS-1:0] valid_sr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_sr <= '0;
        else     valid_sr <= {valid_sr[LEVELS-2:0], valid_in};
    end

    assign valid_out = valid_sr[LEVELS-1];

    // -------------------------------------------------------------------------
    //  Compile-time helpers (PURE MATH FIX for Quartus II)
    // -------------------------------------------------------------------------
    
    // FIX 1: A binary reduction tree of N inputs always requires exactly N - 1 operations!
    localparam int TOT_PORTS = N - 1;

    // FIX 2: Replaced recursion/loops with pure closed-form mathematical equations.
    // Quartus II perfectly evaluates pure arithmetic for localparams.
    
    // Mathematically, repeated ceiling division by 2 equals ceil(N / 2^lvl)
    function automatic int nodes_at(input int lvl, input int n_total);
        return (n_total + (1 << lvl) - 1) >> lvl;
    endfunction

    // Every pair addition reduces the total node count by exactly 1.
    // Thus, the total pairs processed prior to this level is simply
    // the starting node count minus the current node count!
    function automatic int port_offset(input int lvl, input int n_total);
        return n_total - nodes_at(lvl, n_total);
    endfunction

    // -------------------------------------------------------------------------
    //  Shared multi-port ROM
    // -------------------------------------------------------------------------
    logic [WIDTH-1:0] rom_addr [0:TOT_PORTS-1];
    logic [WIDTH-1:0] rom_data [0:TOT_PORTS-1];

    rom_phi_plus_mp #(
        .INT_BITS  (INT_BITS),
        .FRAC_BITS (FRAC_BITS),
        .PORTS     (TOT_PORTS)
    ) rom_inst (
        .clk  (clk),
        .rst_n  (rst_n),
        .addr (rom_addr),
        .data (rom_data)
    );

    // -------------------------------------------------------------------------
    //  Pipeline node storage
    // -------------------------------------------------------------------------
    logic [WIDTH-1:0] node [0:LEVELS][0:N-1];

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_input
            assign node[0][i] = x[i];
        end
    endgenerate

    // -------------------------------------------------------------------------
    //  Tree instantiation
    // -------------------------------------------------------------------------
    genvar lvl, j;
    generate
        for (lvl = 0; lvl < LEVELS; lvl++) begin : gen_level
            localparam int NIN    = nodes_at(lvl, N);
            localparam int NPAIRS = NIN / 2;
            localparam int POFF   = port_offset(lvl, N);
            localparam bit ODD    = (NIN % 2 == 1);

            for (j = 0; j < NPAIRS; j++) begin : gen_pair
                log_add_pair #(INT_BITS, FRAC_BITS) cell_ (
                    .clk       (clk),
                    .rst_n       (rst_n),
                    .a         (node[lvl  ][2*j    ]),
                    .b         (node[lvl  ][2*j + 1]),
                    .rom_addr_o(rom_addr[POFF + j]  ),
                    .rom_data_i(rom_data[POFF + j]  ),
                    .sum       (node[lvl+1][j]       )
                );
            end

            if (ODD) begin : gen_pass
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) node[lvl+1][NPAIRS] <= '0;
                    else     node[lvl+1][NPAIRS] <= node[lvl][NIN-1];
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    //  Final result
    // -------------------------------------------------------------------------
    assign sum = node[LEVELS][0];

endmodule


`timescale 1ns / 1ps
module log_adder_n_tb;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int N         = 8;
    localparam int INT_BITS  = 6;
    localparam int FRAC_BITS = 11;

    // -------------------------------------------------------------------------
    // Derived constants
    // -------------------------------------------------------------------------
    localparam int WIDTH   = INT_BITS + FRAC_BITS;
    localparam int LATENCY = $clog2(N);   // one log_add_pair stage per tree level

    localparam logic [WIDTH-1:0] NEG_INF = {WIDTH{1'b1}};

    // -------------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------------
    logic             clk, rst_n;
    logic             valid_in, valid_out;
    logic [WIDTH-1:0] x   [0:N-1];
    logic [WIDTH-1:0] sum;

    // -------------------------------------------------------------------------
    // Clock & cycle counter
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    int cycles = 0;
    always @(posedge clk) cycles++;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    log_adder_n #(
        .N         (N),
        .INT_BITS  (INT_BITS),
        .FRAC_BITS (FRAC_BITS)
    ) UUT (
        .clk       (clk),
        .rst_n       (rst_n),
        .valid_in  (valid_in),
        .x         (x),
        .valid_out (valid_out),
        .sum       (sum)
    );

    // -------------------------------------------------------------------------
    // Conversion helpers
    // -------------------------------------------------------------------------
    function automatic real to_real(input logic [WIDTH-1:0] val);
        if (val == NEG_INF) return 0.0;
        return 2.0 ** ($signed(val) * 1.0 / (2.0 ** FRAC_BITS));
    endfunction

    function automatic real to_log(input logic [WIDTH-1:0] val);
        if (val == NEG_INF) return -999.0;
        return $signed(val) * 1.0 / (2.0 ** FRAC_BITS);
    endfunction

    function automatic real expected_sum(input logic [WIDTH-1:0] inputs [0:N-1]);
        automatic real s = 0.0;
        for (int k = 0; k < N; k++) s += to_real(inputs[k]);
        return s;
    endfunction

    // -------------------------------------------------------------------------
    // Input helpers
    // -------------------------------------------------------------------------
    task automatic fill_all(input logic [WIDTH-1:0] val);
        for (int k = 0; k < N; k++) x[k] = val;
    endtask

    task automatic fill_seq(input real base_exp, input real step_exp);
        for (int k = 0; k < N; k++)
            x[k] = $rtoi((base_exp + k * step_exp) * (2.0 ** FRAC_BITS));
    endtask

    // -------------------------------------------------------------------------
    // Run-test task
    // -------------------------------------------------------------------------
    task automatic run_test(input string desc, input real expected_real);
        automatic logic [WIDTH-1:0] snap [0:N-1];
        automatic real exp_val;

        // snapshot inputs and expected before they can change
        for (int k = 0; k < N; k++) snap[k] = x[k];
        exp_val = expected_real;

        // single-cycle valid pulse
        @(posedge clk); #1;
        valid_in = 1'b1;
        @(posedge clk); #1;
        valid_in = 1'b0;

        // wait for valid_out
        @(posedge valid_out); #1;

        $display("------------------------------------------------------------------");
        $display("  TEST : %s", desc);
        $display("  Input log2 values:");
        for (int k = 0; k < N; k++)
            $display("    x[%0d] = %7.3f  (linear = %0.4f)",
                     k, to_log(snap[k]), to_real(snap[k]));
        $display("  Expected linear sum = %0.6f", exp_val);
        $display("  GOT    sum_log2 = %7.3f   sum_real = %0.6f   cycle = %0d",
                 to_log(sum), to_real(sum), cycles);

        // tolerance check: 0.5% relative, or absolute 0.001 if near zero
        begin
            automatic real got   = to_real(sum);
            automatic real err   = (exp_val > 0.001) ?
                                   ((got - exp_val) / exp_val) : (got - exp_val);
            if (err < 0.0) err = -err;
            if (err > 0.005)
                $display("  ** MISMATCH ** relative error = %0.4f%%", err * 100.0);
            else
                $display("  PASS  (error = %0.4f%%)", err * 100.0);
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sequence
    // -------------------------------------------------------------------------
    initial begin
        rst_n      = 0;
        valid_in = 0;
        fill_all('0);
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        $display("");
        $display("==================================================================");
        $display("  LOG_ADDER_N TB   N=%0d  INT_BITS=%0d  FRAC_BITS=%0d  WIDTH=%0d",
                 N, INT_BITS, FRAC_BITS, WIDTH);
        $display("  Auto-computed pipeline latency = %0d cycles", LATENCY);
        $display("  Handshake: valid_in pulse -> wait for valid_out");
        $display("==================================================================\n");

        // T1: All log2(1) = 0  =>  N x 1.0
        fill_all(17'b0);
        run_test($sformatf("All ones: %0d x 1.0 = %0d", N, N), real'(N));

        // T2: All log2(2) = 1  =>  N x 2.0
        fill_all((1 << FRAC_BITS));
        run_test($sformatf("All twos: %0d x 2.0 = %0d", N, 2*N), real'(2*N));

        // T3: All NEG_INF  =>  0
        fill_all(NEG_INF);
        run_test("All NEG_INF: sum = 0", 0.0);

        // T4: One real, rest NEG_INF  =>  8.0
        fill_all(NEG_INF);
        x[0] = (3 << FRAC_BITS);   // log2(8)
        run_test("Single real x[0]=8, rest NEG_INF", 8.0);

        // T5: Two equal values, rest NEG_INF  =>  16.0
        fill_all(NEG_INF);
        x[0] = (3 << FRAC_BITS);
        x[1] = (3 << FRAC_BITS);
        run_test("Two reals: 8 + 8 = 16, rest NEG_INF", 16.0);

        // T6: Geometric decay x[k] = 2^(-k/4)
        begin
            automatic real exp_sum = 0.0;
            for (int k = 0; k < N; k++) begin
                x[k] = $rtoi((-k * 1.0 / 4.0) * (2.0 ** FRAC_BITS));
                exp_sum += 2.0 ** (-k * 1.0 / 4.0);
            end
            run_test($sformatf("Geometric decay 2^(-k/4), k=0..%0d", N-1), exp_sum);
        end

        // T7: Large dominant + small values
        begin
            automatic real exp_sum;
            x[0] = (10 << FRAC_BITS);                 // log2(1024)
            for (int k = 1; k < N; k++)
                x[k] = ($signed(-5 << FRAC_BITS));    // log2(1/32)
            exp_sum = 1024.0 + (N-1) * (2.0 ** (-5.0));
            run_test($sformatf("1024 + %0dx(1/32): large dominates", N-1), exp_sum);
        end

        $display("------------------------------------------------------------------");
        $display("\n==================================================================");
        $display("  ALL TESTS DONE");
        $display("==================================================================\n");
        $finish;
    end

endmodule

`timescale 1ns / 1ps
module log_adder_n_acc_tb;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int N         = 8;          // DUT width (7 real inputs + 1 feedback)
    localparam int N_REAL    = N - 1;      // 7 user inputs per vector
    localparam int INT_BITS  = 6;
    localparam int FRAC_BITS = 11;
    localparam int WIDTH     = INT_BITS + FRAC_BITS;
    localparam int LATENCY   = $clog2(N);

    localparam logic [WIDTH-1:0] NEG_INF = {WIDTH{1'b1}};

    localparam int N_VECS = 7;

    function automatic real get_vec_exp(input int v, input int k);
        case (v)
            0: return 0.0;
            1: return 1.0;
            2: return 2.0;
            3: return k * 1.0; // 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            4: return -1.0;
            5: return 3.0;
            6: begin
                case (k)
                    0: return -2.0;
                    1: return 0.0;
                    2: return 2.0;
                    3: return -2.0;
                    4: return 0.0;
                    5: return 2.0;
                    6: return 0.0;
                    default: return 0.0;
                endcase
            end
            default: return 0.0;
        endcase
    endfunction

    // -------------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------------
    logic             clk, rst_n;
    logic             valid_in, valid_out;
    logic [WIDTH-1:0] x   [0:N-1];
    logic [WIDTH-1:0] sum;

    // Feedback register
    logic [WIDTH-1:0] feedback;

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
    log_adder_n #(
        .N         (N),
        .INT_BITS  (INT_BITS),
        .FRAC_BITS (FRAC_BITS)
    ) UUT (
        .clk       (clk),
        .rst_n       (rst_n),
        .valid_in  (valid_in),
        .x         (x),
        .valid_out (valid_out),
        .sum       (sum)
    );

    // -------------------------------------------------------------------------
    // Conversion helpers
    // -------------------------------------------------------------------------
    function automatic real to_real(input logic [WIDTH-1:0] val);
        if (val == NEG_INF) return 0.0;
        return 2.0 ** ($signed(val) * 1.0 / (2.0 ** FRAC_BITS));
    endfunction

    function automatic real to_log(input logic [WIDTH-1:0] val);
        if (val == NEG_INF) return -999.0;
        return $signed(val) * 1.0 / (2.0 ** FRAC_BITS);
    endfunction

    // -------------------------------------------------------------------------
    // Apply one vector + feedback, pulse valid, wait for valid_out, update fb
    // -------------------------------------------------------------------------
    task automatic run_vector(
        input int  vec_idx,
        input real expected_accum   // expected running linear accumulator
    );
        automatic logic [WIDTH-1:0] snap_x   [0:N-1];
        automatic logic [WIDTH-1:0] snap_fb = feedback;

        // Fetch inputs safely via function
        for (int k = 0; k < N_REAL; k++)
            x[k] = $rtoi(get_vec_exp(vec_idx, k) * (2.0 ** FRAC_BITS));
        x[N-1] = feedback;

        // Snapshot for display
        for (int k = 0; k < N; k++) snap_x[k] = x[k];

        // Single-cycle valid pulse
        @(posedge clk); #1;
        valid_in = 1'b1;
        @(posedge clk); #1;
        valid_in = 1'b0;

        // Wait for result
        @(posedge valid_out); #1;

        // Update feedback register with this result
        feedback = sum;

        // Display
        $display("------------------------------------------------------------------");
        $display("  VECTOR %0d", vec_idx);
        $display("  Inputs (log2 | linear):");
        for (int k = 0; k < N_REAL; k++)
            $display("    x[%0d] = %7.3f  |  %0.4f", k, to_log(snap_x[k]), to_real(snap_x[k]));
        $display("    x[%0d] = %7.3f  |  %0.4f  <-- feedback",
                 N-1, to_log(snap_fb), to_real(snap_fb));
        $display("  Expected running sum = %0.6f", expected_accum);
        $display("  GOT      sum_log2 = %7.3f   sum_real = %0.6f   cycle = %0d",
                 to_log(sum), to_real(sum), cycles);

        begin
            automatic real got = to_real(sum);
            automatic real err = (expected_accum > 0.001) ?
                                 ((got - expected_accum) / expected_accum) :
                                 (got - expected_accum);
            if (err < 0.0) err = -err;
            if (err > 0.005)
                $display("  ** MISMATCH ** relative error = %0.4f%%", err * 100.0);
            else
                $display("  PASS  (error = %0.4f%%)", err * 100.0);
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sequence
    // -------------------------------------------------------------------------
    initial begin
        rst_n      = 0;
        valid_in = 0;
        feedback = NEG_INF;   // accumulator starts at log(0)
        for (int k = 0; k < N; k++) x[k] = '0;

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        $display("");
        $display("==================================================================");
        $display("  LOG_ADDER_N FEEDBACK TB");
        $display("  N=%0d  N_REAL=%0d  INT_BITS=%0d  FRAC_BITS=%0d  WIDTH=%0d",
                 N, N_REAL, INT_BITS, FRAC_BITS, WIDTH);
        $display("  LATENCY=%0d cycles  |  feedback slot: x[%0d]", LATENCY, N-1);
        $display("  7 vectors fed sequentially; feedback accumulates across them.");
        $display("==================================================================\n");

        // Pre-compute running expected sums
        begin
            automatic real accum = 0.0;
            automatic real vec_sum;

            for (int v = 0; v < N_VECS; v++) begin
                vec_sum = 0.0;
                // Accumulate safely via function
                for (int k = 0; k < N_REAL; k++)
                    vec_sum += 2.0 ** get_vec_exp(v, k);
                accum += vec_sum;
                run_vector(v, accum);
            end
        end

        $display("------------------------------------------------------------------");
        $display("\n==================================================================");
        $display("  ALL VECTORS DONE  |  Final accumulator = %0.6f", to_real(feedback));
        $display("==================================================================\n");
        $finish;
    end

endmodule

