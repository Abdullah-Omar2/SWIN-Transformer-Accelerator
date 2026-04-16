module quantizer #(
    parameter int W_INPUT  = 32,
    parameter int W_OUTPUT = 8,
    parameter int W_SHIFT  = 8,
    parameter int NUM_VALS = 7
)(
    input  logic signed [W_INPUT-1:0]  in_values [0:NUM_VALS-1],
    input  logic [W_SHIFT-1:0]                       shift_amt, // Always positive: Performs Right Shift
    output logic signed [W_OUTPUT-1:0] out_values [0:NUM_VALS-1]
);

    localparam logic signed [W_OUTPUT-1:0] MAX_OUT = (1 << (W_OUTPUT-1)) - 1;
    localparam logic signed [W_OUTPUT-1:0] MIN_OUT = -(1 << (W_OUTPUT-1));

    // Encapsulate the per-element logic in an automatic function.
    // This completely resolves vlog-2583 by isolating variables.
    function automatic logic signed [W_OUTPUT-1:0] quantize_element(
        input logic signed [W_INPUT-1:0] val,
        input logic [W_SHIFT-1:0]        shift
    );
        logic signed [W_INPUT:0]   temp_val; // Extra bit for overflow/math
        logic                      v_sign;
        logic [W_INPUT-1:0]        abs_in;
        logic [W_INPUT-1:0]        mask;
        logic [W_INPUT-1:0]        lsb_bits;
        logic [W_INPUT-1:0]        half_val;
        logic                      is_greater_than_half;
        logic                      is_half_and_odd;
        logic signed [W_INPUT:0]   rounded_val;

        // 1. Extract sign and absolute value
        v_sign = val[W_INPUT-1];
        abs_in = v_sign ? -val : val;

        // 2. Handle shift logic and rounding components
        if (shift > 0) begin
            // Create mask safely using W_INPUT width to prevent overflow during calculation
            mask     = ({{(W_INPUT-1){1'b0}},1'b1} << shift) - 1'b1;
            half_val = ({{(W_INPUT-1){1'b0}},1'b1} << (shift - 1));
            
            lsb_bits = abs_in & mask;
            temp_val = $signed({1'b0, (abs_in >> shift)});

            // Convergent Rounding (Ties to Even) logic
            is_greater_than_half = (lsb_bits > half_val);
            is_half_and_odd      = (lsb_bits == half_val) && temp_val[0];

            if (is_greater_than_half || is_half_and_odd) begin
                rounded_val = temp_val + 1'b1;
            end else begin
                rounded_val = temp_val;
            end
        end else begin
            // shift is 0: No shift
            rounded_val = $signed({1'b0, abs_in});
        end
        
        if (v_sign) begin
            rounded_val = -rounded_val;
        end
        
        if (rounded_val > $signed(MAX_OUT)) begin
            return MAX_OUT;
        end else if (rounded_val < $signed(MIN_OUT)) begin
            return MIN_OUT;
        end else begin
            return rounded_val[W_OUTPUT-1:0];
        end
    endfunction

    genvar i;
    generate
        for (i = 0; i < NUM_VALS; i++) begin : gen_shifters
            assign out_values[i] = quantize_element(in_values[i], shift_amt);
        end
    endgenerate

endmodule

module quantizer_cu #(
    parameter int DEPTH = 10
)(
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    next_shift,
    
    // Output Address for the buffer
    output logic [$clog2(DEPTH)-1:0] rd_addr
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_addr <= '0;
        end else if (next_shift) begin
            // Wrap around to 0 if we are at the last location
            if (rd_addr == (DEPTH - 1)) begin
                rd_addr <= '0;
            end else begin
                rd_addr <= rd_addr + 1;
            end
        end
    end

endmodule

module quantizer_core #(
    parameter int W_INPUT  = 32,
    parameter int W_OUTPUT = 8,
    parameter int W_SHIFT  = 8,
    parameter int NUM_VALS = 7,
    parameter int DEPTH    = 10
)(
    input  logic                                     clk,
    input  logic                                     rst_n,
    
    // Control interface
    input  logic                                     next_shift,
    
    // Interface to external Buffer
    output logic [$clog2(DEPTH)-1:0]                 buf_rd_addr,
    input  logic [W_SHIFT-1:0]                       buf_rd_data, // comes from buffer rd_data
    
    // Data interface
    input  logic signed [W_INPUT-1:0]  in_values [0:NUM_VALS-1],
    output logic signed [W_OUTPUT-1:0] out_values [0:NUM_VALS-1]
);

    // 1. Instantiate the Control Unit to manage buffer addresses
    quantizer_cu #(
        .DEPTH(DEPTH)
    ) u_cu (
        .clk(clk),
        .rst_n(rst_n),
        .next_shift(next_shift),
        .rd_addr(buf_rd_addr)
    );

    // 2. Instantiate the Quantizer to process values using shift_amt from buffer
    quantizer #(
        .W_INPUT(W_INPUT),
        .W_OUTPUT(W_OUTPUT),
        .W_SHIFT(W_SHIFT),
        .NUM_VALS(NUM_VALS)
    ) u_quant (
        .in_values(in_values),
        .shift_amt(buf_rd_data), // The shift amount is fed directly from the buffer
        .out_values(out_values)
    );

endmodule

`timescale 1ns/1ps

module tb_quantizer;

    // ==========================================
    // System Parameters
    // ==========================================
    localparam int W_INPUT  = 32;
    localparam int W_OUTPUT = 8;
    localparam int NUM_VALS = 7;
    
    localparam int DEPTH    = 10;
    localparam int W_SHIFT  = 8; 

    // ==========================================
    // Signals
    // ==========================================
    logic clk;
    logic rst_n;

    // Buffer Write Interface
    logic                     wr_en;
    logic [$clog2(DEPTH)-1:0] wr_addr;
    logic [W_SHIFT-1:0]       wr_data;

    // System Control & Data
    logic                     next_shift;
    
    // UPDATED: Interfaces changed to Unpacked Arrays to match version 2
    logic signed [W_INPUT-1:0]  in_values  [0:NUM_VALS-1];
    logic signed [W_OUTPUT-1:0] out_values [0:NUM_VALS-1];

    // Interconnects
    logic [$clog2(DEPTH)-1:0] sys_to_buf_addr;
    logic [W_SHIFT-1:0]       buf_to_sys_data;

    // ==========================================
    // Instantiations
    // ==========================================

    // 1. The new combined System (CU + Quantizer)
    quantizer_core #(
        .W_INPUT(W_INPUT),
        .W_OUTPUT(W_OUTPUT),
        .W_SHIFT(W_SHIFT),
        .NUM_VALS(NUM_VALS),
        .DEPTH(DEPTH)
    ) u_sys (
        .clk(clk),
        .rst_n(rst_n),
        .next_shift(next_shift),
        .buf_rd_addr(sys_to_buf_addr),
        .buf_rd_data(buf_to_sys_data),
        .in_values(in_values),
        .out_values(out_values)
    );

    // 2. The Buffer (Standalone storage)
    // Note: Ensure your buffer module is compiled alongside this
    buffer #(
        .DEPTH(DEPTH),
        .NUM_BYTES(1),
        .BYTE_WIDTH(W_SHIFT)
    ) u_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .rd_en(1'b1), 
        .rd_addr(sys_to_buf_addr),
        .rd_data(buf_to_sys_data)
    );

    // ==========================================
    // Clock Generation
    // ==========================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ==========================================
    // Test Sequence
    // ==========================================
    initial begin
        // Initialize
        rst_n      = 0;
        wr_en      = 0;
        wr_addr    = 0;
        wr_data    = 0;
        next_shift = 0;
        
        // Initialize unpacked array
        for (int i = 0; i < NUM_VALS; i++) in_values[i] = '0;

        @(posedge clk);
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("\n--- Step 1: Populating the Buffer ---");
        for (int i = 0; i < DEPTH; i++) begin
            wr_en   = 1;
            wr_addr = i;
            wr_data = $urandom_range(0, 10); 
            @(posedge clk);
        end
        wr_en = 0;

        @(posedge clk); 
        
        $display("\n--- Step 2: Testing Integrated System (Unpacked Interface) ---");
        
        for (int i = 0; i < DEPTH + 5; i++) begin 
            // 1. Randomize input vector [-10000, 10000]
            for (int j = 0; j < NUM_VALS; j++) begin
                in_values[j] = $signed($urandom_range(0, 20000)) - 10000;
            end

            // 2. Small delay to allow combinatorial logic to settle (quantizer is purely comb)
            #1;

            // 3. Display status
            $display("T=%0t | ADDR_REQ: %0d | SHIFT_ACTIVE: %0d", $time, sys_to_buf_addr, buf_to_sys_data);
            $write("  IN:  ");
            for (int j = 0; j < NUM_VALS; j++) $write("%6d ", $signed(in_values[j]));
            $write("\n  OUT: ");
            for (int j = 0; j < NUM_VALS; j++) $write("%6d ", $signed(out_values[j]));
            $display("\n---------------------------------------------------------");

            // 4. Trigger next shift
            next_shift = 1;
            @(posedge clk);
            next_shift = 0;
            
            // 5. Wait for buffer read latency (if buffer is registered)
            @(posedge clk);
        end

        $display("\n==== ALL SYSTEM TESTS FINISHED ====");
        $finish;
    end

endmodule