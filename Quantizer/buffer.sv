module buffer #(
    parameter int DEPTH      = 16,   // number of 56-bit words ? change freely
    parameter int NUM_BYTES  = 7,    // elements per word (7 × 8 = 56)
    parameter int BYTE_WIDTH = 8     // bits per element

)(
    input  logic                             clk,
    input  logic                             rst_n,     // active-low sync reset

    // Write port
    input  logic                             wr_en,
    input  logic [$clog2(DEPTH)-1:0]         wr_addr,
    input  logic [(NUM_BYTES*BYTE_WIDTH)-1:0] wr_data,  // 56-bit word

    // Read port
    input  logic                             rd_en,
    input  logic [$clog2(DEPTH)-1:0]         rd_addr,
    output logic [(NUM_BYTES*BYTE_WIDTH)-1:0] rd_data  // 56-bit word
);

    // --------------------------------------------------------
    //  Local parameters
    // --------------------------------------------------------
    localparam int DATA_WIDTH = NUM_BYTES * BYTE_WIDTH;   // 56
    localparam int ADDR_WIDTH = $clog2(DEPTH);

    // --------------------------------------------------------
    //  Storage array  ? DEPTH words, each DATA_WIDTH bits wide
    // --------------------------------------------------------
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // --------------------------------------------------------
    //  Write logic (synchronous)
    // --------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Optionally zero-init memory on reset
            for (int i = 0; i < DEPTH; i++)
                mem[i] <= '0;
        end else begin
            if (wr_en) begin
                mem[wr_addr] <= wr_data;
            end
        end
    end

    // --------------------------------------------------------
    //  Read logic (synchronous, registered output)
    //  Output is 0 if the location has never been written
    // --------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else begin
            if (rd_en) begin
                rd_data <= mem[rd_addr];
            end
        end
    end

endmodule


// ============================================================
//  Testbench ? 56-bit Parameterized Buffer
// ============================================================

`timescale 1ns/1ps

module tb_buffer;

    // --------------------------------------------------------
    //  Parameters matching the DUT
    // --------------------------------------------------------
    localparam int DEPTH      = 16;
    localparam int NUM_BYTES  = 7;
    localparam int BYTE_WIDTH = 8;
    localparam int DATA_WIDTH = NUM_BYTES * BYTE_WIDTH;   // 56
    localparam int ADDR_WIDTH = $clog2(DEPTH);

    // --------------------------------------------------------
    //  DUT signals
    // --------------------------------------------------------
    logic                         clk    = 0;
    logic                         rst_n;
    logic                         wr_en;
    logic [ADDR_WIDTH-1:0]        wr_addr;
    logic [DATA_WIDTH-1:0]        wr_data;
    logic                         rd_en;
    logic [ADDR_WIDTH-1:0]        rd_addr;
    logic [DATA_WIDTH-1:0]        rd_data;

    // --------------------------------------------------------
    //  Clock ? 10 ns period
    // --------------------------------------------------------
    always #5 clk = ~clk;

    // --------------------------------------------------------
    //  DUT instantiation
    // --------------------------------------------------------
    buffer #(
        .DEPTH      (DEPTH),
        .NUM_BYTES  (NUM_BYTES),
        .BYTE_WIDTH (BYTE_WIDTH)
    ) dut (
        .clk     (clk),
        .rst_n     (rst_n),
        .wr_en   (wr_en),
        .wr_addr (wr_addr),
        .wr_data (wr_data),
        .rd_en   (rd_en),
        .rd_addr (rd_addr),
        .rd_data (rd_data)
    );

    // --------------------------------------------------------
    //  Task : write one 56-bit word
    // --------------------------------------------------------
    task automatic write_word(
        input logic [ADDR_WIDTH-1:0]  addr,
        input logic [DATA_WIDTH-1:0]  data
    );
        @(posedge clk);
        wr_en   <= 1'b1;
        wr_addr <= addr;
        wr_data <= data;
        @(posedge clk);
        wr_en   <= 1'b0;
    endtask

    // --------------------------------------------------------
    //  Task : read one 56-bit word and check expected value
    // --------------------------------------------------------
    task automatic read_word(
        input logic [ADDR_WIDTH-1:0]  addr,
        input logic [DATA_WIDTH-1:0]  expected
    );
        @(posedge clk);
        rd_en   <= 1'b1;
        rd_addr <= addr;
        @(posedge clk);
        rd_en <= 1'b0;
        // Data is registered ? available one cycle after rd_en
        @(posedge clk);
        if (rd_data === expected)
            $display("[PASS] addr=%0d  got=0x%014h  expected=0x%014h",
                     addr, rd_data, expected);
        else
            $error("[FAIL] addr=%0d  got=0x%014h  expected=0x%014h",
                   addr, rd_data, expected);
    endtask

    // --------------------------------------------------------
    //  Stimulus
    // --------------------------------------------------------
    initial begin
        // Defaults
        rst_n   = 1;
        wr_en   = 0;
        rd_en   = 0;
        wr_addr = '0;
        wr_data = '0;
        rd_addr = '0;

        // ---- Reset ----
        repeat(4) @(posedge clk);
        rst_n = 0;
        @(posedge clk);

        // ---- Write a few 56-bit words ----
        // Each word is 7 bytes shown as 14 hex digits
        write_word(4'd0,  56'hAABBCCDDEEFF11);
        write_word(4'd1,  56'h112233445566AA);
        write_word(4'd5,  56'hDEADBEEFCAFE00);
        write_word(4'd15, 56'h0102030405060F);

        // ---- Fill remaining locations to trigger full flag ----
        for (int i = 2; i < DEPTH; i++) begin
            if (i != 5 && i != 15)
                write_word(i[ADDR_WIDTH-1:0], (i * 56'h01010101010101));
        end

        @(posedge clk);

        // ---- Read back and verify ----
        read_word(4'd0,  56'hAABBCCDDEEFF11);
        read_word(4'd1,  56'h112233445566AA);
        read_word(4'd5,  56'hDEADBEEFCAFE00);
        read_word(4'd15, 56'h0102030405060F);

        // ---- Overwrite address 0 and verify ----
        write_word(4'd0, 56'hFFFFFFFFFFFFFF);
        read_word (4'd0, 56'hFFFFFFFFFFFFFF);

        // ---- Reset and check empty ----
        @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        rst_n = 0;
        @(posedge clk);

        repeat(5) @(posedge clk);
        $display("\n=== Simulation complete ===");
        $finish;
    end

endmodule