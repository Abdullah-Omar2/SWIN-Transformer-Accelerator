// =============================================================================
// ilb_patch_embed_buf.sv
//
// Intermediate Layer Buffer — Patch Embedding Convolution Output
//
// Captures the result of the first (and only) convolution stage: a 4×4,
// stride-4 convolution of the 224×224×3 input image using 96 kernels,
// producing a (56×56, 96) INT8 feature map.
//
// ── Convolution scheduling ────────────────────────────────────────────────
//   • 12 PE blocks (4 per input channel) share computation.
//   • Each cycle, 7 spatially-adjacent output pixels are produced simultaneously
//     (one per PE row), covering columns [col_grp*7 .. col_grp*7 + 6].
//   • 56 columns / 7 per cycle = 8 cycles per spatial row.
//   • 56 rows × 8 cycles = 448 cycles per output channel.
//   • Channels are processed sequentially: ch0 fully written → ch1 → … → ch95.
//   • Total write cycles: 96 × 448 = 43,008.
//
// ── Memory layout — channel-first, row-major ──────────────────────────────
//   byte_addr(ch, row, col) = ch × (H×W) + row × W + col
//
//   Channel k occupies byte addresses [k×3136 .. (k+1)×3136 − 1].
//   Total capacity: 96 × 56 × 56 = 301,056 bytes  (≈ 294 KB).
//
//   This layout matches the spec requirement:
//     "the first element is v1 … till v3136 for the first channel,
//      the second channel starts from v3137 … and so on".
//
// ── Write port (wr_*) ─────────────────────────────────────────────────────
//   Each cycle when wr_en is asserted, exactly 7 INT8 pixels belonging to
//   the same output channel and spatial row are written.
//
//   wr_ch       [6:0] : output channel being written  (0 .. 95)
//   wr_row      [5:0] : spatial row being written     (0 .. 55)
//   wr_col_grp  [2:0] : column group (0 .. 7).
//                       Pixel column = wr_col_grp × N_ROWS + i  (i=0..6)
//   wr_data[i]  [7:0] : quantised INT8 value for column offset i.
//
// ── Read port (rd_*) ──────────────────────────────────────────────────────
//   32-bit word read (4 packed INT8 bytes), 1-cycle registered latency.
//   rd_addr [16:0] is a WORD address (byte_addr >> 2).
//   Address range : 0 .. 75,263  (301,056 / 4).
//
//   Used by:
//     • ilb_patch_merge_buf controller for the spatial-merge DMA.
//     • Off-chip flush (future MWU path).
//
// ── Status / handshake ────────────────────────────────────────────────────
//   embed_done  : 1-cycle pulse when the last write of ch-95 completes.
//                 Signals the controller to begin spatial-merge.
//   flush_req   : asserted by controller to begin sequential off-chip export.
//   flush_done  : asserted for 1 cycle when every word has been clocked out.
//                 (Actual off-chip bus is external to this module.)
//
// =============================================================================

module ilb_patch_embed_buf #(
    parameter int H      = 56,    // spatial height of feature map
    parameter int W      = 56,    // spatial width  of feature map
    parameter int C      = 96,    // number of output channels
    parameter int N_ROWS = 7      // MMU burst height (PE rows)
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // WRITE PORT
    // Receives N_ROWS quantised INT8 pixels per cycle from the post-quantiser
    // path.  The controller advances wr_col_grp every cycle, then wr_row,
    // then wr_ch, exactly mirroring the convolution scheduling above.
    // =========================================================================
    input  logic        wr_en,
    input  logic [6:0]  wr_ch,           // output channel  0 .. C-1
    input  logic [5:0]  wr_row,          // spatial row     0 .. H-1
    input  logic [2:0]  wr_col_grp,      // column group    0 .. (W/N_ROWS)-1
    input  logic [7:0]  wr_data [0:N_ROWS-1],  // 7 INT8 pixel values

    // =========================================================================
    // READ PORT — 32-bit word, 1-cycle latency
    // rd_addr is a WORD (4-byte) index into the flat channel-first layout.
    // byte_addr = rd_addr << 2.
    // =========================================================================
    input  logic        rd_en,
    input  logic [16:0] rd_addr,         // word addr   0 .. TOTAL_WORDS-1
    output logic [31:0] rd_data,

    // =========================================================================
    // STATUS
    // =========================================================================
    output logic        embed_done,      // 1-cycle pulse: full feature map written
    input  logic        flush_req,       // controller requests off-chip DMA
    output logic        flush_done       // 1-cycle pulse: flush complete
);

    // ── Derived sizing ────────────────────────────────────────────────────────
    localparam int PIXELS_PER_CH = H * W;             // 3136
    localparam int TOTAL_BYTES   = C * PIXELS_PER_CH; // 301 056
    localparam int TOTAL_WORDS   = TOTAL_BYTES / 4;   // 75 264
    localparam int COL_GROUPS    = W / N_ROWS;         // 8  (56/7)

    // ── Backing store — flat byte array ───────────────────────────────────────
    // Address mapping: mem[ ch * PIXELS_PER_CH + row * W + col ]
    logic [7:0] mem [0:TOTAL_BYTES-1];

    initial begin
        for (int i = 0; i < TOTAL_BYTES; i++) mem[i] = '0;
    end

    // =========================================================================
    // Write logic — 7 bytes per cycle
    //
    // For group g and row offset r:
    //   col      = wr_col_grp * N_ROWS + r
    //   byte_addr = wr_ch * PIXELS_PER_CH + wr_row * W + col
    //
    // Guard: col < W prevents writes on the last partial group if W % N_ROWS ≠ 0.
    // (W=56, N_ROWS=7  →  56%7==0, so no partial group; guard is safety net.)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int col   = int'(wr_col_grp) * N_ROWS + r;
                automatic int baddr = int'(wr_ch)  * PIXELS_PER_CH
                                    + int'(wr_row) * W
                                    + col;
                if (col < W)
                    mem[baddr] <= wr_data[r];
            end
        end
    end

    // =========================================================================
    // Read logic — 4 packed INT8 bytes, 1-cycle registered latency
    //
    // Little-endian packing:
    //   rd_data[7:0]   ← mem[ rd_addr*4 + 0 ]   (lowest byte / lowest channel)
    //   rd_data[31:24] ← mem[ rd_addr*4 + 3 ]
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else if (rd_en) begin
            automatic int bbase = int'(rd_addr) * 4;
            rd_data <= { mem[bbase + 3],
                         mem[bbase + 2],
                         mem[bbase + 1],
                         mem[bbase    ] };
        end
    end

    // =========================================================================
    // embed_done — 1-cycle pulse on the clock after the final write
    //
    // The final write occurs when all three indices are at their maximum:
    //   wr_ch      == C-1         (channel 95)
    //   wr_row     == H-1         (row 55)
    //   wr_col_grp == COL_GROUPS-1 (group 7, covering cols 49..55)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            embed_done <= 1'b0;
        else
            embed_done <= wr_en
                          && (wr_ch      == 7'(C - 1))
                          && (wr_row     == 6'(H - 1))
                          && (wr_col_grp == 3'(COL_GROUPS - 1));
    end

    // =========================================================================
    // Off-chip flush — sequential word read-out (placeholder)
    //
    // When flush_req is asserted the module walks flush_addr from 0 to
    // TOTAL_WORDS-1, presenting each word on rd_data in turn.  The external
    // MWU / DMA engine reads rd_data each cycle.  flush_done pulses for 1
    // cycle when the last word has been presented.
    //
    // Note: The actual off-chip bus and MWU interface are external to this
    // module.  This block simply sequences the address counter; the controller
    // is expected to gate flush_req appropriately.
    // =========================================================================
    logic [16:0] flush_addr_r;
    logic        flushing_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            flushing_r    <= 1'b0;
            flush_addr_r  <= '0;
            flush_done    <= 1'b0;
        end else begin
            flush_done <= 1'b0;

            if (flush_req && !flushing_r) begin
                // Rising edge of flush_req: start the sequential walk
                flushing_r   <= 1'b1;
                flush_addr_r <= '0;
            end else if (flushing_r) begin
                if (flush_addr_r == 17'(TOTAL_WORDS - 1)) begin
                    flushing_r   <= 1'b0;
                    flush_addr_r <= '0;
                    flush_done   <= 1'b1;
                end else begin
                    flush_addr_r <= flush_addr_r + 1'b1;
                end
            end
        end
    end

endmodule
// =============================================================================
// End of ilb_patch_embed_buf.sv
// =============================================================================
