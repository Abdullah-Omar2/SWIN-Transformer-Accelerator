// =============================================================================
// ilb_patch_merge_buf.sv
//
// Intermediate Layer Buffer — Patch Merging
//
// This buffer orchestrates the two-step patch-merging operation that
// downsamples the Swin-Transformer feature map between stages.
//
// ── Overview of patch merging ─────────────────────────────────────────────
//   Input  (from ilb_patch_embed_buf / MLP output): (56×56, 96)  INT8
//   After Phase A (Spatial Merge — done by memory controller):  (28×28, 384) INT8
//   After Phase B (FC layer — done by MMU externally):          (28×28, 192) INT8
//
// ── Phase A — Spatial Merge (Concatenation) ───────────────────────────────
//   The controller reads from ilb_patch_embed_buf and writes into this buffer
//   performing a spatial downsampling + channel expansion:
//
//   For every output patch (pr, pc)  where pr,pc ∈ [0..27]:
//
//     Group 0 → in spatial pos (2·pr,   2·pc  ), channels 0..95
//     Group 1 → in spatial pos (2·pr,   2·pc+1), channels 0..95
//     Group 2 → in spatial pos (2·pr+1, 2·pc  ), channels 0..95
//     Group 3 → in spatial pos (2·pr+1, 2·pc+1), channels 0..95
//
//   Concatenation:  out[pr][pc][g·96 + ch] = in[2·pr+(g>>1)][2·pc+(g&1)][ch]
//   Result shape: 28 × 28 × (4 × 96) = 28 × 28 × 384 = 301,056 bytes.
//
//   This operation requires NO arithmetic — it is a pure memory-to-memory
//   rearrangement driven entirely by the controller addressing.  Hence
//   "done by the memory" as specified.
//
//   Write interface: spa_wr_* — one INT8 byte per cycle, flat byte address.
//   The controller increments spa_wr_addr sequentially as it reads from the
//   embed buffer (iterating: pr, pc, group, channel).
//
//   Phase-A storage layout (patch-major, channel-minor):
//     byte_addr = pr × (W_OUT × C_SPA) + pc × C_SPA + merge_ch
//     where merge_ch = group × C_IN + ch   (0..383)
//     Total: 28 × 28 × 384 = 301,056 bytes.
//
// ── Phase-A off-chip export ───────────────────────────────────────────────
//   After spa_done pulses, the controller DMA's (28×28, 384) to off-chip
//   memory via the pa_rd_* port.  The actual DMA engine is external.
//
// ── Phase B — Store FC-layer output ───────────────────────────────────────
//   The MMU executes the fully-connected layer (384 → 192) externally, reading
//   (28×28, 384) from off-chip.  Its 7-output-per-cycle results are written
//   back into THIS buffer through the fc_wr_* port (same burst style as the
//   other ILB sub-buffers).
//
//   Phase-B storage layout (patch-major, channel-minor):
//     byte_addr = patch × C_FC + fc_col
//     patch     = fc_wr_row_base + r   (r = 0..N_ROWS-1)
//     Total:  28 × 28 × 192 = 150,528 bytes.
//
//   Because 150,528 < 301,056 the same backing store is reused from address 0
//   after the Phase-A data has been flushed to off-chip.
//
// ── Phase-B read port (pb_rd_*) ───────────────────────────────────────────
//   After fc_done the downstream Swin-block stages read (28×28, 192) from
//   this buffer using patch + column addressing.  Returns 4 packed INT8 bytes.
//
// ── Status signals ────────────────────────────────────────────────────────
//   spa_done   : 1-cycle pulse when the final Phase-A byte is written.
//   fc_done    : 1-cycle pulse when the final Phase-B byte is written.
//   flush_req  : asserted to begin sequential word read-out of Phase-A result.
//   flush_done : 1-cycle pulse when Phase-A flush completes.
//
// ── Sizing summary ────────────────────────────────────────────────────────
//   Backing store : 301,056 bytes  (≈ 294 KB)
//   Phase A writes: 301,056 bytes  (1 byte/cycle → 301,056 cycles)
//   Phase B writes: 150,528 bytes  (7 bytes/cycle → ~21,504 write-groups)
//   32-bit data bus throughout.
//
// =============================================================================

module ilb_patch_merge_buf #(
    parameter int H_IN   = 56,   // input spatial height
    parameter int W_IN   = 56,   // input spatial width
    parameter int C_IN   = 96,   // input channels
    parameter int H_OUT  = 28,   // output spatial height  (= H_IN / 2)
    parameter int W_OUT  = 28,   // output spatial width   (= W_IN / 2)
    parameter int C_SPA  = 384,  // spatial-merge output channels (= 4 × C_IN)
    parameter int C_FC   = 192,  // FC-layer output channels
    parameter int N_ROWS = 7     // MMU burst height (matches output_buffer N_OUT)
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // PHASE-A WRITE PORT — Spatial-merge byte stream
    //
    // The controller computes the destination address for each byte as it
    // streams data from ilb_patch_embed_buf:
    //
    //   spa_wr_addr = pr × (W_OUT × C_SPA) + pc × C_SPA + g × C_IN + ch
    //
    // where (pr, pc) is the 28×28 output patch, g ∈ {0,1,2,3} is the 2×2
    // group index, and ch ∈ {0..95} is the channel within that group.
    //
    // One byte is written per cycle; spa_wr_addr increments each cycle.
    // Address range: 0 .. SPA_BYTES-1 = 0 .. 301,055.
    // =========================================================================
    input  logic        spa_wr_en,
    input  logic [17:0] spa_wr_addr,   // flat byte address  0 .. 301,055
    input  logic [7:0]  spa_wr_data,   // one INT8 byte

    // =========================================================================
    // PHASE-A READ PORT — 32-bit word, 1-cycle latency
    //
    // Used for:
    //   • Off-chip DMA of (28×28, 384) result (driven by flush sequencer).
    //   • Any controller inspection of Phase-A content.
    //
    // pa_rd_addr [16:0] : WORD address  0 .. SPA_WORDS-1  (= 0 .. 75,263)
    // =========================================================================
    input  logic        pa_rd_en,
    input  logic [16:0] pa_rd_addr,
    output logic [31:0] pa_rd_data,

    // =========================================================================
    // PHASE-B WRITE PORT — FC-layer result  (7 INT8 bytes per cycle)
    //
    // Matches the burst style of ilb_proj_buf / ilb_ffn1_buf.
    //
    // fc_wr_row_base [9:0] : first patch of burst  (0, 7, 14, …, 777)
    //                        patch = fc_wr_row_base + r,  r = 0..N_ROWS-1
    //                        Total patches: 28×28 = 784.
    // fc_wr_col      [7:0] : byte column within the 192-channel vector (0..191)
    // fc_wr_data[r]  [7:0] : INT8 value for patch (row_base + r), column fc_col
    //
    // Storage: mem[ patch × C_FC + fc_wr_col ]
    // =========================================================================
    input  logic        fc_wr_en,
    input  logic [9:0]  fc_wr_row_base,  // 0 .. FC_PATCHES-N_ROWS  (= 777)
    input  logic [7:0]  fc_wr_col,       // 0 .. C_FC-1             (= 191)
    input  logic [7:0]  fc_wr_data [0:N_ROWS-1],

    // =========================================================================
    // PHASE-B READ PORT — 4 packed INT8 bytes, 1-cycle latency
    //
    // Downstream (next Swin-block input loader) reads the (28×28, 192) result.
    //
    // pb_rd_patch [9:0] : patch index  0 .. 783
    // pb_rd_col   [7:0] : byte column, WORD-aligned (0, 4, 8, …, 188)
    //                     i.e., the byte-column of the first byte in the word.
    // pb_rd_data [31:0] : { ch+3, ch+2, ch+1, ch } packed INT8
    // =========================================================================
    input  logic        pb_rd_en,
    input  logic [9:0]  pb_rd_patch,
    input  logic [7:0]  pb_rd_col,
    output logic [31:0] pb_rd_data,

    // =========================================================================
    // STATUS
    // =========================================================================
    output logic        spa_done,    // 1-cycle pulse: Phase-A fully written
    output logic        fc_done,     // 1-cycle pulse: Phase-B fully written
    input  logic        flush_req,   // request off-chip DMA of Phase-A result
    output logic        flush_done   // 1-cycle pulse: flush complete
);

    // ── Sizing ────────────────────────────────────────────────────────────────
    localparam int SPA_BYTES   = H_OUT * W_OUT * C_SPA;  // 301 056
    localparam int SPA_WORDS   = SPA_BYTES / 4;           // 75 264
    localparam int FC_PATCHES  = H_OUT * W_OUT;            // 784
    localparam int FC_BYTES    = FC_PATCHES * C_FC;        // 150 528
    localparam int MEM_BYTES   = SPA_BYTES;                // max(SPA, FC)

    // ── Backing store ─────────────────────────────────────────────────────────
    // Single flat byte array shared by both phases (Phase B fits within
    // Phase A's footprint so the same RAM is safely reused after flush).
    logic [7:0] mem [0:MEM_BYTES-1];

    initial begin
        for (int i = 0; i < MEM_BYTES; i++) mem[i] = '0;
    end

    // =========================================================================
    // Phase-A Write — one byte per cycle at flat byte address
    // =========================================================================
    always_ff @(posedge clk) begin
        if (spa_wr_en) begin
            mem[spa_wr_addr] <= spa_wr_data;
        end
    end

    // =========================================================================
    // Phase-A Read — 32-bit word, 1-cycle registered latency
    // Little-endian: rd_data[7:0] ← mem[addr*4], rd_data[31:24] ← mem[addr*4+3]
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pa_rd_data <= '0;
        end else if (pa_rd_en) begin
            automatic int bbase = int'(pa_rd_addr) * 4;
            pa_rd_data <= { mem[bbase + 3],
                            mem[bbase + 2],
                            mem[bbase + 1],
                            mem[bbase    ] };
        end
    end

    // =========================================================================
    // Phase-B Write — 7 INT8 bytes per cycle (FC-layer result burst)
    //
    // Byte address: patch * C_FC + fc_wr_col
    // Guard on patch < FC_PATCHES handles last burst group safely.
    // =========================================================================
    always_ff @(posedge clk) begin
        if (fc_wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(fc_wr_row_base) + r;
                if (patch < FC_PATCHES) begin
                    automatic int baddr = patch * C_FC + int'(fc_wr_col);
                    mem[baddr] <= fc_wr_data[r];
                end
            end
        end
    end

    // =========================================================================
    // Phase-B Read — 4 packed INT8 bytes, 1-cycle registered latency
    //
    // Byte base = pb_rd_patch * C_FC + pb_rd_col
    // pb_rd_col must be word-aligned (caller's responsibility).
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pb_rd_data <= '0;
        end else if (pb_rd_en) begin
            automatic int bbase = int'(pb_rd_patch) * C_FC + int'(pb_rd_col);
            pb_rd_data <= { mem[bbase + 3],
                            mem[bbase + 2],
                            mem[bbase + 1],
                            mem[bbase    ] };
        end
    end

    // =========================================================================
    // spa_done — 1-cycle pulse when the final Phase-A byte is written
    //
    // The last byte is the 384th channel of the last output patch (783):
    //   spa_wr_addr = 783 * 384 + 383 = 300,671 + 383 = 301,055 = SPA_BYTES-1
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            spa_done <= 1'b0;
        else
            spa_done <= spa_wr_en && (spa_wr_addr == 18'(SPA_BYTES - 1));
    end

    // =========================================================================
    // fc_done — 1-cycle pulse when the final Phase-B burst write completes
    //
    // The last write group has fc_wr_row_base = FC_PATCHES - N_ROWS = 777
    // (patches 777..783) and fc_wr_col = C_FC - 1 = 191.
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fc_done <= 1'b0;
        else
            fc_done <= fc_wr_en
                       && (fc_wr_row_base == 10'(FC_PATCHES - N_ROWS))
                       && (fc_wr_col      ==  8'(C_FC - 1));
    end

    // =========================================================================
    // Off-chip flush sequencer — Phase-A DMA placeholder
    //
    // When flush_req is asserted (and a flush is not already running), the
    // module increments flush_addr_r from 0 to SPA_WORDS-1, making each word
    // available on pa_rd_data for the external DMA engine to capture.
    // flush_done pulses for exactly 1 cycle after the last word is presented.
    // =========================================================================
    logic [16:0] flush_addr_r;
    logic        flushing_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            flushing_r   <= 1'b0;
            flush_addr_r <= '0;
            flush_done   <= 1'b0;
        end else begin
            flush_done <= 1'b0;

            if (flush_req && !flushing_r) begin
                flushing_r   <= 1'b1;
                flush_addr_r <= '0;
            end else if (flushing_r) begin
                if (flush_addr_r == 17'(SPA_WORDS - 1)) begin
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
// End of ilb_patch_merge_buf.sv
// =============================================================================
