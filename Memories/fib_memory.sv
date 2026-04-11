// =============================================================================
// fib_memory.sv  (rev 6 — MODE 1 Patch Merging correctly documented)
//
// ── What changed from rev 5 ───────────────────────────────────────────────
//   DOCUMENTATION FIX — MODE 1 comment was wrong.
//
//   Rev 5 stated: "The FIB is NOT used by Patch Merging."
//   This contradicts the accelerator paper:
//
//     "The dataflow of the Patch Merging computations mode is similar to
//      the mode Patch Embedding.  MRU reads data from external memory
//      and writes it into the FIB; DSU sends data and weight parameters
//      to the MMU."
//
//   The FIB IS used in Patch Merging.  The MRU loads the Phase-A spatial-
//   merge result (C_SPA bytes per patch) from off-chip into the FIB before
//   the FC computation begins.  The DSU then reads the FIB exactly as it
//   does for Conv and Swin Block modes — via Port A with a flat word address.
//
//   No RTL change is required: Port A already serves Mode 1 correctly.
//   The controller supplies the right flat word addresses; the FIB is
//   agnostic to which mode is active.
//
// ── Why DEPTH and AW are unchanged across all modes and stages ───────────
//
//   The FIB is loaded ONCE per top-level mode from off-chip memory, then
//   read throughout that mode's entire computation.  Its DEPTH is the max
//   of all per-mode footprints:
//
//   MODE 0  Patch Embedding  (Stage 1 only — architecture is fixed)
//     Input: 224×224×3 image, CHW, 4 bytes/word
//     224×224×3 / 4 = 37,632 words  [0..37,631]
//
//   MODE 1  Patch Merging  (3 PM stages between Swin stages 1–4)
//     The FIB holds the Phase-A spatial-merge result: C_SPA bytes per patch.
//     The MRU loads this from external memory before the FC pass begins.
//     Layout: patch-major, feature-minor (identical structure to Swin mode).
//     Address formula (Port A):
//       addr = patch_idx × (C_SPA/4) + k_word
//       patch_idx = pr × W_OUT + pc
//
//     PM1 (56×56→28×28, C_SPA=384):  28×28×384/4  = 75,264 words  ← PM max
//     PM2 (28×28→14×14, C_SPA=768):  14×14×768/4  = 37,632 words
//     PM3 (14×14→7×7, C_SPA=1536):    7×7×1536/4  = 18,816 words
//
//     PM1 footprint = 75,264 words — equal to the Swin Block Stage 1 size.
//     All PM stages fit within DEPTH = 75,264. ✓
//
//   MODE 2 / 3  Swin Block  (W-MSA / SW-MSA, all stages)
//     The FIB holds the feature map X that is the MSA input.
//     Layout: patch-major, feature-minor.
//     Address formula (Port A):
//       addr = patch_idx × (C/4) + k_word
//       patch_idx = row × W_FM + col
//
//     Stage 1 (56×56, C=96):    56×56×96/4  = 75,264 words  ← DEPTH ceiling
//     Stage 2 (28×28, C=192):   28×28×192/4 = 37,632 words
//     Stage 3 (14×14, C=384):   14×14×384/4 =  9,408 words
//     Stage 4 (7×7,   C=768):    7×7×768/4  =  9,408 words
//
//   Maximum across all modes: 75,264 words (Mode 0/1 PM1 / Mode 2 Stage 1).
//   DEPTH = 75,264.  AW = 17  (2^17 = 131,072 ≥ 75,264).
//   Both are UNCHANGED from rev 5.
//
//   Note: Modes 1 and 2/3 share the same patch-major flat layout, so the
//   controller uses the same address formula for both; only the spatial
//   dimensions (W_FM/W_OUT) and channel word count (C/4 or C_SPA/4) differ.
//
// ── Port A address formula (Modes 0 / 1 / 2 / 3) ─────────────────────────
//
//   Mode 0 (Conv, CHW layout):
//     addr = ch × C_IMG_CH_WORDS + in_row × C_IMG_W_WORDS + col_word
//     C_IMG_CH_WORDS = 224×224/4 = 12,544
//     C_IMG_W_WORDS  = 224/4     = 56
//
//   Mode 1 (Patch Merging — patch-major):
//     addr = patch_idx × (C_SPA/4) + k_word
//     PM1: word_cnt=96,  PM2: word_cnt=192,  PM3: word_cnt=384
//
//   Mode 2/3 (Swin Block — patch-major, same formula as Mode 1):
//     addr = patch_idx × (C/4) + k_word
//     Stage 1: word_cnt=24, Stage 2: 48, Stage 3: 96, Stage 4: 192
//
// ── Port B (SW-MSA cyclic shift) — Modes 2/3 only, unchanged ─────────────
//   Port B is not used in Modes 0 or 1.
//   fm_h / fm_w runtime ports set the active spatial dimensions per stage:
//     Stage 1: fm_h=56, fm_w=56   Stage 2: fm_h=28, fm_w=28
//     Stage 3: fm_h=14, fm_w=14   Stage 4: fm_h=7,  fm_w=7
//   shift_h = shift_w = 3 (always floor(M/2) for M=7).
// =============================================================================

module fib_memory #(
    parameter int  DEPTH   = 75264,
    parameter int  AW      = 17,
    parameter int  FM_H    = 56,    // Stage 1 spatial height (max)
    parameter int  FM_W    = 56,    // Stage 1 spatial width  (max)
    parameter int  FM_CH_W = 24     // max channel words per patch (Stage1: 96/4)
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── Write port (DMA / CPU) ────────────────────────────────────────────
    input  logic [AW-1:0] wr_addr,
    input  logic [31:0]   wr_data,
    input  logic          wr_en,

    // ════════════════════════════════════════════════════════════════════
    // PORT A — direct-address read
    //
    // Used by ALL four modes:
    //   Mode 0  Conv (Patch Embedding) — CHW image
    //   Mode 1  Patch Merging (FC pass) — Phase-A spatial-merge result
    //   Mode 2  W-MSA  — feature map X, patch-major
    //   Mode 3  SW-MSA — feature map X, patch-major (shift applied at FIB level via Port B)
    //
    // The controller computes rd_addr from the mode-specific formula
    // described in the header; the FIB is agnostic to which mode is active.
    // ════════════════════════════════════════════════════════════════════
    input  logic [AW-1:0] rd_addr,
    input  logic          rd_en,
    output logic [31:0]   rd_data,

    // ════════════════════════════════════════════════════════════════════
    // PORT B — cyclic-shifted read  (SW-MSA, modes 2/3 only)
    //
    // NOT used in Mode 0 (Conv) or Mode 1 (Patch Merging).
    //
    // Controller provides LOGICAL (row, col, k_word) in shifted space.
    // Runtime fm_h / fm_w let the same hardware serve all Swin stages:
    //   Stage1: fm_h=56, fm_w=56   Stage2: fm_h=28, fm_w=28
    //   Stage3: fm_h=14, fm_w=14   Stage4: fm_h=7,  fm_w=7
    //
    // Physical address:
    //   phys_row  = (sw_rd_row + shift_h) % fm_h    [conditional subtract]
    //   phys_col  = (sw_rd_col + shift_w) % fm_w
    //   phys_addr = (phys_row * fm_w + phys_col) * sw_rd_k_word_cnt + sw_rd_k_word
    //
    // sw_rd_k_word_cnt: channel words per patch for active Swin stage (driven by ctrl)
    //   Stage1=24, Stage2=48, Stage3=96, Stage4=192
    //
    // shift_h = shift_w = 3 (always floor(M/2) for window size M=7).
    // ════════════════════════════════════════════════════════════════════
    input  logic [$clog2(FM_H)-1:0]    sw_rd_row,
    input  logic [$clog2(FM_W)-1:0]    sw_rd_col,
    input  logic [$clog2(FM_CH_W)-1:0] sw_rd_k_word,
    input  logic                        sw_rd_en,
    output logic [31:0]                 sw_rd_data,

    // Runtime spatial dimensions for Port B (set per stage, stable during round)
    input  logic [$clog2(FM_H)-1:0]    shift_h,       // always 3 for M=7
    input  logic [$clog2(FM_W)-1:0]    shift_w,       // always 3 for M=7
    input  logic [$clog2(FM_H):0]      fm_h,          // active stage height
    input  logic [$clog2(FM_W):0]      fm_w,          // active stage width
    input  logic [$clog2(FM_CH_W*8)-1:0] sw_k_word_cnt // channel words/patch
);

    logic [31:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── Write ─────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end

    // ── Port A: direct read (1-cycle latency) ─────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      rd_data <= '0;
        else if (rd_en)  rd_data <= mem[rd_addr];
    end

    // ── Port B: cyclic-shifted read ────────────────────────────────────────
    // Modulo by single conditional subtraction (shift < fm_h,w guaranteed).
    logic [$clog2(FM_H):0]   row_sum;
    logic [$clog2(FM_W):0]   col_sum;
    logic [$clog2(FM_H)-1:0] phys_row;
    logic [$clog2(FM_W)-1:0] phys_col;
    logic [AW-1:0]            sw_phys_addr;

    always_comb begin
        row_sum  = {1'b0, sw_rd_row} + {1'b0, shift_h};
        col_sum  = {1'b0, sw_rd_col} + {1'b0, shift_w};

        phys_row = (row_sum >= fm_h)
                   ? row_sum[$clog2(FM_H)-1:0] - fm_h[$clog2(FM_H)-1:0]
                   : row_sum[$clog2(FM_H)-1:0];

        phys_col = (col_sum >= fm_w)
                   ? col_sum[$clog2(FM_W)-1:0] - fm_w[$clog2(FM_W)-1:0]
                   : col_sum[$clog2(FM_W)-1:0];

        sw_phys_addr = AW'(phys_row) * AW'(fm_w) * AW'(sw_k_word_cnt)
                     + AW'(phys_col) * AW'(sw_k_word_cnt)
                     + AW'(sw_rd_k_word);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)          sw_rd_data <= '0;
        else if (sw_rd_en)   sw_rd_data <= mem[sw_phys_addr];
    end

endmodule
