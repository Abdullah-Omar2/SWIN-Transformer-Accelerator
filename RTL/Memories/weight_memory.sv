// =============================================================================
// weight_memory.sv  (rev 6 — Patch Merging address map corrected)
//
// ── What changed from rev 5 ───────────────────────────────────────────────
//
//   BUG FIX — Patch Merging weight layout was doubly wrong.
//
//   Ground truth from ilb_patch_merge_buf:
//     Phase-A : spatial 2×2 concat  →  C_SPA = 4 × C_IN bytes per patch
//     Phase-B : single FC layer     →  C_SPA inputs → C_FC outputs
//     (no second FC phase exists in ilb_patch_merge_buf)
//
//   Error 1 — phantom "W2":
//     rev 5 listed both W1 (narrow) and W2 (wide) for PM.
//     Patch Merging has ONE weight matrix W [C_SPA × C_FC].
//     There is no W2.
//
//   Error 2 — inverted column word count:
//     A weight matrix column = one output neuron = C_SPA input weights.
//     col_words = C_SPA / 4  (not C_FC / 4 as rev 5 stated).
//       PM1: col_words = 384/4  =  96   (rev 5 said  48 — WRONG)
//       PM2: col_words = 768/4  = 192   (rev 5 said  96 — WRONG)
//       PM3: col_words = 1536/4 = 384   (rev 5 said 192 — WRONG)
//
//   Consequence: PM total words halved (phantom W2 removed):
//     PM1:  96 w/col ×  192 cols =  18,432 words  (rev 5:  36,864 — WRONG)
//     PM2: 192 w/col ×  384 cols =  73,728 words  (rev 5: 147,456 — WRONG)
//     PM3: 384 w/col ×  768 cols = 294,912 words  (rev 5: 589,824 — WRONG)
//
//   DEPTH and AW are unchanged — Stage 4 Swin still dominates at 1,769,472.
//
// ── Sizing summary ────────────────────────────────────────────────────────
//
//   Mode 2'b00  Patch Embedding  (Stage 1 only)
//     96 kernels × 12 words/kernel = 1,152 words  [0..1151]
//
//   Mode 2'b01  Patch Merging  (single FC: W [C_SPA × C_FC], col-major)
//     PM1: C_SPA=384,  C_FC=192,  col_words= 96, total=  18,432 words
//     PM2: C_SPA=768,  C_FC=384,  col_words=192, total=  73,728 words
//     PM3: C_SPA=1536, C_FC=768,  col_words=384, total= 294,912 words  ← PM max
//
//   Mode 2'b10  Swin Transformer Block
//     Stage 4 (C=768, FFN=3072): 1,769,472 words  ← DEPTH ceiling
//
//   Maximum: Stage 4 Swin = 1,769,472 words
//   AW = 21  (2^21 = 2,097,152 ≥ 1,769,472)
//
// ── Address map — Patch Merging (mode 2'b01) ──────────────────────────────
//
//   W_BASE = 0 for all PM stages.
//   Column c (output neuron c): addr = c * col_words + word_in_col
//     col_words = C_SPA / 4  (driven by controller per stage)
//
//   PM1: col_words= 96, n_cols=192, [0..18431]
//   PM2: col_words=192, n_cols=384, [0..73727]
//   PM3: col_words=384, n_cols=768, [0..294911]
//
// ── Address map — Swin Block (mode 2'b10) ─────────────────────────────────
//
//   For stage channel depth C and FFN depth FFN_C=4C:
//     col_words = C/4
//
//   W_Q    [0                                   .. C*col_words-1]
//   W_K    [  C*col_words                       .. 2*C*col_words-1]
//   W_V    [2*C*col_words                       .. 3*C*col_words-1]
//   W_Proj [3*C*col_words                       .. 4*C*col_words-1]
//   W_FFN1 [4*C*col_words                       .. 4*C*col_words+FFN_C*col_words-1]
//   W_FFN2 [4*C*col_words+FFN_C*col_words       .. 4*C*col_words+2*FFN_C*col_words-1]
//
//   Stage 1 (C=96):   WQ/K/V/P=0/2304/4608/6912,    FFN1=9216,   FFN2=18432
//   Stage 2 (C=192):  WQ/K/V/P=0/9216/18432/27648,  FFN1=36864,  FFN2=73728
//   Stage 3 (C=384):  WQ/K/V/P=0/36864/73728/110592,FFN1=147456, FFN2=294912
//   Stage 4 (C=768):  WQ/K/V/P=0/147456/294912/442368,FFN1=589824,FFN2=1179648
//
// NOTE: Bias values not stored here — dedicated bias_buffer module.
// =============================================================================

module weight_memory #(
    parameter int DEPTH = 1769472,
    parameter int AW    = 21
)(
    input  logic          clk,
    input  logic          rst_n,

    input  logic [AW-1:0] wr_addr,
    input  logic [31:0]   wr_data,
    input  logic          wr_en,

    input  logic [AW-1:0] rd_addr,
    input  logic          rd_en,
    output logic [31:0]   rd_data
);

    logic [31:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    always_ff @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      rd_data <= '0;
        else if (rd_en)  rd_data <= mem[rd_addr];
    end

endmodule
