// =============================================================================
// ilb_swin_block.sv
//
// Intermediate Layer Buffer — Swin Transformer Block  (extended)
//
// This is the top-level ILB wrapper.  It instantiates ALL seven sub-buffers
// that together cover:
//   • Patch Embedding convolution output          (ilb_patch_embed_buf)
//   • Patch Merging spatial concat + FC result    (ilb_patch_merge_buf)
//   • One complete Swin Block (W-MSA or SW-MSA + FFN):
//       QKV / Proj, Attention scores, Context vectors,
//       Proj+residual, FFN1-GELU intermediate.
//
// ── Sub-buffer inventory ─────────────────────────────────────────────────
//
//   Buffer               Module                Shape           Bytes
//   ───────────────────  ────────────────────  ──────────────  ─────────
//   Patch embed output   ilb_patch_embed_buf   56×56×96        301,056 B
//   Patch merge store    ilb_patch_merge_buf   28×28×384(A)    301,056 B
//                                              28×28×192(B)    150,528 B
//   QKV / Proj store     ilb_qkv_buf           49×96             4,704 B
//   Attention scores     ilb_score_buf         49×49×32          9,604 B
//   Context vectors      ilb_context_buf       49×96             4,704 B
//   Proj + residual      ilb_proj_buf          49×96             4,704 B
//   FFN1 intermediate    ilb_ffn1_buf          49×384           18,816 B
//   ───────────────────  ────────────────────  ──────────────  ─────────
//   Total on-chip peak                                         794,952 B (≈ 776 KB)
//   (patch_embed and patch_merge Phase-A overlap in practice; both cannot
//    be simultaneously active since patch_merge writes begin only after
//    patch_embed is flushed.)
//
// ── Stage-0 : Patch Embedding ─────────────────────────────────────────────
//
//   Step  Operation                  Read from        Write to
//   ────  ─────────────────────────  ───────────────  ─────────────────────
//   PE-1  Conv 4×4×3, stride 4,      Input SRAM/FIB   output_buffer (OB)
//         96 kernels → (56×56,96)    (224×224×3)      ↓ quantiser
//   PE-2  Quantise + store           output_buffer    ilb_patch_embed_buf
//         (7 pixels/cycle, 8 cyc/    (7 pixels/cyc,   wr_ch/wr_row/wr_col_grp
//          row, 448 cyc/ch)           43,008 cyc total)
//
// ── Stage-1 : Patch Merging ──────────────────────────────────────────────
//
//   Step  Operation                  Read from              Write to
//   ────  ─────────────────────────  ─────────────────────  ──────────────────
//   PM-1  Spatial Merge (by memory)  ilb_patch_embed_buf    ilb_patch_merge_buf
//         Concatenate 2×2 pixels     (rd_en / rd_addr)      (spa_wr_en/addr/data)
//         → (28×28, 384)             1 byte/cyc read        1 byte/cyc write
//   PM-2  Flush (28×28,384)         ilb_patch_merge_buf    Off-chip memory
//         to off-chip               (pa_rd_* / flush_req)  (MWU — external)
//   PM-3  FC layer 384→192 (by MMU) Off-chip (28×28,384)   output_buffer
//         28×28 patches × FC kernel  (external to this ILB) ↓ quantiser
//   PM-4  Store FC result            output_buffer           ilb_patch_merge_buf
//         (7 patches/cyc,            (fc_wr_en/row/col)      fc_wr_data[7]
//          ~21,504 write groups)
//
// ── Stage-2 : Swin Block (W-MSA / SW-MSA + FFN) ───────────────────────────
//
//   Step  Operation             Read from          Write to
//   ────  ───────────────────── ─────────────────  ──────────────────
//   1     X × W_Q → Q           FIB (X)            ilb_qkv_buf
//   2     X × W_K → K           FIB (X)            ilb_qkv_buf (swap)
//   3     X × W_V → V           FIB (X)            ilb_qkv_buf (swap)
//   4a    Q_h × K_h^T → S_h     ilb_qkv_buf (Q,K)  ilb_score_buf
//   4b    mask(S_h) [SW only]    ilb_score_buf RMW  ilb_score_buf (in-place)
//   4c    Softmax(S_h)           ilb_score_buf      ilb_score_buf (in-place)
//   5     S_h × V_h → A_h       ilb_score_buf,     ilb_context_buf
//                                ilb_qkv_buf (V)
//   6     concat(A)×W_P→M_out   ilb_context_buf    ilb_qkv_buf (Proj)
//   7     M_out + X → Out_MSA   ilb_qkv_buf + FIB  ilb_proj_buf (RMW)
//   8     Out_MSA × W_FFN1      ilb_proj_buf        → GCU directly
//   9     GELU(step8)           GCU output          ilb_ffn1_buf
//   10    Z × W_FFN2 → F_out    ilb_ffn1_buf        → ilb_proj_buf (RMW)
//   11    F_out + Out_MSA       ilb_proj_buf RMW    ilb_proj_buf (in-place)
//   12    flush to off-chip     ilb_proj_buf        MWU → off-chip
//
// ── Connections to the wider system ──────────────────────────────────────
//   All sub-buffer ports are exposed flat at this top level.
//   unified_controller drives all addresses, enables, and phase sequencing.
//   The residual adder and off-chip DMA engine live in full_system_top.
//   MMU is separate (mmu_top.sv) — NOT touched by this file.
//
// ── What this module does NOT contain ────────────────────────────────────
//   • MMU / PE array             — mmu_top.sv (untouched)
//   • Softmax unit (SCU)         — separate
//   • GELU unit (GCU)            — separate
//   • Mask buffer                — mask_buffer.sv (separate)
//   • Residual adder             — combinatorial, in full_system_top
//   • Output quantiser / post-proc — full_system_top pipeline
//   • MWU / off-chip DMA engine  — full_system_top
//
// =============================================================================

module ilb_swin_block #(
    // ── Patch Embedding / Merging ──────────────────────────────────────────
    parameter int H_EMB  = 56,    // embed feature map height
    parameter int W_EMB  = 56,    // embed feature map width
    parameter int C_EMB  = 96,    // embed output channels
    parameter int H_MRG  = 28,    // merge output spatial height
    parameter int W_MRG  = 28,    // merge output spatial width
    parameter int C_SPA  = 384,   // merge Phase-A channels (4 × C_EMB)
    parameter int C_FC   = 192,   // merge Phase-B FC channels
    // ── Swin Block ────────────────────────────────────────────────────────
    parameter int N_PATCHES  = 49,
    parameter int C_MSA      = 96,
    parameter int C_FFN      = 384,
    parameter int N_HEADS    = 3,
    parameter int HEAD_DIM   = 32,
    parameter int N_ROWS     = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // ilb_patch_embed_buf — Patch Embedding output  (56×56×96 INT8)
    // =========================================================================
    // Write port (from post-quantiser output path, 7 pixels/cycle)
    input  logic        emb_wr_en,
    input  logic [6:0]  emb_wr_ch,           // channel   0 .. 95
    input  logic [5:0]  emb_wr_row,          // row       0 .. 55
    input  logic [2:0]  emb_wr_col_grp,      // col group 0 .. 7
    input  logic [7:0]  emb_wr_data [0:N_ROWS-1],

    // Read port (32-bit word, used by patch-merge controller)
    input  logic        emb_rd_en,
    input  logic [16:0] emb_rd_addr,         // word addr 0 .. 75263
    output logic [31:0] emb_rd_data,

    // Status
    output logic        emb_embed_done,      // pulse: all 96ch written
    input  logic        emb_flush_req,
    output logic        emb_flush_done,

    // =========================================================================
    // ilb_patch_merge_buf — Patch Merging store  (28×28×384 / 28×28×192)
    // =========================================================================
    // Phase-A write (spatial merge, 1 byte/cycle)
    input  logic        mrg_spa_wr_en,
    input  logic [17:0] mrg_spa_wr_addr,     // byte addr 0 .. 301055
    input  logic [7:0]  mrg_spa_wr_data,

    // Phase-A read (32-bit word for off-chip DMA)
    input  logic        mrg_pa_rd_en,
    input  logic [16:0] mrg_pa_rd_addr,      // word addr 0 .. 75263
    output logic [31:0] mrg_pa_rd_data,

    // Phase-B write (FC-layer result, 7 bytes/cycle from output buffer)
    input  logic        mrg_fc_wr_en,
    input  logic [9:0]  mrg_fc_wr_row_base,  // patch 0 .. 777
    input  logic [7:0]  mrg_fc_wr_col,       // channel byte 0 .. 191
    input  logic [7:0]  mrg_fc_wr_data [0:N_ROWS-1],

    // Phase-B read (4-byte word for downstream Swin input)
    input  logic        mrg_pb_rd_en,
    input  logic [9:0]  mrg_pb_rd_patch,     // patch 0 .. 783
    input  logic [7:0]  mrg_pb_rd_col,       // byte col (word-aligned)
    output logic [31:0] mrg_pb_rd_data,

    // Status
    output logic        mrg_spa_done,        // pulse: Phase-A complete
    output logic        mrg_fc_done,         // pulse: Phase-B complete
    input  logic        mrg_flush_req,
    output logic        mrg_flush_done,

    // =========================================================================
    // ilb_qkv_buf — Q / K / V / Proj output  (49 × 96 INT8)
    // =========================================================================
    // Bank swap
    input  logic        qkv_swap,

    // Write port
    input  logic        qkv_wr_en,
    input  logic [5:0]  qkv_wr_patch_base,
    input  logic [6:0]  qkv_wr_col,
    input  logic [7:0]  qkv_wr_data [0:N_ROWS-1],

    // Read Port A — patch-sequential (Q rows, Proj rows)
    input  logic        qkv_rd_en_a,
    input  logic [5:0]  qkv_rd_patch_a,
    input  logic [6:0]  qkv_rd_col_a,
    output logic [31:0] qkv_rd_data_a,

    // Read Port B — column-slice (K^T columns, V columns)
    input  logic        qkv_rd_en_b,
    input  logic [5:0]  qkv_rd_row_b,
    input  logic [6:0]  qkv_rd_col_b,
    output logic [31:0] qkv_rd_data_b,

    // =========================================================================
    // ilb_score_buf — Attention scores  (49 × 49 INT32)
    // =========================================================================
    // Valid lifecycle
    input  logic        score_commit,
    input  logic        score_clear,
    output logic        score_valid,

    // Write port — QK^T accumulation
    input  logic        score_wr_en,
    input  logic [5:0]  score_wr_row_base,
    input  logic [5:0]  score_wr_col,
    input  logic [31:0] score_wr_data [0:N_ROWS-1],

    // RMW port — mask application (SW-MSA) and Softmax write-back
    input  logic        score_rmw_rd_en,
    input  logic [11:0] score_rmw_addr,
    output logic [31:0] score_rmw_rd_data,
    input  logic        score_rmw_wr_en,
    input  logic [31:0] score_rmw_wr_data,

    // Sequential read — Softmax input / S×V operand
    input  logic        score_rd_en,
    input  logic [11:0] score_rd_addr,
    output logic [31:0] score_rd_data,

    // =========================================================================
    // ilb_context_buf — Context vectors  (49 × 96 INT8, 3 heads concat)
    // =========================================================================
    // Write port — S×V output per head
    input  logic        ctx_wr_en,
    input  logic [5:0]  ctx_wr_patch_base,
    input  logic [1:0]  ctx_wr_head,
    input  logic [4:0]  ctx_wr_head_col,
    input  logic [7:0]  ctx_wr_data [0:N_ROWS-1],

    // Read port — load concat(A) into ibuf for Proj
    input  logic        ctx_rd_en,
    input  logic [5:0]  ctx_rd_patch,
    input  logic [6:0]  ctx_rd_col,
    output logic [31:0] ctx_rd_data,

    // =========================================================================
    // ilb_proj_buf — Proj output + MSA residual + FFN residual  (49 × 96 INT8)
    // =========================================================================
    // Write port — Proj MMU output, or FFN2+residual final result
    input  logic        proj_wr_en,
    input  logic [5:0]  proj_wr_patch_base,
    input  logic [6:0]  proj_wr_col,
    input  logic [7:0]  proj_wr_data [0:N_ROWS-1],

    // Read port — FFN input loading, or MWU flush
    input  logic        proj_rd_en,
    input  logic [5:0]  proj_rd_patch,
    input  logic [6:0]  proj_rd_col,
    output logic [31:0] proj_rd_data,

    // RMW port — residual additions (MSA: step 7; FFN: step 11)
    input  logic        proj_rmw_rd_en,
    input  logic [10:0] proj_rmw_addr,
    output logic [31:0] proj_rmw_rd_data,
    input  logic        proj_rmw_wr_en,
    input  logic [31:0] proj_rmw_wr_data,

    // =========================================================================
    // ilb_ffn1_buf — FFN1 GELU output  (49 × 384 INT8)
    // =========================================================================
    // Write port — GCU (GELU) output
    input  logic        ffn1_wr_en,
    input  logic [5:0]  ffn1_wr_patch_base,
    input  logic [8:0]  ffn1_wr_col,
    input  logic [7:0]  ffn1_wr_data [0:N_ROWS-1],

    // Read port — FFN2 activation input
    input  logic        ffn1_rd_en,
    input  logic [5:0]  ffn1_rd_patch,
    input  logic [7:0]  ffn1_rd_col_word,
    output logic [31:0] ffn1_rd_data
);

    // ── Patch Embedding buffer ────────────────────────────────────────────
    ilb_patch_embed_buf #(
        .H      (H_EMB),
        .W      (W_EMB),
        .C      (C_EMB),
        .N_ROWS (N_ROWS)
    ) u_patch_embed (
        .clk          (clk),
        .rst_n        (rst_n),
        .wr_en        (emb_wr_en),
        .wr_ch        (emb_wr_ch),
        .wr_row       (emb_wr_row),
        .wr_col_grp   (emb_wr_col_grp),
        .wr_data      (emb_wr_data),
        .rd_en        (emb_rd_en),
        .rd_addr      (emb_rd_addr),
        .rd_data      (emb_rd_data),
        .embed_done   (emb_embed_done),
        .flush_req    (emb_flush_req),
        .flush_done   (emb_flush_done)
    );

    // ── Patch Merging buffer ──────────────────────────────────────────────
    ilb_patch_merge_buf #(
        .H_IN   (H_EMB),
        .W_IN   (W_EMB),
        .C_IN   (C_EMB),
        .H_OUT  (H_MRG),
        .W_OUT  (W_MRG),
        .C_SPA  (C_SPA),
        .C_FC   (C_FC),
        .N_ROWS (N_ROWS)
    ) u_patch_merge (
        .clk              (clk),
        .rst_n            (rst_n),
        .spa_wr_en        (mrg_spa_wr_en),
        .spa_wr_addr      (mrg_spa_wr_addr),
        .spa_wr_data      (mrg_spa_wr_data),
        .pa_rd_en         (mrg_pa_rd_en),
        .pa_rd_addr       (mrg_pa_rd_addr),
        .pa_rd_data       (mrg_pa_rd_data),
        .fc_wr_en         (mrg_fc_wr_en),
        .fc_wr_row_base   (mrg_fc_wr_row_base),
        .fc_wr_col        (mrg_fc_wr_col),
        .fc_wr_data       (mrg_fc_wr_data),
        .pb_rd_en         (mrg_pb_rd_en),
        .pb_rd_patch      (mrg_pb_rd_patch),
        .pb_rd_col        (mrg_pb_rd_col),
        .pb_rd_data       (mrg_pb_rd_data),
        .spa_done         (mrg_spa_done),
        .fc_done          (mrg_fc_done),
        .flush_req        (mrg_flush_req),
        .flush_done       (mrg_flush_done)
    );

    // =========================================================================
    // Sub-buffer instantiations  (Swin Block)
    // =========================================================================

    // ── QKV / Proj buffer ─────────────────────────────────────────────────
    ilb_qkv_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_MSA),
        .N_ROWS    (N_ROWS)
    ) u_qkv (
        .clk             (clk),
        .rst_n           (rst_n),
        .swap            (qkv_swap),
        .wr_en           (qkv_wr_en),
        .wr_patch_base   (qkv_wr_patch_base),
        .wr_col          (qkv_wr_col),
        .wr_data         (qkv_wr_data),
        .rd_en_a         (qkv_rd_en_a),
        .rd_patch_a      (qkv_rd_patch_a),
        .rd_col_a        (qkv_rd_col_a),
        .rd_data_a       (qkv_rd_data_a),
        .rd_en_b         (qkv_rd_en_b),
        .rd_row_b        (qkv_rd_row_b),
        .rd_col_b        (qkv_rd_col_b),
        .rd_data_b       (qkv_rd_data_b)
    );

    // ── Attention score buffer ─────────────────────────────────────────────
    ilb_score_buf #(
        .N_PATCHES (N_PATCHES),
        .N_ROWS    (N_ROWS)
    ) u_score (
        .clk             (clk),
        .rst_n           (rst_n),
        .score_commit    (score_commit),
        .score_clear     (score_clear),
        .score_valid     (score_valid),
        .wr_en           (score_wr_en),
        .wr_row_base     (score_wr_row_base),
        .wr_col          (score_wr_col),
        .wr_data         (score_wr_data),
        .rmw_rd_en       (score_rmw_rd_en),
        .rmw_addr        (score_rmw_addr),
        .rmw_rd_data     (score_rmw_rd_data),
        .rmw_wr_en       (score_rmw_wr_en),
        .rmw_wr_data     (score_rmw_wr_data),
        .rd_en           (score_rd_en),
        .rd_addr         (score_rd_addr),
        .rd_data         (score_rd_data)
    );

    // ── Context vector buffer ─────────────────────────────────────────────
    ilb_context_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_MSA),
        .HEAD_DIM  (HEAD_DIM),
        .N_HEADS   (N_HEADS),
        .N_ROWS    (N_ROWS)
    ) u_ctx (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (ctx_wr_en),
        .wr_patch_base   (ctx_wr_patch_base),
        .wr_head         (ctx_wr_head),
        .wr_head_col     (ctx_wr_head_col),
        .wr_data         (ctx_wr_data),
        .rd_en           (ctx_rd_en),
        .rd_patch        (ctx_rd_patch),
        .rd_col          (ctx_rd_col),
        .rd_data         (ctx_rd_data)
    );

    // ── Proj + residual buffer ─────────────────────────────────────────────
    ilb_proj_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_MSA),
        .N_ROWS    (N_ROWS)
    ) u_proj (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (proj_wr_en),
        .wr_patch_base   (proj_wr_patch_base),
        .wr_col          (proj_wr_col),
        .wr_data         (proj_wr_data),
        .rd_en           (proj_rd_en),
        .rd_patch        (proj_rd_patch),
        .rd_col          (proj_rd_col),
        .rd_data         (proj_rd_data),
        .rmw_rd_en       (proj_rmw_rd_en),
        .rmw_addr        (proj_rmw_addr),
        .rmw_rd_data     (proj_rmw_rd_data),
        .rmw_wr_en       (proj_rmw_wr_en),
        .rmw_wr_data     (proj_rmw_wr_data)
    );

    // ── FFN1 GELU intermediate buffer ─────────────────────────────────────
    ilb_ffn1_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_FFN),
        .N_ROWS    (N_ROWS)
    ) u_ffn1 (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (ffn1_wr_en),
        .wr_patch_base   (ffn1_wr_patch_base),
        .wr_col          (ffn1_wr_col),
        .wr_data         (ffn1_wr_data),
        .rd_en           (ffn1_rd_en),
        .rd_patch        (ffn1_rd_patch),
        .rd_col_word     (ffn1_rd_col_word),
        .rd_data         (ffn1_rd_data)
    );

endmodule
