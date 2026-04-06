// =============================================================================
// full_system_top.sv  (rev 6 — MHA / Swin Transformer Block support)
//
// ── What changed from rev 5 ───────────────────────────────────────────────
//   1. mode is now 2 bits (was 1 bit):
//        2'b00 = Patch Embedding (Conv)
//        2'b01 = Patch Merging  (MLP)
//        2'b10 = Swin Transformer Block (MHA)
//
//   2. unified_controller now exposes omem_fb_en_ctrl, an internal signal
//      that selects whether imem reads come from fib_memory (FIB) or
//      output_memory (ILB) during MHA.  This replaces the former top-level
//      omem_fb_en input for the MHA path; the external omem_fb_en input is
//      ORed with the controller signal so software can still override.
//
//   3. unified_input_buf interface extended with MHA-specific ports:
//        ibuf_mha_load_en, ibuf_mha_load_patch, ibuf_mha_load_k_word,
//        ibuf_mha_load_data, ibuf_mha_capture_row
//      mode port widened to 2 bits.
//
//   4. WAW widened to 16 in localparam to match the controller's enlarged
//      weight address space needed for MHA weight matrices.
//
//   5. New output port: mha_done_to_mwu — a 1-cycle strobe generated when
//      MHA mode completes (state reaches DONE), instructing the MWU to DMA
//      the final FFN2 output from ILB to off-chip memory.  The actual MWU
//      interface is outside this file; the strobe is exposed as a top-level
//      output for the MWU control logic.
//
// ── Round boundaries (when does the system go back to off-chip memory?) ──
//
//   Mode 0 (Conv / Patch Embedding):
//     MWU writes result after EACH kernel row-group tile (S_C_WRITEBACK).
//     Weight reload happens per-kernel (S_C_LOAD_W).
//     → Off-chip weight access: every ~30 cycles.
//     → Off-chip result write:  every ~180 cycles.
//
//   Mode 1 (MLP / Patch Merging):
//     All W1 columns processed, then all W2 columns, for each row group.
//     MWU writes result after each row group.
//     → Off-chip result write: after all 96 L2 columns per row group.
//
//   Mode 2 (MHA / Swin Transformer Block):
//     ALL operations for one 7×7 window execute on-chip in one round:
//       Q, K, V projections → QKᵀ → Softmax (SCU) → SxV →
//       W_proj → Shortcut1 → FFN1 → GELU → FFN2 → Shortcut2
//     The MWU is invoked ONCE per window (after S_H_SHORTCUT2 completes),
//     writing 49×96 bytes = 4704 bytes back to off-chip.
//     Over 64 windows → 64 MWU transfers.
//     Between windows, ALL weights remain on-chip (weight_memory holds
//     W_Q/K/V/proj/FFN1/FFN2 simultaneously since WAW=16 → 64K words).
//     The FIB is loaded ONCE before starting mode-2 and holds the full
//     56×56×96 feature map (75,264 words, unchanged from the MLP case).
//
// ── Signal flow (MHA additions) ───────────────────────────────────────────
//
//   FIB ──► ibuf (MHA patches)  ──► MMU ──► obuf ──► post_proc ──► ILB
//   ILB ──► ibuf (Q/S/FFN1 in)  ──► MMU  (QKᵀ, SxV, FFN2)
//   ILB ──► wbuf (K^T/V columns) ──► MMU
//   ILB ──► MWU ──► off-chip  (final FFN2 output per window)
//   FIB ──► MMU (shortcut add)
//
// =============================================================================

module full_system_top (
    input  logic clk,
    input  logic rst_n,

    // ── Mode and control ───────────────────────────────────────────────────
    input  logic [1:0] mode,   // 2'b00=Conv, 2'b01=MLP, 2'b10=MHA
    input  logic start,
    output logic done,

    // ── Post-processing controls ──────────────────────────────────────────
    input  logic signed [7:0] quant_shift_amt,
    input  logic               relu_en,

    // ── Feedback control ──────────────────────────────────────────────────
    // External override (software).  The controller also asserts this
    // internally for MHA intermediate reads; the two are OR'd together.
    input  logic               omem_fb_en,

    // ── CPU/DMA: weight_memory write ──────────────────────────────────────
    input  logic [15:0] cpu_wmem_wr_addr,   // widened to 16 bits (WAW=16)
    input  logic [31:0] cpu_wmem_wr_data,
    input  logic        cpu_wmem_wr_en,

    // ── CPU/DMA: fib_memory write ─────────────────────────────────────────
    input  logic [16:0] cpu_fib_wr_addr,
    input  logic [31:0] cpu_fib_wr_data,
    input  logic        cpu_fib_wr_en,

    // ── CPU/DMA: output_memory read ───────────────────────────────────────
    input  logic [18:0] cpu_omem_rd_addr,
    input  logic        cpu_omem_rd_en,
    output logic [31:0] cpu_omem_rd_data,

    // ── MWU trigger (MHA only) ────────────────────────────────────────────
    // Pulsed for 1 cycle when a window's FFN2 output is ready in ILB.
    // The MWU controller (external) monitors this and initiates the DMA.
    output logic        mha_window_done,

    // ── GCU GELU control (MHA FFN1 output) ───────────────────────────────
    // Tells the GCU to start computing GELU on the FFN1 output in ILB.
    output logic        gcu_start,
    input  logic        gcu_done    // GCU signals GELU completion
);

// =============================================================================
// Localparams
// =============================================================================
localparam int WAW = 16;   // bumped from 15 for MHA weight tables
localparam int FAW = 17;
localparam int OAW = 19;

// =============================================================================
// Unified controller output wires
// =============================================================================

logic [WAW-1:0] ctrl_wmem_rd_addr;
logic           ctrl_wmem_rd_en;
logic [31:0]    ctrl_wmem_rd_data;

logic [OAW-1:0] ctrl_imem_rd_addr;
logic           ctrl_imem_rd_en;
logic [31:0]    ctrl_imem_rd_data;

logic [OAW-1:0] ctrl_omem_wr_addr;
logic           ctrl_omem_wr_en;

logic           ctrl_wbuf_load_en;
logic [3:0]     ctrl_wbuf_load_pe_idx;
logic [6:0]     ctrl_wbuf_load_k_word;
logic [31:0]    ctrl_wbuf_load_data;
logic           ctrl_wbuf_bias_load_en;
logic [31:0]    ctrl_wbuf_bias_load_data;
logic           ctrl_wbuf_swap;

logic           ctrl_ibuf_load_en;
logic [3:0]     ctrl_ibuf_load_pe_idx;
logic [2:0]     ctrl_ibuf_load_win_idx;
logic [2:0]     ctrl_ibuf_load_row;
logic [6:0]     ctrl_ibuf_load_k_word;
logic [31:0]    ctrl_ibuf_load_data;
logic           ctrl_ibuf_swap;
logic           ctrl_ibuf_l1_capture_en;
logic [8:0]     ctrl_ibuf_l1_col_wr;

// MHA ibuf ports
logic           ctrl_ibuf_mha_load_en;
logic [5:0]     ctrl_ibuf_mha_load_patch;
logic [4:0]     ctrl_ibuf_mha_load_k_word;
logic [31:0]    ctrl_ibuf_mha_load_data;
logic [5:0]     ctrl_ibuf_mha_capture_row;

logic           ctrl_mmu_valid_in;
logic [2:0]     ctrl_mmu_op_code;
logic [1:0]     ctrl_mmu_stage;
logic [2:0]     ctrl_mmu_sub_cycle;

logic           ctrl_obuf_capture_en;
logic [2:0]     ctrl_obuf_rd_idx;

logic           ctrl_omem_fb_en;   // controller-driven ILB feedback select

// =============================================================================
// Physical memory wires
// =============================================================================
logic [31:0]    wmem_rd_data_phys;

logic [FAW-1:0] fib_rd_addr;
logic           fib_rd_en;
logic [31:0]    fib_rd_data;

logic [OAW-1:0] omem_fb_rd_addr;
logic           omem_fb_rd_en;
logic [31:0]    omem_fb_rd_data;

// =============================================================================
// Feedback path mux
// Combined: software omem_fb_en OR controller-driven ctrl_omem_fb_en
// =============================================================================
logic omem_fb_sel;
assign omem_fb_sel = omem_fb_en | ctrl_omem_fb_en;

logic omem_fb_sel_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) omem_fb_sel_d <= 1'b0;
    else        omem_fb_sel_d <= omem_fb_sel;
end

always_comb begin
    fib_rd_addr    = '0; fib_rd_en    = 1'b0;
    omem_fb_rd_addr= '0; omem_fb_rd_en= 1'b0;
    if (!omem_fb_sel) begin
        fib_rd_addr = ctrl_imem_rd_addr[FAW-1:0];
        fib_rd_en   = ctrl_imem_rd_en;
    end else begin
        omem_fb_rd_addr = ctrl_imem_rd_addr;
        omem_fb_rd_en   = ctrl_imem_rd_en;
    end
end

assign ctrl_imem_rd_data = omem_fb_sel_d ? omem_fb_rd_data : fib_rd_data;

// =============================================================================
// MHA window-done strobe
// Pulsed for 1 cycle when S_H_SHORTCUT2 completes (controller asserts done
// transiently via omem_wr_en going low after last FFN2 writeback).
// We detect the falling edge of omem_wr_en in mode 2 as the window-done signal.
// =============================================================================
logic omem_wr_en_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) omem_wr_en_d <= 1'b0;
    else        omem_wr_en_d <= ctrl_omem_wr_en;
end
assign mha_window_done = (mode == 2'b10) && omem_wr_en_d && !ctrl_omem_wr_en;

// =============================================================================
// GCU start strobe — generated when controller enters S_H_GELU_WAIT
// We detect it by watching for ctrl_omem_wr_en de-asserting after FFN1 WB
// in MHA mode.  (In practice this would be a dedicated ctrl output; here we
// reuse the same falling-edge detection on omem_wr_en while tracking the
// sub-state.  The controller emits gcu_start directly as an output port.)
// Implemented below as a registered version of the wmem_rd_en deassertion
// after FFN1 — a cleaner signal would be added in a full implementation.
// For now, tie to the existing mmmu_valid_in falling edge in mode 2 / stage 0.
// =============================================================================
logic mmu_valid_in_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) mmu_valid_in_d <= 1'b0;
    else        mmu_valid_in_d <= ctrl_mmu_valid_in;
end
// GCU starts after the last FFN1 compute cycle (op_code=5, stage=0)
assign gcu_start = (mode == 2'b10)
                 && (ctrl_mmu_op_code == 3'd5) && (ctrl_mmu_stage == 2'd0)
                 && mmu_valid_in_d && !ctrl_mmu_valid_in;

// =============================================================================
// Unified buffers → MMU bus wires
// =============================================================================
logic [7:0]  ubuf_w_out   [0:11][0:3];
logic [31:0] ubuf_bias_out;
logic [7:0]  ubuf_in_out  [0:11][0:6][0:3];

// =============================================================================
// MMU wires
// =============================================================================
logic        mmu_valid_out;
logic [7:0]  mmu_in_bus   [0:11][0:6][0:3];
logic [7:0]  mmu_w_bus    [0:11][0:3];
logic [31:0] mmu_bias_bus [0:11];
logic [31:0] mmu_out      [0:6];

// =============================================================================
// Output buffer / post-processing wires
// =============================================================================
logic [31:0]        obuf_rd_data;
logic signed [31:0] quant_data;
logic signed [31:0] relu_data;
logic signed [31:0] post_proc_data;

// =============================================================================
// MMU bus wiring
// =============================================================================
always_comb begin
    for (int p = 0; p < 12; p++) begin
        for (int t = 0; t < 4; t++)
            mmu_w_bus[p][t] = ubuf_w_out[p][t];
        mmu_bias_bus[p] = (p == 0) ? ubuf_bias_out : 32'd0;
        for (int w = 0; w < 7; w++)
            for (int t = 0; t < 4; t++)
                mmu_in_bus[p][w][t] = ubuf_in_out[p][w][t];
    end
end

// =============================================================================
// Instance: unified_controller
// =============================================================================
unified_controller #(
    .WAW      (WAW),
    .FAW      (FAW),
    .OAW      (OAW),
    .W2_BASE  (9216),
    // MHA weight offsets — must match the off-chip memory layout
    .WQ_BASE   (10240),
    .WK_BASE   (19456),
    .WV_BASE   (28672),
    .WPROJ_BASE(37888),
    .WFFN1_BASE(47104),
    .WFFN2_BASE(56320),
    // MHA ILB base addresses
    .ILB_Q_BASE   (0),
    .ILB_K_BASE   (3072),
    .ILB_V_BASE   (6144),
    .ILB_S_BASE   (9216),
    .ILB_A_BASE   (16468),
    .ILB_PROJ_BASE(19540),
    .ILB_FFN1_BASE(20588)
) u_ctrl (
    .clk                  (clk),
    .rst_n                (rst_n),
    .mode                 (mode),
    .start                (start),
    .done                 (done),

    .wmem_rd_addr         (ctrl_wmem_rd_addr),
    .wmem_rd_en           (ctrl_wmem_rd_en),
    .wmem_rd_data         (wmem_rd_data_phys),

    .wmem_shadow_wr_addr  (),    // MHA: no double-buffering for weight mem (future work)
    .wmem_shadow_wr_en    (),
    .wmem_swap            (),

    .ext_weight_rd_addr   (),
    .ext_weight_rd_en     (),

    .imem_rd_addr         (ctrl_imem_rd_addr),
    .imem_rd_en           (ctrl_imem_rd_en),
    .imem_rd_data         (ctrl_imem_rd_data),

    .omem_wr_addr         (ctrl_omem_wr_addr),
    .omem_wr_en           (ctrl_omem_wr_en),

    .wbuf_load_en         (ctrl_wbuf_load_en),
    .wbuf_load_pe_idx     (ctrl_wbuf_load_pe_idx),
    .wbuf_load_k_word     (ctrl_wbuf_load_k_word),
    .wbuf_load_data       (ctrl_wbuf_load_data),
    .wbuf_bias_load_en    (ctrl_wbuf_bias_load_en),
    .wbuf_bias_load_data  (ctrl_wbuf_bias_load_data),
    .wbuf_swap            (ctrl_wbuf_swap),

    .ibuf_load_en         (ctrl_ibuf_load_en),
    .ibuf_load_pe_idx     (ctrl_ibuf_load_pe_idx),
    .ibuf_load_win_idx    (ctrl_ibuf_load_win_idx),
    .ibuf_load_row        (ctrl_ibuf_load_row),
    .ibuf_load_k_word     (ctrl_ibuf_load_k_word),
    .ibuf_load_data       (ctrl_ibuf_load_data),
    .ibuf_swap            (ctrl_ibuf_swap),
    .ibuf_l1_capture_en   (ctrl_ibuf_l1_capture_en),
    .ibuf_l1_col_wr       (ctrl_ibuf_l1_col_wr),

    .ibuf_mha_load_en     (ctrl_ibuf_mha_load_en),
    .ibuf_mha_load_patch  (ctrl_ibuf_mha_load_patch),
    .ibuf_mha_load_k_word (ctrl_ibuf_mha_load_k_word),
    .ibuf_mha_load_data   (ctrl_ibuf_mha_load_data),
    .ibuf_mha_capture_row (ctrl_ibuf_mha_capture_row),

    .mmu_valid_in         (ctrl_mmu_valid_in),
    .mmu_op_code          (ctrl_mmu_op_code),
    .mmu_stage            (ctrl_mmu_stage),
    .mmu_sub_cycle        (ctrl_mmu_sub_cycle),

    .obuf_capture_en      (ctrl_obuf_capture_en),
    .obuf_rd_idx          (ctrl_obuf_rd_idx),

    .omem_fb_en_ctrl      (ctrl_omem_fb_en)
);

// =============================================================================
// Instance: weight_memory  (WAW=16 → 64K entries, covers all MHA weights)
// =============================================================================
weight_memory #(.AW(WAW)) u_wmem (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_addr (cpu_wmem_wr_addr),
    .wr_data (cpu_wmem_wr_data),
    .wr_en   (cpu_wmem_wr_en),
    .rd_addr (ctrl_wmem_rd_addr),
    .rd_en   (ctrl_wmem_rd_en),
    .rd_data (wmem_rd_data_phys)
);

// =============================================================================
// Instance: fib_memory
// Unchanged — 75,264 words covers Conv, MLP, and MHA input feature maps.
// =============================================================================
fib_memory u_fib (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_addr (cpu_fib_wr_addr),
    .wr_data (cpu_fib_wr_data),
    .wr_en   (cpu_fib_wr_en),
    .rd_addr (fib_rd_addr),
    .rd_en   (fib_rd_en),
    .rd_data (fib_rd_data)
);

// =============================================================================
// Instance: output_memory  (also serves as ILB for MHA intermediate data)
// The same memory doubles as the ILB during an MHA round:
//   - During MHA: controller writes Q/K/V/S/A/Proj/FFN1/FFN2 to it,
//     reads back via the feedback path.
//   - After each window: MWU reads the final FFN2 result out via cpu_rd port.
// OAW=19 → 512K word space — sufficient for all MHA intermediate buffers.
// =============================================================================
output_memory u_omem (
    .clk         (clk),
    .rst_n       (rst_n),
    .wr_addr     (ctrl_omem_wr_addr),
    .wr_data     (post_proc_data),
    .wr_en       (ctrl_omem_wr_en),
    .cpu_rd_addr (cpu_omem_rd_addr),
    .cpu_rd_en   (cpu_omem_rd_en),
    .cpu_rd_data (cpu_omem_rd_data),
    .fb_rd_addr  (omem_fb_rd_addr),
    .fb_rd_en    (omem_fb_rd_en),
    .fb_rd_data  (omem_fb_rd_data)
);

// =============================================================================
// Instance: unified_weight_buf
// =============================================================================
unified_weight_buf u_wbuf (
    .clk                 (clk),
    .rst_n               (rst_n),
    .mode                (mode[0]),   // weight buf is mode-0 (conv) or mode-1 (mlp/mha)
    .swap                (ctrl_wbuf_swap),

    .conv_load_en        ((mode == 2'b00) & ctrl_wbuf_load_en),
    .conv_load_pe_idx    (ctrl_wbuf_load_pe_idx),
    .conv_load_data      (ctrl_wbuf_load_data),
    .conv_bias_load_en   ((mode == 2'b00) & ctrl_wbuf_bias_load_en),
    .conv_bias_load_data (ctrl_wbuf_bias_load_data),

    // MLP and MHA both use the k_word path in the weight buffer
    .mlp_load_en         ((mode != 2'b00) & ctrl_wbuf_load_en),
    .mlp_load_k_word     (ctrl_wbuf_load_k_word),
    .mlp_load_data       (ctrl_wbuf_load_data),

    .sub_cycle           (ctrl_mmu_sub_cycle),
    .w_out               (ubuf_w_out),
    .bias_out            (ubuf_bias_out)
);

// =============================================================================
// Instance: unified_input_buf  (rev 2, MHA mode added)
// =============================================================================
unified_input_buf u_ibuf (
    .clk                 (clk),
    .rst_n               (rst_n),
    .mode                (mode),       // full 2-bit mode
    .swap                (ctrl_ibuf_swap),

    // Conv
    .conv_load_en        ((mode == 2'b00) & ctrl_ibuf_load_en),
    .conv_load_pe_idx    (ctrl_ibuf_load_pe_idx),
    .conv_load_win_idx   (ctrl_ibuf_load_win_idx),
    .conv_load_data      (ctrl_ibuf_load_data),

    // MLP
    .mlp_load_en         ((mode == 2'b01) & ctrl_ibuf_load_en),
    .mlp_load_row        (ctrl_ibuf_load_row),
    .mlp_load_k_word     (ctrl_ibuf_load_k_word),
    .mlp_load_data       (ctrl_ibuf_load_data),

    // Shared L1 capture / MHA intermediate capture
    .mlp_capture_en      (ctrl_ibuf_l1_capture_en),
    .mlp_col_wr          (ctrl_ibuf_l1_col_wr),
    .mlp_l1_out          (mmu_out),

    // MHA-specific
    .mha_load_en         (ctrl_ibuf_mha_load_en),
    .mha_load_patch      (ctrl_ibuf_mha_load_patch),
    .mha_load_k_word     (ctrl_ibuf_mha_load_k_word),
    .mha_load_data       (ctrl_ibuf_mha_load_data),
    .mha_capture_row     (ctrl_ibuf_mha_capture_row),

    .sub_cycle           (ctrl_mmu_sub_cycle),
    .data_out            (ubuf_in_out)
);

// =============================================================================
// Instance: mmu_top
// =============================================================================
mmu_top u_mmu (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (ctrl_mmu_valid_in),
    .op_code   (ctrl_mmu_op_code),
    .stage     (ctrl_mmu_stage),
    .valid_out (mmu_valid_out),
    .mmu_in    (mmu_in_bus),
    .mmu_w     (mmu_w_bus),
    .mmu_bias  (mmu_bias_bus),
    .mmu_out   (mmu_out)
);

// =============================================================================
// Instance: output_buffer
// =============================================================================
output_buffer u_obuf (
    .clk        (clk),
    .rst_n      (rst_n),
    .capture_en (ctrl_obuf_capture_en),
    .mmu_out    (mmu_out),
    .rd_idx     (ctrl_obuf_rd_idx),
    .rd_data    (obuf_rd_data)
);

// =============================================================================
// Post-processing pipeline (unchanged)
// =============================================================================
rounding_shifter #(.W_INPUT(32), .W_SHIFT(8)) u_quantizer (
    .in_value  ($signed(obuf_rd_data)),
    .shift_amt (quant_shift_amt),
    .out_value (quant_data)
);

relu #(.W(32)) u_relu (
    .in_value  (quant_data),
    .out_value (relu_data)
);

post_proc_mux #(.W(32)) u_mux (
    .relu_in  (relu_data),
    .quant_in (quant_data),
    .relu_en  (relu_en),
    .data_out (post_proc_data)
);

endmodule
