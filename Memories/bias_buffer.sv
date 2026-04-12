// =============================================================================
// bias_buffer.sv  (rev 4 — bug fixes from sizing-table audit)
//
// ── What changed from rev 3 ───────────────────────────────────────────────
//   BUG FIX 1 · Stage 1 QK^T bias count corrected:
//     Was  Stage1 = 2,401  (49×49, 1 head only)
//     Now  Stage1 = 7,203  (3 heads × 49×49)   ← controller must preload
//     all head slices before asserting bb_op_start for Stage 1 attention.
//     Stage 2 (14,406), Stage 3 (28,812), Stage 4 (57,624) were already
//     correct and are unchanged.
//
//   BUG FIX 2 · Patch Merging output mux corrected:
//     Was  mode=2'b11 → broadcast bias_reg[0] to all 7 lanes  (Conv path)
//     Now  mode=2'b11 → bias_out[k] = bias_reg[k]  (per-column, MLP path)
//     PM Linear layers (384→192, 768→384, 1536→768) have one unique bias
//     value per output column, identical in structure to an MLP FC layer.
//     Broadcasting a single value corrupted all but the first column.
//
//   BUG FIX 3 · Removed spurious "PM FC2 bias → BB_MLP_L2_BASE" comment:
//     Each Patch Merging block contains exactly ONE linear layer.  There is
//     no PM FC2.  The BB_MLP_L2_BASE slot is used exclusively for MLP/FFN
//     FC2 and Output Projection (see Output Proj mapping below).
//
//   BUG FIX 4 · Output Projection bias mapping documented and clarified:
//     Output Proj biases are 96 / 192 / 384 / 768 per stage.  They are
//     stored at BB_MLP_L1_BASE (same region as MLP FC1 / PM) and served
//     with mode=2'b01, since they are a plain per-column linear bias.
//     BB_CONV_BASE (96 entries) must NOT be used for Output Proj in
//     Stages 2-4 as it cannot hold more than 96 values.
//
// ── Revised Memory Map ────────────────────────────────────────────────────
//
//   Address range        Entries  Content
//   ─────────────────────────────────────────────────────────────────────────
//      0 ..     95          96    Conv  (96 output channels)
//     96 ..   3167        3072    MLP/FFN FC1  │ also PM Linear
//                                              │ also Output Projection
//                                              │ (max = Stage4: 3072 / 768)
//   3168 ..   3935         768    MLP/FFN FC2  (max = Stage4: 768 outputs)
//   3936 ..  61559       57624    MHA QK^T     (max = Stage4: 24h×49×49)
//                                   Stage1 =  7,203  (3h  × 49×49)  ← FIX
//                                   Stage2 = 14,406  (6h  × 49×49)
//                                   Stage3 = 28,812  (12h × 49×49)
//                                   Stage4 = 57,624  (24h × 49×49)
//                                 Stage-specific portions written before
//                                 each attention round.
//  61560 ..  65535        3976    Reserved
//   ─────────────────────────────────────────────────────────────────────────
//   Total used: 61,560  ≤  DEPTH = 65,536  ✓
//
//   Shared-region scheduling (controller responsibility):
//     BB_MLP_L1_BASE holds one of {MLP FC1, PM Linear, Output Proj} at a
//     time; these operations never overlap so there is no conflict.
//     BB_MLP_L2_BASE holds MLP FC2 only; PM has no second linear layer.
//
// ── QK^T bias layout (row-major, all heads contiguous) ───────────────────
//   bias[head h, query q, key k] →
//       address  BB_MHA_QKT_BASE + h*(49*49) + q*49 + k
//   The controller must write ALL heads before asserting bb_op_start.
//   Per-stage head counts:  S1=3  S2=6  S3=12  S4=24
//
// ── Per-operation output behaviour ───────────────────────────────────────
//   Conv          (mode=2'b00):        broadcast bias_reg[0] to all 7 lanes
//   MLP/FFN FC    (mode=2'b01):        bias_out[k] = bias_reg[k], k=0..6
//   MHA QK^T      (mode=2'b10, op=3):  bias_out[k] = bias_reg[k], k=0..6
//   MHA other ops (mode=2'b10, op≠3):  bias_out[k] = 0
//   PM Linear     (mode=2'b11):        bias_out[k] = bias_reg[k], k=0..6
//   Output Proj   (mode=2'b01):        bias_out[k] = bias_reg[k], k=0..6
//     ↑ Output Proj uses the same mode as MLP FC; controller selects the
//       correct base address (BB_MLP_L1_BASE) and stage-specific count.
//
// ── Interface (unchanged from rev 3) ─────────────────────────────────────
//   cpu_wr_addr  : 16 bits
//   bb_op_base_addr : 16 bits
// =============================================================================

module bias_buffer #(
    parameter int AW              = 16,    // 65,536 entries
    parameter int DW              = 32,
    parameter int BB_CONV_BASE    = 0,
    parameter int BB_MLP_L1_BASE  = 96,    // MLP FC1 / PM Linear / Output Proj
    parameter int BB_MLP_L2_BASE  = 3168,  // MLP FC2 only  (96 + 3072)
    parameter int BB_MHA_QKT_BASE = 3936   // (3168 + 768)
)(
    input  logic           clk,
    input  logic           rst_n,

    // ── CPU / DMA preload ─────────────────────────────────────────────────
    input  logic [AW-1:0]  cpu_wr_addr,
    input  logic [DW-1:0]  cpu_wr_data,
    input  logic           cpu_wr_en,

    // ── Controller interface ───────────────────────────────────────────────
    // mode: 2'b00 = Conv
    //       2'b01 = MLP FC / Output Projection  (per-column)
    //       2'b10 = MHA
    //       2'b11 = PM Linear                   (per-column)
    input  logic [1:0]     mode,
    input  logic [2:0]     mmu_op_code,

    input  logic           bb_op_start,
    input  logic [AW-1:0]  bb_op_base_addr,
    input  logic           bb_advance,

    // ── Status ────────────────────────────────────────────────────────────
    output logic           bias_ready,

    // ── MMU bias output (7 values → mmu_bias[0:6]) ───────────────────────
    output logic [DW-1:0]  bias_out [0:6]
);

    // ── Internal RAM: 65536 × 32-bit ─────────────────────────────────────
    localparam int DEPTH = 1 << AW;  // 65,536

    logic [DW-1:0] bias_mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) bias_mem[i] = '0;
    end

    always_ff @(posedge clk) begin
        if (cpu_wr_en)
            bias_mem[cpu_wr_addr] <= cpu_wr_data;
    end

    // ── 7-element register bank ───────────────────────────────────────────
    logic [DW-1:0] bias_reg [0:6];

    // ── FSM ───────────────────────────────────────────────────────────────
    typedef enum logic [1:0] {
        S_IDLE    = 2'd0,
        S_LOADING = 2'd1,
        S_READY   = 2'd2
    } fsm_t;

    fsm_t           fsm;
    logic [AW-1:0]  load_base;
    logic [2:0]     load_cnt_rd;
    logic [2:0]     load_cnt_wr;
    logic [AW-1:0]  ram_rd_addr;

    // ── Registered RAM read address → 1-cycle latency ────────────────────
    logic [DW-1:0]  ram_rd_data;
    always_ff @(posedge clk) begin
        ram_rd_data <= bias_mem[ram_rd_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm         <= S_IDLE;
            load_base   <= '0;
            load_cnt_rd <= '0;
            load_cnt_wr <= '0;
            bias_ready  <= 1'b0;
            for (int i = 0; i < 7; i++) bias_reg[i] <= '0;
        end else begin
            case (fsm)
                S_IDLE: begin
                    bias_ready <= 1'b0;
                    if (bb_op_start) begin
                        load_base   <= bb_op_base_addr;
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                        fsm         <= S_LOADING;
                    end
                end

                S_LOADING: begin
                    // Override: restart if bb_op_start during load
                    if (bb_op_start) begin
                        load_base   <= bb_op_base_addr;
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                    end else begin
                        // Advance read pointer (issues reads ahead)
                        if (load_cnt_rd < 3'd6)
                            load_cnt_rd <= load_cnt_rd + 3'd1;

                        // Capture arriving data (1 cycle after read)
                        load_cnt_wr <= load_cnt_rd;
                        if (load_cnt_wr <= 3'd6)
                            bias_reg[load_cnt_wr] <= ram_rd_data;

                        // Done after write-counter hits 6
                        if (load_cnt_wr == 3'd6) begin
                            bias_ready <= 1'b1;
                            fsm        <= S_READY;
                        end
                    end
                end

                S_READY: begin
                    if (bb_op_start) begin
                        // Re-arm immediately
                        load_base   <= bb_op_base_addr;
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                        bias_ready  <= 1'b0;
                        fsm         <= S_LOADING;
                    end else if (bb_advance) begin
                        // Advance to next group of 7 entries
                        load_base   <= load_base + AW'(7);
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                        bias_ready  <= 1'b0;
                        fsm         <= S_LOADING;
                    end
                end

                default: fsm <= S_IDLE;
            endcase
        end
    end

    // ── RAM read address mux ──────────────────────────────────────────────
    assign ram_rd_addr = load_base + AW'(load_cnt_rd);

    // ── Output mux ────────────────────────────────────────────────────────
    // Conv          (mode 00): broadcast bias_reg[0] to all 7 lanes
    // MLP/FFN / Proj(mode 01): bias_out[k] = bias_reg[k]  (per-column)
    // MHA QK^T      (mode 10, op=3): bias_out[k] = bias_reg[k]
    // MHA other     (mode 10, op≠3): bias_out[k] = 0
    // PM Linear     (mode 11): bias_out[k] = bias_reg[k]  (per-column) ← FIX
    //   PM has one unique bias value per output column, not a broadcast.
    always_comb begin
        for (int k = 0; k < 7; k++) begin
            case (mode)
                2'b00:    bias_out[k] = bias_reg[0];                                    // Conv broadcast
                2'b01:    bias_out[k] = bias_reg[k];                                    // MLP FC / Output Proj
                2'b10:    bias_out[k] = (mmu_op_code == 3'd3) ? bias_reg[k] : '0;      // MHA
                2'b11:    bias_out[k] = bias_reg[k];                                    // PM Linear (per-column)
                default:  bias_out[k] = '0;
            endcase
        end
    end

endmodule