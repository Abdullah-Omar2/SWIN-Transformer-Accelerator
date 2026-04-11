// =============================================================================
// bias_buffer.sv  (rev 2 - Added Patch Merging Support)
//
// Dedicated bias storage and delivery unit for the MMU's 7 bias inputs.
// Loaded from CPU/DMA before operation start, then autonomously streams
// the correct 7-wide bias vector to the MMU during computation.
//
// ── Memory Map (32-bit entry addresses) ───────────────────────────────────
//   Conv        [     0 ..    95]    96 entries   1 bias per output channel
//   MLP  L1     [    96 ..   479]   384 entries   1 bias per output column
//   MLP  L2     [   480 ..   575]    96 entries   1 bias per output column
//   MHA  QK^T   [   576 ..  2976]  2401 entries   1 bias per 49×49 element
//   PM   FC     [  2977 ..  3168]   192 entries   1 bias per output channel
//   Reserved    [  3169 ..  4095]   for future use
//
// ── Per-operation behaviour ────────────────────────────────────────────────
//
//   Conv (mode=2'b00) & PM (mode=2'b11):
//     • One 32-bit bias per output channel.
//     • All 7 bias_out[k] slots carry the SAME value (spatial broadcast).
//     • rd_ptr advances by 1 on each bb_advance pulse.
//     • bb_advance should be pulsed by the controller at the
//       "next kernel" state transition.
//
//   MLP (mode=2'b01):
//     • One bias per output column; 384 for L1, 96 for L2.
//     • bias_out[k] = bias_mem[rd_ptr + k], k = 0..6.
//     • rd_ptr advances by 7 on each bb_advance pulse.
//
//   MHA (mode=2'b10):
//     • Bias is applied ONLY during the QK^T sub-operation.
//     • bias_out[k] = bias_mem[rd_ptr + k], k = 0..6.
//     • rd_ptr advances by 7 on each bb_advance pulse.
//
// =============================================================================

module bias_buffer #(
    parameter int AW             = 12,   // address width → 4 096 entries max
    parameter int DW             = 32,   // data width in bits
    parameter int BB_CONV_BASE   = 0,    // start of Conv bias region
    parameter int BB_MLP_L1_BASE = 96,   // start of MLP L1 bias region
    parameter int BB_MLP_L2_BASE = 480,  // start of MLP L2 bias region
    parameter int BB_MHA_QKT_BASE= 576,  // start of MHA QK^T bias region
    parameter int BB_PM_BASE     = 2977  // start of Patch Merging FC bias region
)(
    input  logic           clk,
    input  logic           rst_n,

    // ── CPU / DMA preload port ─────────────────────────────────────────────
    input  logic [AW-1:0]  cpu_wr_addr,   // entry address (0 .. 4095)
    input  logic [DW-1:0]  cpu_wr_data,   // 32-bit bias value
    input  logic           cpu_wr_en,     // write strobe (active high)

    // ── Controller interface ───────────────────────────────────────────────
    input  logic [1:0]     mode,          // 2'b00=Conv 2'b01=MLP 2'b10=MHA 2'b11=PM
    input  logic [2:0]     mmu_op_code,   // forwarded from ctrl_mmu_op_code

    input  logic           bb_op_start,
    input  logic [AW-1:0]  bb_op_base_addr, 

    input  logic           bb_advance,

    // ── Status ────────────────────────────────────────────────────────────
    output logic           bias_ready,    // 1 = bias_out is valid for MMU

    // ── MMU bias output ───────────────────────────────────────────────────
    output logic [DW-1:0]  bias_out [0:6] // directly fed to mmu_bias[0:6]
);

// =============================================================================
// Derived parameters
// =============================================================================
localparam int DEPTH = 1 << AW;   // 4096 32-bit entries
localparam int NOUT  = 7;         // MMU bias slots
localparam int LCNT_W = 3;        // ceil(log2(NOUT)) = 3

// =============================================================================
// Internal bias RAM
// =============================================================================
logic [DW-1:0] bias_mem [0:DEPTH-1];
logic [DW-1:0] ram_rd_data;

always_ff @(posedge clk) begin
    if (cpu_wr_en)
        bias_mem[cpu_wr_addr] <= cpu_wr_data;
end

logic [AW-1:0] ram_rd_addr;
always_ff @(posedge clk) begin
    ram_rd_data <= bias_mem[ram_rd_addr];
end

// =============================================================================
// Output register bank  (7 × 32-bit)
// =============================================================================
logic [DW-1:0] bias_reg [0:NOUT-1];
logic [AW-1:0] load_base;
logic [AW-1:0] rd_ptr;

logic [LCNT_W-1:0] load_cnt_rd;   // issues RAM reads  (0..6)
logic [LCNT_W-1:0] load_cnt_wr;   // captures RAM data (1..7, 7 = done)
logic              load_cnt_wr_vld;

// =============================================================================
// Advance step
//   Conv & PM → 1 (single broadcast bias per channel applied spatially)
//   MLP & MHA → 7 (full column group of 7 biases)
// NOTE: If your MMU maps Patch Merging channel-wise (like MLP), change 
// the condition below to: (mode == 2'b00) ? AW'(1) : AW'(NOUT);
// =============================================================================
logic [AW-1:0] adv_step;
assign adv_step = (mode == 2'b00 || mode == 2'b11) ? AW'(1) : AW'(NOUT);

// =============================================================================
// FSM
// =============================================================================
typedef enum logic [1:0] {
    S_IDLE    = 2'b00,
    S_LOADING = 2'b01,
    S_READY   = 2'b10
} bb_state_t;

bb_state_t state;

// ── State transitions ─────────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
    end else begin
        case (state)
            S_IDLE: begin
                if (bb_op_start)
                    state <= S_LOADING;
            end
            S_LOADING: begin
                if (bb_op_start)
                    state <= S_LOADING;           
                else if (load_cnt_wr == LCNT_W'(NOUT-1) && load_cnt_wr_vld)
                    state <= S_READY;             
            end
            S_READY: begin
                if (bb_op_start || bb_advance)
                    state <= S_LOADING;
            end
            default: state <= S_IDLE;
        endcase
    end
end

// ── load_base: first entry address of the current group ──────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        load_base <= AW'(BB_CONV_BASE);
    end else if (bb_op_start) begin
        load_base <= bb_op_base_addr;
    end else if (state == S_READY && bb_advance) begin
        load_base <= load_base + adv_step;
    end
end

// ── rd_ptr and load_cnt_rd ────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_ptr      <= AW'(BB_CONV_BASE);
        load_cnt_rd <= '0;
    end else if (bb_op_start) begin
        rd_ptr      <= bb_op_base_addr;
        load_cnt_rd <= '0;
    end else if (state == S_READY && bb_advance) begin
        rd_ptr      <= load_base + adv_step;
        load_cnt_rd <= '0;
    end else if (state == S_LOADING && load_cnt_rd != LCNT_W'(NOUT-1)) begin
        rd_ptr      <= rd_ptr + AW'(1);
        load_cnt_rd <= load_cnt_rd + LCNT_W'(1);
    end
end

// ── ram_rd_addr ──────────────────────────────────────────────────────────
always_comb begin
    if (state == S_LOADING || (bb_op_start) ||
        (state == S_READY && bb_advance))
        ram_rd_addr = rd_ptr;
    else
        ram_rd_addr = rd_ptr;   // hold (harmless extra read)
end

// ── load_cnt_wr ──────────────────────────────────────────────────────────
logic [LCNT_W-1:0] load_cnt_rd_d;
logic              loading_d;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        load_cnt_rd_d <= '0;
        loading_d     <= 1'b0;
    end else begin
        load_cnt_rd_d <= load_cnt_rd;
        loading_d     <= (state == S_LOADING) && !bb_op_start;
    end
end

assign load_cnt_wr     = load_cnt_rd_d;
assign load_cnt_wr_vld = loading_d;

// ── bias_reg ─────────────────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (int k = 0; k < NOUT; k++)
            bias_reg[k] <= '0;
    end else if (load_cnt_wr_vld) begin
        bias_reg[load_cnt_wr] <= ram_rd_data;
    end
end

// =============================================================================
// bias_ready
// =============================================================================
assign bias_ready = (state == S_READY);

// =============================================================================
// Output mux
// =============================================================================
always_comb begin
    for (int k = 0; k < NOUT; k++) begin
        unique case (mode)
            2'b00:   bias_out[k] = bias_reg[0];                      // Conv: broadcast
            2'b01:   bias_out[k] = bias_reg[k];                      // MLP: per-column
            2'b10:   bias_out[k] = (mmu_op_code == 3'd3)             // MHA: QK^T only
                                    ? bias_reg[k] : DW'(0);
            2'b11:   bias_out[k] = bias_reg[0];                      // PM: broadcast (assuming spatial mapping) 
                                                                     // NOTE: Change to bias_reg[k] if mapped like MLP!
            default: bias_out[k] = DW'(0);
        endcase
    end
end

endmodule
// =============================================================================
// End of bias_buffer.sv
// =============================================================================