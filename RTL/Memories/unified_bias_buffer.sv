// =============================================================================
// unified_bias_buffer.sv
//
// Sits between bias_buffer and mmu_top.
//
// ── Responsibilities ─────────────────────────────────────────────────────────
//  1. Latches the 7-wide bias group from bias_buffer the moment bias_ready
//     asserts, and holds it stable in bias_hold[0:6] for the MMU.
//
//  2. Counts valid_out pulses (one per completed MMU output group).
//     When the count reaches spatial_count, it pulses bb_advance for one
//     cycle so bias_buffer loads the next 7-entry group.
//
//  3. Presents mmu_bias[0:6] to mmu_top and drives mmu_bias_valid so the
//     MMU knows whether the bias values are ready.  Outputs are forced to
//     zero while the buffer is reloading.
//
// ── Advance cadence (spatial_count) ─────────────────────────────────────────
//
//   Operation             spatial_count   Reason
//   ─────────────────────────────────────────────────────────────────────────
//   Conv  Stage 1         3136            56×56 spatial positions per channel
//   Conv  Stage 2          784            28×28
//   Conv  Stage 3          196            14×14
//   Conv  Stage 4           49             7×7
//   MLP FC1/FC2              1            bias changes every 7 output columns
//   Output Projection        1            same structure as MLP FC
//   PM Linear                1            same structure as MLP FC
//   MHA QK^T                 1            bias changes every 7 keys
//   ─────────────────────────────────────────────────────────────────────────
//   12-bit counter → max 4095 ≥ 3136  ✓
//
// ── Timing ───────────────────────────────────────────────────────────────────
//   bb_advance is a 1-cycle pulse.  bias_buffer requires 8 cycles to reload
//   (7 RAM reads + 1 write-through into bias_reg).  The controller must not
//   assert valid_out during that reload window; for MLP/PM/QK^T (where
//   spatial_count=1) this means one group of 8+ idle cycles between successive
//   MMU output beats, which is met naturally by the MMU's pipeline depth.
//
// ── Interface ────────────────────────────────────────────────────────────────
//   bb_op_start   Shared with bias_buffer.  Resets the latch and counter;
//                 bias_buffer will begin loading from the new base address.
//   spatial_count Set by the controller before asserting bb_op_start; can
//                 change between operations.
// =============================================================================

module unified_bias_buffer #(
    parameter int DW = 32          // data width — must match bias_buffer DW
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── bias_buffer interface ─────────────────────────────────────────────
    input  logic [DW-1:0] bb_bias_out  [0:6],  // current 7-entry group
    input  logic          bb_bias_ready,         // group is stable in bias_reg
    output logic          bb_advance,            // 1-cycle pulse → load next group

    // ── controller interface ──────────────────────────────────────────────
    input  logic          bb_op_start,           // new operation; shared with bias_buffer
    input  logic          valid_out,             // mmu_top produced one output group
    input  logic [11:0]   spatial_count,         // valid_outs before advance (1–3136)

    // ── mmu_top interface ─────────────────────────────────────────────────
    output logic [DW-1:0] mmu_bias     [0:6],   // bias values to PE[0]
    output logic          mmu_bias_valid         // high when mmu_bias is stable
);

    // ── 7-entry hold register ─────────────────────────────────────────────
    logic [DW-1:0] bias_hold [0:6];
    logic          hold_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hold_valid <= 1'b0;
            for (int k = 0; k < 7; k++) bias_hold[k] <= '0;
        end else begin
            if (bb_op_start) begin
                // New operation — invalidate until bias_buffer reloads
                hold_valid <= 1'b0;

            end else if (bb_bias_ready && !hold_valid) begin
                // Fresh group arrived from bias_buffer — latch it
                for (int k = 0; k < 7; k++) bias_hold[k] <= bb_bias_out[k];
                hold_valid <= 1'b1;

            end else if (bb_advance) begin
                // Current group consumed — mark invalid while buffer reloads.
                // bias_hold retains old values (harmless; mmu_bias is zeroed).
                hold_valid <= 1'b0;
            end
        end
    end

    // ── Spatial-position counter & advance pulse ──────────────────────────
    //
    //   Counts valid_out pulses up to spatial_count, then pulses bb_advance.
    //   The counter and bb_advance are only active while hold_valid is high
    //   (i.e. while a valid bias group is loaded and the MMU is running).
    //
    logic [11:0] pos_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos_cnt    <= 12'd0;
            bb_advance <= 1'b0;
        end else begin
            bb_advance <= 1'b0;                         // default: no pulse

            if (bb_op_start) begin
                pos_cnt <= 12'd0;

            end else if (hold_valid && valid_out) begin
                if (pos_cnt == spatial_count - 12'd1) begin
                    pos_cnt    <= 12'd0;
                    bb_advance <= 1'b1;                 // one-cycle pulse
                end else begin
                    pos_cnt <= pos_cnt + 12'd1;
                end
            end
        end
    end

    // ── Output assignments ────────────────────────────────────────────────
    //   mmu_bias is zeroed when hold_valid is low (buffer reloading or idle).
    //   The mode-based broadcast / per-column mux is already done inside
    //   bias_buffer's output combinational block; bias_dispatcher is mode-
    //   agnostic and simply forwards whatever arrives in bb_bias_out.
    always_comb begin
        mmu_bias_valid = hold_valid;
        for (int k = 0; k < 7; k++)
            mmu_bias[k] = hold_valid ? bias_hold[k] : '0;
    end

endmodule