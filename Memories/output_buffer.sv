// =============================================================================
// output_buffer.sv  (rev 2 — MHA ILB raw-write support)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   Added raw_rd_data output port.
//
//   In Conv / MLP mode the datapath is:
//       buf[rd_idx] → rd_data → quantiser → ReLU → post_proc_mux
//                                         → output_memory.wr_data
//
//   In MHA mode ALL intermediate results (Q, K, V, attention scores S,
//   context A, W_proj output, FFN1/FFN2 outputs) must be stored in the ILB
//   (output_memory) as raw 32-bit accumulator values — no quantisation, no
//   ReLU.  The new raw_rd_data port carries the same buf[rd_idx] word on a
//   dedicated wire so that full_system_top can route it directly to
//   output_memory.ilb_raw_wr_data, bypassing the post-proc chain entirely.
//
//   The selection between the two write paths is made inside output_memory
//   by the ilb_wr_bypass signal (asserted for all MHA writes).
//
// ── Port summary ──────────────────────────────────────────────────────────
//   rd_data      [31:0]  (existing) → post_proc pipeline  (Conv / MLP)
//   raw_rd_data  [31:0]  (new)      → ilb_raw_wr_data     (MHA all steps)
//
// =============================================================================

module output_buffer #(
    parameter N_OUT = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Capture from MMU ──────────────────────────────────────────────────
    input  logic        capture_en,
    input  logic [31:0] mmu_out [0:N_OUT-1],

    // ── Sequential read-out ───────────────────────────────────────────────
    input  logic [2:0]  rd_idx,
    output logic [31:0] rd_data,       // → quantiser/ReLU/post_proc_mux (Conv/MLP)
    output logic [31:0] raw_rd_data    // → ilb_raw_wr_data bypass port   (MHA)
);

    logic [31:0] buf [0:N_OUT-1];

    // ── Capture register ──────────────────────────────────────────────────
    // Latches the 7-element row-group produced by the MMU when capture_en is
    // asserted.  For QKV / SxV this fires every 2 cycles; for QK^T it fires
    // every cycle (one full column per cycle in attention mode).
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N_OUT; i++) buf[i] <= '0;
        end else if (capture_en) begin
            for (int i = 0; i < N_OUT; i++) buf[i] <= mmu_out[i];
        end
    end

    // ── Read outputs ──────────────────────────────────────────────────────
    // Both signals expose buf[rd_idx]; the distinction is how they are used
    // downstream:
    //   rd_data     goes through the quantiser → ReLU → post_proc_mux chain.
    //   raw_rd_data bypasses that chain and writes straight to the ILB via
    //               output_memory.ilb_raw_wr_data when ilb_wr_bypass = 1.
    assign rd_data     = buf[rd_idx];
    assign raw_rd_data = buf[rd_idx];

endmodule