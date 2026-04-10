// =============================================================================
// output_memory.sv  (rev 4 — MHA ILB raw-write support)
//
// ── What changed from rev 3 ───────────────────────────────────────────────
//
//   1.  New Port 1 inputs: ilb_raw_wr_data and ilb_wr_bypass
//       -------------------------------------------------------
//       In Conv / MLP mode the engine write path is:
//           post_proc_mux → wr_data → mem[wr_addr]
//       which applies quantisation and optional ReLU before storage.
//
//       In MHA mode ALL intermediate matrices (Q, K, V, attention scores S,
//       context A = S×V, W_proj output, FFN1 and FFN2 outputs) must be kept
//       as raw 32-bit accumulator words in the ILB region of this memory.
//       Applying the quantiser or ReLU would corrupt them.
//
//       When ilb_wr_bypass = 1 the write mux selects ilb_raw_wr_data
//       (connected to output_buffer.raw_rd_data in full_system_top) instead
//       of wr_data.  wr_addr and wr_en are shared by both paths — the
//       controller already computes the correct ILB address.
//
//       ilb_wr_bypass is driven by ctrl_ilb_wr_bypass in full_system_top:
//           assign ctrl_ilb_wr_bypass = (mode == 2'b10);
//       This covers every MHA write.  The signal is 0 for Conv and MLP.
//
//   2.  ILB address map (for reference — enforced by unified_controller
//       parameters, not by this module)
//       -------------------------------------------------------------------
//       The lower ~50 K words of this memory serve as the Intermediate Layer
//       Buffer during an MHA round.  They are reused for every 7×7 window.
//
//       Region             Dimensions      Words    Word base (from ctrl)
//       ─────────────────────────────────────────────────────────────────
//       Q   (3 heads)      3 × 49 × 32    4 704    ILB_Q_BASE    =     0
//       K   (3 heads)      3 × 49 × 32    4 704    ILB_K_BASE    =  3 072
//       V   (3 heads)      3 × 49 × 32    4 704    ILB_V_BASE    =  6 144
//       S   QK^T (3 hd)    3 × 49 × 49    7 203    ILB_S_BASE    =  9 216
//       A   S×V  (3 hd)    3 × 49 × 32    4 704    ILB_A_BASE    = 16 468
//       PROJ output        49 × 96        4 704    ILB_PROJ_BASE = 19 540
//       FFN1 output        49 × 384      18 816    ILB_FFN1_BASE = 20 588
//       ─────────────────────────────────────────────────────────────────
//       Total ILB peak                   49 539    (<< DEPTH = 301 056)
//
//       Words [49 539 .. 301 055] remain available for Conv / MLP results
//       exactly as before.
//
//   3.  K^T transpose read — no memory change required
//       -----------------------------------------------
//       To load K_h^T into the weight buffer for the QK^T step, the
//       controller reads K_h from the ILB in column-major order by computing
//       transposed addresses on the fb_rd_addr port:
//           for col j in 0..48:
//               for row i in 0..31:
//                   fb_rd_addr = ILB_K_BASE + h*49*32 + col*32 + row
//                             (reads K[i][j] presenting it as K^T[j][i])
//       The memory responds identically to any valid fb_rd_addr — no
//       dedicated transpose hardware is needed here.
//
//   4.  Split Q / K / V into heads — no memory change required
//       -------------------------------------------------------
//       After the 49×96 matrix is stored, each head h accesses columns
//       [h*32 .. (h+1)*32-1] via base-address offsets already baked into
//       the controller ILB_*_BASE parameters.  The 49×96 flat layout means:
//           word(row r, col c) = base + r * 96 + c    (row-major)
//           Head h, col c' (0..31): c = h*32 + c'
//       No physical rearrangement is needed; the controller computes the
//       correct strided address for each head transparently.
//
// ── Port additions (Port 1 only) ─────────────────────────────────────────
//   ilb_raw_wr_data [31:0]  raw accumulator from output_buffer.raw_rd_data
//   ilb_wr_bypass           1 → write ilb_raw_wr_data  (MHA all steps)
//                           0 → write wr_data           (Conv / MLP)
//
// ── Unchanged ports ───────────────────────────────────────────────────────
//   Port 2 — CPU / DMA read  (cpu_rd_addr / cpu_rd_en / cpu_rd_data)
//   Port 3 — Engine feedback read (fb_rd_addr / fb_rd_en / fb_rd_data)
//   DEPTH = 301 056,  AW = 19  — unchanged; ILB fits in existing space.
//
// =============================================================================

module output_memory #(
    parameter DEPTH = 301056,
    parameter AW    = 19
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── Port 1 : Engine write ──────────────────────────────────────────────
    input  logic [AW-1:0]  wr_addr,
    input  logic [31:0]    wr_data,            // post_proc_data  (Conv/MLP)
    input  logic [31:0]    ilb_raw_wr_data,    // raw accumulator (MHA all steps)
    input  logic           ilb_wr_bypass,      // 1 → select ilb_raw_wr_data
    input  logic           wr_en,              // write strobe (both paths share)

    // ── Port 2 : CPU / DMA read ───────────────────────────────────────────
    input  logic [AW-1:0]  cpu_rd_addr,
    input  logic           cpu_rd_en,
    output logic [31:0]    cpu_rd_data,

    // ── Port 3 : Engine feedback read (ILB read-back for MHA / layer chain)
    input  logic [AW-1:0]  fb_rd_addr,
    input  logic           fb_rd_en,
    output logic [31:0]    fb_rd_data
);

    logic [31:0] mem [0:DEPTH-1];

    // ── Zero-initialise (simulation) ──────────────────────────────────────
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── Write data mux ────────────────────────────────────────────────────
    // ilb_wr_bypass = 1  →  MHA intermediate store: raw 32-bit accumulator,
    //                        no quantisation, no ReLU.
    // ilb_wr_bypass = 0  →  Conv / MLP final store: post-processed data.
    logic [31:0] effective_wr_data;
    assign effective_wr_data = ilb_wr_bypass ? ilb_raw_wr_data : wr_data;

    // ── Write (Port 1) ────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= effective_wr_data;
    end

    // ── CPU read (Port 2, 1-cycle latency) ───────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)           cpu_rd_data <= '0;
        else if (cpu_rd_en)   cpu_rd_data <= mem[cpu_rd_addr];
    end

    // ── Feedback read (Port 3, 1-cycle latency) ───────────────────────────
    // Used for:
    //   • Conv / MLP layer chaining  (existing behaviour)
    //   • MHA ILB read-back:
    //       – Loading Q_h / S_h into the input buffer (row-major fb_rd_addr)
    //       – Loading K_h^T into the weight buffer    (column-major fb_rd_addr
    //         — transposed access pattern issued by the controller)
    //       – Loading V_h into the weight buffer      (row-major fb_rd_addr)
    //       – Loading PROJ/FFN intermediates          (row-major fb_rd_addr)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)           fb_rd_data <= '0;
        else if (fb_rd_en)    fb_rd_data <= mem[fb_rd_addr];
    end

endmodule