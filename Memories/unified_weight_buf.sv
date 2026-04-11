// =============================================================================
// unified_weight_buf.sv  (rev 8 — Patch Merging MLP mode corrected)
//
// ── What changed from rev 7 ───────────────────────────────────────────────
//
//   BUG FIX — MLP mode (Patch Merging) had two errors.
//
//   Ground truth from ilb_patch_merge_buf:
//     Patch Merging applies ONE FC layer: X [patches × C_SPA] × W [C_SPA × C_FC]
//     → output [patches × C_FC].
//     There is no second weight matrix.  The old "W1 narrow col / W2 wide col"
//     distinction was invented and incorrect.
//
//   Error 1 — spurious W1/W2 sub-mode:
//     rev 7 described mlp_sub_mode as selecting W1 (narrow col) vs W2 (wide col).
//     This distinction does not exist.  There is one FC weight matrix with
//     col_words = C_SPA / 4 for whatever PM stage is active.
//     REMOVED: mlp_sub_mode port.
//
//   Error 2 — MAX_BYTES sized for FFN1 (Swin), not PM:
//     rev 7 set MAX_BYTES = 3072  claiming "Stage 4 FFN1 col: 768 words × 4 B".
//     But Swin Block operations use mode 2'b10, NOT mode 2'b01.
//     MAX_BYTES for mode 2'b01 (MLP) must cover the widest PM column:
//       PM3 C_SPA = 1536 bytes  →  MAX_BYTES = 1536  for MLP mode
//     Swin FFN1 (mode 2'b10) uses msa_col_bytes and is unaffected.
//     FIX: MAX_BYTES = 3072 kept as the compile-time bank ceiling (it is the
//     dominant of 1536 for MLP and 3072 for Swin FFN1 in mode 2'b10), but
//     the MLP read guard now correctly uses mlp_col_bytes = C_SPA (not C_FC).
//
//   Error 3 — mlp_load_k_word range documented incorrectly:
//     rev 7 said "0..767" referencing Stage 4 FFN row width.
//     For PM, mlp_load_k_word addresses words within a C_SPA column:
//       PM1: 0..95   (384/4 = 96 words)
//       PM2: 0..191  (768/4 = 192 words)
//       PM3: 0..383  (1536/4 = 384 words)
//     Maximum = 383, so 9 bits suffice.  The port stays 10 bits (superset).
//
//   Error 4 — mlp_col_bytes meaning was wrong:
//     rev 7: "mlp_col_bytes = C_FC bytes (output width)".
//     CORRECT: mlp_col_bytes = C_SPA bytes (input width per output column).
//     The controller must now drive mlp_col_bytes = C_SPA of the active stage:
//       PM1: 384   PM2: 768   PM3: 1536
//
// ── Valid-byte table — MLP mode (corrected) ───────────────────────────────
//
//   PM stage │ C_SPA │ C_FC  │ col_words (=C_SPA/4) │ mlp_col_bytes
//   ─────────┼───────┼───────┼─────────────────────┼──────────────
//     PM1     │  384  │  192  │         96           │     384
//     PM2     │  768  │  384  │        192           │     768
//     PM3     │ 1536  │  768  │        384           │    1536
//
// ── Valid-byte table — SWIN mode (unchanged) ──────────────────────────────
//
//   Stage │ QKV/Proj (B) │ QK^T (B) │ SxV (B) │ FFN1 (B) │ FFN2 (B)
//   ──────┼──────────────┼──────────┼─────────┼──────────┼──────────
//     1   │     96       │    32    │   52    │   384    │    96
//     2   │    192       │    32    │   52    │   768    │   192
//     3   │    384       │    32    │   52    │  1536    │   384
//     4   │    768       │    32    │   52    │  3072    │   768
//
// ── MAX_BYTES ─────────────────────────────────────────────────────────────
//   MAX_BYTES = 3072 — dominant is Stage 4 FFN1 col in Swin mode (2'b10).
//   MLP mode (2'b01) only ever loads up to PM3 C_SPA = 1536 bytes per column;
//   the upper 1536 bytes of the bank are unused in MLP mode but do no harm.
//
// ── shadow_clr unchanged ──────────────────────────────────────────────────
//   Pulse once before each shadow fill to zero stale bytes.
//   Still critical for QK^T (32 B load, 16 B stale) and SxV (49→52 B pad).
// =============================================================================

module unified_weight_buf #(
    parameter int MAX_BYTES = 3072,  // Stage 4 Swin FFN1 col (mode 2'b10 dominant)
    parameter int N_PE      = 12,
    parameter int N_TAP     = 4
)(
    input  logic        clk,
    input  logic        rst_n,

    // 2'b00=CONV  2'b01=MLP/Patch Merging  2'b10=Swin Block
    input  logic [1:0]  mode,

    input  logic        swap,

    // Pulse before each shadow fill — zeros stale bytes
    input  logic        shadow_clr,

    // ── CONV load  (mode 2'b00) ───────────────────────────────────────────
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,
    input  logic [31:0] conv_load_data,

    // ── MLP / Patch Merging load  (mode 2'b01) ────────────────────────────
    //
    // One FC weight matrix W [C_SPA × C_FC], column-major.
    // Each call to mlp_load_en fills one word (4 bytes) of the current column.
    //
    // mlp_load_k_word : word index within the column, 0 .. C_SPA/4 - 1
    //   PM1: 0..95   PM2: 0..191   PM3: 0..383   (10 bits covers all)
    //
    // mlp_col_bytes : valid byte count = C_SPA of active PM stage (stable per col)
    //   PM1: 384   PM2: 768   PM3: 1536
    //   Controller drives this once at round start and holds it stable.
    //
    // NOTE: The old mlp_sub_mode (W1/W2 split) is removed — there is no W2.
    input  logic        mlp_load_en,
    input  logic [9:0]  mlp_load_k_word,  // word index 0..383 (PM3 max)
    input  logic [31:0] mlp_load_data,
    input  logic [10:0] mlp_col_bytes,    // C_SPA: 384/768/1536 (was 12 bits, now 11 covers 1536)

    // ── Swin Block load  (mode 2'b10) ─────────────────────────────────────
    // Unchanged from rev 7.
    input  logic        msa_load_en,
    input  logic [9:0]  msa_load_word,    // 0..767 (Stage4 FFN1: 768 words)
    input  logic [31:0] msa_load_data,

    // 2'b00=QKV/Proj/FFN2  2'b01=QK^T  2'b10=SxV  2'b11=FFN1
    input  logic [1:0]  msa_sub_mode,

    // Exact valid bytes for current Swin column (set stable per col):
    //   sub-mode 2'b00: C bytes (QKV/Proj/FFN2)
    //   sub-mode 2'b01: 32     (QK^T, constant)
    //   sub-mode 2'b10: 52     (SxV,  constant)
    //   sub-mode 2'b11: FFN_C bytes (FFN1)
    input  logic [11:0] msa_col_bytes,    // 32..3072

    // Sub-cycle counter — selects 48-byte window into active bank
    input  logic [6:0]  sub_cycle,        // up to 64 sub-cycles (Stage4 FFN1)

    // Weight output to MMU (bias_out removed — bias_buffer is separate)
    output logic [7:0]  w_out [0:N_PE-1][0:N_TAP-1]
);

// =============================================================================
// Fixed valid-byte constants (stage-invariant)
// =============================================================================
localparam int BYTES_CONV = N_PE * N_TAP;  // 48 B, 1 sub-cycle
localparam int BYTES_QKT  = 32;            // d_head=32, constant all stages
localparam int BYTES_SV   = 52;            // 49 padded to 52, constant all stages

// =============================================================================
// Double-banked storage
// =============================================================================
logic [7:0]  bank [0:1][0:MAX_BYTES-1];
logic        active;
logic        shadow;
assign shadow = ~active;

always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) active <= 1'b0;
    else if (swap) active <= shadow;

// =============================================================================
// Shadow bank write  (shadow_clr highest priority)
// =============================================================================
always_ff @(posedge clk) begin
    if (shadow_clr) begin
        for (int i = 0; i < MAX_BYTES; i++)
            bank[shadow][i] <= 8'h00;
    end else begin
        case (mode)

            // ── CONV: 48 bytes, 1 sub-cycle, no bias ──────────────────────
            2'b00: begin
                if (conv_load_en) begin
                    bank[shadow][conv_load_pe_idx * N_TAP    ] <= conv_load_data[ 7: 0];
                    bank[shadow][conv_load_pe_idx * N_TAP + 1] <= conv_load_data[15: 8];
                    bank[shadow][conv_load_pe_idx * N_TAP + 2] <= conv_load_data[23:16];
                    bank[shadow][conv_load_pe_idx * N_TAP + 3] <= conv_load_data[31:24];
                end
            end

            // ── MLP / Patch Merging: single FC weight column ───────────────
            // Each word fills 4 consecutive bytes of the column.
            // mlp_load_k_word is the word index (0 .. C_SPA/4 - 1).
            // Valid byte range = mlp_col_bytes = C_SPA of active PM stage.
            2'b01: begin
                if (mlp_load_en) begin
                    bank[shadow][mlp_load_k_word * N_TAP    ] <= mlp_load_data[ 7: 0];
                    bank[shadow][mlp_load_k_word * N_TAP + 1] <= mlp_load_data[15: 8];
                    bank[shadow][mlp_load_k_word * N_TAP + 2] <= mlp_load_data[23:16];
                    bank[shadow][mlp_load_k_word * N_TAP + 3] <= mlp_load_data[31:24];
                end
            end

            // ── Swin Block: all sub-modes share byte-level write ───────────
            2'b10: begin
                if (msa_load_en) begin
                    bank[shadow][msa_load_word * N_TAP    ] <= msa_load_data[ 7: 0];
                    bank[shadow][msa_load_word * N_TAP + 1] <= msa_load_data[15: 8];
                    bank[shadow][msa_load_word * N_TAP + 2] <= msa_load_data[23:16];
                    bank[shadow][msa_load_word * N_TAP + 3] <= msa_load_data[31:24];
                end
            end

            default: ;
        endcase
    end
end

// =============================================================================
// Active-bank read-out → w_out[pe][tap]
//
// k = sub_cycle * (N_PE * N_TAP) + pe * N_TAP + tap
// w_out[pe][tap] = (k < valid_bytes) ? bank[active][k] : 8'h00
//
// MLP valid_bytes = mlp_col_bytes = C_SPA of active PM stage
//   (NOT C_FC — the column width is the INPUT dimension of the FC layer)
// =============================================================================
always_comb begin
    for (int pe = 0; pe < N_PE; pe++)
        for (int tap = 0; tap < N_TAP; tap++)
            w_out[pe][tap] = 8'h00;

    case (mode)

        // ── CONV: 48 bytes, always sub_cycle=0 ────────────────────────────
        2'b00: begin
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < BYTES_CONV) ? bank[active][k] : 8'h00;
                end
        end

        // ── MLP / Patch Merging ───────────────────────────────────────────
        // valid_bytes = mlp_col_bytes = C_SPA (not C_FC).
        // Sub-cycle count = ceil(C_SPA / 48):
        //   PM1: ceil(384/48) = 8   PM2: ceil(768/48) = 16   PM3: ceil(1536/48) = 32
        2'b01: begin
            automatic int vb_mlp = int'(mlp_col_bytes);
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < vb_mlp) ? bank[active][k] : 8'h00;
                end
        end

        // ── Swin Block ────────────────────────────────────────────────────
        // QK^T and SxV have fixed sizes (constant across all stages).
        // QKV/Proj/FFN1/FFN2 widths come from msa_col_bytes.
        2'b10: begin
            automatic int vb_msa;
            case (msa_sub_mode)
                2'b00:   vb_msa = int'(msa_col_bytes);  // QKV / Proj / FFN2
                2'b01:   vb_msa = BYTES_QKT;             // 32 B
                2'b10:   vb_msa = BYTES_SV;              // 52 B
                2'b11:   vb_msa = int'(msa_col_bytes);  // FFN1
                default: vb_msa = 0;
            endcase
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < vb_msa) ? bank[active][k] : 8'h00;
                end
        end

        default: ;
    endcase
end

// =============================================================================
// Simulation assertion
// =============================================================================
// synthesis translate_off
always_ff @(posedge clk) begin
    if (mode == 2'b00 && sub_cycle != 7'd0)
        $error("[unified_weight_buf] CONV mode: sub_cycle=%0d but must be 0", sub_cycle);
end
// synthesis translate_on

endmodule
