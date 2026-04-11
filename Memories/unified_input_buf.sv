// =============================================================================
// unified_input_buf.sv  (rev 6 — Patch Merging MLP mode corrected)
//
// ── What changed from rev 5 ───────────────────────────────────────────────
//
//   BUG FIX — MLP mode (Patch Merging) had two errors in K_MAX and
//   mlp_k_bytes semantics.
//
//   Ground truth from ilb_patch_merge_buf:
//     Phase-A output  = C_SPA bytes per patch (spatial concat result)
//     FC input row    = C_SPA bytes per patch
//     FC output row   = C_FC  bytes per patch
//
//   The input to the FC (and hence to the ibuf in MLP mode) is the
//   Phase-A output: C_SPA bytes per patch, NOT C_FC.
//
//   Error 1 — K_MAX sized for Swin FFN1, not PM:
//     rev 5: K_MAX = 3072  "Stage 4 FFN1 input row: 768 words × 4 B"
//     Swin FFN1 uses mode 2'b10 (not 2'b01), so K_MAX for the MLP-mode
//     bank layout must cover only PM stage input rows:
//       PM1 C_SPA = 384, PM2 = 768, PM3 = 1536  → K_MAX = 1536
//
//     With K_MAX = 3072, row addresses in MLP mode were WRONG:
//       row 1 started at byte 3072 instead of 1536 (for PM3)
//       row 6 started at byte 18432 instead of 9216 (for PM3)
//     All rows beyond row 0 were written and read at incorrect addresses.
//     FIX: K_MAX = 1536  (PM3 C_SPA dominant in MLP mode)
//
//   Error 2 — mlp_k_bytes meaning was wrong:
//     rev 5 comment: "mlp_k_bytes = valid byte count per MLP input row (96..3072)"
//     For PM, the input row is C_SPA bytes (Phase-A result), NOT C_FC bytes.
//     mlp_k_bytes must be C_SPA of the active PM stage:
//       PM1: 384   PM2: 768   PM3: 1536
//     FIX: comment corrected; mlp_k_bytes port width reduced to [10:0] (covers 1536).
//
//   Other sizes (MHA rows, BANK_BYTES, Swin modes) are unchanged.
//   BANK_BYTES remains 37,632 bytes (49 patches × 768 bytes, MHA dominant).
//     Check: 7 × 1536 = 10,752 bytes for PM3 < 37,632 — fits.
//
// ── Bank sizing ───────────────────────────────────────────────────────────
//   MHA  max: 49 patches × 768 bytes      = 37,632 bytes  ← dominant
//   MLP  max: 7 rows     × 1536 bytes     = 10,752 bytes  (PM3)
//   CONV max: 12 PEs     × 7 wins × 4 tap =    336 bytes
//   BANK_BYTES = 37,632  (MHA dominant, unchanged)
//
// ── K_MAX change ──────────────────────────────────────────────────────────
//   K_MAX  = 1536   was 3072 — corrected to PM3 C_SPA (MLP mode dominant)
//   K_MHA  = 768    unchanged (Stage 4 MHA patch feature bytes)
//
// ── Modes ─────────────────────────────────────────────────────────────────
//   2'b00  CONV   — Patch Embedding
//   2'b01  MLP    — Patch Merging FC input (C_SPA bytes per row)
//   2'b10  W-MSA  — all Swin stages
//   2'b11  SW-MSA — all Swin stages
// =============================================================================

module unified_input_buf #(
    parameter int N_ROWS   = 7,
    parameter int K_MAX    = 1536,  // CORRECTED: PM3 C_SPA = 1536 bytes (was 3072)
    parameter int N_PE     = 12,
    parameter int N_WIN    = 7,
    parameter int N_TAP    = 4,
    parameter int MHA_ROWS = 49,
    parameter int K_MHA    = 768,   // Stage 4 max patch feature bytes (unchanged)
    parameter int SHIFT    = 3
)(
    input  logic        clk,
    input  logic        rst_n,

    // 2'b00=CONV, 2'b01=MLP, 2'b10=W-MSA, 2'b11=SW-MSA
    input  logic [1:0]  mode,
    input  logic        swap,

    // ── Runtime row-width signals (stable per round) ───────────────────────
    //
    // mlp_k_bytes: valid byte count per MLP input row = C_SPA of active PM stage
    //   PM1=384  PM2=768  PM3=1536
    //   CORRECTED: was described as "96..3072" (wrong); now "384..1536" (PM only)
    //   Port width reduced to 11 bits (covers 1536; was 12 bits for 3072).
    //
    // mha_k_bytes: valid byte count per patch = C of active Swin stage
    //   Stage1=96  Stage2=192  Stage3=384  Stage4=768  (unchanged)
    input  logic [10:0] mlp_k_bytes,    // 384..1536 (C_SPA of active PM stage)
    input  logic [9:0]  mha_k_bytes,    // 96..768   (unchanged)

    // ═════════════════════════════════════════════════════════════════════
    // CONV load port  (mode 2'b00)
    // ═════════════════════════════════════════════════════════════════════
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,
    input  logic [2:0]  conv_load_win_idx,
    input  logic [31:0] conv_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // MLP load port  (mode 2'b01 — Patch Merging FC input)
    //
    // Input data is Phase-A output from ilb_patch_merge_buf:
    //   one word (4 bytes) of the C_SPA feature vector for one of the 7 rows.
    //
    // mlp_load_k_word: word index within the C_SPA-byte row
    //   PM1: 0..95   PM2: 0..191   PM3: 0..383
    //   10-bit port covers PM3 max = 383 (unchanged).
    // ═════════════════════════════════════════════════════════════════════
    input  logic        mlp_load_en,
    input  logic [2:0]  mlp_load_row,
    input  logic [9:0]  mlp_load_k_word,   // 0..383 for PM3 (10 bits, unchanged)
    input  logic [31:0] mlp_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // MHA / SW-MSA load port  (mode 2'b10 or 2'b11)
    // Unchanged from rev 5.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        mha_load_en,
    input  logic [5:0]  mha_load_patch,
    input  logic [7:0]  mha_load_k_word,   // 0..191 (Stage4: 768/4=192 words)
    input  logic [31:0] mha_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // Capture port  (all non-CONV modes)
    // cap_col_byte: byte offset within the row
    //   MLP: 0..C_SPA-1 (0..1535 for PM3)   MHA: 0..K_MHA-1 (0..767)
    //   11 bits covers max(1535, 767) = 1535.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        cap_en,
    input  logic [10:0] cap_col_byte,          // 0..1535 (was 12 bits for 3071)
    input  logic [31:0] cap_data [0:N_ROWS-1],

    input  logic [5:0]  mha_cap_patch_base,    // 0,7,...,42 (unchanged)

    // ═════════════════════════════════════════════════════════════════════
    // MHA patch-group read base (unchanged)
    // ═════════════════════════════════════════════════════════════════════
    input  logic [5:0]  mha_patch_base,

    input  logic [2:0]  sub_cycle,

    output logic [7:0]  data_out [0:N_PE-1][0:N_WIN-1][0:N_TAP-1],

    // SW-MSA mask outputs (unchanged)
    output logic [5:0]  mask_req_patch,
    output logic [1:0]  sw_patch_region
);

    // ── Bank sizing ───────────────────────────────────────────────────────
    // MHA dominant: 49 × 768 = 37,632 bytes
    // MLP (PM3):    7  × 1536 = 10,752 bytes  (fits within MHA footprint)
    localparam int BANK_BYTES = MHA_ROWS * K_MHA;  // 37,632 (unchanged)

    logic [7:0] bank [0:1][0:BANK_BYTES-1];
    logic       active;
    logic       shadow;
    assign shadow = ~active;

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) active <= 1'b0;
        else if (swap) active <= shadow;

    // =========================================================================
    // Write logic
    // =========================================================================
    always_ff @(posedge clk) begin
        case (mode)

            // ── CONV ──────────────────────────────────────────────────────
            2'b00: begin
                if (conv_load_en) begin
                    automatic int base = int'(conv_load_pe_idx) * (N_WIN * N_TAP)
                                       + int'(conv_load_win_idx) * N_TAP;
                    bank[shadow][base    ] <= conv_load_data[ 7: 0];
                    bank[shadow][base + 1] <= conv_load_data[15: 8];
                    bank[shadow][base + 2] <= conv_load_data[23:16];
                    bank[shadow][base + 3] <= conv_load_data[31:24];
                end
            end

            // ── MLP / Patch Merging ───────────────────────────────────────
            // Bank layout: row × K_MAX + k_word × N_TAP
            // K_MAX = 1536 (CORRECTED from 3072) — matches PM3 C_SPA row width.
            // Valid bytes per row = mlp_k_bytes = C_SPA of active stage.
            //
            // Row address for row r, word w:  r * K_MAX + w * 4
            //   PM1 (C_SPA=384): row 1 starts at byte 1536, row 6 at byte 9216
            //   PM2 (C_SPA=768): row 1 starts at byte 1536, row 6 at byte 9216
            //   PM3 (C_SPA=1536):row 1 starts at byte 1536, row 6 at byte 9216
            // All PM stages address rows at the same K_MAX=1536 stride.
            2'b01: begin
                if (mlp_load_en) begin
                    automatic int base = int'(mlp_load_row)    * K_MAX
                                       + int'(mlp_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mlp_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mlp_load_data[15: 8];
                    bank[shadow][base + 2] <= mlp_load_data[23:16];
                    bank[shadow][base + 3] <= mlp_load_data[31:24];
                end
                if (cap_en) begin
                    for (int r = 0; r < N_ROWS; r++) begin
                        automatic int addr = r * K_MAX + int'(cap_col_byte);
                        bank[shadow][addr] <= cap_data[r][7:0];
                    end
                end
            end

            // ── W-MSA and SW-MSA ──────────────────────────────────────────
            // Bank layout unchanged: patch × K_MHA + k_word × N_TAP
            2'b10, 2'b11: begin
                if (mha_load_en) begin
                    automatic int base = int'(mha_load_patch) * K_MHA
                                       + int'(mha_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mha_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mha_load_data[15: 8];
                    bank[shadow][base + 2] <= mha_load_data[23:16];
                    bank[shadow][base + 3] <= mha_load_data[31:24];
                end
                if (cap_en) begin
                    for (int r = 0; r < N_ROWS; r++) begin
                        automatic int patch = int'(mha_cap_patch_base) + r;
                        automatic int addr  = patch * K_MHA + int'(cap_col_byte);
                        if (patch < MHA_ROWS)
                            bank[shadow][addr] <= cap_data[r][7:0];
                    end
                end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Read logic (combinatorial)
    // =========================================================================
    always_comb begin
        for (int pe = 0; pe < N_PE; pe++)
            for (int win = 0; win < N_WIN; win++)
                for (int tap = 0; tap < N_TAP; tap++)
                    data_out[pe][win][tap] = 8'h00;

        case (mode)

            // ── CONV ──────────────────────────────────────────────────────
            2'b00: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++)
                            data_out[pe][win][tap] =
                                bank[active][ pe*(N_WIN*N_TAP) + win*N_TAP + tap ];
            end

            // ── MLP / Patch Merging ───────────────────────────────────────
            // Valid guard: k < mlp_k_bytes = C_SPA of active stage.
            // addr = win * K_MAX + k   (K_MAX = 1536 — CORRECTED)
            2'b01: begin
                automatic int vk_mlp = int'(mlp_k_bytes);
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k    = int'(sub_cycle)*(N_PE*N_TAP) + pe*N_TAP + tap;
                            automatic int addr = win * K_MAX + k;
                            data_out[pe][win][tap] = (k < vk_mlp) ? bank[active][addr] : 8'h00;
                        end
            end

            // ── W-MSA ─────────────────────────────────────────────────────
            2'b10: begin
                automatic int vk_mha = int'(mha_k_bytes);
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k     = int'(sub_cycle)*(N_PE*N_TAP) + pe*N_TAP + tap;
                            automatic int patch = int'(mha_patch_base) + win;
                            automatic int addr  = patch * K_MHA + k;
                            data_out[pe][win][tap] =
                                (k < vk_mha && patch < MHA_ROWS)
                                ? bank[active][addr] : 8'h00;
                        end
            end

            // ── SW-MSA ────────────────────────────────────────────────────
            2'b11: begin
                automatic int vk_sw = int'(mha_k_bytes);
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k     = int'(sub_cycle)*(N_PE*N_TAP) + pe*N_TAP + tap;
                            automatic int patch = int'(mha_patch_base) + win;
                            automatic int addr  = patch * K_MHA + k;
                            data_out[pe][win][tap] =
                                (k < vk_sw && patch < MHA_ROWS)
                                ? bank[active][addr] : 8'h00;
                        end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // SW-MSA mask outputs (unchanged)
    // =========================================================================
    always_comb begin
        mask_req_patch  = 6'h00;
        sw_patch_region = 2'b00;
        if (mode == 2'b11) begin
            mask_req_patch = mha_patch_base;
            automatic int row_in_win = int'(mha_patch_base) / N_WIN;
            automatic int col_in_win = int'(mha_patch_base) % N_WIN;
            sw_patch_region[1] = (row_in_win >= SHIFT) ? 1'b1 : 1'b0;
            sw_patch_region[0] = (col_in_win >= SHIFT) ? 1'b1 : 1'b0;
        end
    end

endmodule
