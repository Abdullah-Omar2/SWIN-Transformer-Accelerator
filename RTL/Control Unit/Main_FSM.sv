// ============================================================================
// Module      : Main_FSM.sv (Container Interface version — v2)
// Description : Main Controller FSM for Swin-Transformer Accelerator.
//               Orchestrates PE, MSA (Window×Head loop), MLP, and PM sequences.
//
// Key architectural changes vs v1:
//   - Q/K/V projections computed per-window (window loop wraps all 6 phases).
//   - Attention phases (QxKT, Softmax, SxV) loop over heads inside each window.
//   - Discrete RESIDUAL_ADD_1/2 states removed; residual_start is now asserted
//     combinationally during the SxV phase (inline datapath adder) and after layer 1 in MLP.
//   - WINDOW_START state (4'd13) added to initialize each window pass.
// ============================================================================

module Main_FSM #(
    parameter NUM_STAGES = 4,
    parameter int BLOCKS_PER_STAGE [0:NUM_STAGES-1] = '{2, 2, 6, 2}
)(
    input  logic        clk,
    input  logic        rst_n,

    // Host interface
    input  logic        start,
    output logic        done,

    // Container Interfaces
    main_fsm_if.master  main_if,
    msa_fsm_if.main     msa_if,
    mlp_fsm_if.main     mlp_if,

    // Shared feedback from physical hardware (top level)
    input  logic        valid_out,       // MMU done (shared)
    input  logic        stored_done,     // IB row stored (shared)
    input  logic        ag_mem_done,     // AG: tiles ready in buffers
    input  logic        ag_layer_done,   // AG: entire layer/window done
    input  logic        shift_done,      // Cyclic shift buffer done
    input  logic        residual_done,   // MLP internal residual done
    input  logic        write_done,      // Output write-back done

    // Softmax block interface (surfaced here, not buried as tie-offs)
    input  logic        softmax_ready,   // Softmax unit is ready to accept data
    input  logic        softmax_done     // Softmax computation finished
);

// ── AG Opcodes for Main-owned ops ─────────────────────────────────────────
localparam logic [2:0] AG_OP_PE = 3'b100;
localparam logic [2:0] AG_OP_PM = 3'b100;

// ── State Encoding ─────────────────────────────────────────────────────────
typedef enum logic [3:0] {
    IDLE            = 4'd0,
    PATCH_EMB_LOAD  = 4'd1,   // PE : load tile from AG
    PATCH_EMB_EXEC  = 4'd2,   // PE : run MMU (Conv streaming, op=3'b000)
    BLOCK_START     = 4'd3,   // Init block / window / head counters
    HEAD_START      = 4'd4,   // MSA: set phase=3 (QxKT) for current head
    CYCLIC_SHIFT    = 4'd5,   // SW-MSA: forward cyclic shift
    MSA_PHASE_START = 4'd6,   // MSA: pulse msa_start with current phase opcode
    MSA_PHASE_WAIT  = 4'd7,   // MSA: wait msa_done; branch on phase/head/window
    REVERSE_SHIFT   = 4'd8,   // SW-MSA: reverse cyclic shift
    // 4'd9 : unused (freed from RESIDUAL_ADD_1)
    MLP_EXEC        = 4'd10,  // Trigger MLP sub-FSM
    PM_LOAD         = 4'd11,  // PM : load concat tile
    PM_EXEC_TILE    = 4'd12,  // PM : run MMU (Concat-FC, op=3'b101)
    WINDOW_START    = 4'd13,  // MSA: begin new window; reset head; set phase=0
    BLOCK_END       = 4'd14,  // Increment block_idx; return to BLOCK_START
    WRITE_OUTPUT    = 4'd15   // Write final results; pulse done
} state_t;

state_t state, next_state;

// ── Counters / Registers ───────────────────────────────────────────────────
logic [1:0] stage_idx;          // Current Swin stage  (0–3)
logic [3:0] block_idx;          // Block within stage
logic [2:0] msa_phase_counter;  // Current MSA phase (0=Q … 5=SxV)
logic [4:0] head_idx;           // Current attention head
logic [4:0] max_heads;          // Head limit for current stage
logic [6:0] window_idx;         // Current window index
logic [6:0] max_windows;        // Window limit for current stage
logic       is_sw_msa;

assign is_sw_msa             = block_idx[0];   // Odd blocks are SW-MSA
assign msa_if.msa_opcode     = msa_phase_counter;
assign msa_if.msa_head_idx   = head_idx;
assign msa_if.msa_window_idx = window_idx;
assign msa_if.is_sw_msa      = is_sw_msa;      // Route to Attn Mask logic

// ── Head Limit Decoder ─────────────────────────────────────────────────────
always_comb begin
    case (stage_idx)
        2'd0: max_heads = 5'd3;
        2'd1: max_heads = 5'd6;
        2'd2: max_heads = 5'd12;
        2'd3: max_heads = 5'd24;
        default: max_heads = 5'd3;
    endcase
end

// ── Window Limit Decoder ───────────────────────────────────────────────────
// num_windows = (spatial_res / 7)^2
always_comb begin
    case (stage_idx)
        2'd0: max_windows = 7'd64;  // (56/7)^2 = 64
        2'd1: max_windows = 7'd16;  // (28/7)^2 = 16
        2'd2: max_windows = 7'd4;   // (14/7)^2 =  4
        2'd3: max_windows = 7'd1;   // ( 7/7)^2 =  1
        default: max_windows = 7'd64;
    endcase
end

// ── State Register ─────────────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else        state <= next_state;
end

// ── Next-State Logic ───────────────────────────────────────────────────────
always_comb begin
    next_state = state;
    case (state)

        // ── Idle ────────────────────────────────────────────────────────
        IDLE: begin
            if (start)
                next_state = PATCH_EMB_LOAD;
        end

        // ── Patch Embedding ──────────────────────────────────────────────
        PATCH_EMB_LOAD: begin
            if (ag_mem_done)
                next_state = PATCH_EMB_EXEC;
        end

        PATCH_EMB_EXEC: begin
            if (stored_done) begin
                if (ag_layer_done)
                    next_state = BLOCK_START;
                else
                    next_state = PATCH_EMB_LOAD;
            end
        end

        // ── Block Sequencer ──────────────────────────────────────────────
        BLOCK_START: begin
            if (block_idx >= BLOCKS_PER_STAGE[stage_idx]) begin
                // All blocks done for this stage
                if (stage_idx == NUM_STAGES - 1)
                    next_state = WRITE_OUTPUT;
                else
                    next_state = PM_LOAD;       // Patch Merging to next stage
            end else begin
                if (is_sw_msa)
                    next_state = CYCLIC_SHIFT;
                else
                    next_state = WINDOW_START;  // Start Window 0 of W-MSA block
            end
        end

        // ── SW-MSA cyclic shift ──────────────────────────────────────────
        CYCLIC_SHIFT: begin
            if (shift_done)
                next_state = WINDOW_START;      // Enter window loop after shift
        end

        // ── Window Loop Entry ────────────────────────────────────────────
        // Resets head_idx and phase counter; unconditional 1-cycle pass-through.
        WINDOW_START: begin
            next_state = MSA_PHASE_START;       // Start Phase 0 (Q projection)
        end

        // ── Head Loop Entry ──────────────────────────────────────────────
        // After Q/K/V projections, jump into attention at Phase 3.
        HEAD_START: begin
            next_state = MSA_PHASE_START;       // Start Phase 3 (QxKT)
        end

        // ── MSA Phase Dispatch ───────────────────────────────────────────
        MSA_PHASE_START: begin
            next_state = MSA_PHASE_WAIT;
        end

        // ── MSA Phase Completion & Branching ─────────────────────────────
        MSA_PHASE_WAIT: begin
            if (msa_if.done) begin
                case (msa_phase_counter)
                    // Projection phases: Q→K→V (window-level)
                    3'd0,
                    3'd1: next_state = MSA_PHASE_START; // Advance to next projection

                    3'd2: next_state = HEAD_START;       // V done → enter head loop

                    // Attention phases: QxKT→Softmax→SxV (head-level)
                    3'd3,
                    3'd4: next_state = MSA_PHASE_START; // Advance within attention

                    3'd5: begin // SxV done → decide next step
                        if (head_idx < max_heads - 5'd1)
                            next_state = HEAD_START;     // More heads → next head
                        else if (window_idx < max_windows - 7'd1)
                            next_state = WINDOW_START;   // More windows → next window
                        else if (is_sw_msa)
                            next_state = REVERSE_SHIFT;  // SW-MSA: undo shift
                        else
                            next_state = MLP_EXEC;       // W-MSA: done, run MLP
                    end

                    default: next_state = MSA_PHASE_START;
                endcase
            end
        end

        // ── SW-MSA reverse shift ─────────────────────────────────────────
        REVERSE_SHIFT: begin
            if (shift_done)
                next_state = MLP_EXEC; // Residual is inline; go direct to MLP
        end

        // ── MLP Sub-FSM ──────────────────────────────────────────────────
        MLP_EXEC: begin
            if (mlp_if.done)
                next_state = BLOCK_END;
        end

        // ── Patch Merging ────────────────────────────────────────────────
        PM_LOAD: begin
            if (ag_mem_done)
                next_state = PM_EXEC_TILE;
        end

        PM_EXEC_TILE: begin
            if (stored_done) begin                
                if (ag_layer_done)
                    next_state = BLOCK_START; // PM done → back to block loop
                else
                    next_state = PM_LOAD;
            end
        end

        // ── Block / Stage Advance ─────────────────────────────────────────
        BLOCK_END: next_state = BLOCK_START;

        // ── Final Write-Back ──────────────────────────────────────────────
        WRITE_OUTPUT: begin
            if (write_done)
                next_state = IDLE;
        end

        default: next_state = IDLE;
    endcase
end

// ── Counter Updates ────────────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        stage_idx         <= 2'd0;
        block_idx         <= 4'd0;
        msa_phase_counter <= 3'd0;
        head_idx          <= 5'd0;
        window_idx        <= 7'd0;
    end else begin
        case (state)

            IDLE: if (start) begin
                stage_idx         <= 2'd0;
                block_idx         <= 4'd0;
                msa_phase_counter <= 3'd0;
                head_idx          <= 5'd0;
                window_idx        <= 7'd0;
            end

            BLOCK_START: begin
                // Reset all MSA loop counters at the start of each block
                msa_phase_counter <= 3'd0;
                head_idx          <= 5'd0;
                window_idx        <= 7'd0;
            end

            WINDOW_START: begin
                // Begin a new window: Q projection starts at phase 0
                msa_phase_counter <= 3'd0;
                head_idx          <= 5'd0;   // Reset head for each new window
            end

            HEAD_START: begin
                // Begin attention for current head: jump to Phase 3 (QxKT)
                msa_phase_counter <= 3'd3;
            end

            MSA_PHASE_WAIT: begin
                if (msa_if.done) begin
                    case (msa_phase_counter)
                        // Projection advance
                        3'd0,
                        3'd1: msa_phase_counter <= msa_phase_counter + 3'd1;

                        3'd2: ; // V done — HEAD_START will set phase to 3

                        // Attention advance
                        3'd3,
                        3'd4: msa_phase_counter <= msa_phase_counter + 3'd1;

                        3'd5: begin // SxV done — update head / window
                            if (head_idx < max_heads - 5'd1) begin
                                head_idx <= head_idx + 5'd1;
                                // phase will be reset to 3 in HEAD_START
                            end else begin
                                head_idx <= 5'd0;
                                if (window_idx < max_windows - 7'd1)
                                    window_idx <= window_idx + 7'd1;
                                    // phase will be reset to 0 in WINDOW_START
                            end
                        end

                        default: ;
                    endcase
                end
            end

            BLOCK_END: block_idx <= block_idx + 4'd1;

            PM_EXEC_TILE: begin
                // Patch Merging complete → advance stage
                if (stored_done && ag_layer_done) begin
                    stage_idx <= stage_idx + 2'd1;
                    block_idx <= 4'd0;
                end
            end

            default: ;
        endcase
    end
end

// ── Output Logic ───────────────────────────────────────────────────────────
always_comb begin
    // ── Safe Defaults ────────────────────────────────────────────────────
    main_if.active           = 1'b0;
    main_if.ag_op_start      = 3'b000;
    main_if.ag_load_next     = 1'b0;
    main_if.mmu_valid_in     = 1'b0;
    main_if.mmu_op_code      = 3'b001;
    main_if.mmu_stage        = stage_idx;
    main_if.mmu_a_sel        = 1'b1;
    main_if.msa_mmu_bias_sel = 1'b1;
    main_if.shift_start      = 1'b0;
    main_if.shift_direction  = 1'b0;
    main_if.residual_start   = 1'b0;  // Repurposed: inline residual_mode_en
    main_if.write_start      = 1'b0;
    msa_if.start             = 1'b0;
    mlp_if.start             = 1'b0;

    // ── Active flag (Main FSM owns shared buses during PE / PM) ──────────
    case (state)
        PATCH_EMB_LOAD, PATCH_EMB_EXEC,
        PM_LOAD, PM_EXEC_TILE: main_if.active = 1'b1;
        default:               main_if.active = 1'b0;
    endcase

    // ── Per-state Output Assignments ─────────────────────────────────────
    case (state)

        PATCH_EMB_LOAD: begin
            main_if.ag_op_start  = AG_OP_PE;
            main_if.ag_load_next = 1'b1;
        end

        PATCH_EMB_EXEC: begin
            main_if.mmu_a_sel    = 1'b0;    // Feature Input Buffer
            main_if.mmu_valid_in = 1'b1;
            main_if.mmu_op_code  = 3'b000;  // Dedicated Conv streaming mode
            main_if.mmu_stage    = 2'd0;    // PE is always Stage 0
        end

        CYCLIC_SHIFT: begin
            main_if.shift_start     = 1'b1;
            main_if.shift_direction = 1'b0; // Forward
        end

        MSA_PHASE_START: msa_if.start = 1'b1; // 1-cycle trigger to MSA sub-FSM

        REVERSE_SHIFT: begin
            main_if.shift_start     = 1'b1;
            main_if.shift_direction = 1'b1; // Reverse
        end

        MLP_EXEC: mlp_if.start = 1'b1; // 1-cycle trigger to MLP sub-FSM

        PM_LOAD: begin
            main_if.ag_op_start  = AG_OP_PM;
            main_if.ag_load_next = 1'b1;
        end

        PM_EXEC_TILE: begin
            main_if.mmu_valid_in = 1'b1;
            main_if.mmu_op_code  = 3'b101;  // Concat-FC (PM uses 4C dot product)
            main_if.mmu_stage    = stage_idx;
        end

        WRITE_OUTPUT: main_if.write_start = 1'b1;

        default: ;
    endcase

    // ── Inline MSA Residual Enable ────────────────────────────────────────
    // Assert residual_start (used as residual_mode_en) during Phase 5 (SxV)
    // so the datapath adder accumulates: IB_stored[x] + SxV_output[x].
    // The adder is wired on the MMU's 7 output ports at the top level.
    if (state == MSA_PHASE_WAIT && msa_phase_counter == 3'd5)
        main_if.residual_start = 1'b1;

    // ── Phase-dependent bias/data mux for MSA phases ──────────────────────
    // QxKT (3) and SxV (5) read from IB; all others use weight bias buffer.
    case (msa_phase_counter)
        3'd3, 3'd5: main_if.msa_mmu_bias_sel = 1'b0; // IB path
        default:    main_if.msa_mmu_bias_sel = 1'b1; // Weight bias buffer
    endcase
end

// ═════════════════════════════════════════════════════════════════════════════
// Gated Sub-FSM Feedback
// Main FSM routes shared physical signals only to the currently active sub-FSM.
// Idle sub-FSMs see 0 on all feedback inputs → no spurious state transitions.
// ═════════════════════════════════════════════════════════════════════════════

// MLP: live only during MLP_EXEC
assign mlp_if.ag_mem_done   = (state == MLP_EXEC) ? ag_mem_done   : 1'b0;
assign mlp_if.ag_layer_done = (state == MLP_EXEC) ? ag_layer_done : 1'b0;
assign mlp_if.mmu_valid_out = (state == MLP_EXEC) ? valid_out     : 1'b0;
assign mlp_if.stored_done   = (state == MLP_EXEC) ? stored_done   : 1'b0;
assign mlp_if.residual_done = (state == MLP_EXEC) ? residual_done : 1'b0;
assign mlp_if.swin_stage    = stage_idx;

// MSA: live only during MSA_PHASE_WAIT (sub-FSM actively running a tile loop)
assign msa_if.ag_mem_done   = (state == MSA_PHASE_WAIT) ? ag_mem_done   : 1'b0;
assign msa_if.ag_layer_done = (state == MSA_PHASE_WAIT) ? ag_layer_done : 1'b0;
assign msa_if.mmu_valid_out = (state == MSA_PHASE_WAIT) ? valid_out     : 1'b0;
assign msa_if.stored_done   = (state == MSA_PHASE_WAIT) ? stored_done   : 1'b0;
assign msa_if.softmax_ready = (state == MSA_PHASE_WAIT) ? softmax_ready : 1'b0;
assign msa_if.softmax_done  = (state == MSA_PHASE_WAIT) ? softmax_done  : 1'b0;

// Main: always connected (PE / PM use these directly)
assign main_if.shift_done    = shift_done;
assign main_if.residual_done = residual_done;
assign main_if.write_done    = write_done;

assign done = (state == WRITE_OUTPUT) && write_done;

endmodule
