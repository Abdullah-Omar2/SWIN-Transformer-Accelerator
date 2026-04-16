// ============================================================================
// Module      : Control_Unit.sv (Container Interface version)
// Description : Top-level Control Unit Wrapper for the Swin-Transformer.
//               Instantiates Main, MLP, and MSA FSMs and handles shared bus
//               multiplexing (AG, MMU, IB).
// ============================================================================

module Control_Unit #(
    parameter NUM_STAGES = 4,
    parameter int BLOCKS_PER_STAGE [0:NUM_STAGES-1] = '{2, 2, 6, 2}
)(
    input  logic        clk,
    input  logic        rst_n,

    // Host
    input  logic        start,
    output logic        done,
    
    // Data-Path MUX Selects (derived from FSM containers)
    output logic        mmu_a_sel,
    output logic        mmu_bias_sel,
    // AG Context (exposed for the Unified AG)
    output logic [1:0]  swin_stage,
    output logic [4:0]  msa_head_idx,
    output logic [6:0]  msa_window_idx,

    // =====================================================================
    // Address Generator — ONE shared interface
    // =====================================================================
    output logic [2:0]  ag_op_start,       // Init AG for new operation/phase
    output logic        ag_load_next,      // Fetch next tile pair
    input  logic        ag_mem_done,       // AG: tiles ready
    input  logic        ag_layer_done,     // AG: all tiles done (level signal)

    // =====================================================================
    // MMU — ONE shared interface
    // =====================================================================
    output logic        mmu_valid_in,      // 1-cycle pulse to start MMU
    output logic [2:0]  mmu_op_code,       // Operation type
    output logic [1:0]  mmu_stage,         // Swin-T stage for latency scaling
    input  logic        mmu_valid_out,     // MMU done

    // =====================================================================
    // Intermediate Buffer — ONE shared interface
    // =====================================================================
    input  logic        stored_done,       // IB: output row stored

    // =====================================================================
    // Shared Single-Source Signals (driven to Main FSM for gating)
    // =====================================================================
    input  logic        shift_done,        // Cyclic Shift Buffer done
    output logic        shift_start,       // Cyclic Shift start
    output logic        shift_direction,   // 0=forward, 1=reverse

    input  logic        residual_done,     // Residual Adder done
    output logic        residual_start,    // Post-MSA Residual start

    input  logic        write_done,        // Output write done
    output logic        write_start,       // Output write start

    // =====================================================================
    // Attn Mask control
    // =====================================================================
    output logic        is_sw_msa,         // 1=SW-MSA block, 0=W-MSA block → Attn Mask select

    // =====================================================================
    // Softmax block
    // =====================================================================
    output logic        softmax_start,     // Trigger softmax computation (from MSA FSM)
    input  logic        softmax_ready,     // Softmax unit ready to accept
    input  logic        softmax_done       // Softmax computation complete
);

// =====================================================================
// Internal Container Interfaces
// =====================================================================
main_fsm_if  main_if();
msa_fsm_if   msa_if();
mlp_fsm_if   mlp_if();

// =====================================================================
// Centralized Multiplexing Logic (Priority: Main > MSA > MLP)
// =====================================================================

// AG Bus Muxing
assign ag_op_start  = main_if.active ? main_if.ag_op_start  :
                      msa_if.start   ? msa_if.ag_op_start   :
                                       mlp_if.ag_op_start;

assign ag_load_next = main_if.ag_load_next | msa_if.ag_load_next | mlp_if.ag_load_next;

// MMU Bus Muxing
assign mmu_valid_in = main_if.mmu_valid_in | msa_if.mmu_valid_in | mlp_if.mmu_valid_in;

assign mmu_op_code  = main_if.active ? main_if.mmu_op_code  :
                      msa_if.start   ? msa_if.mmu_op_code   : 
                                       mlp_if.mmu_op_code;

assign mmu_stage    = main_if.active ? main_if.mmu_stage    :
                                       mlp_if.mmu_stage;


// Data-Path MUX Selects
assign mmu_a_sel    = main_if.mmu_a_sel; 
assign mmu_bias_sel = main_if.active ? 1'b1 : // PE/PM always Bias Buffer
                      msa_if.start   ? main_if.msa_mmu_bias_sel : // MSA is phase-dependent
                                       1'b1; // MLP always Bias Buffer

// AG Context Mapping
assign swin_stage      = mlp_if.swin_stage;
assign msa_head_idx    = msa_if.msa_head_idx;
assign msa_window_idx  = msa_if.msa_window_idx;
assign is_sw_msa       = msa_if.is_sw_msa;        // Route to Attn Mask hardware
assign softmax_start   = msa_if.softmax_start;    // Route MSA FSM trigger to top-level

// =====================================================================
// Sub-module Instantiations
// =====================================================================

// Main FSM
Main_FSM #(
    .NUM_STAGES       (NUM_STAGES),
    .BLOCKS_PER_STAGE (BLOCKS_PER_STAGE)
) u_main_fsm (
    .clk              (clk),
    .rst_n            (rst_n),
    .start            (start),
    .done             (done),
    .main_if          (main_if.master),
    .msa_if           (msa_if.main),
    .mlp_if           (mlp_if.main),
    // Shared physical feedback — gating done inside Main_FSM
    .valid_out        (mmu_valid_out),
    .stored_done      (stored_done),
    .ag_mem_done      (ag_mem_done),
    .ag_layer_done    (ag_layer_done),
    .shift_done       (shift_done),
    .residual_done    (residual_done),
    .write_done       (write_done),
    // Softmax block — now proper ports, not internal tie-offs
    .softmax_ready    (softmax_ready),
    .softmax_done     (softmax_done)
);

// Expose shift/residual/write outputs to top-level ports
assign shift_start      = main_if.shift_start;
assign shift_direction  = main_if.shift_direction;
assign residual_start   = main_if.residual_start | mlp_if.residual_start;
assign write_start      = main_if.write_start;

// MLP Sub-FSM
MLP_FSM u_mlp_fsm (
    .clk                (clk),
    .rst_n              (rst_n),
    .mlp_if             (mlp_if.sub)
);

// MSA Sub-FSM
MSA_FSM_V1 u_msa_fsm (
    .clk                (clk),
    .rst_n              (rst_n),
    .msa_if             (msa_if.sub)
);

endmodule
