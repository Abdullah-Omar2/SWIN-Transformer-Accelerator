// =============================================================================
// Module      : cu_interfaces.sv
// Description : SystemVerilog interfaces for "Container-based" FSM signaling.
//               Each FSM (Main, MLP, MSA) has its own interface containing
//               all signals it needs to drive or read.
// =============================================================================

// -----------------------------------------------------------------------------
// MLP Container Interface
// -----------------------------------------------------------------------------
interface mlp_fsm_if;
    // FSM Signaling
    logic        start;
    logic        done;
    logic [1:0]  swin_stage;

    // Shared AG requests (local copies)
    logic [2:0]  ag_op_start;
    logic        ag_load_next;
    logic        ag_mem_done;
    logic        ag_layer_done;

    // Shared MMU requests
    logic        mmu_valid_in;
    logic [2:0]  mmu_op_code;
    logic [1:0]  mmu_stage;
    logic        mmu_valid_out;

    // Shared IB requests
    logic        stored_done;

    // Local Residual control
    logic        residual_start;
    logic        residual_done;   // PLACEHOLDER: tied 1'b0; MLP FSM does not consume

    // Main FSM Port (Master)
    modport main (
        output start,
        input  done,
        output swin_stage,
        // Gated feedback: Main FSM writes these into the interface
        output ag_mem_done, ag_layer_done,
        output mmu_valid_out,
        output stored_done,
        output residual_done
    );

    // MLP FSM Port (Slave)
    modport sub (
        input  start,
        output done,
        input  swin_stage,

        output ag_op_start, ag_load_next, input ag_mem_done, input ag_layer_done,
        output mmu_valid_in, output mmu_op_code, output mmu_stage, input mmu_valid_out,
        input stored_done,
        output residual_start, input residual_done
    );
endinterface

// -----------------------------------------------------------------------------
// MSA Container Interface
// -----------------------------------------------------------------------------
interface msa_fsm_if;
    // FSM Signaling
    logic        start;
    logic        done;
    logic [2:0]  msa_opcode;
    logic [4:0]  msa_head_idx;
    logic [6:0]  msa_window_idx;
    logic        is_sw_msa;        // Forwarded to Attn Mask: 1=SW-MSA, 0=W-MSA

    // Shared AG requests
    logic [2:0]  ag_op_start;
    logic        ag_load_next;
    logic        ag_mem_done;
    logic        ag_layer_done;

    // Shared MMU requests
    logic        mmu_valid_in;
    logic [2:0]  mmu_op_code;
    logic        mmu_valid_out;

    // Shared IB requests
    logic        stored_done;

    // Local Softmax control
    logic        softmax_start;
    logic        softmax_ready;    // PLACEHOLDER: tied 1'b1 until Softmax block integrated
    logic        softmax_done;     // PLACEHOLDER: tied 1'b0 until Softmax block integrated

    // Main FSM Port (Master)
    modport main (
        output start,
        input  done,
        output msa_opcode,
        output msa_head_idx,
        output msa_window_idx,
        output is_sw_msa,
        // Gated feedback: Main FSM writes these into the interface
        output ag_mem_done, ag_layer_done,
        output mmu_valid_out,
        output stored_done,
        output softmax_ready,
        output softmax_done
    );

    // MSA FSM Port (Slave)
    modport sub (
        input  start,
        output done,
        input  msa_opcode,
        input  msa_head_idx,
        input  msa_window_idx,
        input  is_sw_msa,

        output ag_op_start, ag_load_next, input ag_mem_done, input ag_layer_done,
        output mmu_valid_in, input mmu_valid_out,
        input stored_done,
        output softmax_start, input softmax_ready, input softmax_done
    );
endinterface

// -----------------------------------------------------------------------------
// Main FSM Container (for PE/PM operations)
// -----------------------------------------------------------------------------
interface main_fsm_if;
    // Common
    logic        active;           // High when Main FSM drives shared bus (PE/PM)

    // Shared AG requests (Main versions)
    logic [2:0]  ag_op_start;
    logic        ag_load_next;

    // Shared MMU requests
    logic        mmu_valid_in;
    logic [2:0]  mmu_op_code;
    logic [1:0]  mmu_stage;

    // Shared MMU Data MUX Selects
    logic        mmu_a_sel;        // 0=FIB (patch/attention data), 1=Weight Buffer
    logic        msa_mmu_bias_sel; // 0=IB path (QxKT/SxV phases), 1=Weight Bias Buffer

    // Local Shift control
    logic        shift_start;
    logic        shift_done;
    logic        shift_direction;  // 0=Forward (CYCLIC_SHIFT), 1=Reverse (REVERSE_SHIFT)

    // Local Residual (post-MSA inline adder enable)
    logic        residual_start;   // Level signal during MSA Phase 5 (SxV)
    logic        residual_done;    // From residual adder (not consumed by FSM currently)

    // Local Write control
    logic        write_start;
    logic        write_done;

    // Main FSM provides everything here
    modport master (
        output active,
        output ag_op_start, ag_load_next,
        output mmu_valid_in, output mmu_op_code, output mmu_stage,
        output mmu_a_sel, output msa_mmu_bias_sel,
        output shift_start, input  shift_done, output shift_direction,
        output residual_start, input  residual_done,
        output write_start, input  write_done
    );
endinterface
