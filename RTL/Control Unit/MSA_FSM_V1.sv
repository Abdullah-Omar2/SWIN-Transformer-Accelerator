// =============================================================================
// Module      : MSA_FSM_V1  (Container Interface version)
// Description : Sub-FSM for one MSA phase in the Swin-Transformer accelerator.
// =============================================================================

module MSA_FSM_V1 (
    input  logic        clk,
    input  logic        rst_n,

    // Container Interface
    msa_fsm_if.sub      msa_if
);

// PHASE indices (captures from Main Controller)
localparam logic [2:0] PHASE_Q       = 3'b000;
localparam logic [2:0] PHASE_K       = 3'b001;
localparam logic [2:0] PHASE_V       = 3'b010;
localparam logic [2:0] PHASE_QKT     = 3'b011;
localparam logic [2:0] PHASE_SOFTMAX = 3'b100;
localparam logic [2:0] PHASE_SxV     = 3'b101;

// State Encoding  
typedef enum logic [2:0] {
    MSA_IDLE         = 3'd0,  // Wait for trigger
    MSA_MMU_START    = 3'd1,  // Pulse ag_op_start for AG initialization
    MSA_MMU_LOAD     = 3'd2,  // Pulse ag_load_next; wait ag_mem_done
    MSA_MMU_EXEC     = 3'd3,  // Wait stored_done; check ag_layer_done
    MSA_SOFTMAX_EXEC = 3'd4,  // Wait softmax_ready; pulse softmax_start
    MSA_SOFTMAX_WAIT = 3'd5,  // Wait softmax_done
    MSA_FINISH       = 3'd6   // Signal msa_if.done
} msa_state_t;

msa_state_t  curr_state, next_state;
logic [2:0]  opcode_r;

// State Transition
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        curr_state <= MSA_IDLE;
        opcode_r   <= 3'b000;
    end else begin
        curr_state <= next_state;
        if (curr_state == MSA_IDLE && msa_if.start)
            opcode_r <= msa_if.msa_opcode;
    end
end

// Next-State Logic
always_comb begin
    next_state = curr_state;
    case (curr_state)
        MSA_IDLE: begin
            if (msa_if.start) begin
                if (msa_if.msa_opcode == PHASE_SOFTMAX) 
                    next_state = MSA_SOFTMAX_EXEC;
                else                                    
                    next_state = MSA_MMU_START;
            end
        end

        MSA_MMU_START: next_state = MSA_MMU_LOAD;

        MSA_MMU_LOAD: begin
            if (msa_if.ag_mem_done) 
                next_state = MSA_MMU_EXEC;
        end

        MSA_MMU_EXEC: begin
            if (msa_if.stored_done) begin
                if (msa_if.ag_layer_done) 
                    next_state = MSA_FINISH;
                else                     
                    next_state = MSA_MMU_LOAD;
            end
        end

        MSA_SOFTMAX_EXEC: begin
            if (msa_if.softmax_ready) 
                next_state = MSA_SOFTMAX_WAIT;
        end

        MSA_SOFTMAX_WAIT: begin
            if (msa_if.softmax_done)  
                next_state = MSA_FINISH;
        end

        MSA_FINISH: next_state = MSA_IDLE;

        default: next_state = MSA_IDLE;
    endcase
end

// Output logic
always_comb begin
    // Defaults
    msa_if.done           = 1'b0;
    msa_if.ag_op_start    = 3'b000;
    msa_if.ag_load_next   = 1'b0;
    msa_if.mmu_valid_in   = 1'b0;
    msa_if.softmax_start  = 1'b0;

    case (curr_state)
        MSA_MMU_START: begin
            msa_if.ag_op_start = opcode_r; 
        end

        MSA_MMU_LOAD: begin
            msa_if.ag_load_next = 1'b1;
            if (msa_if.ag_mem_done) 
                msa_if.mmu_valid_in = 1'b1;
        end

        MSA_SOFTMAX_EXEC: begin
            if (msa_if.softmax_ready) 
                msa_if.softmax_start = 1'b1;
        end

        MSA_FINISH: begin
            msa_if.done = 1'b1;
        end
        default: ;
    endcase
end

endmodule
