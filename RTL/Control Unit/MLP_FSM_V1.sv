// =============================================================================
// Module      : MLP_FSM  (Container Interface version - v2)
// Description : Manages the two-layer MLP sequence (FC L1 -> FC L2).
//               Residual addition is now INLINE during the L2 execution pass.
// =============================================================================

module MLP_FSM (
    input  logic        clk,
    input  logic        rst_n,

    // Container Interface
    mlp_fsm_if.sub      mlp_if
);

// State Encoding  
typedef enum logic [3:0] {
    MLP_IDLE          = 4'd0,
    // ---- L1 layer ----
    MLP_L1_START      = 4'd1,
    MLP_L1_LOAD       = 4'd2,
    MLP_L1_EXEC       = 4'd3,
    // ---- Transition ----
    MLP_IDLE_L2       = 4'd4,
    // ---- L2 layer ----
    MLP_L2_START      = 4'd5,
    MLP_L2_LOAD       = 4'd6,
    MLP_L2_EXEC       = 4'd7,
    // ---- Finalization ----
    MLP_FINISH        = 4'd10
} mlp_state_t;

// AG Opcodes for MLP
localparam logic [2:0] OP_MLP_L1 = 3'b110;
localparam logic [2:0] OP_MLP_L2 = 3'b111;

// MMU Opcodes
localparam logic [2:0] MMU_OP_FC_L1 = 3'b001;
localparam logic [2:0] MMU_OP_FC_L2 = 3'b101;

mlp_state_t curr_state, next_state;

// State transition
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) curr_state <= MLP_IDLE;
    else        curr_state <= next_state;
end

// Next-State Logic
always_comb begin
    next_state = curr_state;
    case (curr_state)
        MLP_IDLE: begin 
            if (mlp_if.start)
                next_state = MLP_L1_START; 
        end 

        MLP_L1_START: next_state = MLP_L1_LOAD;
        
        MLP_L1_LOAD: begin   
            if (mlp_if.ag_mem_done) 
                next_state = MLP_L1_EXEC;
        end 
        
        MLP_L1_EXEC: begin  
            if (mlp_if.stored_done) begin
                if (mlp_if.ag_layer_done) 
                    next_state = MLP_IDLE_L2;
                else                    
                    next_state = MLP_L1_LOAD;
                end
        end 
        
        MLP_IDLE_L2:  next_state = MLP_L2_START;
        
        MLP_L2_START: next_state = MLP_L2_LOAD;
       
        MLP_L2_LOAD: begin 
            if (mlp_if.ag_mem_done) 
                next_state = MLP_L2_EXEC;
        end
       
        MLP_L2_EXEC: begin   
            if (mlp_if.stored_done) begin
                if (mlp_if.ag_layer_done) 
                    next_state = MLP_FINISH; // DIRECT TO FINISH (Residual is inline)
                else                     
                    next_state = MLP_L2_LOAD;
            end
        end
        
        MLP_FINISH:    next_state = MLP_IDLE;
        default:       next_state = MLP_IDLE;
    endcase
end

// MMU Stage passthrough
assign mlp_if.mmu_stage = mlp_if.swin_stage; 

// Output Logic
always_comb begin
    // Defaults
    mlp_if.done           = 1'b0;
    mlp_if.ag_op_start    = 3'b000;
    mlp_if.ag_load_next   = 1'b0;
    mlp_if.mmu_valid_in   = 1'b0;
    mlp_if.mmu_op_code    = MMU_OP_FC_L1;
    mlp_if.residual_start = 1'b0;

    case (curr_state)
        MLP_L1_START: mlp_if.ag_op_start = OP_MLP_L1;
        
        MLP_L1_LOAD: begin
            mlp_if.ag_load_next = 1'b1;
            if (mlp_if.ag_mem_done) mlp_if.mmu_valid_in = 1'b1;
            mlp_if.mmu_op_code   = MMU_OP_FC_L1;
        end
        
        MLP_L1_EXEC: begin
            mlp_if.mmu_op_code   = MMU_OP_FC_L1;
        end

        MLP_L2_START: mlp_if.ag_op_start = OP_MLP_L2;
        
        MLP_L2_LOAD: begin
            mlp_if.ag_load_next = 1'b1;
            if (mlp_if.ag_mem_done) mlp_if.mmu_valid_in = 1'b1;
            
            mlp_if.mmu_op_code    = MMU_OP_FC_L2;
            mlp_if.residual_start = 1'b1; // ACTIVATE INLINE ADDER
        end
        
        MLP_L2_EXEC: begin
            mlp_if.mmu_op_code    = MMU_OP_FC_L2;
            mlp_if.residual_start = 1'b1; // ACTIVATE INLINE ADDER
        end
        
        MLP_FINISH:    mlp_if.done           = 1'b1;
        
        default: ;
    endcase
end

endmodule
