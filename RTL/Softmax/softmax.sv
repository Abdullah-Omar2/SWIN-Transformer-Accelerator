// -------------------------------------------------------------------------
// Control Unit for MAC and ACC/DIV pipeline (VECTOR STRIP-MINING ARCHITECTURE)
// -------------------------------------------------------------------------
module softmax_cu #(
    parameter int NUM_LOCATIONS = 7,
    parameter int BUFFER_DEPTH  = 21
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        mask_en_i,

    // Buffer Interface
    output logic                             buf_rd_en,
    output logic [$clog2(BUFFER_DEPTH)-1:0]  buf_rd_addr,
    output logic                             buf_wr_en,
    output logic [$clog2(BUFFER_DEPTH)-1:0]  buf_wr_addr,
    
    output logic                             mask_buf_rd_en,
    output logic [$clog2(BUFFER_DEPTH)-1:0]  mask_buf_rd_addr,

    // Vector Register Control (Spatial Scaling Hooks)
    output logic                             vec_load_en,
    output logic [$clog2(NUM_LOCATIONS)-1:0] vec_load_addr,
    output logic [$clog2(NUM_LOCATIONS)-1:0] vec_store_addr,

    // max_and_clamp Interface
    output logic        mac_valid_i,
    output logic        mac_mode_i,
    input  logic        mac_valid_o,

    // accum_and_div Interface
    output logic        acc_valid_i,
    output logic        acc_mode_i,
    output logic        acc_in_sel, // OPTIMIZATION 4: 1 = bypass vec_reg and chain directly from mac_out
    input  logic        acc_ready,
    input  logic        acc_valid_o,

    // Chunk isolation reset
    output logic        compute_rst_n_o,

    // TB control
    output logic        done
);

    localparam int BUF_ADDR_W = $clog2(BUFFER_DEPTH);
    localparam int VEC_ADDR_W = $clog2(NUM_LOCATIONS);

    // Optimized State Machine for Task-Level Pipelining
    typedef enum logic [3:0] {
        ST_IDLE,
        ST_LOAD,        
        ST_LOAD_WAIT,   
        ST_MAX,         
        ST_CLAMP,       
        ST_ACC_WAIT,    // Combined Wait for chained MAC -> ACC flow
        ST_DIV_PIPE,    // DIV flow
        ST_STORE,       
        ST_DONE
    } state_t;

    state_t state;
    
    logic [BUF_ADDR_W-1:0] base_addr; // Tracks which 7-location chunk we are processing
    logic [VEC_ADDR_W:0]   cnt;       // Tracks the 0..6 intra-chunk offset
    logic                  rd_valid_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            base_addr <= '0;
            cnt       <= '0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        state     <= ST_LOAD;
                        base_addr <= '0;
                        cnt       <= '0;
                    end
                end
                
                ST_LOAD: begin
                    if (cnt == NUM_LOCATIONS - 1) begin
                        state <= ST_LOAD_WAIT;
                        cnt   <= '0;
                    end else begin
                        cnt <= cnt + 1;
                    end
                end
                ST_LOAD_WAIT: begin
                    state <= ST_MAX;
                end
                
                ST_MAX: begin
                    state <= ST_CLAMP;
                end
                
                ST_CLAMP: begin
                    state <= ST_ACC_WAIT;
                end
                
                ST_ACC_WAIT: begin
                    // This state handles the combinational overlapping.
                    // Once the accumulator tree is ready, we transition to DIV.
                    if (acc_ready) state <= ST_DIV_PIPE;
                end
                
                ST_DIV_PIPE: begin
                    if (acc_valid_o) begin
                        state <= ST_STORE;
                        cnt   <= '0;
                    end
                end
                
                ST_STORE: begin
                    if (cnt == NUM_LOCATIONS - 1) begin
                        // Chunk loop logic! Check if there is another full chunk in the buffer
                        if (base_addr + NUM_LOCATIONS < BUFFER_DEPTH) begin
                            // Use explicit cast rather than array slicing to prevent bit width warnings
                            base_addr <= base_addr + NUM_LOCATIONS;
                            cnt       <= '0;
                            state     <= ST_LOAD; // Loop back for the next chunk!
                        end else begin
                            state <= ST_DONE;
                            cnt   <= '0;
                        end
                    end else begin
                        cnt <= cnt + 1;
                    end
                end
                
                ST_DONE: begin
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

    // Offset latency tracker to catch the memory output into the Vector Reg safely
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_q    <= 1'b0;
            vec_load_addr <= '0;
        end else begin
            rd_valid_q    <= buf_rd_en;
            if (buf_rd_en) vec_load_addr <= cnt[VEC_ADDR_W-1:0]; // Use intra-chunk offset
        end
    end
    assign vec_load_en = rd_valid_q;

    always_comb begin
        // Construct full buffer memory addresses combining chunk base and row offset.
        buf_rd_addr    = base_addr + cnt;
        buf_wr_addr    = base_addr + cnt;
        vec_store_addr = cnt[VEC_ADDR_W-1:0];
        
        buf_rd_en      = 1'b0;
        buf_wr_en      = 1'b0;
        mac_valid_i    = 1'b0;
        mac_mode_i     = 1'b0;
        acc_valid_i    = 1'b0;
        acc_mode_i     = 1'b0;
        acc_in_sel     = 1'b0;
        done           = 1'b0;
        mask_buf_rd_en   = 1'b0;
        mask_buf_rd_addr = base_addr + cnt;
        
        // Assert a reset signal strictly for the isolated compute datapath 
        // to flush the registers before starting a new chunk.
        compute_rst_n_o  = !(state == ST_LOAD && cnt == '0);

        case (state)
            ST_LOAD: begin 
                buf_rd_en = 1'b1;
                mask_buf_rd_en = mask_en_i;
            end
            
            ST_MAX: begin
                mac_valid_i = 1'b1;
                mac_mode_i  = 1'b0; // MAX
            end
            
            ST_CLAMP: begin
                mac_valid_i = 1'b1;
                mac_mode_i  = 1'b1; // CLAMP
            end
            
            ST_ACC_WAIT: begin
                // OVERLAPPING PHASES: Task-Level Pipelining
                // Chain the result into ACC instantly
                if (mac_valid_o) begin
                    acc_valid_i = 1'b1;
                    acc_mode_i  = 1'b0; // ACCUMULATE
                    acc_in_sel  = 1'b1; // BYPASS REG
                end
            end
            
            ST_DIV_PIPE: begin
                acc_valid_i = 1'b1;
                acc_mode_i  = 1'b1; // DIVIDE
            end

            ST_STORE: buf_wr_en = 1'b1;
            ST_DONE:  done      = 1'b1;
            default: ;
        endcase
    end

endmodule


// -------------------------------------------------------------------------
// Softmax Core (Combined Top-Level Module)
// -------------------------------------------------------------------------
module softmax_core #(
    parameter int NUM_LOCATIONS = 7,
    parameter int BUFFER_DEPTH  = 21,
    parameter int N_ELEMENTS    = 7,
    parameter int BYTE_W        = 8
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                start,
    input  logic                                mask_en_i,

    // Buffer read interface
    output logic                                buf_rd_en,
    output logic [$clog2(BUFFER_DEPTH)-1:0]     buf_rd_addr,
    input  logic [(N_ELEMENTS*BYTE_W)-1:0]      buf_rd_data,

    // Buffer write interface
    output logic                                buf_wr_en,
    output logic [$clog2(BUFFER_DEPTH)-1:0]     buf_wr_addr,
    output logic [(N_ELEMENTS*BYTE_W)-1:0]      buf_wr_data,
    
    output logic                                mask_buf_rd_en,
    output logic [$clog2(BUFFER_DEPTH)-1:0]     mask_buf_rd_addr,
    input  logic [N_ELEMENTS-1:0]               mask_buf_rd_data,

    output logic                                done
);
    localparam int N_TOTAL    = NUM_LOCATIONS * N_ELEMENTS;
    localparam int DATA_W     = N_ELEMENTS * BYTE_W;
    localparam int VEC_ADDR_W = $clog2(NUM_LOCATIONS);
    
    localparam logic signed [BYTE_W-1:0] MASK_VAL = {1'b1, {BYTE_W-1{1'b0}}};

    // Internal control signals
    logic cu_vec_load_en;
    logic [VEC_ADDR_W-1:0] cu_vec_load_addr;
    logic [VEC_ADDR_W-1:0] cu_vec_store_addr;
    logic cu_acc_in_sel;
    logic cu_compute_rst_n;
    
    logic mac_valid_i, mac_mode_i, mac_valid_o;
    logic acc_valid_i, acc_mode_i, acc_valid_o, acc_ready;

    // The Massive 49-Element Vector Register
    logic signed [BYTE_W-1:0] vec_reg [N_TOTAL];
    logic signed [BYTE_W-1:0] mac_out [N_TOTAL];
    logic signed [BYTE_W-1:0] acc_out [N_TOTAL];
    logic signed [BYTE_W-1:0] acc_in_mux [N_TOTAL];

    // Data Forwarding Mux
    always_comb begin
        for (int i = 0; i < N_TOTAL; i++) begin
            acc_in_mux[i] = cu_acc_in_sel ? mac_out[i] : vec_reg[i];
        end
    end

    // Parallel-to-Serial Write-back logic (Vector Register -> Buffer slice)
    genvar j;
    generate
        for (j = 0; j < N_ELEMENTS; j++) begin : pack_wr_data
            assign buf_wr_data[j*BYTE_W +: BYTE_W] = vec_reg[cu_vec_store_addr * N_ELEMENTS + j];
        end
    endgenerate

    // Vector Register Pipeline Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N_TOTAL; i++) vec_reg[i] <= '0;
        end else begin
            // 1. Serial-to-Parallel Load (Overwrites previous chunk cleanly)
            if (cu_vec_load_en) begin
                for (int col = 0; col < N_ELEMENTS; col++) begin
                    if (mask_en_i && mask_buf_rd_data[col])
                        vec_reg[cu_vec_load_addr * N_ELEMENTS + col] <= MASK_VAL;
                    else
                        vec_reg[cu_vec_load_addr * N_ELEMENTS + col] <= buf_rd_data[col*BYTE_W +: BYTE_W];
                end
            end 
            // 2. Compute Pipeline write-backs
            else if (mac_valid_o && !acc_valid_o) begin
                for (int i = 0; i < N_TOTAL; i++) vec_reg[i] <= mac_out[i];
            end else if (acc_valid_o) begin
                for (int i = 0; i < N_TOTAL; i++) vec_reg[i] <= acc_out[i];
            end
        end
    end

    // Sub-module Instantiations
    softmax_cu #(
        .NUM_LOCATIONS(NUM_LOCATIONS),
        .BUFFER_DEPTH(BUFFER_DEPTH)
    ) u_cu (
        .clk(clk), .rst_n(rst_n), .start(start),
        .mask_en_i(mask_en_i), .mask_buf_rd_en(mask_buf_rd_en), .mask_buf_rd_addr(mask_buf_rd_addr),
        .buf_rd_en(buf_rd_en), .buf_rd_addr(buf_rd_addr),
        .buf_wr_en(buf_wr_en), .buf_wr_addr(buf_wr_addr),
        .vec_load_en(cu_vec_load_en), .vec_load_addr(cu_vec_load_addr), .vec_store_addr(cu_vec_store_addr),
        .mac_valid_i(mac_valid_i), .mac_mode_i(mac_mode_i), .mac_valid_o(mac_valid_o),
        .acc_valid_i(acc_valid_i), .acc_mode_i(acc_mode_i), .acc_in_sel(cu_acc_in_sel), 
        .acc_ready(acc_ready), .acc_valid_o(acc_valid_o),
        .compute_rst_n_o(cu_compute_rst_n),
        .done(done)
    );

    max_and_clamp #(
        .DATA_W (BYTE_W),
        .N      (N_TOTAL) 
    ) u_mac (
        .clk(clk), .rst_n(!((!rst_n) | (!cu_compute_rst_n))), .mode_i(mac_mode_i), .valid_i(mac_valid_i), 
        .in_i(vec_reg), .clamp_o(mac_out), .valid_o(mac_valid_o)
    );

    accum_and_div #(
        .INT_BITS  (6),
        .FRAC_BITS (11),
        .N_IN      (N_TOTAL), 
        .DATA_W    (BYTE_W)
    ) u_acc (
        .clk(clk), .rst_n(!((!rst_n) | (!cu_compute_rst_n))), .mode_i(acc_mode_i), .valid_i(acc_valid_i), 
        .in_i(acc_in_mux), .out_o(acc_out), .valid_o(acc_valid_o), .ready(acc_ready)
    );

endmodule


// -------------------------------------------------------------------------
// Top Level Testbench
// -------------------------------------------------------------------------
module tb_softmax;

    localparam int NUM_LOCATIONS = 7;
    localparam int BUFFER_DEPTH  = 21; // MULTI-CHUNK CAPACITY (3 Arrays of 7)
    
    localparam int BUF_ADDR_W = $clog2(BUFFER_DEPTH);
    localparam int VEC_ADDR_W = $clog2(NUM_LOCATIONS);
    
    localparam int NUM_CHUNKS = BUFFER_DEPTH / NUM_LOCATIONS;
    
    localparam int N_ELEMENTS = 7;
    localparam int BYTE_W = 8;
    localparam int DATA_W = N_ELEMENTS * BYTE_W;
    localparam int MASK_DATA_W = N_ELEMENTS;
    
    // SPATIAL SCALING: Compute vector scales strictly to the 1-chunk size (49 elements)
    localparam int N_TOTAL = NUM_LOCATIONS * N_ELEMENTS; 
    
    int cycles = 0;

    logic clk;
    logic rst_n;
    logic start;
    logic done;
    logic mask_en;

    // Buffer signals (combined from TB and Core)
    logic                 final_buf_wr_en;
    logic [BUF_ADDR_W-1:0] final_buf_wr_addr;
    logic [DATA_W-1:0]    final_buf_wr_data;
    logic                 final_buf_rd_en;
    logic [BUF_ADDR_W-1:0] final_buf_rd_addr;
    logic [DATA_W-1:0]    buf_rd_data;

    // Core outputs to Buffer
    logic                 core_buf_rd_en;
    logic [BUF_ADDR_W-1:0] core_buf_rd_addr;
    logic                 core_buf_wr_en;
    logic [BUF_ADDR_W-1:0] core_buf_wr_addr;
    logic [DATA_W-1:0]    core_buf_wr_data;

    // TB stimulus signals
    logic                 tb_buf_rd_en;
    logic [BUF_ADDR_W-1:0] tb_buf_rd_addr;
    logic                 tb_buf_wr_en;
    logic [BUF_ADDR_W-1:0] tb_buf_wr_addr;
    logic [DATA_W-1:0]    tb_buf_wr_data;
    
    logic                    core_mask_buf_rd_en;
    logic [BUF_ADDR_W-1:0]   core_mask_buf_rd_addr;
    logic [MASK_DATA_W-1:0]  mask_buf_rd_data;
    
    logic                    tb_mask_buf_wr_en;
    logic [BUF_ADDR_W-1:0]  tb_mask_buf_wr_addr;
    logic [MASK_DATA_W-1:0]  tb_mask_buf_wr_data;

    // Multiplex Buffer accesses between Testbench and Core
    assign final_buf_wr_en   = tb_buf_wr_en ? 1'b1           : core_buf_wr_en;
    assign final_buf_wr_addr = tb_buf_wr_en ? tb_buf_wr_addr : core_buf_wr_addr;
    assign final_buf_wr_data = tb_buf_wr_en ? tb_buf_wr_data : core_buf_wr_data;
    
    assign final_buf_rd_en   = tb_buf_rd_en ? 1'b1           : core_buf_rd_en;
    assign final_buf_rd_addr = tb_buf_rd_en ? tb_buf_rd_addr : core_buf_rd_addr;

    // Instantiate Memory Buffer
    buffer #(.DEPTH(BUFFER_DEPTH), .NUM_BYTES(N_ELEMENTS), .BYTE_WIDTH(BYTE_W)) u_buf (
        .clk(clk), .rst_n(rst_n), .wr_en(final_buf_wr_en), .wr_addr(final_buf_wr_addr), 
        .wr_data(final_buf_wr_data), .rd_en(final_buf_rd_en), .rd_addr(final_buf_rd_addr), .rd_data(buf_rd_data)
    );
    
    buffer #(
        .DEPTH      (BUFFER_DEPTH),
        .NUM_BYTES  (1),             // one 7-bit "byte" per word
        .BYTE_WIDTH (N_ELEMENTS)     // 7 bits  ?  word width = 7
    ) u_mask_buf (
        .clk     (clk),
        .rst_n     (rst_n),
        // Write port ? TB only
        .wr_en   (tb_mask_buf_wr_en),
        .wr_addr (tb_mask_buf_wr_addr),
        .wr_data (tb_mask_buf_wr_data),
        // Read port ? core only
        .rd_en   (core_mask_buf_rd_en),
        .rd_addr (core_mask_buf_rd_addr),
        .rd_data (mask_buf_rd_data)
    );

    // Instantiate the Combined Softmax Core
    softmax_core #(
        .NUM_LOCATIONS(NUM_LOCATIONS),
        .BUFFER_DEPTH(BUFFER_DEPTH),
        .N_ELEMENTS(N_ELEMENTS),
        .BYTE_W(BYTE_W)
    ) u_core (
        .clk(clk), .rst_n(rst_n), .start(start),
        .mask_en_i(mask_en), .mask_buf_rd_en(core_mask_buf_rd_en), 
        .mask_buf_rd_addr (core_mask_buf_rd_addr), .mask_buf_rd_data (mask_buf_rd_data), 
        .buf_rd_en(core_buf_rd_en), .buf_rd_addr(core_buf_rd_addr), .buf_rd_data(buf_rd_data),
        .buf_wr_en(core_buf_wr_en), .buf_wr_addr(core_buf_wr_addr), .buf_wr_data(core_buf_wr_data),
        .done(done)
    );

    // Task for reading real buffer memory using a ZERO-TIME PEEK mechanism
    // This perfectly prevents the testbench from consuming cycles and lagging behind the hardware!
    task print_buffer_decimal(input string label);
        logic signed [BYTE_W-1:0] val;
        $display("\n--- %s ---", label);
        for (int row = 0; row < BUFFER_DEPTH; row++) begin
            $write("[%0d]Row [%02d]: ", cycles, row);
            for (int col = 0; col < N_ELEMENTS; col++) begin
                // Peek directly into the buffer's memory array without consuming simulation time
                val = u_buf.mem[row][col*BYTE_W +: BYTE_W];
                $write("%4d ", val);
            end
            $display("");
            
            // Add visual separator between Array chunks
            if ((row + 1) % NUM_LOCATIONS == 0 && row != BUFFER_DEPTH - 1) begin
                $display("  -----------------------------------");
            end
        end
    endtask
    
    task print_mask_buffer(input string label);
        $display("\n--- %s ---", label);
        // Only print rows 0..(NUM_CHUNKS*NUM_LOCATIONS-1) for brevity
        for (int row = 0; row < BUFFER_DEPTH; row++) begin
            $write("[mask] Row [%02d]: ", row);
            for (int col = 0; col < N_ELEMENTS; col++) begin
                // The word is stored as a single 7-bit value; bit[col] = mask for element col
                $write("%0d ", u_mask_buf.mem[row][col]);
            end
            $display("");
            if ((row + 1) % NUM_LOCATIONS == 0 && row != BUFFER_DEPTH - 1)
                $display("  -----------------------------------");
        end
    endtask

    // Task for peeking into the core's massive internal 49-element Vector Register
    task print_vector_decimal(input string label, input int chunk);
        logic signed [BYTE_W-1:0] val;
        $display("\n--- %s ---", label);
        for (int row = 0; row < NUM_LOCATIONS; row++) begin
            // Map the internal vector row back to its global buffer row for clear logging
            $write("[%0d]VecRow [%0d] (Buffer Row %02d): ", cycles, row, chunk * NUM_LOCATIONS + row);
            for (int col = 0; col < N_ELEMENTS; col++) begin
                // Use hierarchical reference to reach inside the core!
                val = u_core.vec_reg[row * N_ELEMENTS + col];
                $write("%4d ", val);
            end
            $display("");
        end
    endtask

    always #5 clk = ~clk;
    always @(posedge clk) cycles++;

    initial begin
        clk = 0; rst_n = 0; start = 0; mask_en = 0;
        tb_buf_wr_en = 0; tb_buf_rd_en = 0;
        tb_mask_buf_wr_en = 0;
        #20 rst_n = 1;
        
        for (int row = 0; row < BUFFER_DEPTH; row++) begin
            logic [DATA_W-1:0] init_word;
            for (int col = 0; col < N_ELEMENTS; col++) begin
                init_word[col*BYTE_W +: BYTE_W] = $urandom_range(16) - 8;
            end
            @(posedge clk);
            tb_buf_wr_en   <= 1;
            tb_buf_wr_addr <= row[BUF_ADDR_W-1:0];
            tb_buf_wr_data <= init_word;
        end
        @(posedge clk); 
        tb_buf_wr_en <= 0;
        #1;
        
        for (int row = 0; row < BUFFER_DEPTH; row++) begin
            logic [MASK_DATA_W-1:0] mword;
            if (row < NUM_LOCATIONS)                          // chunk 0
                mword = 7'b000_0001;                          // element 0 masked
            else if (row < 2 * NUM_LOCATIONS)                 // chunk 1
                mword = 7'b000_0000;                          // no masking
            else                                              // chunk 2
                mword = 7'b010_1000;                          // elements 3 & 5 masked
            @(posedge clk);
            tb_mask_buf_wr_en   <= 1;
            tb_mask_buf_wr_addr <= row[BUF_ADDR_W-1:0];
            tb_mask_buf_wr_data <= mword;
        end
        @(posedge clk);
        tb_mask_buf_wr_en <= 0;
        #1; 

        print_buffer_decimal("INITIAL BUFFER CONTENTS");
        print_mask_buffer("MASK BUFFER CONTENTS");
        mask_en = 1'b1;
        $display("\n[%0d]--- STARTING VECTOR PIPELINE (Processing %0d Chunks) ---", cycles, NUM_CHUNKS);
        @(posedge clk); start <= 1;
        @(posedge clk); start <= 0;

        // SEQUENTIAL MONITORING LOOP
        begin
            for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) begin
                $display("\n[%0d] === EXECUTING CHUNK %0d === (Buffer Rows %0d to %0d)", cycles, chunk, chunk*NUM_LOCATIONS, (chunk+1)*NUM_LOCATIONS-1);
                
                // 1. MONITOR MAX PHASE
                wait (u_core.mac_valid_i === 1'b1 && u_core.mac_mode_i === 1'b0);
                $display("\n[%0d] Monitoring EXECUTION PIPE (Massive 49-Element Parallel Processing):", cycles);
                
                @(posedge clk); #1; // evaluate tree combinational delay
                $display(" [%0d] 1. Global Max computed across all 49 elements instantly: %d", cycles, u_core.u_mac.max_reg);
                
                // 2. MONITOR CLAMP PHASE
                wait (u_core.mac_valid_o === 1'b1);
                @(posedge clk); #1; // wait for vector register to safely latch the clamped results
                print_vector_decimal("2. CLAMP PHASE COMPLETED (49 elements clamped simultaneously & fed to Accumulator)", chunk);
                
                // 3. MONITOR ACCUMULATION PHASE
                wait (u_core.acc_ready === 1'b1);
                @(posedge clk); #1;
                $display("\n [%0d] 3. Accumulator tree finished. Global Running Accumulator (Real): %f", cycles, $itor($signed(u_core.u_acc.accum_reg)) / 2048.0);
                
                // 4. WAIT FOR DIVISION AND STORE
                wait (u_core.acc_valid_o === 1'b1);
                $display(" [%0d] 4. Division Phase complete. Writing back to buffer...", cycles);
                
                // Wait for the final ST_STORE instruction of this chunk to complete
                wait(u_core.buf_wr_en === 1'b1 && u_core.buf_wr_addr == (chunk * NUM_LOCATIONS + NUM_LOCATIONS - 1));
                @(posedge clk); #1;
                
                // 5. PRINT UPDATED BUFFER (Zero simulation time elapsed during this task call)
                print_buffer_decimal($sformatf("BUFFER CONTENTS AFTER CHUNK %0d COMPLETION", chunk));
            end
        end

        wait(done);
        @(posedge clk); #1;
        $display("\n[%0d]--- PIPELINE COMPLETELY FINISHED ---", cycles);
        #20 $finish;
    end

endmodule