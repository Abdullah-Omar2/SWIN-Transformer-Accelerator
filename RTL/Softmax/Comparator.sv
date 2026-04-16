module tree_comparator #(
    parameter int DATA_W = 8,
    parameter int N      = 10
)(
    input  wire  signed [DATA_W-1:0] in [0:N-1],
    output logic signed [DATA_W-1:0] max_out
);
    localparam int STAGES = (N == 1) ? 1 : $clog2(N);
    logic signed [DATA_W-1:0] stage [0:STAGES][0:N-1];

    always_comb begin
        int nc;
        for (int i = 0; i < N; i++)
            stage[0][i] = in[i];

        for (int s = 0; s < STAGES; s++) begin
            nc = (N + (1 << s) - 1) >> s;
            for (int i = 0; i < nc/2; i++)
                stage[s+1][i] = (stage[s][2*i] > stage[s][2*i+1])
                                 ? stage[s][2*i] : stage[s][2*i+1];
            if (nc % 2 == 1)
                stage[s+1][nc/2] = stage[s][nc-1];
        end
        // *** FIXED: moved outside the loop ***
        max_out = stage[STAGES][0];
    end
endmodule

`timescale 1ns / 1ps

module tree_comparator_tb;

    parameter int DATA_W = 8;
    parameter int N = 10;

    logic signed [DATA_W-1:0] in [0:N-1];
    logic signed [DATA_W-1:0] max_out;

    tree_comparator #(
        .DATA_W(DATA_W),
        .N(N)
    ) dut (
        .in(in),
        .max_out(max_out)
    );

    initial begin
      
      logic signed [DATA_W-1:0] golden_max;
      repeat (100) begin
        
	      in[0] = $urandom_range(0, 2 * ((1<<DATA_W)-1) - ((1<<DATA_W)-1));
        golden_max = in[0];
        for (int i = 1; i < N; i++) begin
          in[i] = $urandom_range(0, 2 * ((1<<DATA_W)-1) - ((1<<DATA_W)-1));
          if (in[i] > golden_max)
          golden_max = in[i];
        end
      end

      #1;
        
      if (max_out !== golden_max) begin
        $error("Mismatch! expected=%0d got=%0d",
                  golden_max, max_out);
          $stop;
      end

      $display("All random tests passed");
      $stop;
    end


endmodule

