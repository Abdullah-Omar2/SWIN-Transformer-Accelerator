module rom_phi_plus_mp #(
    parameter int INT_BITS  = 6,
    parameter int FRAC_BITS = 11,
    parameter int PORTS     = 24          // one per level-0 pair
)(
    input  logic clk, rst_n,
    input  logic [INT_BITS+FRAC_BITS-1:0] addr [0:PORTS-1],
    output logic [INT_BITS+FRAC_BITS-1:0] data [0:PORTS-1]
);
    localparam int WIDTH  = INT_BITS + FRAC_BITS;
    localparam int AWIDTH = WIDTH - 2;
    localparam int DEPTH  = 2 ** AWIDTH;

    localparam int T0 = 23611;
    localparam int T1 = 21562;
    localparam int T2 = 20364;
    localparam int T3 = 19513;
    localparam int T4 = 18854;

    // Single shared array ? loaded once, read by every port.
    // The synthesiser will map this to BRAM or distributed RAM banks.
    logic [WIDTH-1:0] rom [0:DEPTH-1];
    initial $readmemh("rom_phi_plus.hex", rom);

    genvar p;
    generate
        for (p = 0; p < PORTS; p++) begin : gen_rport
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    data[p] <= '0;
                end else begin
                    if      (addr[p] >= (T0)) data[p] <= '0;
                    else if (addr[p] >= (T1)) data[p] <= (1);
                    else if (addr[p] >= (T2)) data[p] <= (2);
                    else if (addr[p] >= (T3)) data[p] <= (3);
                    else if (addr[p] >= (T4)) data[p] <= (4);
                    else                            data[p] <= rom[addr[p]];
                end
            end
        end
    endgenerate
endmodule
