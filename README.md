# SWIN-Transformer-Accelerator

## Overview

This repository contains the hardware accelerator design for the Swin Transformer, a hierarchical vision transformer known for its efficiency and performance in various computer vision tasks. The project includes both Python modeling for Post-Training Quantization (PTQ) and Register-Transfer Level (RTL) implementation of the accelerator's core components.

## Repository Structure

The repository is organized into the following main directories:

*   `Python Modeling/`: Contains Python scripts for the Swin Transformer model, including Post-Training Quantization (PTQ) steps.
    *   `PTQ/`: Subdirectory containing the quantization steps (e.g., `step1`, `step9`).
*   `RTL/`: Contains the Register-Transfer Level (RTL) implementation of the hardware accelerator components.
    *   `Control Unit/`: Modules related to the control logic of the accelerator, including FSMs for different operations.
    *   `MMU/`: Matrix Multiplication Unit (MMU) and its sub-components.
    *   `Memories/`: Various buffer and memory modules used in the accelerator.
    *   `Quantizer/`: Quantization logic implemented in RTL.
    *   `ReLU/`: Rectified Linear Unit (ReLU) implementation.
    *   `Softmax/`: Softmax function implementation.
*   `Useful Doc/`: Contains design documents, logs, and other supplementary materials.
    *   `bias_buffer_design_doc.md`: Design document for the bias buffer.
    *   `swin_sizing_table.html`: HTML table detailing Swin Transformer sizing.
    *   `ops_table.html`: HTML table detailing operations.
    *   `control_guide_for_shift_buffer.txt`: Guide for the shift buffer control.
    *   `model_logs.txt`: Logs from model execution.
    *   `MHA.docx`: Document related to Multi-Head Attention.
    *   `RUNTIME FLOW.jpg`: Image illustrating runtime flow.
    *   `phases_Modified_V3.pdf`: PDF document on modified phases.
    *   `stm.xlsx`: Excel sheet for state machine.
    *   `unified_input_buf.svg`: SVG for unified input buffer.
    *   `unified_weight_buf.html`: HTML for unified weight buffer.
*   `arch.jpg`: Overall architecture diagram.
*   `presentation.pdf`: Project presentation.

## Key Features

### Hardware Accelerator (RTL)

The RTL implementation focuses on accelerating the Swin Transformer's computational bottlenecks. Key components include:

*   **Matrix Multiplication Unit (MMU)**: Designed for efficient matrix operations, a core part of transformer models. The `mmu.sv` module implements the MMU, which processes `mmu_in`, `mmu_w`, and `mmu_bias` to produce `mmu_out` [1]. It utilizes Processing Elements (PEs) and an adder tree for parallel computation.
*   **Control Unit**: Manages the overall flow and operation of the accelerator. Files like `unified_controller.sv`, `Main_FSM.sv`, `MLP_FSM_V1.sv`, and `MSA_FSM_V1.sv` define the state machines and control logic for various operations, including Multi-Head Attention (MHA) and Multi-Layer Perceptron (MLP) [2].
*   **Memories and Buffers**: A collection of specialized memory modules (`Memories/` directory) such as `bias_buffer.sv`, `unified_input_buf.sv`, `unified_weight_buf.sv`, and `ilb_swin_block.sv` are designed to store and provide data efficiently to the computational units. The `bias_buffer.sv` is a dedicated on-chip memory unit that stores bias parameters for different operations (Conv, MLP, MHA) and delivers a 7-element bias vector to the MMU [3].
*   **Quantizer**: Implements the quantization logic (`Quantizer/Quantizer.sv`) to handle reduced precision data, crucial for hardware efficiency. It supports configurable input and output widths, as well as shift amounts for quantization [4].
*   **Activation Functions**: Includes RTL implementations of common activation functions like ReLU (`ReLU/RELU.sv`) and Softmax (`Softmax/softmax.sv`).

### Python Modeling (PTQ)

The `Python Modeling/PTQ` directory provides a step-by-step process for Post-Training Quantization of the Swin Transformer. This involves converting a pre-trained floating-point model to a lower-precision fixed-point representation suitable for hardware deployment, while minimizing accuracy loss.

## Design Documentation

The `Useful Doc/` directory contains critical design documents that elaborate on the architectural decisions and implementation details:

*   **Bias Buffer Design Document (`bias_buffer_design_doc.md`)**: Provides a detailed overview of the bias buffer's functionality, bias requirements per operation, internal memory map, module interface, datapath architecture, and Finite State Machine (FSM) [3].
*   **Swin Sizing Table (`swin_sizing_table.html`)**: Offers insights into the sizing and memory requirements for different stages of the Swin Transformer.
*   **Operations Table (`ops_table.html`)**: Details the various operations performed by the accelerator.
*   **Control Guide for Shift Buffer (`control_guide_for_shift_buffer.txt`)**: Explains the control mechanisms for the shift buffer, including insertion points and parameters for the `unified_controller.sv` [2].