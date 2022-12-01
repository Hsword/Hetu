# Overlap Test
Modern GPUs simultaneously performing compute kernels and communication primitives lead to slowdown for both sides.
This directory contains scripts to test such overlap slowdown coefficient.
In our evaluations, the overlap slowdown coefficient of computation and communication is $1.3 \times$.

## Usage
- Run `sh test_overlap.sh` to test overlap slowdown coefficient.

