# gpu_ops
This directory contains executor and operators for computation and communication. Though the name of directory is "gpu_ops", in each operator we call different API for computation in NumPy(CPU), DNNL(CPU), CUDA(GPU) according to the context specified in executor and the environment.

## Executor
* Defined in executor.py, contains all the configurations and controls the training/inference process.

## Operators
### Computation
| Operator | NumPy(CPU) | DNNL(CPU) | CUDA(GPU) | CUDA Backend |
| :----: | :----: | :----: | :----: | :----: |
| AddByConstOp | ✔ | ✔ | ✔ | / |
| AddOp | ✔ | ✔ | ✔ | / |
| Avg_Pool2dOp | ✔ | ✔ | ✔ | CuDNN |
| Avg_Pool2d_GradientOp | ✔ | ✔ | ✔ | CuDNN |
| BatchMatMulOp | ✔ | ✖ | ✔ | CuBLAS |
| Batch_NormalizationOp | ✔ | ✔ | ✔ | CuDNN |
| Batch_Normalization_GradientOp | ✔ | ✔ | ✔ | CuDNN |
| BinaryCrossEntropyOp | ✔ | ✖ | ✔ | / |
| BroadcastToOp | ✔ | ✖ | ✔ | / |
| BroadcastShapeOp | ✔ | ✖ | ✔ | / |
| ConcatOp | ✔ | ✔ | ✔ | / |
| Concat_gradientOP | ✔ | ✔ | ✔ | / |
| ConcatenateOp | ✔ | ✖ | ✔ | / |
| Concatenate_gradientOP | ✔ | ✖ | ✔ | / |
| Conv2dOp | ✔ | ✔ | ✔ | CuDNN |
| Conv2dAddBiasOp | ✔ | ✖ | ✔ | CuDNN |
| Conv2d_Gradient_of_DataOp | ✔ | ✔ | ✔ | CuDNN |
| Conv2d_Gradient_of_FilterOp | ✔ | ✔ | ✔ | CuDNN |
| Conv2d_BroadcastToOp | ✔ | ✖ | ✔ | / |
| Conv2d_ReduceSumOp | ✔ | ✖ | ✔ | / |
| CsrmvOp | ✔ | ✖ | ✔ | / |
| CsrmmOp | ✔ | ✖ | ✔ | / |
| DistGCN_15dOp | ✖ | ✖ | ✔ | / |
| DivOp | ✔ | ✔ | ✔ | / |
| DivConstOp | ✔ | ✔ | ✔ | / |
| DropoutOp | ✔ | ✔ | ✔ | CuRAND |
| Dropout_GradientOp | ✔ | ✔ | ✔ | CuRAND |
| EmbeddingLookUp | ✔ | ✖ | ✔ | / |
| EmbeddingLookUp_Gradient | ✔ | ✖ | ✔ | / |
| Instance_Normalization2dOp | ✖ | ✖ | ✔ | CuDNN |
| Instance_Normalization2d_GradientOp | ✖ | ✖ | ✔ | CuDNN |
| Layer_NormalizationOp | ✔ | ✖ | ✔ | CuDNN |
| Layer_Normalization_GradientOp | ✔ | ✖ | ✔ | CuDNN |
| LeakyReluOp | ✖ | ✖ | ✔ | / |
| LeakyReluGradientOp | ✖ | ✖ | ✔ | / |
| LinearOp | ✔ | ✖ | ✔ | CuBLAS |
| MatrixDotOp | ✔ | ✖ | ✔ | / |
| MatMulOp | ✔ | ✔ | ✔ | CuBLAS |
| Max_Pool2dOp | ✔ | ✔ | ✔ | CuDNN |
| Max_Pool2d_GradientOp | ✔ | ✔ | ✔ | CuDNN |
| MulByConstOp | ✔ | ✔ | ✔ | / |
| MulOp | ✔ | ✔ | ✔ | / |
| OneHotOp | ✔ | ✖ | ✔ | / |
| OnesLikeOp | ✔ | ✔ | ✔ | / |
| OppositeOp | ✔ | ✔ | ✔ | / |
| PadOp | ✔ | ✔ | ✔ | / |
| Pad_GradientOp | ✔ | ✔ | ✔ | / |
| ReduceMeanOp | ✔ | ✖ | ✔ | CuDNN |
| ReduceSumOp | ✔ | ✖ | ✔ | CuDNN |
| ReduceSumAxisZeroOp | ✔ | ✔ | ✔ | / |
| ReluOp | ✔ | ✔ | ✔ | / |
| ReluGradientOp | ✔ | ✔ | ✔ | / |
| Array_ReshapeOp | ✔ | ✔ | ✔ | / |
| SigmoidOp | ✔ | ✔ | ✔ | / |
| SliceOp | ✔ | ✖ | ✔ | / |
| SliceGradientOp | ✔ | ✖ | ✔ | / |
| SoftmaxOp | ✔ | ✔ | ✔ | CuDNN |
| SoftmaxGradientOp | ✔ | ✖ | ✔ | CuDNN |
| SoftmaxCrossEntropyOp | ✔ | ✔ | ✔ | CuDNN (Optional) |
| SoftmaxCrossEntropyGradientOp | ✔ | ✖ | ✔ | CuDNN (Optional) |
| SplitOp | ✔ | ✖ | ✔ | / |
| SplitGradientOp | ✔ | ✖ | ✔ | / |
| SqrtOp | ✔ | ✔ | ✔ | / |
| SumOp | ✔ | ✖ | ✔ | / |
| ReciprocalSqrtOp | ✔ | ✔ | ✔ | / |
| TanhOp | ✔ | ✔ | ✔ | / |
| TanhGradientOp | ✔ | ✖ | ✔ | / |
| TransposeOp | ✔ | ✔ | ✔ | / |
| WhereOp | ✔ | ✖ | ✔ | / |
| ZerosLikeOp | ✔ | ✔ | ✔ | / |
| OptimizerOp | ✔ | ✔ | ✔ | / |
| OptimizerOp for sparse | ✔ | ✖ | ✔ | / |
| DataloaderOp | ✔ | ✔ | / | / |

### Communication
* DataH2DOp
* DataD2HOp
* DataD2HSparseOp
* AllReduceCommunicateOp
* ParameterServerCommunicateOp
* PipelineSendOp
* PipelineReceiveOp
* Dispatch
