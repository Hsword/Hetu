/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_runtime_api.h
 * \brief DL runtime library.
 *
 */

#ifndef HETUSYS_RUNTIME_C_RUNTIME_API_H_
#define HETUSYS_RUNTIME_C_RUNTIME_API_H_

#ifdef __cplusplus
#define HETUSYS_EXTERN_C extern "C"
#else
#define HETUSYS_EXTERN_C
#endif

#include "dlarray.h"
#include <stddef.h>
#include <stdint.h>

HETUSYS_EXTERN_C {
    // type of array index.
    typedef int64_t index_t;

    // the array handle
    typedef DLArray *DLArrayHandle;
    typedef DLStream *DLStreamHandle;
    int DLStreamCreate(size_t dev_id, DLStreamHandle * handle);
    int DLStreamDestroy(DLStreamHandle handle);
    int DLStreamSync(DLStreamHandle handle);

    typedef DLEvent *DLEventHandle;
    int DLEventCreate(size_t dev_id, DLEventHandle * handle);
    int DLEventDestroy(DLEventHandle handle);
    int DLEventRecord(DLStreamHandle stream_andle, DLEventHandle event_handle);
    int DLEventSync(DLEventHandle handle);
    int DLEventElapsedTime(DLEventHandle start, DLEventHandle ending,
                           float *duration);

    // gpu chunk
    void clear_chunk();

    int DLGpuAbs(const DLArrayHandle input, DLArrayHandle output,
                 DLStreamHandle stream_handle);
    int DLGpuAbsGradient(const DLArrayHandle grad, const DLArrayHandle input,
                         DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuAddmm(const DLArrayHandle input, const DLArrayHandle matA,
                   const DLArrayHandle matB, float alpha, float beta,
                   DLArrayHandle matC, DLStreamHandle stream_handle);
    int DLGpuAddmmGradient(const DLArrayHandle input, DLArrayHandle output,
                           int axis, float beta, DLStreamHandle stream_handle);
    int DLGpuArange(float start, float end, float step, DLArrayHandle output,
                    DLStreamHandle stream_handle);

    int DLGpuArgsort(const DLArrayHandle input, DLArrayHandle output,
                     DLArrayHandle index, DLArrayHandle output_index, int dim,
                     bool descending, DLStreamHandle stream_handle);
    // Array related apis for quick proptying
    /*!
     * \brief Allocate a nd-array's memory,
     *  including space of shape, of given spec.
     *
     * \param shape The shape of the array, the data content will be copied to
     * out \param ndim The number of dimension of the array. \param ctx The ctx
     * this array sits on. \param out The output handle. \return 0 when success,
     * -1 when failure happens
     */
    int DLArrayAlloc(const index_t *shape, const index_t *stride, index_t ndim,
                     DLContext ctx, DLArrayHandle *out, int nbits);

    /*!
     * \brief Free the DL Array.
     * \param handle The array handle to be freed.
     * \return 0 when success, -1 when failure happens
     */
    int DLArrayFree(DLArrayHandle handle);

    /*!
     * \brief Copy the array, both from and to must be valid during the copy.
     * \param from The array to be copied from.
     * \param to The target space.
     * \param stream The stream where the copy happens, can be NULL.
     * \return 0 when success, -1 when failure happens
     */
    int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                          DLStreamHandle stream);
    int DLArrayCopyFromToOffset(DLArrayHandle from, size_t foffset,
                                DLArrayHandle to, size_t toffset,
                                size_t copy_size, DLStreamHandle stream);

    /*!
     * \brief Set all array elements to given value.
     * \param arr The array to be Set.
     * \param value The target value.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuArraySet(DLArrayHandle arr, float value,
                      DLStreamHandle stream_handle);

    int DLGpuBaddbmm(const DLArrayHandle input, const DLArrayHandle matA,
                     const DLArrayHandle matB, float alpha, float beta,
                     DLArrayHandle matC, DLStreamHandle stream_handle);

    int DLGpuBool(const DLArrayHandle input, DLArrayHandle output,
                  DLStreamHandle stream_handle);
    int DLGpuBoolVal(const DLArrayHandle input, float val, DLArrayHandle output,
                     int cond, DLStreamHandle stream_handle);
    int DLGpuBoolMatrix(const DLArrayHandle input_A,
                        const DLArrayHandle input_B, DLArrayHandle output,
                        int cond, DLStreamHandle stream_handle);
    /*!
     * \brief Broadcast input array to output array.
     * \param input The input array.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output,
                         DLStreamHandle stream_handle);

    int DLGpuClamp(const DLArrayHandle input, float min, float max,
                   DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuClampMin(const DLArrayHandle input, float min,
                      DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuClampMax(const DLArrayHandle input, float max,
                      DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuClampMat(const DLArrayHandle input, const DLArrayHandle min_mat,
                      const DLArrayHandle max_mat, DLArrayHandle output,
                      DLStreamHandle stream_handle);
    int DLGpuConstPow(const DLArrayHandle input, float val,
                      DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuConstPowGradient(
        const DLArrayHandle input, const DLArrayHandle grad, float val,
        DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuCos(const DLArrayHandle input, DLArrayHandle output,
                 DLStreamHandle stream_handle);
    /*!
     * \brief Reduce sum input array by axis=0 and store to output.
     * \param input The input array.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output,
                               DLStreamHandle stream_handle);
    int _DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output,
                                DLArrayHandle arr_workspace,
                                DLStreamHandle stream_handle);
    int DLGpuExp(const DLArrayHandle input, DLArrayHandle output,
                 DLStreamHandle stream_handle);
    int DLGpuFloor(const DLArrayHandle input, DLArrayHandle output,
                   DLStreamHandle stream_handle);
    int DLGpuGather(const DLArrayHandle input, const DLArrayHandle index,
                    DLArrayHandle output, int dim,
                    DLStreamHandle stream_handle);
    int DLGpuGatherGradient(const DLArrayHandle input,
                            const DLArrayHandle index, DLArrayHandle output,
                            int dim, DLStreamHandle stream_handle);
    int DLGpuMaskedFill(const DLArrayHandle input, const DLArrayHandle mask,
                        float val, DLArrayHandle output,
                        DLStreamHandle stream_handle);
    /*!
     * \brief Elementwise add two matrices and store to output.
     * \param matA The left input array.
     * \param matB The right input array.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuMatrixElementwiseAdd(
        const DLArrayHandle matA, const DLArrayHandle matB,
        DLArrayHandle output, bool lazy_input, DLStreamHandle stream_handle);
    int DLGpuMatrixElementwiseAddSimple(
        const DLArrayHandle matA, const DLArrayHandle matB,
        DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuMatrixElementwiseAddLazy(
        const DLArrayHandle matA, const DLArrayHandle matB,
        DLArrayHandle output, const DLArrayHandle gpu_buf,
        DLStreamHandle stream_handle);

    /*!
     * \brief Add matrix by const and store to output.
     * \param input The input array.
     * \param val The constant.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                         DLArrayHandle output,
                                         DLStreamHandle stream_handle);

    /*!
     * \brief Elementwise multiply two matrices and store to output.
     * \param matA The left input array.
     * \param matB The right input array.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuMatrixElementwiseMultiply(
        const DLArrayHandle matA, const DLArrayHandle matB,
        DLArrayHandle output, DLStreamHandle stream_handle);

    /*!
     * \brief Multiply matrix by const and store to output.
     * \param input The input array.
     * \param val The constant.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                                   DLArrayHandle output,
                                   DLStreamHandle stream_handle);
    int DLGpuMatrixMultiplyByConstInt(const DLArrayHandle input, int val,
                                      DLArrayHandle output,
                                      DLStreamHandle stream_handle);

    /*!
     * \brief Elementwise divide two matrices and store to output.
     * \param matA The dividend array.
     * \param matB The divisor array.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuMatrixElementwiseDivide(
        const DLArrayHandle matA, const DLArrayHandle matB,
        DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuMatrixElementwiseDivideHandleZero(
        const DLArrayHandle matA, const DLArrayHandle matB,
        DLArrayHandle output, DLStreamHandle stream_handle);

    /*!
     * \brief Divide const by matrix and store to output.
     * \param input The input array.
     * \param val The constant.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuMatrixDivConst(float val, const DLArrayHandle input,
                            DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuMin(const DLArrayHandle input, DLArrayHandle output_val, int dim,
                 DLStreamHandle stream_handle);
    int DLGpuMinMat(const DLArrayHandle matA, const DLArrayHandle matB,
                    DLArrayHandle output, DLStreamHandle stream_handle);
    /*!
     * \brief Compute opposite number on all array elements, and store to
     * output. \param input The input array. \param output The output value.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuOpposite(const DLArrayHandle input, DLArrayHandle output,
                      DLStreamHandle stream_handle);

    int DLGpuIsPositive(const DLArrayHandle input, DLArrayHandle output,
                        DLStreamHandle stream_handle);
    int DLGpuBinaryStepBackward(const DLArrayHandle input, DLArrayHandle output,
                                DLStreamHandle stream_handle);
    int DLGpuOuter(const DLArrayHandle input, const DLArrayHandle vector,
                   DLArrayHandle output, DLStreamHandle stream_handle);

    /*!
     * \brief Matrix multiply two matrices and store to output.
     * \param matA The left input array.
     * \param transposeA Whether matA needs to be transposed
     * \param matB The right input array.
     * \param transposeB Whether matB needs to be transposed
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                            const DLArrayHandle matB, bool transposeB,
                            DLArrayHandle matC, DLStreamHandle stream_handle);

    int DLGpuPow(const DLArrayHandle input, DLArrayHandle output, float exp,
                 DLStreamHandle stream_handle);
    int DLGpuPowGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                         DLArrayHandle output, float exp,
                         DLStreamHandle stream_handle);

    /*!
     * \brief Compute sqrt on all array elements, and store to output.
     * \param input The input array.
     * \param output The output value.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuSqrt(const DLArrayHandle input, DLArrayHandle output,
                  DLStreamHandle stream_handle);

    /*!
     * \brief Compute reciprocal sqrt on all array elements, and store to
     * output. \param input The input array. \param output The output value.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuReciprocalSqrt(const DLArrayHandle input, DLArrayHandle output,
                            DLStreamHandle stream_handle);

    /*!
     * \brief Compute relu on all array elements, and store to output.
     * \param input The input array.
     * \param output The output value.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output,
                  DLStreamHandle stream_handle);

    /*!
     * \brief Compute relu gradient, and store to output.
     * \param input The input array.
     * \param in_grad The input gradients value.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuReluGradient(const DLArrayHandle input,
                          const DLArrayHandle in_grad, DLArrayHandle output,
                          DLStreamHandle stream_handle);

    /*!
     * \brief Compute leaky relu on all array elements, and store to output.
     * \param input The input array.
     * \param alpha The val to multiple when x < 0
     * \param output The output value.
     * \return 0 when success, -1 when failure happens
     */

    int DLGpuGelu(const DLArrayHandle input, DLArrayHandle output,
                  DLStreamHandle stream_handle);
    /*!
     * \brief Compute gelu, and store to output.
     * \param input The input array.
     * \param in_grad The input gradients value.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */

    int DLGpuGeluGradient(const DLArrayHandle input,
                          const DLArrayHandle in_grad, DLArrayHandle output,
                          DLStreamHandle stream_handle);
    /*!
     * \brief Compute gelu gradient, and store to output.
     * \param input The input array.
     * \param in_grad The input gradients value.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuLeakyRelu(const DLArrayHandle input, const float alpha,
                       DLArrayHandle output, DLStreamHandle stream_handle);

    /*!
     * \brief Compute leaky relu gradient, and store to output.
     * \param input The input array.
     * \param in_grad The input gradients value.
     * \param alpha The val to multiple when x < 0
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuLeakyReluGradient(
        const DLArrayHandle input, const DLArrayHandle in_grad,
        const float alpha, DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuSin(const DLArrayHandle input, DLArrayHandle output,
                 DLStreamHandle stream_handle);
    /*!
     * \brief Compute tanh on matrix, and store to output.
     * \param input The input array.
     * \param output The output value.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuTanh(const DLArrayHandle input, DLArrayHandle output,
                  DLStreamHandle stream_handle);

    /*!
     * \brief Compute the gradient of tanh on matrix, and store to output.
     * \param forward The forward matrix.
     * \param grad The gradient matrix.
     * \param output The output value.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuTanhGradient(const DLArrayHandle forward, const DLArrayHandle grad,
                          DLArrayHandle output, DLStreamHandle stream_handle);

    /*!
     * \brief Compute sigmoid, and store to output.
     * \param input The input array.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */
    int DLGpuSigmoid(const DLArrayHandle input, DLArrayHandle output,
                     DLStreamHandle stream_handle);

    /*!
     * \brief Compute softmax, and store to output.
     * \param input The input array.
     * \param output The output array.
     * \return 0 when success, -1 when failure happens
     */

    int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output,
                     DLStreamHandle stream_handle);

    /*!
     * \brief Compute softmax_cross_entropy.
     *  np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
     * \param input_a The y array.
     * \param input_b The y_ array.
     * \param output The output value.
     * \return 0 when success, -1 when failure happens
     */

    int DLGpuSoftmaxCrossEntropySparse(
        const DLArrayHandle input_a, const DLArrayHandle input_b,
        const int ignored_index, DLArrayHandle output,
        DLStreamHandle stream_handle);
    int DLGpuSoftmaxCrossEntropySparse_Gradient(
        const DLArrayHandle input_a, const DLArrayHandle input_b,
        const DLArrayHandle input_c, const int ignored_index,
        DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuSoftmaxCrossEntropy(
        const DLArrayHandle input_a, const DLArrayHandle input_b,
        DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuSoftmaxCrossEntropy_Gradient(
        const DLArrayHandle input_a, const DLArrayHandle input_b,
        const DLArrayHandle input_c, DLArrayHandle output,
        DLStreamHandle stream_handle);

    int DLGpuEmbeddingLookUp(const DLArrayHandle input, const DLArrayHandle ids,
                             DLArrayHandle output,
                             DLStreamHandle stream_handle);

    int DLGpuCSREmbeddingLookUp(
        const DLArrayHandle value, const DLArrayHandle row,
        const DLArrayHandle col, const DLArrayHandle ids,
        DLArrayHandle input_grad, DLStreamHandle stream_handle);

    int DLGpuCOOEmbeddingLookUp(
        const DLArrayHandle value, const DLArrayHandle row,
        const DLArrayHandle col, const DLArrayHandle ids,
        DLArrayHandle input_grad, DLStreamHandle stream_handle);

    int DLGpuMinDist(const DLArrayHandle lookup, const DLArrayHandle key,
                     DLArrayHandle codebook, DLArrayHandle indices,
                     DLArrayHandle output, bool mode,
                     DLStreamHandle stream_handle);

    int DLGpuCrossEntropy(const DLArrayHandle input_y,
                          const DLArrayHandle label, DLArrayHandle output,
                          DLStreamHandle stream_handle);

    int DLGpuCrossEntropyGradient(
        const DLArrayHandle grad, const DLArrayHandle input_y,
        const DLArrayHandle label, DLArrayHandle output,
        DLStreamHandle stream_handle);

    int DLGpuCrossEntropySparse(const DLArrayHandle input_y,
                                const DLArrayHandle label,
                                const int ignored_index, DLArrayHandle output,
                                DLStreamHandle stream_handle);

    int DLGpuCrossEntropySparseGradient(
        const DLArrayHandle grad, const DLArrayHandle input_y,
        const DLArrayHandle label, const int ignored_index,
        DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuConv2d(const DLArrayHandle input_x, const DLArrayHandle input_f,
                    DLArrayHandle output, DLArrayHandle workspace_arr,
                    const int padding, const int stride,
                    DLStreamHandle stream_handle);

    int DLGpuConv2d_Gradient_of_Filter(
        const DLArrayHandle input_x, const DLArrayHandle gradient_y,
        DLArrayHandle gradient_f, DLArrayHandle workspace_im2col,
        DLArrayHandle workspace_batch_filter, const int padding,
        const int stride, DLStreamHandle stream_handle);

    int DLGpuConv2d_Gradient_of_Data(
        const DLArrayHandle input_f, const DLArrayHandle gradient_y,
        DLArrayHandle gradient_x, DLArrayHandle workspace_im2col,
        const int padding, const int stride, DLStreamHandle stream_handle);

    int DLGpuAvgerage_Pooling2d(
        const DLArrayHandle input, const size_t kernel_H, const size_t kernel_W,
        DLArrayHandle output, const size_t padding, const size_t stride,
        DLStreamHandle stream_handle);

    int DLGpuAvgerage_Pooling2d_gradient(
        const DLArrayHandle gradient_Y, const size_t kernel_H,
        const size_t kernel_W, DLArrayHandle gradient_X, const size_t padding,
        const size_t stride, DLStreamHandle stream_handle);

    int DLGpuMax_Pooling2d(const DLArrayHandle input, const int kernel_H,
                           const int kernel_W, DLArrayHandle output,
                           const int padding, const int stride,
                           DLStreamHandle stream_handle);

    int DLGpuMax_Pooling2d_gradient(
        const DLArrayHandle input, const DLArrayHandle output_grad,
        const int kernel_H, const int kernel_W, DLArrayHandle input_grad,
        const int padding, const int stride, DLStreamHandle stream_handle);

    int DLGpuReshape(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                     DLStreamHandle stream_handle);

    int DLGpuConv2d_broadcast_to(const DLArrayHandle input_x,
                                 DLArrayHandle output_y,
                                 DLStreamHandle stream_handle);

    int DLGpuConv2d_reduce_sum(const DLArrayHandle input_x,
                               DLArrayHandle output_y,
                               DLStreamHandle stream_handle);

    int DLGpuTopKIdx(const DLArrayHandle input, DLArrayHandle output_idx, int k,
                     DLStreamHandle stream_handle);

    int DLGpuTopKVal(const DLArrayHandle input, DLArrayHandle output_idx,
                     DLArrayHandle output_val, int k,
                     DLStreamHandle stream_handle);

    int DLGpuScatter(const DLArrayHandle target, int dim, DLArrayHandle index,
                     DLArrayHandle src, DLStreamHandle stream_handle);

    int DLGpuMinusByConst(const DLArrayHandle input, DLArrayHandle output,
                          float val, DLStreamHandle stream_handle);
    int DLGpuMinusElewise(const DLArrayHandle input1, DLArrayHandle input2,
                          DLArrayHandle output, DLStreamHandle stream_handle);
    int DLGpuNorm(const DLArrayHandle in_arr, DLArrayHandle out_arr, int axis,
                  int p, DLStreamHandle stream_handle);
    int DLGpuNormGradient(const DLArrayHandle in_arr,
                          const DLArrayHandle in_arr_y,
                          const DLArrayHandle grad_y, DLArrayHandle out_arr,
                          int axis, int p, DLStreamHandle stream_handle);
    int DLGpuClone(const DLArrayHandle input, DLArrayHandle output,
                   DLStreamHandle stream_handle);

    int DLGpuMax(const DLArrayHandle input, DLArrayHandle output_val, int dim,
                 DLStreamHandle stream_handle);
    int DLGpuMaxMat(const DLArrayHandle matA, const DLArrayHandle matB,
                    DLArrayHandle output, DLStreamHandle stream_handle);

    int CuDNN_DLGpuConv2d(
        const DLArrayHandle input_x, const DLArrayHandle input_f,
        DLArrayHandle output, const int padding_h, const int padding_w,
        const int stride_h, const int stride_w, DLStreamHandle stream_handle);

    int CuDNN_DLGpuConv2d_Gradient_of_Filter(
        const DLArrayHandle input_x, const DLArrayHandle gradient_y,
        DLArrayHandle gradient_f, const int padding_h, const int padding_w,
        const int stride_h, const int stride_w, DLStreamHandle stream_handle);

    int CuDNN_DLGpuConv2d_Gradient_of_Data(
        const DLArrayHandle input_f, const DLArrayHandle gradient_y,
        DLArrayHandle gradient_x, const int padding_h, const int padding_w,
        const int stride_h, const int stride_w, DLStreamHandle stream_handle);

    int CuDNN_DLGpuAvgerage_Pooling2d(
        const DLArrayHandle input, const size_t kernel_H, const size_t kernel_W,
        DLArrayHandle output, const size_t padding, const size_t stride,
        DLStreamHandle stream_handle);

    int CuDNN_DLGpuAvgerage_Pooling2d_gradient(
        const DLArrayHandle output_Y, const DLArrayHandle gradient_Y,
        const DLArrayHandle input_X, const size_t kernel_H,
        const size_t kernel_W, DLArrayHandle gradient_X, const size_t padding,
        const size_t stride, DLStreamHandle stream_handle);

    int CuDNN_DLGpuDropout(const DLArrayHandle input_X, const float dropout,
                           DLArrayHandle output_Y, int *reserve_size,
                           void **reserve_space, int first_time,
                           DLStreamHandle stream_handle);

    int CuDNN_DLGpuDropout_gradient(const DLArrayHandle output_Y,
                                    const float dropout, DLArrayHandle input_X,
                                    int *reserve_size, void **reserve_space,
                                    DLStreamHandle stream_handle);

    int CuDNN_DLGpuMax_Pooling2d(
        const DLArrayHandle input, const size_t kernel_H, const size_t kernel_W,
        DLArrayHandle output, const size_t padding, const size_t stride,
        DLStreamHandle stream_handle);

    int CuDNN_DLGpuMax_Pooling2d_gradient(
        const DLArrayHandle output_Y, const DLArrayHandle gradient_Y,
        const DLArrayHandle input_X, const size_t kernel_H,
        const size_t kernel_W, DLArrayHandle gradient_X, const size_t padding,
        const size_t stride, DLStreamHandle stream_handle);

    int CuDNN_DLGpuBatch_Normalization(
        const DLArrayHandle input_X, const DLArrayHandle bn_scale,
        const DLArrayHandle bn_bias, DLArrayHandle output_Y, double momentum,
        double eps, DLArrayHandle running_mean, DLArrayHandle running_var,
        DLArrayHandle save_mean, DLArrayHandle save_var,
        DLStreamHandle stream_handle);

    int CuDNN_DLGpuBatch_Normalization_gradient(
        const DLArrayHandle gradient_Y, const DLArrayHandle input_X,
        const DLArrayHandle bn_scale, DLArrayHandle gradient_X,
        DLArrayHandle gradient_bn_scale, DLArrayHandle gradient_bn_bias,
        double eps, DLArrayHandle save_mean, DLArrayHandle save_var,
        DLStreamHandle stream_handle);

    int CuDNN_DLGpuBatch_Normalization_inference(
        const DLArrayHandle input_X, const DLArrayHandle bn_scale,
        const DLArrayHandle bn_bias, DLArrayHandle output_Y, double eps,
        DLArrayHandle estimated_mean, DLArrayHandle estimated_var,
        DLStreamHandle stream_handle);

    int DLGpuPad(const DLArrayHandle input_X, DLArrayHandle output_Y,
                 int *paddings, int pad_len, size_t mode, float constant_values,
                 DLStreamHandle stream_handle);

    int DLGpuPad_gradient(
        const DLArrayHandle output_gradient_Y, DLArrayHandle input_gradient_X,
        int *paddings, int pad_len, size_t mode, DLStreamHandle stream_handle);

    int DLGpuConcat(const DLArrayHandle input_x, const DLArrayHandle input_y,
                    DLArrayHandle output, int axis,
                    DLStreamHandle stream_handle);

    int DLGpuConcat_gradient(const DLArrayHandle output_gradient,
                             DLArrayHandle input_gradient, int axis, int id,
                             DLStreamHandle stream_handle);

    int DLGpuConcatenate(const DLArrayHandle input, DLArrayHandle output,
                         int axis, int offset, DLStreamHandle stream_handle);

    int DLGpuConcatenate_gradient(const DLArrayHandle o_grad,
                                  DLArrayHandle i_grad, int axis, int offset,
                                  DLStreamHandle stream_handle);

    int DLGpuBicubicInterpolate(const DLArrayHandle input, DLArrayHandle output,
                                bool align_corners,
                                DLStreamHandle stream_handle);
    int DLGpuBicubicInterpolateGradient(
        DLArrayHandle input, const DLArrayHandle output, bool align_corners,
        DLStreamHandle stream_handle);

    int DLGpuRoll(const DLArrayHandle input, int *shift, int *axis, int nums,
                  DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuTranspose(const DLArrayHandle input, DLArrayHandle output,
                       int *perm, DLStreamHandle stream_handle);

    int DLGpuTransposeSimple(const DLArrayHandle input, DLArrayHandle output,
                             const DLArrayHandle gpu_buffer,
                             DLStreamHandle stream_handle);

    int CuSparse_DLGpuCsrmv(
        const DLArrayHandle data_handle, const DLArrayHandle row_handle,
        const DLArrayHandle col_handle, int nrow, int ncol, bool transpose,
        const DLArrayHandle input_handle, DLArrayHandle output_handle,
        DLStreamHandle stream_handle);

    int CuSparse_DLGpuCsrmm(
        const DLArrayHandle data_handle, const DLArrayHandle row_handle,
        const DLArrayHandle col_handle, int nrow, int ncol, bool transposeA,
        const DLArrayHandle matB, bool transposeB, DLArrayHandle matC,
        int start_pos, int end_pos, DLStreamHandle stream_handle);

    int DLGpuSlice(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                   int64_t *begin_pos, DLStreamHandle stream_handle);

    int DLGpuSliceSimple(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                         const DLArrayHandle gpu_buf,
                         DLStreamHandle stream_handle);

    int DLGpuSliceGradient(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                           int64_t *begin_pos, DLStreamHandle stream_handle);

    int DLGpuSliceGradientSimple(
        const DLArrayHandle in_arr, DLArrayHandle out_arr,
        const DLArrayHandle gpu_buf, DLStreamHandle stream_handle);

    int DLGpuWhere(const DLArrayHandle cond, const DLArrayHandle arr1,
                   const DLArrayHandle arr2, DLArrayHandle output,
                   DLStreamHandle stream_handle);
    int DLGpuWhereConst(const DLArrayHandle cond, const DLArrayHandle arr1,
                        float const_attr, DLArrayHandle output,
                        DLStreamHandle stream_handle);

    int DLGpuBatchMatrixMultiply(
        const DLArrayHandle matA, bool transposeA, const DLArrayHandle matB,
        bool transposeB, DLArrayHandle matC, DLStreamHandle stream_handle);

    int DLGpuLayerNormalization(
        const DLArrayHandle in_arr, const DLArrayHandle ln_scale,
        const DLArrayHandle ln_bias, DLArrayHandle mean, DLArrayHandle var,
        DLArrayHandle out_arr, float eps, DLStreamHandle stream_handle);

    int DLGpuLayerNormalizationGradient(
        const DLArrayHandle out_grads, const DLArrayHandle in_arr,
        const DLArrayHandle ln_scale, DLArrayHandle grad_arr,
        DLArrayHandle grad_scale, DLArrayHandle grad_bias,
        const DLArrayHandle mean_arr, const DLArrayHandle var_arr, float eps,
        DLStreamHandle stream_handle);

    int DLGpuInstanceNormalization2d(
        const DLArrayHandle in_arr, DLArrayHandle mean, DLArrayHandle var,
        DLArrayHandle out_arr, float eps, DLStreamHandle stream_handle);

    int DLGpuInstanceNormalization2dGradient(
        const DLArrayHandle out_grads, const DLArrayHandle in_arr,
        DLArrayHandle grad_arr, const DLArrayHandle mean_arr,
        const DLArrayHandle var_arr, float eps, DLStreamHandle stream_handle);

    int DLGpuBroadcastShape(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                            int *add_axes, DLStreamHandle stream_handle);

    int DLGpuBroadcastShapeSimple(
        const DLArrayHandle in_arr, DLArrayHandle out_arr,
        const DLArrayHandle out_strides, const DLArrayHandle in_dims,
        DLStreamHandle stream_handle);

    int DLGpuPower(const DLArrayHandle in_arr, DLArrayHandle out_arr, float p,
                   DLStreamHandle stream_handle);

    int DLGpuReduceSum(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                       int *axes, int num_ax, DLStreamHandle stream_handle);

    int DLGpuReduceMean(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                        int *axes, int num_ax, DLStreamHandle stream_handle);

    int DLGpuReduceMin(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                       int *axes, int num_ax, DLStreamHandle stream_handle);

    int DLGpuReduceMul(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                       int *axes, int num_ax, DLStreamHandle stream_handle);

    int DLGpuReduceNorm1(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                         int *axes, int num_ax, DLStreamHandle stream_handle);
    int DLGpuReduceNorm2(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                         int *axes, int num_ax, DLStreamHandle stream_handle);
    int DLGpuReduceNorm2Raw(const DLArrayHandle in_arr, DLArrayHandle out_ptr,
                            int *axes, int num_ax, int offset,
                            DLStreamHandle stream_handle);

    int DLGpuAdd_(DLArrayHandle arr, const DLArrayHandle adder,
                  const DLArrayHandle alpha, float cons,
                  DLStreamHandle stream_handle);

    int DLGpuDivMul(DLArrayHandle arr, const DLArrayHandle alpha, float cons,
                    DLStreamHandle stream_handle);

    int DLGpuArrayLazyCallback(const DLArrayHandle from, DLArrayHandle to,
                               DLStreamHandle stream_handle);

    int IndexedSlicesOneSideAdd(
        const DLArrayHandle indices, const DLArrayHandle values,
        DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuReduceIndexedSlice(
        const DLArrayHandle in_indices, const DLArrayHandle in_values,
        DLArrayHandle out_indices, DLArrayHandle out_values,
        DLArrayHandle workspace, size_t storage_size, int end_bit,
        DLStreamHandle stream_handle);

    int DLGpuReduceIndexedSliceGetWorkspaceSize(size_t ind_size, size_t * size);

    int DLGpuReduceIndexedSliceWithEmbedding(
        const DLArrayHandle in_indices, const DLArrayHandle in_values,
        const DLArrayHandle parameters, DLArrayHandle out_indices,
        DLArrayHandle out_values, DLArrayHandle out_params,
        DLArrayHandle workspace, size_t storage_size, int end_bit,
        DLStreamHandle stream_handle);

    int DLGpuSGDUpdateIndexedSlices(
        const DLArrayHandle indices, const DLArrayHandle grads,
        const DLArrayHandle params, DLArrayHandle output, float lr,
        DLStreamHandle stream_handle);

    int DLGpuAdaGradUpdateIndexedSlices(
        const DLArrayHandle indices, const DLArrayHandle grads,
        const DLArrayHandle params, DLArrayHandle output, float lr,
        DLArrayHandle accum, float epsilon, DLStreamHandle stream_handle);

    int DLGpuAdamUpdateIndexedSlices(
        const DLArrayHandle indices, const DLArrayHandle grads,
        const DLArrayHandle params, DLArrayHandle output, float lr,
        DLArrayHandle m, DLArrayHandle v, DLArrayHandle maxv, float beta1,
        float beta2, DLArrayHandle betats, float epsilon,
        DLStreamHandle stream_handle);

    int DLGpuDropout(const DLArrayHandle input, const float dropout,
                     DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuDropoutGradient_recompute(
        const DLArrayHandle grad, const float dropout, DLArrayHandle output,
        unsigned long long seed_seqnum, DLStreamHandle stream_handle);

    int DLGpuDropoutGradient(const DLArrayHandle grad,
                             const DLArrayHandle fw_output, const float dropout,
                             DLArrayHandle output,
                             DLStreamHandle stream_handle);

    int CuDNN_DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output,
                           DLStreamHandle stream_handle);

    int CuDNN_DLGpuSoftmaxGradient(const DLArrayHandle y_arr,
                                   const DLArrayHandle dy, DLArrayHandle dx,
                                   DLStreamHandle stream_handle);

    int CuDNN_DLGpuLogSoftmax(const DLArrayHandle input, DLArrayHandle output,
                              DLStreamHandle stream_handle);
    int CuDNN_DLGpuLogSoftmaxGradient(const DLArrayHandle y_arr,
                                      const DLArrayHandle dy, DLArrayHandle dx,
                                      DLStreamHandle stream_handle);

    int CuDNN_DLGpuSoftmaxEntropy(
        const DLArrayHandle input_y, const DLArrayHandle label,
        DLArrayHandle output, DLStreamHandle stream_handle);

    int CuDNN_DLGpuSoftmaxEntropyGradient(
        const DLArrayHandle grad, const DLArrayHandle input_y,
        const DLArrayHandle label, DLArrayHandle output,
        DLStreamHandle stream_handle);

    int DLGpuOneHot(const DLArrayHandle input, DLArrayHandle output,
                    DLStreamHandle stream_handle);

    int DLGpuLinear(const DLArrayHandle matA, bool transposeA,
                    const DLArrayHandle matB, bool transposeB,
                    const DLArrayHandle bias, DLArrayHandle matC,
                    DLStreamHandle stream_handle);

    int Cudnn_Conv2dAddBias(
        const DLArrayHandle input_x, const DLArrayHandle input_f,
        const DLArrayHandle bias, DLArrayHandle output, const int padding_h,
        const int padding_w, const int stride_h, const int stride_w,
        DLStreamHandle stream_handle);

    int DLGpuTrilLookup(const DLArrayHandle input, DLArrayHandle output,
                        int offset, DLStreamHandle stream_handle);
    int DLGpuTrilLookupGradient(const DLArrayHandle input, DLArrayHandle output,
                                int offset, DLStreamHandle stream_handle);

    int DLGpuSign(const DLArrayHandle input, DLArrayHandle output,
                  DLStreamHandle stream_handle);

    int DLGpuClipping(DLArrayHandle arr, float min_value, float max_value,
                      DLStreamHandle stream_handle);

    // Initializers
    int DLGpuNormalInit(DLArrayHandle arr, const float mean, const float stddev,
                        DLStreamHandle stream_handle);

    int DLGpuUniformInit(DLArrayHandle arr, const float lb, const float ub,
                         DLStreamHandle stream_handle);

    int DLGpuTruncatedNormalInit(DLArrayHandle arr, const float mean,
                                 const float stddev,
                                 DLStreamHandle stream_handle);

    int DLGpuReversedTruncatedNormalInit(DLArrayHandle arr, const float mean,
                                         const float stddev,
                                         DLStreamHandle stream_handle);

    int DLGpuGumbelInit(DLArrayHandle arr, DLStreamHandle stream_handle);

    int DLGpuRandomInt(DLArrayHandle arr, const int lb, const int ub,
                       DLStreamHandle stream_handle);

    // Optimizer Ops
    int DLGpuSparseSet(DLArrayHandle table, const DLArrayHandle indices,
                       const DLArrayHandle data, DLStreamHandle stream_handle);
    int AddL2Regularization(const DLArrayHandle param, DLArrayHandle grad,
                            float l2reg, DLStreamHandle stream_handle);
    int AddL2RegularizationSparse(
        const DLArrayHandle param, const DLArrayHandle grad_indices,
        DLArrayHandle grad_values, float l2reg, DLStreamHandle stream_handle);
    int SGDOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                           float lr, DLStreamHandle stream_handle);
    int SGDOptimizerSparseUpdate(DLArrayHandle param,
                                 const DLArrayHandle grad_indices,
                                 const DLArrayHandle grad_values, float lr,
                                 DLStreamHandle stream_handle);

    int MomentumOptimizerUpdate(
        DLArrayHandle param, const DLArrayHandle grad, DLArrayHandle velocity,
        float lr, float momentum, bool nesterov, DLStreamHandle stream_handle);

    int MomentumOptimizerSparseUpdate(
        DLArrayHandle param, const DLArrayHandle grad_indices,
        const DLArrayHandle grad_values, DLArrayHandle velocity, float lr,
        float momentum, bool nesterov, DLStreamHandle stream_handle);

    int AdaGradOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                               DLArrayHandle acc, float lr, float eps,
                               DLStreamHandle stream_handle);

    int AdaGradOptimizerSparseUpdate(
        DLArrayHandle param, const DLArrayHandle grad_indices,
        const DLArrayHandle grad_values, DLArrayHandle acc, float lr, float eps,
        DLStreamHandle stream_handle);

    int BetatsUpdate(DLArrayHandle betats, float beta1, float beta2,
                     DLStreamHandle stream_handle);

    int AdamOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                            DLArrayHandle expavg, DLArrayHandle expavgsq,
                            DLArrayHandle maxv, float lr, float beta1,
                            float beta2, DLArrayHandle beta_ts, float eps,
                            DLStreamHandle stream_handle);

    int AdamOptimizerSparseUpdate(
        DLArrayHandle param, const DLArrayHandle grad_indices,
        const DLArrayHandle grad_values, DLArrayHandle expavg,
        DLArrayHandle expavgsq, DLArrayHandle maxv, float lr, float beta1,
        float beta2, DLArrayHandle beta_ts, float eps,
        DLStreamHandle stream_handle);

    int AdamWOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                             DLArrayHandle expavg, DLArrayHandle expavgsq,
                             float lr, float beta1, float beta2,
                             DLArrayHandle beta_ts, float eps,
                             float weight_decay, DLStreamHandle stream_handle);
    int AdamWOptimizerSparseUpdate(
        DLArrayHandle param, const DLArrayHandle grad_indices,
        const DLArrayHandle grad_values, DLArrayHandle expavg,
        DLArrayHandle expavgsq, float lr, float beta1, float beta2,
        DLArrayHandle beta_ts, float eps, float weight_decay,
        DLStreamHandle stream_handle);

    int LambOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                            DLArrayHandle expavg, DLArrayHandle expavgsq,
                            float lr, float beta1, float beta2,
                            DLArrayHandle beta_ts, float eps,
                            float weight_decay, DLStreamHandle stream_handle);
    int LambOptimizerSparseUpdate(
        DLArrayHandle param, const DLArrayHandle grad_indices,
        const DLArrayHandle grad_values, DLArrayHandle expavg,
        DLArrayHandle expavgsq, float lr, float beta1, float beta2,
        DLArrayHandle beta_ts, float eps, float weight_decay,
        DLStreamHandle stream_handle);

    int DeduplicateIndexedSlices(
        const DLArrayHandle origin, const DLArrayHandle inverse,
        DLArrayHandle compressed, DLStreamHandle stream_handle);

    int IndexedSlices2Dense(
        const DLArrayHandle values, const DLArrayHandle indices,
        DLArrayHandle new_values, DLStreamHandle stream_handle);

    int DLGpuAssignWithIndexedSlices(
        DLArrayHandle embedding, const DLArrayHandle indices,
        const DLArrayHandle values, DLStreamHandle stream_handle);

    int DLGpuAssignQuantizedEmbeddingUnified(
        DLArrayHandle embedding, const DLArrayHandle indices,
        const DLArrayHandle values, float scale, float minele, int digit,
        bool stochastic, DLStreamHandle stream_handle);

    int DLGpuAssignQuantizedEmbedding(
        DLArrayHandle embedding, const DLArrayHandle indices,
        const DLArrayHandle values, const DLArrayHandle qparam, int digit,
        bool stochastic, DLStreamHandle stream_handle);

    int DLGPUReorderIntoLookup(
        const DLArrayHandle idoffsets, const DLArrayHandle dedupemb,
        DLArrayHandle lookup, DLStreamHandle stream_handle);

    int DLGPUAssignALPTEmbedding(
        DLArrayHandle embedding, const DLArrayHandle indices,
        const DLArrayHandle values, const DLArrayHandle scale, float middle,
        int digit, bool stochastic, DLStreamHandle stream_handle);

    int DLGpuUniqueIndices(const DLArrayHandle indices, DLArrayHandle output,
                           DLArrayHandle idoffsets, DLArrayHandle workspace,
                           size_t storage_size, int end_bit,
                           DLStreamHandle stream_handle);

    int DLGpuGetUniqueWorkspaceSize(size_t ind_size, size_t * size);

    int DLGpuDedupLookup(const DLArrayHandle lookups,
                         const DLArrayHandle idoffsets, DLArrayHandle output,
                         DLStreamHandle stream_handle);

    int DLGpuDedupGrad(const DLArrayHandle grad, const DLArrayHandle idoffsets,
                       DLArrayHandle output, DLStreamHandle stream_handle);

    // DNNL Ops
    int DnnlMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                           const DLArrayHandle matB, bool transposeB,
                           const DLArrayHandle matC);

    int DnnlMatrixElementwiseMultiplyByConst(const DLArrayHandle mat, float val,
                                             DLArrayHandle output);

    int DnnlMatrixElementwiseMultiply(const DLArrayHandle matA,
                                      const DLArrayHandle matB,
                                      DLArrayHandle output);

    int DnnlMatrixElementwiseAddByConst(const DLArrayHandle mat, float val,
                                        DLArrayHandle output);

    int DnnlMatrixElementwiseAdd(const DLArrayHandle matA,
                                 const DLArrayHandle matB,
                                 DLArrayHandle output);

    int DnnlMatrixElementwiseDivideByConst(const DLArrayHandle mat, float val,
                                           DLArrayHandle output);

    int DnnlMatrixElementwiseDivide(const DLArrayHandle matA,
                                    const DLArrayHandle matB,
                                    DLArrayHandle output);

    int cpu_BroadcastTo(const DLArrayHandle in_arr, DLArrayHandle out_arr);

    int cpu_ReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output);

    int cpu_ArraySet(DLArrayHandle input, float value);

    int cpu_Reshape(const DLArrayHandle in_arr, DLArrayHandle out_arr);

    int DnnlSoftmax(const DLArrayHandle input, DLArrayHandle output);

    int DnnlSoftmaxCrossEntropy(const DLArrayHandle A, const DLArrayHandle B,
                                DLArrayHandle output);

    int DnnlSqrt(const DLArrayHandle input, DLArrayHandle output);

    int DnnlReciprocalSqrt(const DLArrayHandle input, DLArrayHandle output);

    int DnnlTanh(const DLArrayHandle input, DLArrayHandle output);

    int DnnlOpposite(const DLArrayHandle input, DLArrayHandle output);

    int DnnlSigmoid(const DLArrayHandle input, DLArrayHandle output);

    int DnnlConv2d(const DLArrayHandle input_x, DLArrayHandle input_f,
                   DLArrayHandle output, const int padding, const int stride);

    int DnnlConv2d_Gradient_of_Filter(
        const DLArrayHandle input_x, const DLArrayHandle gradient_y,
        DLArrayHandle gradient_f, const int padding, const int stride);

    int DnnlConv2d_Gradient_of_Data(
        const DLArrayHandle input_f, const DLArrayHandle gradient_y,
        DLArrayHandle gradient_x, const int padding, const int stride);

    int DnnlAvgPool(const DLArrayHandle input, const int kernel_H,
                    const int kernel_W, DLArrayHandle output, const int padding,
                    const int stride);

    int DnnlAvgPool_Gradient(const DLArrayHandle gradient_Y, const int kernel_H,
                             const int kernel_W, DLArrayHandle gradient_X,
                             const int padding, const int stride);

    int DnnlMaxPool(const DLArrayHandle input, const int kernel_H,
                    const int kernel_W, DLArrayHandle output, const int padding,
                    const int stride);

    int DnnlMaxPool_Gradient(const DLArrayHandle input,
                             const DLArrayHandle input_grad, const int kernel_H,
                             const int kernel_W, DLArrayHandle output_grad,
                             const int padding, const int stride);

    int DnnlRelu(const DLArrayHandle input, DLArrayHandle output);

    int DnnlRelu_Gradient(const DLArrayHandle input,
                          const DLArrayHandle in_grad, DLArrayHandle output);

    int DnnlBatchNorm(const DLArrayHandle input, const DLArrayHandle bn_scale,
                      const DLArrayHandle bn_bias, DLArrayHandle output,
                      DLArrayHandle running_mean, DLArrayHandle running_var,
                      DLArrayHandle save_mean, DLArrayHandle save_var,
                      float momentum, float eps);
    int DnnlBatchNorm_Gradient(
        const DLArrayHandle grad_y, const DLArrayHandle input,
        const DLArrayHandle bn_scale, DLArrayHandle grad_x,
        DLArrayHandle grad_scale, DLArrayHandle grad_bias, DLArrayHandle mean,
        DLArrayHandle var, const float eps);
    int DnnlBatchNorm_Inference(
        const DLArrayHandle input, const DLArrayHandle bn_scale,
        const DLArrayHandle bn_bias, DLArrayHandle output, DLArrayHandle mean,
        DLArrayHandle var, float eps);

    int DnnlConcat(const DLArrayHandle input_x, const DLArrayHandle input_y,
                   DLArrayHandle output, int axis);
    int cpu_Concat_Gradient(const DLArrayHandle output_gradient,
                            DLArrayHandle input_gradient, int axis, int id);

    int cpu_Dropout(const DLArrayHandle input_X, float dropout,
                    DLArrayHandle output_Y);
    int cpu_Dropout_Gradient(const DLArrayHandle output_Y, float dropout,
                             DLArrayHandle input_X,
                             unsigned long long seed_seqnum);

    int cpu_Pad(const DLArrayHandle input_X, DLArrayHandle output_Y,
                int *paddings, int pad_len, size_t mode, float constant_values);
    int cpu_Pad_Gradient(const DLArrayHandle output_gradient_Y,
                         DLArrayHandle input_gradient_X, int *paddings,
                         int pad_len, size_t mode);

    int cpu_Transpose(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                      int *perm);

    int cpu_EmbeddingLookup(const DLArrayHandle in_mat, const DLArrayHandle ids,
                            DLArrayHandle out_mat);

    int cpu_IndexedSlices2Dense(const DLArrayHandle indices,
                                const DLArrayHandle values,
                                DLArrayHandle output);
    int cpu_AddL2Regularization(const DLArrayHandle param,
                                const DLArrayHandle grad, float l2reg);
    int cpu_SparseAddToDense(const DLArrayHandle indices,
                             const DLArrayHandle grad, DLArrayHandle output);
    int cpu_SGDOptimizerUpdate(const DLArrayHandle param,
                               const DLArrayHandle grad, float learning_rate);
    int cpu_SGDOptimizerSparseUpdate(DLArrayHandle param,
                                     const DLArrayHandle grad_indices,
                                     const DLArrayHandle grad_values, float lr);
    int cpu_MomentumOptimizerUpdate(
        DLArrayHandle param, const DLArrayHandle grad, DLArrayHandle velocity,
        float learning_rate, float momentum, bool nesterov);
    int cpu_AdaGradOptimizerUpdate(DLArrayHandle param,
                                   const DLArrayHandle grad, DLArrayHandle acc,
                                   float learning_rate, float eps);
    int cpu_AdaGradOptimizerSparseUpdate(
        DLArrayHandle param, const DLArrayHandle indices,
        const DLArrayHandle values, DLArrayHandle acc, float learning_rate,
        float eps);
    int cpu_BetatsUpdate(DLArrayHandle betats, float beta1, float beta2);
    int cpu_AdamOptimizerUpdate(
        DLArrayHandle param, const DLArrayHandle grad, DLArrayHandle expavg,
        DLArrayHandle expavgsq, DLArrayHandle maxv, float learning_rate,
        float beta1, float beta2, DLArrayHandle beta_ts, float eps);
    int cpu_AdamOptimizerSparseUpdate(
        DLArrayHandle param, const DLArrayHandle indices,
        const DLArrayHandle values, DLArrayHandle expavg,
        DLArrayHandle expavgsq, DLArrayHandle maxv, float learning_rate,
        float beta1, float beta2, DLArrayHandle beta_ts, float eps);

    int cpu_ReduceIndexedSlice(
        const DLArrayHandle in_indices, const DLArrayHandle in_values,
        DLArrayHandle out_indices, DLArrayHandle out_values);

    int cpu_ReduceIndexedSliceWithEmbedding(
        const DLArrayHandle in_indices, const DLArrayHandle in_values,
        const DLArrayHandle in_params, DLArrayHandle out_indices,
        DLArrayHandle out_values, DLArrayHandle out_params);

    int cpu_AssignWithIndexedSlices(DLArrayHandle embedding,
                                    const DLArrayHandle indices,
                                    const DLArrayHandle values);

    int cpu_SGDUpdateIndexedSlices(
        const DLArrayHandle indices, const DLArrayHandle grads,
        const DLArrayHandle params, DLArrayHandle output, float lr);

    int cpu_AdaGradUpdateIndexedSlices(
        const DLArrayHandle indices, const DLArrayHandle grads,
        const DLArrayHandle params, DLArrayHandle output, float lr,
        DLArrayHandle accum, float epsilon);

    int cpu_AdamUpdateIndexedSlices(
        const DLArrayHandle indices, const DLArrayHandle grads,
        const DLArrayHandle params, DLArrayHandle output, float lr,
        DLArrayHandle m, DLArrayHandle v, DLArrayHandle maxv, float beta1,
        float beta2, DLArrayHandle betats, float epsilon);

    int cpu_UniqueIndices(const DLArrayHandle indices, DLArrayHandle output,
                          DLArrayHandle idoffsets);
    int cpu_DedupLookup(const DLArrayHandle lookups,
                        const DLArrayHandle idoffsets, DLArrayHandle output);
    int cpu_DedupGrad(const DLArrayHandle grad, const DLArrayHandle idoffsets,
                      DLArrayHandle output);

    int cpu_NormalInit(DLArrayHandle arr, const float mean, const float stddev);
    int cpu_UniformInit(DLArrayHandle arr, const float lb, const float ub);
    int cpu_TruncatedNormalInit(DLArrayHandle arr, const float mean,
                                const float stddev);
    int cpu_ReversedTruncatedNormalInit(DLArrayHandle arr, const float mean,
                                        const float stddev);

    int DLGpuBinaryCrossEntropy(const DLArrayHandle prediction,
                                const DLArrayHandle label, DLArrayHandle loss,
                                DLStreamHandle stream_handle);

    int DLGpuBinaryCrossEntropy_Gradient(
        const DLArrayHandle prediction, const DLArrayHandle label,
        const DLArrayHandle output_grad, DLArrayHandle output,
        DLStreamHandle stream_handle);

    int DLGpuBinaryCrossEntropyWithLogits(
        const DLArrayHandle prediction, const DLArrayHandle label,
        DLArrayHandle loss, DLStreamHandle stream_handle);

    int DLGpuBinaryCrossEntropyWithLogits_Gradient(
        const DLArrayHandle prediction, const DLArrayHandle label,
        const DLArrayHandle output_grad, DLArrayHandle output,
        DLStreamHandle stream_handle);

    int DLGpuDot(const DLArrayHandle matA, const DLArrayHandle matB,
                 DLArrayHandle output, DLStreamHandle stream_handle);

    int DLGpuArgmax(const DLArrayHandle input, DLArrayHandle output, int dim,
                    DLStreamHandle stream_handle = NULL);

    int DLGpuArgmaxPartial(const DLArrayHandle input,
                           const DLArrayHandle full_mask, DLArrayHandle output,
                           int dim, int topk,
                           DLStreamHandle stream_handle = NULL);

    int DLGpuCumsumWithBias(DLArrayHandle input, DLArrayHandle output,
                            float bias, int dim,
                            DLStreamHandle stream_handle = NULL);

    int DLGpuTopKIdx(const DLArrayHandle input, DLArrayHandle output_idx, int k,
                     DLStreamHandle stream_handle = NULL);

    int DLGpuTopKVal(const DLArrayHandle input, DLArrayHandle output_idx,
                     DLArrayHandle output_val, int k,
                     DLStreamHandle stream_handle = NULL);

    int DLGpuLayoutTransformTop1(
        const DLArrayHandle input, DLArrayHandle indices_s,
        DLArrayHandle location_s, DLArrayHandle output, int capacity,
        DLStreamHandle stream_handle = NULL);

    int DLGpuLayoutTransformTop2(
        const DLArrayHandle input, DLArrayHandle indices_s1,
        DLArrayHandle indices_s2, DLArrayHandle location_s1,
        DLArrayHandle location_s2, DLArrayHandle output, int capacity,
        DLStreamHandle stream_handle = NULL);

    int DLGpuReverseLayoutTransformTop1(
        const DLArrayHandle input, DLArrayHandle indices_s,
        DLArrayHandle location_s, DLArrayHandle gates, DLArrayHandle output,
        int capacity, DLStreamHandle stream_handle = NULL);

    int DLGpuReverseLayoutTransformTop2(
        const DLArrayHandle input, DLArrayHandle indices_s1,
        DLArrayHandle indices_s2, DLArrayHandle location_s1,
        DLArrayHandle location_s2, DLArrayHandle gates_1, DLArrayHandle gates_2,
        DLArrayHandle output, int capacity,
        DLStreamHandle stream_handle = NULL);

    int DLGpuLayoutTransformTop1Gradient(
        const DLArrayHandle input, DLArrayHandle indice, DLArrayHandle location,
        DLArrayHandle output, int capacity,
        DLStreamHandle stream_handle = NULL);

    int DLGpuRepeat(const DLArrayHandle input, DLArrayHandle output,
                    DLStreamHandle stream_handle);
    int DLGpuRepeatGradient(const DLArrayHandle input, DLArrayHandle output,
                            DLStreamHandle stream_handle);

    int DLGpuReverseLayoutTransformTop1GradientData(
        const DLArrayHandle input, DLArrayHandle indice, DLArrayHandle location,
        DLArrayHandle gate, DLArrayHandle output, int capacity,
        DLStreamHandle stream_handle = NULL);

    int DLGpuReverseLayoutTransformTop2GradientData(
        const DLArrayHandle input, DLArrayHandle indice_1,
        DLArrayHandle indice_2, DLArrayHandle location_1,
        DLArrayHandle location_2, DLArrayHandle gate_1, DLArrayHandle gate_2,
        DLArrayHandle output, int capacity,
        DLStreamHandle stream_handle = NULL);

    int DLGpuReverseLayoutTransformTop1GradientGate(
        const DLArrayHandle combined_output, DLArrayHandle expert_output,
        DLArrayHandle indice, DLArrayHandle location, DLArrayHandle output,
        int capacity, DLStreamHandle stream_handle = NULL);

    int DLGpuSliceAssign(const DLArrayHandle input, DLArrayHandle output,
                         float val, int *begin_pos, int *end_pos,
                         DLStreamHandle stream_handle);

    int DLGpuSliceAssignMatrix(
        const DLArrayHandle input_A, const DLArrayHandle input_B,
        DLArrayHandle output, int *begin_pos_A, int *end_pos_A,
        int *begin_pos_B, DLStreamHandle stream_handle);

    int DLGpuSliceByMatrix(const DLArrayHandle in_arr,
                           const DLArrayHandle index1,
                           const DLArrayHandle index2, DLArrayHandle out_arr,
                           DLStreamHandle stream_handle);

    int DLGpuSliceByMatrixGradient(
        const DLArrayHandle in_arr, const DLArrayHandle index1,
        const DLArrayHandle index2, DLArrayHandle out_arr,
        DLStreamHandle stream_handle);

    int DLGpuIndexing(const DLArrayHandle input, DLArrayHandle index,
                      DLArrayHandle output,
                      DLStreamHandle stream_handle = NULL);

    int DLGpuIndexingGrad(const DLArrayHandle output_grad, DLArrayHandle index,
                          DLArrayHandle input_grad,
                          DLStreamHandle stream_handle = NULL);

    int DLGpuScatter1D(const DLArrayHandle input, DLArrayHandle index,
                       DLArrayHandle output,
                       DLStreamHandle stream_handle = NULL);

    int DLGpuScatter1DGrad(const DLArrayHandle output_grad, DLArrayHandle index,
                           DLArrayHandle input_grad,
                           DLStreamHandle stream_handle = NULL);

    int DLGpuLog(const DLArrayHandle input, DLArrayHandle output, float eps,
                 DLStreamHandle stream_handle = NULL);

    int DLGpuLogGrad(const DLArrayHandle output_grad, DLArrayHandle input,
                     DLArrayHandle input_grad, float eps,
                     DLStreamHandle stream_handle = NULL);

    int DLGpuMask(const DLArrayHandle input, const DLArrayHandle mask,
                  DLArrayHandle output, DLStreamHandle stream_handle = NULL);

    int DLGpuExp(const DLArrayHandle input, DLArrayHandle output,
                 DLStreamHandle stream_handle);

    int DLGpuNllLoss(const DLArrayHandle input, DLArrayHandle target,
                     DLArrayHandle output, DLStreamHandle stream_handle = NULL);

    int DLGpuNllLossGrad(const DLArrayHandle output_grad, DLArrayHandle target,
                         DLArrayHandle input,
                         DLStreamHandle stream_handle = NULL);

    int DLGpuReverseLayoutTransformNoGate(
        const DLArrayHandle input, DLArrayHandle indices_s,
        DLArrayHandle location_s, DLArrayHandle output, int capacity,
        DLStreamHandle stream_handle);

    int DLGpuReverseLayoutTransformNoGateGradient(
        const DLArrayHandle input, DLArrayHandle indice, DLArrayHandle location,
        DLArrayHandle output, int capacity, DLStreamHandle stream_handle);

    int DLGpuHA2ALayoutTransform(
        const DLArrayHandle input, DLArrayHandle output, int num_nodes,
        int num_local_gpus, DLStreamHandle stream_handle);

    int DLGpuHA2AReverseLayoutTransform(
        const DLArrayHandle input, DLArrayHandle output, int num_nodes,
        int num_local_gpus, DLStreamHandle stream_handle);

    int DLGpuSamGroupSum(const DLArrayHandle gate, DLArrayHandle output,
                         int num_local_gpus, DLStreamHandle stream_handle);

    int DLGpuGroupTopKIdx(const DLArrayHandle input, DLArrayHandle top1_group,
                          DLArrayHandle output_idx, int k, int num_local_gpus,
                          DLStreamHandle stream_handle);

    int DLGpuSamMax(const DLArrayHandle input, DLArrayHandle top1_group,
                    DLArrayHandle topk_indice, DLArrayHandle output,
                    int num_local_gpus, DLStreamHandle stream_handle);

    int DLGpuSamMaxGrad(const DLArrayHandle output_grad, DLArrayHandle input,
                        DLArrayHandle top1_group, DLArrayHandle topk_indice,
                        DLArrayHandle output, int num_local_gpus,
                        DLStreamHandle stream_handle);

    int DLGpuRobeHash(const DLArrayHandle input,
                      const DLArrayHandle random_numbers, DLArrayHandle output,
                      int length, int dim, int Z, bool use_slot_coef,
                      DLStreamHandle stream_handle);
    int DLGpuRobeSign(const DLArrayHandle input,
                      const DLArrayHandle random_numbers, DLArrayHandle output,
                      int dim, bool use_slot_coef,
                      DLStreamHandle stream_handle);

    int DLGpuModHash(const DLArrayHandle input, DLArrayHandle output,
                     int nembed, DLStreamHandle stream_handle);

    int DLGpuModHashNegative(const DLArrayHandle input, DLArrayHandle output,
                             int nembed, DLStreamHandle stream_handle);

    int DLGpuDivHash(const DLArrayHandle input, DLArrayHandle output,
                     int nembed, DLStreamHandle stream_handle);

    int DLGpuCompoHash(const DLArrayHandle input, DLArrayHandle output,
                       int ntable, int nembed, DLStreamHandle stream_handle);

    int DLGpuLearnHash(const DLArrayHandle input, const DLArrayHandle slope,
                       const DLArrayHandle bias, const DLArrayHandle prime,
                       DLArrayHandle output, int nbucket, bool normal,
                       float eps, DLStreamHandle stream_handle);

    int DLGpuNumLessThan(const DLArrayHandle input, DLArrayHandle middle,
                         DLArrayHandle output, float threshold, int *axes,
                         int num_ax, DLStreamHandle stream_handle);
    int DLGpuSetLessThan(const DLArrayHandle arr, float threshold,
                         DLStreamHandle stream_handle);
    int DLGpuSetMaskLessThan(DLArrayHandle arr, DLArrayHandle mask,
                             float threshold, DLStreamHandle stream_handle);
    int DLGpuGetLargerThan(const DLArrayHandle input,
                           const DLArrayHandle threshold, DLArrayHandle mask,
                           DLStreamHandle stream_handle);
    int DLGpuNumLessThanTensorThreshold(
        const DLArrayHandle input, DLArrayHandle middle, DLArrayHandle output,
        const DLArrayHandle threshold, int *axes, int num_ax,
        DLStreamHandle stream_handle);
    int DLGpuMultiplyGroupingAlpha(
        DLArrayHandle arr, const DLArrayHandle grouping,
        const DLArrayHandle alpha, DLStreamHandle stream_handle);

    int DLGpuDequantize(const DLArrayHandle input, DLArrayHandle output,
                        int digit, float scale, float minele,
                        DLStreamHandle stream_handle);
    int DLGpuDequantizeSigned(const DLArrayHandle input, DLArrayHandle output,
                              int digit, float scale, float middle,
                              DLStreamHandle stream_handle);

    int DLGpuPrepackEmbedding(const DLArrayHandle input, DLArrayHandle output,
                              DLArrayHandle qparams, int digit,
                              DLStreamHandle stream_handle);
    int DLGpuQuantizedEmbeddingLookup(
        const DLArrayHandle input, const DLArrayHandle indices,
        DLArrayHandle output, DLArrayHandle qparams, int digit,
        DLStreamHandle stream_handle);
    int DLGpuQuantizedEmbeddingLookupWithScale(
        const DLArrayHandle input, const DLArrayHandle indices,
        const DLArrayHandle scale, DLArrayHandle output, int digit,
        float middle, DLStreamHandle stream_handle);
    int DLGpuUnifiedQuantizedEmbeddingLookup(
        const DLArrayHandle input, const DLArrayHandle indices,
        DLArrayHandle output, int digit, float scale, float minele,
        DLStreamHandle stream_handle);
    int DLGpuRoundingToInt(const DLArrayHandle input, DLArrayHandle output,
                           float scale, float minele, int digit,
                           bool stochastic, DLStreamHandle stream_handle);
    int DLGpuRoundingToSignedInt(
        const DLArrayHandle input, DLArrayHandle output, float scale,
        float middle, int digit, bool stochastic, DLStreamHandle stream_handle);
    int DLGpuQuantizeEmbeddingWithScale(
        const DLArrayHandle input, const DLArrayHandle scale,
        DLArrayHandle output, float middle, int digit,
        DLStreamHandle stream_handle);
    int DLGpuLSQRounding(const DLArrayHandle input, const DLArrayHandle scale,
                         DLArrayHandle output, int digit, float middle,
                         DLStreamHandle stream_handle);
    int DLGpuLSQRoundingGradient(const DLArrayHandle input,
                                 DLArrayHandle output, int digit,
                                 DLStreamHandle stream_handle);

} // HETUSYS_EXTERN_C

#endif // HETUSYS_RUNTIME_C_RUNTIME_API_H_
