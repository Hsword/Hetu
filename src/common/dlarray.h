/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlarray.h
 * \brief Header that defines array struct.
 */
#ifndef HETUSYS_H_
#define HETUSYS_H_

#ifdef __cplusplus
#define HETUSYS_EXTERN_C extern "C"
#else
#define HETUSYS_EXTERN_C
#endif

#include <stddef.h>
#include <stdint.h>
#include "dispatch.h"

HETUSYS_EXTERN_C {
    /*!
     * \brief The device type in DLContext.
     */
    typedef enum {
        kCPU = 1,
        kGPU = 2,
    } DLDeviceType;

    /*!
     * \brief A Device context for array.
     */
    typedef struct {
        /*! \brief The device index */
        int device_id;
        /*! \brief The device type used in the device. */
        DLDeviceType device_type;
    } DLContext;

    /*!
     * \brief Plain C Array object, does not manage memory.
     */
    typedef struct {
        /*!
         * \brief The opaque data pointer points to the allocated data.
         *  This will be CUDA device pointer or cl_mem handle in OpenCL.
         *  This pointer is always aligns to 256 bytes as in CUDA.
         */
        void *data;
        /*! \brief The device context of the tensor */
        DLContext ctx;
        /*! \brief Number of dimensions */
        int ndim;
        /*! \brief The shape of the tensor */
        int64_t *shape;
        /*! \brief The stride of the tensor */
        int64_t *stride;
        /*! \brief The bytes number of each value */
        int nbits = 32;
        /*! \brief The datatype of the tensor */
        DataType dtype = DataType::FLOAT32;
    } DLArray;

    typedef struct {
        int device_id;
        void *handle;
    } DLStream;

    typedef struct {
        int device_id;
        void *handle;
    } DLEvent;
} // HETUSYS_EXTERN_C
#endif // HETUSYS_H_
