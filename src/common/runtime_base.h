/*!
 *  Copyright (c) 2017 by Contributors
 * \file runtime_base.h
 * \brief Base of all C APIs
 */
#ifndef HETUSYS_RUNTIME_RUNTIME_BASE_H_
#define HETUSYS_RUNTIME_RUNTIME_BASE_H_

#include "c_runtime_api.h"
#include <stdexcept>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*!
 * \brief every function starts with API_BEGIN(), and finishes with API_END()
 *  or API_END_HANDLE_ERROR
 */
#define API_END()                                                              \
    }                                                                          \
    catch (std::runtime_error & _except_) {                                    \
        return HETUSYSAPIHandleException(_except_);                            \
    }                                                                          \
    return 0;

/*!
 * \brief every function starts with API_BEGIN() and finishes with API_END() or
 * API_END_HANDLE_ERROR. The finally clause contains procedure to cleanup states
 * when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize)                                         \
    }                                                                          \
    catch (std::runtime_error & _except_) {                                    \
        Finalize;                                                              \
        return HETUSYSAPIHandleException(_except_);                            \
    }                                                                          \
    return 0;

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int HETUSYSAPIHandleException(const std::runtime_error &e) {
    // TODO
    // TVMAPISetLastError(e.what());
    return -1;
}

#endif // HETUSYS_RUNTIME_RUNTIME_BASE_H_
