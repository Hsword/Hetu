
#ifndef HETYSYS_DISPATCH_H_
#define HETYSYS_DISPATCH_H_

typedef enum {
    FLOAT32 = 0,
    INT32 = 1,
    UINT32 = 2,
    INT8 = 3,
    UINT8 = 4,
    INT16 = 5,
    UINT16 = 6,
} DataType;

template <DataType DTYPE>
struct DataType2SpecMeta;

#define DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DTYPE, SPEC_TYPE)                \
    template <>                                                                \
    struct DataType2SpecMeta<DTYPE> {                                          \
        using spec_type = SPEC_TYPE; /* the specialized type */                \
    }

DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::INT8, int8_t);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::UINT8, uint8_t);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::INT16, int16_t);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::UINT16, uint16_t);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::INT32, int32_t);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::UINT32, uint32_t);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::FLOAT32, float);

// adapted from pytorch
#define HT_DISPATH_CASE(DATA_TYPE, SPEC_TYPE, ...)                             \
    case DATA_TYPE: {                                                          \
        using SPEC_TYPE = DataType2SpecMeta<DATA_TYPE>::spec_type;             \
        __VA_ARGS__();                                                         \
        break;                                                                 \
    }

#define HT_DISPATH_SWITCH(DTYPE, ...)                                          \
    do {                                                                       \
        const auto &_dtype = DTYPE;                                            \
        switch (_dtype) {                                                      \
            __VA_ARGS__                                                        \
        default:                                                               \
            assert(false);                                                     \
        }                                                                      \
    } while (0)

#define HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, ...)                         \
    HT_DISPATH_CASE(DataType::FLOAT32, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_FLOATING_TYPES(DTYPE, SPEC_TYPE, ...)                      \
    HT_DISPATH_SWITCH(DTYPE,                                                   \
                      HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, ...)                          \
    HT_DISPATH_CASE(DataType::INT32, SPEC_TYPE, __VA_ARGS__)                   \
    HT_DISPATH_CASE(DataType::UINT32, SPEC_TYPE, __VA_ARGS__)                  \
    HT_DISPATH_CASE(DataType::INT16, SPEC_TYPE, __VA_ARGS__)                   \
    HT_DISPATH_CASE(DataType::UINT16, SPEC_TYPE, __VA_ARGS__)                  \
    HT_DISPATH_CASE(DataType::INT8, SPEC_TYPE, __VA_ARGS__)                    \
    HT_DISPATH_CASE(DataType::UINT8, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_INTEGER_TYPES(DTYPE, SPEC_TYPE, ...)                       \
    HT_DISPATH_SWITCH(DTYPE,                                                   \
                      HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES(SPEC_TYPE, ...)             \
    HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__)                     \
    HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(DTYPE, SPEC_TYPE, ...)          \
    HT_DISPATH_SWITCH(DTYPE, HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES(       \
                                 SPEC_TYPE, __VA_ARGS__))

#endif
