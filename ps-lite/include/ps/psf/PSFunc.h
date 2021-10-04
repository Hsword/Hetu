#pragma once

#include "common/sarray.h"
#include "ps/base.h"

#include <tuple>
#include <functional>
using std::tuple;
using std::get;
using std::function;

namespace ps {

// PS Functions are declared within groups
// Each function group will act independently, meaning that one call in group A will not affect the state in B
enum class PsfGroup {
    kBaseGroup,
    kParameterServer,
    kSSPControl,
    kPReduceScheduler,
    kNumGroup
};

template<PsfGroup> class PSHandler;

// This is the base class for other PSHandler
template<>
class PSHandler<PsfGroup::kBaseGroup> {
public:
    virtual ~PSHandler<PsfGroup::kBaseGroup>() {};
};

enum PsfType {
    /* Dense ops */
    DensePush,
    DensePull,
    DDPushPull,
    /* Sparse ops */
    SparsePush,
    SparsePull,
    SDPushPull,
    SSPushPull,
    /* misc ops */
    ParamInit,
    ParamClear,
    ParamSave,
    ParamLoad,
    /* cache sparse table */
    kSyncEmbedding,
    kPushEmbedding,
    kPushSyncEmbedding,
    /* SSP support */
    kSSPInit,
    kSSPSync,
    /* Partial Reduce support */
    kPReduceGetPartner,
    kNumPSfunction,
};

template <PsfType>
struct PSFData;
/*
    To define a new PSFunc, we need 3 parts : Request, Response, _callback
    * Request and Response are tuple-like object, and must only use
      scalar types like int, float or Sarray
    * _callback is a function having format void(const Response&, args...)
      where args are some target memory space to write back
    * See examples in dense.h sparse.h ...
*/

template<>
struct PSFData<kNumPSfunction> {};

/*
  getCallBack, use this to bind _callback to the get the real callback which can
  be stored example: getCallBack<DensePull>(target);
*/
template <PsfType ftype, typename... Args>
function<void(const typename PSFData<ftype>::Response &)>
getCallBack(Args &&... args) {
    return std::bind(PSFData<ftype>::_callback, std::placeholders::_1,
                     std::forward<Args>(args)...);
}

const char* getPSFunctionName(const PsfType &ftype);

} // namespace ps


// add hash function so that it can be used in unordered_map
namespace std {
  template <> struct hash<ps::PsfType> {
    size_t operator() (const ps::PsfType &ftype) const { return static_cast<size_t>(ftype); }
  };
}

#include "dense.h"
#include "sparse.h"
#include "misc.h"
#include "cachetable.h"
#include "ssp.h"
#include "preduce.h"
