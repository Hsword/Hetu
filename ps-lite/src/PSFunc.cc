#include "ps/psf/PSFunc.h"

#include <mutex>
#include <unordered_map>

namespace ps {

static std::unordered_map<PsfType, const char*> psfunction_names;

static void initnames(PSFData<kNumPSfunction>) {}

template<PsfType ftype> static void initnames(PSFData<ftype>) {
    psfunction_names[ftype] = PSFData<ftype>::name;
    initnames(PSFData<PsfType(ftype+1)>());
}

const char*
getPSFunctionName(const PsfType &ftype) {
    static std::once_flag flag;
    std::call_once(flag, []() {
        initnames(PSFData<PsfType(0)>());
    });
    return psfunction_names[ftype];
}

}
