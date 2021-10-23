#pragma once

#include "ps/server/PSFHandle.h"
#include "ps/server/ssp_handler.h"
#include "ps/server/preduce_handler.h"
#include "ps/psf/serializer.h"
#include "ps/kvapp.h"
#include <memory>
#include <vector>
#include <utility>
namespace ps {

template <PsfType>
struct KVServerRegisterHelper;

/**
 * \brief A server node for maintaining key-value pairs
 */
class KVServer : public KVApp {
public:
    /**
     * \brief constructor
     * \param app_id the app id, should match with \ref KVWorker's id
     */
    explicit KVServer(int app_id) : KVApp(app_id) {
        KVAppRegisterHelper<PsfType(0), KVServer>::init(this);
        // TODO : change this to index_sequence if c++14 is available
        handler_[static_cast<int>(PsfGroup::kParameterServer)] = std::make_shared<PSHandler<PsfGroup::kParameterServer>>();
        handler_[static_cast<int>(PsfGroup::kSSPControl)] = std::make_shared<PSHandler<PsfGroup::kSSPControl>>();
        handler_[static_cast<int>(PsfGroup::kPReduceScheduler)] = std::make_shared<PSHandler<PsfGroup::kPReduceScheduler>>();
    }

private:
    template <PsfType ftype>
    void onReceive(const Message &msg) {
        typename PSFData<ftype>::Request request;
        typename PSFData<ftype>::Response response;
        tupleDecode(request, msg.data);
        constexpr PsfGroup group = PSFData<ftype>::group;
        auto handler = std::dynamic_pointer_cast<PSHandler<group>>(handler_[static_cast<int>(group)]);
        assert(handler);
        handler->serve(request, response);
        Message rmsg;
        tupleEncode(response, rmsg.data);
        rmsg.meta = msg.meta;
        rmsg.meta.recver = msg.meta.sender;
        rmsg.meta.request = false;
        Postoffice::Get()->van()->Send(rmsg);
    }

    /** \brief request handle */
    std::unordered_map<int, std::shared_ptr<PSHandler<PsfGroup::kBaseGroup>>> handler_;
    template <PsfType, typename>
    friend struct KVAppRegisterHelper;
};

} // namespace ps
