#pragma once

#include "ps/server/PSFHandle.h"
#include "ps/psf/serializer.h"
#include "ps/kvapp.h"
#include <memory>
#include <vector>
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
    }

private:
    template <PsfType ftype>
    void onReceive(const Message &msg) {
        typename PSFData<ftype>::Request request;
        typename PSFData<ftype>::Response response;
        tupleDecode(request, msg.data);
        handler.serve(request, response);
        Message rmsg;
        tupleEncode(response, rmsg.data);
        rmsg.meta = msg.meta;
        rmsg.meta.recver = msg.meta.sender;
        rmsg.meta.request = false;
        Postoffice::Get()->van()->Send(rmsg);
    }

    /** \brief request handle */
    KVServerMatrixHandle handler;
    template <PsfType, typename>
    friend struct KVAppRegisterHelper;
};

} // namespace ps
