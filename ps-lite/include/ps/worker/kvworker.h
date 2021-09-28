#pragma once

#include "ps/psf/PSFunc.h"
#include "ps/psf/serializer.h"
#include "callback_store.h"
#include "ps/kvapp.h"
#include "ps/partitioner.h"
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <string>

namespace ps {

template <PsfType>
struct KVWorkerRegisterHelper;

class KVWorker : private KVApp {
public:
    Partitioner *par;
    /**
     * \brief constructor
     *
     * \param app_id the app id, should match with \ref KVServer's id
     * \param customer_id the customer id which is unique locally
     */
    explicit KVWorker(int app_id, int customer_id) : KVApp(app_id) {
        KVAppRegisterHelper<PsfType(0), KVWorker>::init(this);
        par = new AveragePartitioner(); // now use naive partitioner
    }

    ~KVWorker() {
        delete par;
        if (logOut.is_open())
            logOut.close();
    }

    void startRecord(std::string dirPath) {
        logOut.open(dirPath + "/loads_" + std::to_string(MyRank()) + ".txt");
        assert(logOut.is_open());
    }

    void recordLoads() {
        for (auto iter = loads.begin(); iter != loads.end(); ++iter) {
            logOut << getPSFunctionName(iter->first) << ": " << (iter->second).first
                   << ' ' << (iter->second).second << std::endl;
        }
        logOut << std::endl;
        loads.clear();
    }

    /**
     * \brief Waits until a Request has been finished
     *
     * Sample usage:
     * \code
     *   _kvworker.Wait(ts);
     * \endcode
     *
     * \param timestamp the timestamp returned by kvworker.Request
     */
    void Wait(int timestamp) {
        obj_->WaitRequest(timestamp);
    }
    /**
     * \brief make a new Request
     *
     * Sample usage:
     * \code
     *   int ts = _kvworker.Request<DensePush>(request, callback);
     * \endcode
     *
     * \param request create request by PSFData<PsfType>::Request
     * \param cb the callback returned by getCallback<PSfType>(args...)
     */
    template <PsfType ftype, typename Tuple, typename CallBack>
    int Request(const Tuple &request, const CallBack &cb) {
        int timestamp = obj_->NewRequest(kServerGroup);
        CallbackStore<ftype>::Get()->store(timestamp, cb);
        // Find the server
        Key key = get<0>(request);
        int target_server_id = par->queryServer(key);
        // Create message
        Message msg;
        tupleEncode(request, msg.data);
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].first += x.size();
            }
        }
        msg.meta.app_id = obj_->app_id();
        msg.meta.customer_id = obj_->customer_id();
        msg.meta.timestamp = timestamp;
        msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
        msg.meta.psftype = ftype;
        msg.meta.request = true;
        Postoffice::Get()->van()->Send(msg);
        return timestamp;
    }

private:
    template <PsfType ftype>
    void onReceive(const Message &msg) {
        typename PSFData<ftype>::Response response;
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].second += x.size();
            }
        }
        tupleDecode(response, msg.data);
        int timestamp = msg.meta.timestamp;
        CallbackStore<ftype>::Get()->run(timestamp, response);
    }
    template <PsfType, typename>
    friend struct KVAppRegisterHelper;
    std::unordered_map<PsfType, std::pair<long long, long long>> loads;
    std::ofstream logOut;
};

} // namespace ps
