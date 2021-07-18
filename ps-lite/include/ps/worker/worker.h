#pragma once

#include <cmath>
#include <unordered_map>
#include <vector>
#include <thread>
#include <condition_variable>
#include <unistd.h>

#include "ps/ps.h"
#include "common/dlarray.h"
#include "common/c_runtime_api.h"
#include "ps/worker/PSAgent.h"
#include "ps/server/param.h"
#include "ps/server/optimizer.h"

using namespace ps;

class Worker {
public:
    Worker();

    void parameter_init(int node_name, ParamType ptype, size_t len,
                        size_t width, InitType init_type, double init_a,
                        double init_b, unsigned long long seed, OptType otype,
                        SArray<float> lrs);
    void parameter_save(int node_name, char *address);
    void parameter_load(int node_name, char *address);
    // for data push&pull
    typedef uint64_t query_t;
    /*
      for each indice, call PSAgent::PushData to launch a thread
      hold the return handle in the global map
      immediately return
      user should guaruntee value unchanged until waitdata
      returns:
        an query_t which is a long
        use waitdata(query_t) to wait for its success
    */
    query_t push_data(const long *indices, int index_size, const DLArray *value,
                      const long *lengths);
    // this is almost the same as push_data
    query_t pull_data(const long *indices, int index_size, DLArray *value,
                      const long *lengths);
    /*
      wait_data waits until a query success
    */
    void wait_data(query_t query);

    void pull(int node_name, DLArray *arr);
    void push(int node_name, const DLArray *arr, DLEvent *evt);
    void dd_pushpull(int node_name, const DLArray *in_arr, DLArray *out_arr,
                     DLEvent *evt);
    void sparse_pull(int node_name, const DLArray *index, DLArray *value,
                     size_t index_size);
    void sparse_push(int node_name, const DLArray *index, const DLArray *value,
                     size_t index_size, DLEvent *evt);
    void sd_pushpull(int node_name, const DLArray *index, const DLArray *in_arr,
                     size_t index_size, DLArray *out_arr, DLEvent *evt);
    void ss_pushpull(int node_name, const DLArray *inind, const DLArray *in_arr,
                     const DLArray *outind, DLArray *out_arr, size_t index_size,
                     DLEvent *evt);
    void wait(int node_name);
    void clear(int node_name);
    void clear_on_server(int node_name);

private:
    // used this hold to thread_pool return object
    std::unordered_map<query_t, std::vector<int>> query2timestamp;
    // data_pull & data_push query, increase 1 each call
    query_t next_query = 0;
    // protect query2timestamp and next_query
    std::mutex data_mu;

    // for concurrent parameter push&pull
    std::unordered_map<int, std::future<void>> node2pullthread;
    std::unordered_map<int, std::future<void>> node2pushthread;

    int _thread_num = 3;
};

extern Worker worker;
