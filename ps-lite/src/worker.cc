#include "ps/worker/worker.h"

Worker::Worker() {
}

void Worker::parameter_init(int node_name, ParamType ptype, size_t len,
                            size_t width, InitType init_type, double init_a,
                            double init_b, unsigned long long seed,
                            OptType otype, SArray<float> lrs) {
    PSAgent::Get()->registerTensor(node_name, ptype, len, width);
    PSAgent::Get()->ParameterInit(node_name, init_type, init_a, init_b, seed,
                                  otype, lrs);
    PSAgent::Get()->wait(node_name);
    Postoffice::Get()->Barrier(0, kWorkerGroup);
}

void Worker::parameter_save(int node_name, char *address) {
    PSAgent::Get()->ParameterSave(node_name, address);
}

void Worker::parameter_load(int node_name, char *address) {
    PSAgent::Get()->ParameterLoad(node_name, address);
}

void Worker::push(int node_name, const DLArray *arr, DLEvent *evt) {
    float *data = static_cast<float *>(arr->data);
    node2pushthread[node_name] = ThreadPool::Get()->Enqueue(
        [node_name](float *data, DLEvent *evt) -> void {
            if (evt != NULL)
                DLEventSync(evt);
            PSAgent::Get()->vecDensePush(node_name, data, -node_name);
        },
        data, evt);
}

Worker::query_t Worker::push_data(const long *indices, int index_size,
                                  const DLArray *value, const long *lengths) {
    float *data = static_cast<float *>(value->data);
    data_mu.lock();
    query_t cur_query = next_query++;
    auto &timestamps = query2timestamp[cur_query];
    data_mu.unlock();

    for (int i = 0; i < index_size; i++) {
        Key idx = (Key)indices[i];
        auto len = lengths[i];
        PSAgent::Get()->PushData(idx, data, len, timestamps);
        data += len;
    }
    return cur_query;
}

// this is almost the same as push_data
Worker::query_t Worker::pull_data(const long *indices, int index_size,
                                  DLArray *value, const long *lengths) {
    float *data = static_cast<float *>(value->data);
    data_mu.lock();
    query_t cur_query = next_query++;
    auto &timestamps = query2timestamp[cur_query];
    data_mu.unlock();

    for (int i = 0; i < index_size; i++) {
        Key idx = (Key)indices[i];
        auto len = lengths[i];
        PSAgent::Get()->PullData(idx, data, len, timestamps);
        data += len;
    }
    return cur_query;
}

// wait_data waits until a query success
void Worker::wait_data(query_t query) {
    data_mu.lock();
    auto iter = query2timestamp.find(query);
    if (iter == query2timestamp.end()) {
        data_mu.unlock();
        LG << "Wait on empty query " << query;
        return;
    } else {
        auto timestamps = std::move(iter->second);
        query2timestamp.erase(iter);
        data_mu.unlock();
        for (int t : timestamps) {
            PSAgent::Get()->waitTimestamp(t);
        }
    }
}

void Worker::pull(int node_name, DLArray *arr) {
    float *rets = static_cast<float *>(arr->data);
    node2pullthread[node_name] = ThreadPool::Get()->Enqueue(
        [node_name](float *data, std::future<void> &push_thread) -> void {
            if (push_thread.valid()) {
                push_thread.wait();
                PSAgent::Get()->wait(node_name);
            }
            PSAgent::Get()->vecDensePull(node_name, data, -node_name);
        },
        rets, std::ref(node2pushthread[node_name]));
}

void Worker::dd_pushpull(int node_name, const DLArray *in_arr, DLArray *out_arr,
                         DLEvent *evt) {
    float *in_data = static_cast<float *>(in_arr->data);
    float *out_data = static_cast<float *>(out_arr->data);
    node2pullthread[node_name] = ThreadPool::Get()->Enqueue(
        [node_name](float *in_data, float *out_data, DLEvent *evt) -> void {
            if (evt != NULL)
                DLEventSync(evt);
            PSAgent::Get()->vecDDPushPull(node_name, in_data, out_data,
                                          -node_name);
        },
        in_data, out_data, evt);
}

void Worker::sparse_pull(int node_name, const DLArray *index, DLArray *value,
                         size_t index_size) {
    float *data = static_cast<float *>(value->data);
    float *indices = static_cast<float *>(index->data);
    node2pullthread[node_name] = ThreadPool::Get()->Enqueue(
        [node_name](float *indices, float *data, const size_t index_size,
                    std::future<void> &push_thread) -> void {
            if (push_thread.valid()) {
                push_thread.wait();
                PSAgent::Get()->wait(node_name);
            }
            PSAgent::Get()->vecPullSparse(node_name, indices, data, index_size,
                                          -node_name);
        },
        indices, data, index_size, std::ref(node2pushthread[node_name]));
}

void Worker::sparse_push(int node_name, const DLArray *index,
                         const DLArray *value, size_t index_size,
                         DLEvent *evt) {
    // for gradient value
    // the DLArray_len should be the length of the parameter
    // for sparse push the length of gradient value is not equal to
    // corresponding parameter
    float *data = static_cast<float *>(value->data);
    float *indices = static_cast<float *>(index->data);
    node2pushthread[node_name] = ThreadPool::Get()->Enqueue(
        [node_name](float *indices, float *data, const size_t index_size,
                    DLEvent *evt) -> void {
            if (evt != NULL)
                DLEventSync(evt);
            PSAgent::Get()->vecPushSparse(node_name, indices, data, index_size,
                                          -node_name);
        },
        indices, data, index_size, evt);
}

void Worker::sd_pushpull(int node_name, const DLArray *index,
                         const DLArray *in_arr, size_t index_size,
                         DLArray *out_arr, DLEvent *evt) {
    float *indices = static_cast<float *>(index->data);
    float *in_data = static_cast<float *>(in_arr->data);
    float *out_data = static_cast<float *>(out_arr->data);
    node2pullthread[node_name] = ThreadPool::Get()->Enqueue(
        [node_name](float *indices, float *in_data, size_t ind_size,
                    float *out_data, DLEvent *evt) -> void {
            if (evt != NULL)
                DLEventSync(evt);
            PSAgent::Get()->vecSDPushPull(node_name, indices, in_data, ind_size,
                                          out_data, -node_name);
        },
        indices, in_data, index_size, out_data, evt);
}

void Worker::ss_pushpull(int node_name, const DLArray *inind,
                         const DLArray *in_arr, const DLArray *outind,
                         DLArray *out_arr, size_t index_size, DLEvent *evt) {
    float *inindices = static_cast<float *>(inind->data);
    float *outindices = static_cast<float *>(outind->data);
    float *in_data = static_cast<float *>(in_arr->data);
    float *out_data = static_cast<float *>(out_arr->data);
    node2pullthread[node_name] = ThreadPool::Get()->Enqueue(
        [node_name](float *inindices, float *in_data, float *outindices,
                    float *out_data, size_t index_size, DLEvent *evt) -> void {
            if (evt != NULL)
                DLEventSync(evt);
            PSAgent::Get()->vecSSPushPull(node_name, inindices, in_data,
                                          outindices, out_data, index_size,
                                          -node_name);
        },
        inindices, in_data, outindices, out_data, index_size, evt);
}

void Worker::wait(int node_name) {
    std::future<void> &push_thread = node2pushthread[node_name];
    std::future<void> &pull_thread = node2pullthread[node_name];
    if (push_thread.valid())
        push_thread.wait();
    if (pull_thread.valid())
        pull_thread.wait();
    PSAgent::Get()->wait(node_name);
}

void Worker::clear(int node_name) {
    PSAgent::Get()->clear(node_name);
}

void Worker::clear_on_server(int node_name) {
    PSAgent::Get()->clearOnServer(node_name);
}

Worker worker;
