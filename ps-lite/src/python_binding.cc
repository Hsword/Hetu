#include "ps/worker/worker.h"

#include "ps/ps.h"
#include "ps/server/kvserver.h"

extern "C" {

void Init() {
    if (Postoffice::Get()->van())
        return;
    Start(0);
}

void Finalize() {
    Finalize(0, true);
}

void Pull(int node_name, DLArray *arr) {
    worker.pull(node_name, arr);
}

void Push(int node_name, const DLArray *arr, DLEvent *evt) {
    worker.push(node_name, arr, evt);
}

void DDPushPull(int node_name, const DLArray *in_arr, DLArray *out_arr,
                DLEvent *evt) {
    worker.dd_pushpull(node_name, in_arr, out_arr, evt);
}

void SparsePush(int node_name, const DLArray *index, const DLArray *value,
                DLEvent *evt) {
    size_t index_size = 1;
    for (int i = 0; i < index->ndim; i++)
        index_size *= index->shape[i];
    worker.sparse_push(node_name, index, value, index_size, evt);
}

void SparsePull(int node_name, const DLArray *index, DLArray *value) {
    size_t index_size = 1;
    for (int i = 0; i < index->ndim; i++)
        index_size *= index->shape[i];
    worker.sparse_pull(node_name, index, value, index_size);
}

void SDPushPull(int node_name, const DLArray *index, const DLArray *in_arr,
                DLArray *out_arr, DLEvent *evt) {
    size_t index_size = 1;
    for (int i = 0; i < index->ndim; i++)
        index_size *= index->shape[i];
    worker.sd_pushpull(node_name, index, in_arr, index_size, out_arr, evt);
}

void SSPushPull(int node_name, const DLArray *inindices, const DLArray *in_arr,
                const DLArray *outindices, DLArray *out_arr, DLEvent *evt) {
    size_t index_size = 1;
    assert(inindices->ndim == outindices->ndim);
    for (int i = 0; i < inindices->ndim; ++i) {
        assert(inindices->shape[i] == outindices->shape[i]);
        index_size *= inindices->shape[i];
    }
    worker.ss_pushpull(node_name, inindices, in_arr, outindices, out_arr,
                       index_size, evt);
}

/**
 *   args:
 *       index, example index
 *       value, the example value
 *       length, length of every example
 */
Worker::query_t PushData(const long *index, int index_size,
                         const DLArray *value, const long *length) {
    return worker.push_data(index, index_size, value, length);
}

Worker::query_t PullData(const long *index, int index_size, DLArray *value,
                         const long *length) {
    return worker.pull_data(index, index_size, value, length);
}

void Wait(int node_id) {
    worker.wait(node_id);
}

void WaitData(Worker::query_t query) {
    worker.wait_data(query);
}

void BarrierWorker() {
    Postoffice::Get()->Barrier(0, kWorkerGroup);
}

void InitTensor(int node_name, int ptype, int len, int width, int init_type,
                double init_a, double init_b, unsigned long long seed,
                int otype, float lrs[], int nlr) {
    worker.parameter_init(
        node_name, static_cast<ParamType>(ptype), static_cast<size_t>(len),
        static_cast<size_t>(width), static_cast<InitType>(init_type), init_a,
        init_b, seed, static_cast<OptType>(otype), SArray<float>(lrs, nlr));
}

void Clear(int node_name) {
    worker.clear(node_name);
}

void ClearOnServer(int node_name) {
    worker.clear_on_server(node_name);
}

void SaveParam(int node_name, char *address) {
    worker.parameter_save(node_name, address);
}

void LoadParam(int node_name, char *address) {
    worker.parameter_load(node_name, address);
}

void startRecord(char *dirPath) {
    PSAgent::Get()->startRecord(std::string(dirPath));
}

void getLoads() {
    PSAgent::Get()->getLoads();
}

void ssp_init(Key key, size_t group_size, ssp_version_t tolerance) {
    PSAgent::Get()->SSPInit(key, group_size, tolerance);
}
void ssp_sync(Key key, ssp_version_t version) {
    PSAgent::Get()->SSPSync(key, version);
}

void preduce_get_partner(Key key, int rank, size_t required_worker_num, float wait_time, int* result) {
    PSAgent::Get()->PReduceGetPartner(key, rank, required_worker_num, wait_time, result);
}

void StartServer() {
    auto server = new KVServer(0);
    RegisterExitCallback([server]() { delete server; });
}

int rank() {
    return Postoffice::Get()->my_rank();
}

int nrank() {
    return Postoffice::Get()->num_workers();
}

} // extern "C"
