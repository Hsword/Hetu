#include "binding.h"

#include "lru_cache.h"
#include "lfu_cache.h"
#include "lfuopt_cache.h"
#include "hetu_client.h"

using namespace hetu;

#include "common/thread_pool.h"

PYBIND11_MODULE(hetu_cache, m) {
    m.doc() = "hetu cache C++ plugin"; // optional module docstring

    // Used for Thread pool
    py::class_<wait_t>(m, "_waittype").def("wait", [](const wait_t &w) {
        py::gil_scoped_release release;
        w.wait();
    });

    py::class_<Embedding, EmbeddingPT>(m, "Embedding")
        .def(py::init(&makeEmbedding))
        .def("mean", &Embedding::mean)
        .def("var", &Embedding::var)
        .def("__repr__", &Embedding::__repr__)
        .def_property_readonly("data", &Embedding::PyAPI_data)
        .def_property_readonly("grad", &Embedding::PyAPI_grad)
        .def_property_readonly("key", &Embedding::key)
        .def_property("version", &Embedding::getVersion,
                      &Embedding::setVersion);

    py::class_<CacheBase>(m, "CacheBase")
        .def_property_readonly("limit", &CacheBase::getLimit)
        .def_property_readonly("width", &CacheBase::getWidth)
        .def_property_readonly("perf", &CacheBase::getPerf)
        .def_property("pull_bound", &CacheBase::getPullBound,
                      &CacheBase::setPullBound)
        .def_property("push_bound", &CacheBase::getPushBound,
                      &CacheBase::setPushBound)
        .def_property("perf_enabled", &CacheBase::getPerfEnabled,
                      &CacheBase::setPerfEnabled)
        .def("bypass", &CacheBase::bypass)
        .def("undo_bypass", &CacheBase::undoBypass)
        .def("embedding_lookup", &CacheBase::embeddingLookup)
        .def("embedding_update", &CacheBase::embeddingUpdate)
        .def("embedding_lookup_raw", &CacheBase::embeddingLookupRaw)
        .def("embedding_update_raw", &CacheBase::embeddingUpdateRaw)
        .def("embedding_push_pull_raw", &CacheBase::embeddingPushPullRaw)
        .def("__repr__", &CacheBase::__repr__);

    py::class_<LRUCache, CacheBase>(m, "LRUCache")
        .def(py::init<size_t, size_t, size_t, int>())
        .def("count", &LRUCache::count)
        .def("lookup", &LRUCache::lookup)
        .def("insert", &LRUCache::insert)
        .def("size", &LRUCache::size)
        .def("keys", &LRUCache::PyAPI_keys);

    py::class_<LFUCache, CacheBase>(m, "LFUCache")
        .def(py::init<size_t, size_t, size_t, int>())
        .def("count", &LFUCache::count)
        .def("lookup", &LFUCache::lookup)
        .def("insert", &LFUCache::insert)
        .def("size", &LFUCache::size)
        .def("keys", &LFUCache::PyAPI_keys);

    py::class_<LFUOptCache, CacheBase>(m, "LFUOptCache")
        .def(py::init<size_t, size_t, size_t, int>())
        .def("count", &LFUOptCache::count)
        .def("lookup", &LFUOptCache::lookup)
        .def("insert", &LFUOptCache::insert)
        .def("size", &LFUOptCache::size)
        .def("keys", &LFUOptCache::PyAPI_keys);

    m.def("debug", ps::debug);
} // PYBIND11_MODULE
