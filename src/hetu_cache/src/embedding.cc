#include "embedding.h"
namespace hetu {

EmbeddingPT makeEmbedding(cache_key_t k, version_t version,
                          py::array_t<embed_t> val) {
    assert(val.ndim() == 1);
    PYTHON_CHECK_ARRAY(val);
    auto res = make_shared<Embedding>(k, val.data(), val.shape(0));
    res->setVersion(version);
    return res;
}

} // namespace hetu
