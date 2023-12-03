import numpy as np
import hetu as ht


def test(indshape=(3, 4), dim=2, indrange=8):
    ctx = ht.gpu(0)
    stream = ht.stream.create_stream_handle(ctx)
    indices = np.random.randint(indrange, size=indshape)
    embedding = np.random.random(size=(indrange, dim))
    htindices = ht.array(indices, dtype=np.int32, ctx=ctx)
    htembedding = ht.array(embedding, ctx=ctx)
    htlookups = ht.empty(indshape+(dim,), ctx=ctx)
    ht.gpu_links.embedding_lookup(htembedding, htindices, htlookups, stream)
    htuniques = ht.empty(indices.shape, dtype=np.int32, ctx=ctx)
    ind_size = np.prod(indshape).item()
    htoffsets = ht.empty((2*ind_size+2,), dtype=np.int32, ctx=ctx)
    ws_size = ht.gpu_links.get_unique_workspace_size(ind_size)
    all_ws_size = (ws_size + 3) // 4
    workspace = ht.empty((all_ws_size, ), ctx=ctx)
    ht.gpu_links.unique_indices(
        htindices, htuniques, htoffsets, workspace, ws_size, 32, stream)
    htdeduplookups = ht.empty(htlookups.shape, ctx=ctx)
    ht.gpu_links.deduplicate_lookup(
        htlookups, htoffsets, htdeduplookups, stream)
    htnewlookups = ht.empty(htlookups.shape, ctx=ctx)
    ht.gpu_links.reorder_into_lookup(
        htoffsets, htdeduplookups, htnewlookups, stream)
    stream.sync()
    lookups = htlookups.asnumpy()
    newlookups = htnewlookups.asnumpy()
    # print(lookups)
    # print(newlookups)
    # np.testing.assert_allclose(lookups, htnewlookups.asnumpy())
    assert np.all(lookups == newlookups)


test()
test(indshape=(17, 31), dim=19, indrange=61)
