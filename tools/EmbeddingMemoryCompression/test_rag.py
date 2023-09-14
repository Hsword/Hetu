import numpy as np
from methods.scheduler.deduplication import Deduplicator
from methods.scheduler.quantize import Quantizer
from methods.scheduler.deeplight import Pruner
from methods.scheduler.dpq import ProductQuantizer
from methods.scheduler.mgqe import MagnitudeProductQuantizer
from methods.scheduler.md import SVDDecomposer
from methods.scheduler.md import MagnitudeSVDDecomposer
from methods.scheduler.tensortrain import TTDecomposer


def main():
    test_dedup()
    test_quant()
    test_prune()
    test_pquant()
    test_magpq()
    test_svdde()
    test_magsvd()
    test_ttde()


def test_dedup():
    ori_size = (1234567, 768)
    compress_rate = 0.7
    block_cap = (10000, 100)
    block_num = tuple((o - 1) // b + 1 for o, b in zip(ori_size, block_cap))
    embedding = np.random.normal(
        scale=0.01, size=ori_size).astype(dtype=np.float32)
    compressed_embedding, dup_map = Deduplicator.compress(
        embedding, compress_rate, block_cap)
    print('comp', compressed_embedding.shape, len(dup_map), block_num)
    reconstruct = Deduplicator.decompress(
        compressed_embedding, dup_map, ori_size, block_cap)
    batch_ids = np.random.randint(0, ori_size[0], size=(5))
    reconstruct_batch = Deduplicator.decompress_batch(
        compressed_embedding, batch_ids, dup_map, ori_size, block_cap)

    # testing
    for i in range(block_num[0]):
        for j in range(block_num[1]):
            start_x = i * block_cap[0]
            ending_x = start_x + block_cap[0]
            start_y = j * block_cap[1]
            ending_y = start_y + block_cap[1]
            a = reconstruct[start_x:ending_x, start_y:ending_y]
            b = compressed_embedding[dup_map[i*block_num[1]+j]
                                     ].reshape(block_cap)[:a.shape[0], :a.shape[1]]
            assert np.all(a == b)
    for i, idx in enumerate(batch_ids):
        assert np.all(reconstruct[idx] == reconstruct_batch[i])


def test_quant():
    ori_size = (1234567, 768)
    digit = 8
    middle = 0
    scale = 0.01
    embedding = np.random.normal(
        scale=0.01, size=ori_size).astype(dtype=np.float32)
    compressed_embedding = Quantizer.compress(embedding, digit, middle, scale)
    print(compressed_embedding.shape, embedding.shape, compressed_embedding.dtype)
    reconstruct = Quantizer.decompress(
        compressed_embedding, middle, scale)
    batch_ids = np.random.randint(0, ori_size[0], size=(10,))
    reconstruct_batch = Quantizer.decompress_batch(
        compressed_embedding, batch_ids, middle, scale)

    # testing
    assert np.all((reconstruct - middle) / scale ==
                  compressed_embedding.astype(np.float32))
    for i, idx in enumerate(batch_ids):
        assert np.all(reconstruct[idx] == reconstruct_batch[i])


def test_prune():
    ori_size = (12345, 768)
    compress_rate = 0.5
    # compress_rate = 0.1
    embedding = np.random.normal(
        scale=0.1, size=ori_size).astype(dtype=np.float32)
    compressed_embedding, form = Pruner.compress(embedding, compress_rate)
    print(compressed_embedding.shape, embedding.shape, compressed_embedding.nnz)
    reconstruct = Pruner.decompress(compressed_embedding)
    batch_ids = np.random.randint(0, ori_size[0], size=(10,))
    reconstruct_batch = Pruner.decompress_batch(
        compressed_embedding, batch_ids, form)

    # testing
    assert np.all(reconstruct == compressed_embedding.todense())
    for i, idx in enumerate(batch_ids):
        assert np.all(reconstruct[idx] == reconstruct_batch[i])


def test_pquant():
    ori_size = (12345, 768)
    subvector_num = 32
    subvector_bits = 8
    embedding = np.random.normal(
        scale=0.01, size=ori_size).astype(dtype=np.float32)
    compressed_embedding = ProductQuantizer.compress(
        embedding, subvector_num, subvector_bits)
    reconstruct = ProductQuantizer.decompress(compressed_embedding)
    batch_ids = np.random.randint(0, ori_size[0], size=(10,))
    reconstruct_batch = ProductQuantizer.decompress_batch(
        compressed_embedding, batch_ids)

    # testing
    for i, idx in enumerate(batch_ids):
        assert np.all(reconstruct[idx] == reconstruct_batch[i])


def test_magpq():
    ori_size = (123456, 768)
    subvector_num = 32
    grouped_subvector_bits = (9, 8)
    embedding = np.random.normal(
        scale=0.01, size=ori_size).astype(dtype=np.float32)
    compressed_embedding, remap = MagnitudeProductQuantizer.compress(
        embedding, subvector_num, grouped_subvector_bits)
    reconstruct = MagnitudeProductQuantizer.decompress(
        compressed_embedding, remap)
    batch_ids = np.random.randint(0, ori_size[0], size=(10,))
    reconstruct_batch = MagnitudeProductQuantizer.decompress_batch(
        compressed_embedding, batch_ids, remap)

    # testing
    nembs = [index.ntotal for index in compressed_embedding]
    accum = np.add.accumulate(nembs)
    accum = np.concatenate([(0,), accum])
    import bisect
    for i in range(10):
        indind = bisect.bisect(accum, remap[i]) - 1
        embind = remap[i] - accum[indind]
        assert np.all(
            reconstruct[i] == compressed_embedding[indind].reconstruct(embind.item()))
    for i, idx in enumerate(batch_ids):
        assert np.all(reconstruct[idx] == reconstruct_batch[i])


def test_svdde():
    from time import time
    ori_size = (21015324, 768)
    compress_rate = 0.1
    embedding = np.random.normal(
        scale=0.01, size=ori_size).astype(dtype=np.float32)
    start = time()
    compressed_embedding, projection_matrix = SVDDecomposer.compress(
        embedding, compress_rate)
    ending = time()
    print('time:', ending - start)
    reconstruct = SVDDecomposer.decompress(
        compressed_embedding, projection_matrix)
    batch_ids = np.random.randint(0, ori_size[0], size=(10,))
    reconstruct_batch = SVDDecomposer.decompress_batch(
        compressed_embedding, batch_ids, projection_matrix)

    # testing
    for i, idx in enumerate(batch_ids):
        assert np.all(reconstruct[idx] == reconstruct_batch[i])


def test_magsvd():
    from time import time
    ori_size = (21015, 768)
    compress_rate = 0.1
    embedding = np.random.normal(
        scale=0.01, size=ori_size).astype(dtype=np.float32)
    start = time()
    compressed_embedding, remap = MagnitudeSVDDecomposer.compress(
        embedding, compress_rate, 2)
    ending = time()
    print('time:', ending - start)
    reconstruct = MagnitudeSVDDecomposer.decompress(
        compressed_embedding, remap)
    batch_ids = np.random.randint(0, ori_size[0], size=(10,))
    reconstruct_batch = MagnitudeSVDDecomposer.decompress_batch(
        compressed_embedding, batch_ids, remap)

    # testing
    nembs = [emb.shape[0] for emb, _ in compressed_embedding]
    accum = np.add.accumulate(nembs)
    accum = np.concatenate([(0,), accum])
    import bisect
    for i in range(10):
        indind = bisect.bisect(accum, remap[i]) - 1
        embind = remap[i] - accum[indind]
        emb, proj = compressed_embedding[indind]
        np.testing.assert_allclose(
            reconstruct[i], emb[embind] @ proj, atol=1e-8)
    for i, idx in enumerate(batch_ids):
        np.testing.assert_allclose(
            reconstruct[idx], reconstruct_batch[i], atol=1e-8)


def test_ttde():
    ori_size = (210153, 768)
    compress_rate = 0.1
    embedding = np.random.normal(
        scale=0.01, size=ori_size).astype(dtype=np.float32)
    compressed_embedding = TTDecomposer.compress(embedding, compress_rate)
    reconstruct = TTDecomposer.decompress(compressed_embedding, ori_size)
    batch_ids = np.random.randint(0, ori_size[0], size=(10,))
    reconstruct_batch = TTDecomposer.decompress_batch(
        compressed_embedding, batch_ids, ori_size[1])

    # testing
    for i, idx in enumerate(batch_ids):
        np.testing.assert_allclose(
            reconstruct[idx], reconstruct_batch[i], atol=1e-8)


if __name__ == '__main__':
    main()
