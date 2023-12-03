import numpy as np
from time import time
import bisect
from hetu.random import get_np_rand


class Compressor:
    @staticmethod
    def compress(embedding):
        raise NotImplementedError

    @staticmethod
    def decompress(compressed_embedding):
        raise NotImplementedError

    @staticmethod
    def decompress_batch(compressed_embedding, batch_ids):
        raise NotImplementedError

    @staticmethod
    def split_by_magnitude(embedding, ngroup):
        assert ngroup > 1
        nemb = embedding.shape[0]
        magnitude = np.linalg.norm(embedding, axis=1)
        orders = np.argsort(-magnitude)
        nmag_per_group = sum(magnitude) / ngroup
        accum = np.add.accumulate(magnitude[orders])
        accum = np.concatenate([(0,), accum])
        remap = np.full((nemb,), fill_value=-1, dtype=np.int32)
        return_embs = []
        start_index = 0
        for g in range(1, ngroup):
            cur_value = g * nmag_per_group
            loc = bisect.bisect(accum, cur_value, lo=start_index)
            ending_index = loc
            cur_idx = orders[start_index:ending_index]
            cur_emb = embedding[cur_idx]
            return_embs.append(cur_emb)
            remap[cur_idx] = np.arange(start_index, ending_index)
            start_index = ending_index
        cur_idx = orders[start_index:]
        cur_emb = embedding[cur_idx]
        return_embs.append(cur_emb)
        remap[cur_idx] = np.arange(start_index, len(remap))
        assert np.all(remap >= 0)
        return return_embs, remap

    @staticmethod
    def timing_wrapper(func, *args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        ending = time()
        print('Time usage:', ending - start)
        return result

    @classmethod
    def timing_decompress_batch(cls, compressed_embedding, nemb, batch_size, *args, **kwargs):
        nprs = get_np_rand(1)
        batch_ids = [nprs.randint(0, nemb, size=(batch_size,), dtype=np.int32) for _ in range(100)]
        for i in range(5):
            cls.decompress_batch(compressed_embedding, batch_ids[i], *args, **kwargs)
        start = time()
        for i in range(5, 100):
            cls.decompress_batch(compressed_embedding, batch_ids[i], *args, **kwargs)
        ending = time()
        print(f'Time usage for batch ({batch_size}): {(ending - start) / 95}')
