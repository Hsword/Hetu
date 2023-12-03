from __future__ import absolute_import

from ._base import _LIB, check_call
import ctypes
from numpy.random import RandomState, MT19937, SeedSequence, mtrand


def to_ull(seed):
    if not isinstance(seed, ctypes.c_ulonglong):
        seed = ctypes.c_ulonglong(seed)
    return seed


def set_random_seed(seed):
    seed = to_ull(seed)
    check_call(_LIB.SetRandomSeed(seed))


def reset_seed_seqnum():
    step_seqnum(-get_seed_seqnum())


def get_seed():
    return _LIB.GetSeed()


def get_seed_seqnum():
    return _LIB.GetSeedSeqNum()


def get_seed_status():
    return (get_seed(), get_seed_seqnum())


def step_seqnum(seqnum):
    seqnum = to_ull(seqnum)
    check_call(_LIB.StepSeqNum(seqnum))


def get_np_rand(step=None) -> mtrand.RandomState:
    if step is not None:
        step_seqnum(step)
    return RandomState(MT19937(SeedSequence(get_seed_status())))
