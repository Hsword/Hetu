import hetu as ht
import random


def test_fixed(learning_rate):
    lr_sched = ht.lr.FixedScheduler(learning_rate)
    for i in range(100):
        assert lr_sched.step() == learning_rate
    print('Fixed scheduler with learning_rate {} pass test.'.format(learning_rate))


def test_step(learning_rate, step_size, gamma):
    lr_sched = ht.lr.StepScheduler(learning_rate, step_size, gamma)
    cur_lr = learning_rate / gamma
    for i in range(10):
        cur_lr *= gamma
        cur_lr = max(cur_lr, 1e-8)
        for j in range(step_size):
            sched_lr = lr_sched.step()
            assert sched_lr == cur_lr, 'Got {} and {} at {}:{}.'.format(
                sched_lr, cur_lr, i, j)
    print('Step scheduler with learning_rate {}, step_size {}, gamma {} pass test.'.format(
        learning_rate, step_size, gamma))


def test_multistep(learning_rate, milestones, gamma):
    lr_sched = ht.lr.MultiStepScheduler(learning_rate, milestones, gamma)
    cur_lr = learning_rate
    cur_step = 0
    for ms in milestones:
        while cur_step < ms:
            assert lr_sched.step() == cur_lr
            cur_step += 1
        cur_lr *= gamma
    for i in range(10):
        assert lr_sched.step() == cur_lr
    print('Multistep scheduler with learning_rate {}, milestones {}, gamma {} pass test.'.format(
        learning_rate, str(milestones), gamma))


def test_exponential(learning_rate, gamma):
    lr_sched = ht.lr.ExponentialScheduler(learning_rate, gamma)
    cur_lr = learning_rate
    cur_step = 0
    for i in range(100):
        assert lr_sched.step() == cur_lr
        cur_lr = max(cur_lr * gamma, 1e-8)
    print('Exponential scheduler with learning_rate {}, gamma {} pass test.'.format(
        learning_rate, gamma))


def test_reduce_on_plateau(learning_rate, mode, factor, patience, threshold, threshold_mode, cooldown):
    lr_sched = ht.lr.ReduceOnPlateauScheduler(
        learning_rate, mode, factor, patience, threshold, threshold_mode, cooldown)
    cur_lr = learning_rate
    results = []
    inputs = []
    for i in range(10):
        inputs.append(random.random())
        results.append(lr_sched.step(inputs[-1]))
    print('Please check manually: lr {}, mode {}, factor {}, patience {}, threshold {} {}, cooldown {}'.format(
        learning_rate, mode, factor, patience, threshold_mode, threshold, cooldown))
    print('Inputs:', inputs)
    print('Results:', results)


test_fixed(0.1)
test_step(0.1, 10, 0.1)
test_step(0.1, 10, 0.5)
test_multistep(0.1, [30, 80], 0.1)
test_multistep(0.1, [5, 7, 11, 13], 0.5)
test_exponential(0.1, 0.99)
test_exponential(0.1, 0.5)
test_reduce_on_plateau(0.1, 'min', 0.1, 1, 0.01, 'rel', 2)
test_reduce_on_plateau(0.1, 'min', 0.1, 2, 0.01, 'abs', 1)
test_reduce_on_plateau(0.1, 'max', 0.1, 3, 0.01, 'rel', 3)
test_reduce_on_plateau(0.1, 'max', 0.1, 2, 0.01, 'abs', 0)
