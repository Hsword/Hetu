
class FixedScheduler(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self):
        return self.learning_rate

    def get(self):
        return self.learning_rate


class StepScheduler(FixedScheduler):
    def __init__(self, learning_rate, step_size, gamma=0.1, ending=1e-8):
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.ending = ending
        assert step_size > 0 and gamma > 0
        assert learning_rate > ending, \
            'Initial learning rate should be larger than ending learning rate; got {}, {}'.format(
                learning_rate, ending)
        assert ending >= 0
        self.reach_end = False
        self.cur_step = 0

    def step(self):
        if self.reach_end:
            return self.ending
        if self.cur_step > 0 and self.cur_step % self.step_size == 0:
            self.learning_rate *= self.gamma
            if self.learning_rate <= self.ending:
                self.reach_end = True
                self.learning_rate = self.ending
        self.cur_step += 1
        return self.learning_rate


class MultiStepScheduler(FixedScheduler):
    def __init__(self, learning_rate, milestones, gamma=0.1):
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.gamma = gamma
        assert milestones[0] > 0
        for i in range(1, len(milestones)):
            assert milestones[i] > milestones[i-1]
        self.cur_step = 0

    def step(self):
        if not self.milestones:
            return self.learning_rate
        if self.cur_step == self.milestones[0]:
            self.milestones = self.milestones[1:]
            self.learning_rate *= self.gamma
        self.cur_step += 1
        return self.learning_rate


class ExponentialScheduler(FixedScheduler):
    def __init__(self, learning_rate, gamma=0.9, ending=1e-8):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.ending = ending
        assert gamma > 0
        assert learning_rate > ending, \
            'Initial learning rate should be larger than ending learning rate; got {}, {}'.format(
                learning_rate, ending)
        assert ending >= 0
        self.cur_step = 0
        self.reach_end = False

    def step(self):
        if self.reach_end:
            return self.learning_rate
        prev_lr = self.learning_rate
        self.learning_rate *= self.gamma
        if self.learning_rate <= self.ending:
            self.reach_end = True
            self.learning_rate = self.ending
        return prev_lr


class ReduceOnPlateauScheduler(FixedScheduler):
    def __init__(self, learning_rate, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, ending=1e-8):
        self.learning_rate = learning_rate
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.ending = ending
        assert learning_rate > ending, \
            'Initial learning rate should be larger than ending learning rate; got {}, {}'.format(
                learning_rate, ending)
        assert mode in ('min', 'max')
        assert threshold_mode in ('rel', 'abs')
        assert factor > 0
        assert patience >= 0
        assert threshold >= 0
        assert cooldown >= 0
        assert ending >= 0
        self.step_in_cooldown = -1
        self.patience_step = 0
        self.last_value = None
        self.reach_end = False

    def step(self, value):
        if self.reach_end:
            return self.learning_rate
        if self.step_in_cooldown >= 0:
            self.step_in_cooldown -= 1
            self.last_value = eval(self.mode)(self.last_value, value)
            return self.learning_rate
        if self.last_value is None:
            self.last_value = value
            return self.learning_rate
        if self.mode == 'min':
            larger = self.last_value
            smaller = value
        else:
            larger = value
            smaller = self.last_value
        should_change = False
        if self.threshold_mode == 'rel':
            should_change = larger < (1 - self.threshold) * smaller
        else:
            should_change = larger < smaller - self.threshold
        if should_change:
            if self.patience_step >= self.patience:
                self.patience_step = 0
                self.learning_rate *= self.factor
                if self.learning_rate <= self.ending:
                    self.learning_rate = self.ending
                    self.reach_end = True
                self.step_in_cooldown += self.cooldown
            else:
                self.patience_step += 1
        else:
            self.patience_step = 0
        self.last_value = eval(self.mode)(self.last_value, value)
        return self.learning_rate
