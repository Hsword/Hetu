import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, diminput, dimoutput, dimhidden, nsteps):
        super(RNN, self).__init__()
        self.diminput = diminput
        self.dimoutput = dimoutput
        self.dimhidden = dimhidden
        self.nsteps = nsteps
        self.fc1 = nn.Linear(diminput, dimhidden)
        self.fc2 = nn.Linear(dimhidden*2, dimhidden)
        self.fc3 = nn.Linear(dimhidden, dimoutput)

    def forward(self, x):
        last_state = torch.zeros((x.shape[0], self.dimhidden)).to(x.device)
        for i in range(self.nsteps):
            t = i % self.nsteps
            index = torch.Tensor([idx for idx in range(
                t*self.diminput, (t+1)*self.diminput)]).long().to(x.device)
            cur_x = torch.index_select(x, 1, index)
            h = self.fc1(cur_x)
            s = torch.cat([h, last_state], axis=1)
            s = self.fc2(s)
            last_state = F.relu(s)

        final_state = last_state
        y = self.fc3(final_state)
        return y


def rnn(diminput, dimoutput, dimhidden, nsteps):

    return RNN(diminput, dimoutput, dimhidden, nsteps)
