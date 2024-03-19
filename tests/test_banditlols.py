from __future__ import division, generators, print_function

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import sys
import macarico.util
macarico.util.reseed()

from macarico.base import Example
from macarico.annealing import ExponentialAnnealing, stochastic, EWMA
from macarico.lts.lols import BanditLOLS
from macarico.tasks.sequence_labeler import HammingLoss, HammingLossReference
from macarico.features.sequence import RNN, AttendAt
from macarico.actors.rnn import RNNActor
from macarico.policies.linear import SoftmaxPolicy
from macarico.data.synthetic import make_sequence_mod_data

def test1(learning_method, exploration):
    print()
    print('# testing learning_method=%d exploration=%d' % (learning_method, exploration))
    print()
    n_types = 10
    n_labels = 4
    data = make_sequence_mod_data(100, 6, n_types, n_labels)
    # data = [Example(x, y, n_labels) for x, y in data]

    attention = AttendAt()
    actor = RNNActor([attention], env.n_actions, d_hid=50)

    tRNN = RNNActor([RNN(n_types)], [AttendAt()], n_labels)
    policy = SoftmaxPolicy(tRNN, n_labels)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    p_rollin_ref  = stochastic(ExponentialAnnealing(0.9))
    p_rollout_ref = stochastic(ExponentialAnnealing(0.99999))

    macarico.util.trainloop(
        training_data   = data[:len(data)//2],
        dev_data        = data[len(data)//2:],
        policy          = policy,
        Learner         = lambda: BanditLOLS(HammingLossReference(),
                                             policy,
                                             p_rollin_ref,
                                             p_rollout_ref,
                                             learning_method,  # LEARN_IPS, LEARN_DR, LEARN_BIASED
                                             exploration,
                                             ),
        losses          = HammingLoss(),
        optimizer       = optimizer,
        run_per_epoch   = [p_rollin_ref.step, p_rollout_ref.step],
        train_eval_skip = 10,
    )

if __name__ == '__main__':
    for learning_method in [BanditLOLS.LEARN_BIASED, BanditLOLS.LEARN_IPS, BanditLOLS.LEARN_DR]:
        for exploration in [BanditLOLS.EXPLORE_UNIFORM, BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOLTZMANN_BIASED]:
            test1(learning_method, exploration)


