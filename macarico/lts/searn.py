from __future__ import division, generators, print_function

import numpy as np
import torch
import macarico
from macarico.annealing import stochastic, NoAnnealing
from macarico.util import break_ties_by_policy, argmin

class SEARN(macarico.Learner):
    def __init__(self, policy, reference, p_rollin_ref=stochastic(NoAnnealing(0.5)), weight=stochastic(NoAnnealing(0.5))):
        macarico.Learner.__init__(self)
        self.rollin_ref = p_rollin_ref
        self.policy = policy
        self.policies = [policy]
        self.weights = [weight]
        self.reference = reference
        self.objective = 0.0
        self.disable_ref = False

    def update_policies(self, new_policy, new_weight):
        self.policies.append(new_policy)
        self.weights.append(new_weight)
        self.policy = new_policy
        return

    def forward(self, state):
        ref = break_ties_by_policy(self.reference, self.policy, state, False)
        pol_idx = self.pick_policy() # picks from the distribution of classifiers
        all_policies = [x(state) for x in self.policies]
        pol = all_policies[pol_idx]
        self.objective += self.policy.update(state, ref) # reference computes costs. 'optimal approximation'. policy.update computes losses
        if self.disable_ref:
            return pol
        return ref if self.rollin_ref() else pol

    def pick_policy(self):
        n = len(self.policies)
        beta_bar = self.weights[0].inst.alpha
        probs = [(1 - beta_bar) * beta_bar**(n - s.time) / (1 - beta_bar**n) for s in self.weights]
        # print(probs)
        idx = np.random.choice(range(len(self.policies)), p=probs)
        return idx

    def update_weights(self):
        for w in self.weights:
            w.step()
        return
    
    def get_objective(self, _):
        ret = self.objective
        self.objective = 0.0
        # self.rollin_ref.step()
        return ret