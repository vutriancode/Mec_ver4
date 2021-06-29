from __future__ import division
import numpy as np

from rl.util import *
from rl.policy import EpsGreedyQPolicy


class CustomerEpsGreedyQPolicy(EpsGreedyQPolicy):
    def __init__(self, eps=.1,eps_end=0.01):
        super(CustomerEpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.eps_end = eps_end
        self.lan = 0

    def select_action(self, q_values):
        self.lan +=1
        if self.lan%20000==0 and self.eps >= 2*self.eps_end:
            self.eps =self.eps/2.0

        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(CustomerEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
