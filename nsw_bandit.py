#!/usr/bin/python3
import numpy as np
import cvxpy as cp
from solvers import *
from utils import *

class NSW_Bandit:
    def __init__(self, num_agents, num_arms):
        self.k = num_arms
        self.n = num_agents
        self.mu_matrix = None
        self.opt_p = None
        self.opt_nsw = None

    def set_default_mu(self):
        """ By default, sample each mu uniformly between 0 and 1
        """
        self.mu_matrix = np.random.uniform(size=(self.n, self.k))

    def set_mu_matrix(self, mu_matrix):
        assert(mu_matrix.shape == (self.n, self.k))
        self.mu_matrix = mu_matrix

    def get_opt_p(self, cvx=True):
        if cvx:
            self.opt_p = normalize_p(solve_cvx(self.mu_matrix, self.k, self.n))
        else:
            self.opt_p = normalize_p(solve_torch(self.mu_matrix, self.k, self.n))
        return self.opt_p

    def get_nsw(self, p):
        total = 1
        for i in range(self.n):
            total *= np.matmul(p, self.mu_matrix[i, :])
        return total

    def get_opt_nsw(self):
        """ This one is used for cvxpy - need to do sum of log
        """
        if self.opt_nsw is None:
            if self.opt_p is None:
                self.get_opt_p()
            self.opt_nsw = self.get_nsw(self.opt_p)
        return self.opt_nsw

    def get_sample_p(self, p_):
        """ Right now, we are going to do this sample based. That is, we are going to sample an arm
        from distribution p, pull that arm, and return the corresponding welfare vector. 

        In it also possible to do a number of simulations over the arm pulls so as to get the
        expected welfare vector
        """
        assert(np.sum(p_)-1 <= 0.001 and len(p_)==self.k)
        assert(self.mu_matrix is not None)
        p = normalize_p(p_)
        arm = np.random.choice(self.k, p=p)
        return arm, self.get_sample_arm(arm) 
        
    def get_sample_arm(self, arm):
        """ Given an arm, pull that arm and record the regret 
        """
        mean_reward = self.mu_matrix[:,arm]
        rewards = np.zeros(self.n)
        for i in range(self.n):
            rewards[i] = np.random.choice(2, p=[1-mean_reward[i], mean_reward[i]])
        return rewards


def unit_test():
    bandit_instance = NSW_Bandit(6, 3)
    bandit_instance.set_default_mu()
    p_opt = bandit_instance.get_opt_p(cvx=True)
    nsw = bandit_instance.get_nsw(p_opt)
    print(p_opt, nsw)
    

if __name__ == "__main__":
    unit_test()
