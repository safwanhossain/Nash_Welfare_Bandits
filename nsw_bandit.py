import numpy as np
import cvxpy as cp

class NSW_Bandit:
    def __init__(self, num_arms, num_agents):
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
        assert(mu_matrix.shape == (n, k))
        self.mu_matrix = mu_matrix

    def get_optimal_p(self):
        if self.opt_p is None:
            p = cp.Variable(self.k)
            objective = cp.Maximize(cp.log(self.get_nsw(p, np_out=False)))
            constraints = [cp.sum(p) == 1, p >= 0]
            problem = cp.problem(objective, constraints)
            result = problem.solve()
            print(result)
            self.opt_p = p.value
        return self.opt_p

    def get_nsw(self, p, np_out=True):
        total = 0
        for i in range(self.n):
            if np_out == True:
                total *= np.dot(p, self.mu_matrix[i, :])
            else:
                total *= cp.dot(p, self.mu_matrix[i, :])
        return total

    def get_opt_nsw(self):
        if self.opt_nsw is None:
            if self.opt_p is None:
                self.get_optimal_p()
            self.opt_nsw = get_nsw(self.opt_p)
        return self.opt_nsw

    def get_sample(self, p):
        """ Right now, we are going to do this sample based. That is, we are going to sample an arm
        from distribution p, pull that arm, and return the corresponding welfare vector. 

        In it also possible to do a number of simulations over the arm pulls so as to get the
        expected welfare vector
        """
        assert(np.sum(p) == 1 and len(p) == self.k)
        assert(self.mu_matrix is not None)
        arm = np.random.choice(self.k, p=p)
        mean_reward = self.mu_matrix[:,arm]
        
        rewards = np.zeros(self.n)
        for i in range(self.n):
            rewards[i] = np.random.choice(2, p=[1-mean_reward, mean_reward])
        return rewards


def unit_test():
    bandit_instance = NSW_bandit(6, 3)
    bandit_instance.set_default_mu()
    p_opt = bandit_instance.get_optimal_p()
    print(p_opt)
    nsw = get_nsw(p_opt)
    print(nsw)
    
