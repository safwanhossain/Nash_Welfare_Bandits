#!/usr/bin/python3
import numpy as np
from solvers import *
from nsw_bandit import *
from tqdm import tqdm
import matplotlib.pyplot as plt

class epsilon_greedy:
    def __init__(self, bandit_instance, num_sims=100, T=1000):
        self.bandit_instance = bandit_instance
        self.bandit_instance.set_default_mu()
        self.num_sims = num_sims
        self.T = T
        self.opt_NSW = self.bandit_instance.get_opt_nsw()
        self.opt_p = self.bandit_instance.get_opt_p()
        self.cumulative_regret = np.ones(T)*-1
        self.eps_t = np.ones(T)*-1
        self.n, self.k = bandit_instance.n, bandit_instance.k
        print("Optimal NSW: ", self.opt_NSW)

        self.empirical_rewards = np.ones((num_sims,self.n,self.k))
        self.num_samples = np.zeros((num_sims, self.k))
        self.last_p = np.random.uniform(size=(num_sims, self.k))
        for sim in range(self.num_sims):
            self.last_p[sim] = self.last_p[sim]/np.sum(self.last_p[sim])

    def run(self):
        self.mean_regrets = np.ones(self.T)

        for t in tqdm(range(1, self.T)):
            curr_eps_t = np.power(t, -1/3)*(np.power(2*self.n, 2/3))*(np.power(3*self.k*np.log(t), 1/3))
            curr_eps_t = np.clip(curr_eps_t, 0, 1)
            #print("current epsilon", curr_eps_t)
            self.eps_t[t] = curr_eps_t
            flips = np.random.choice(2, self.num_sims, p=[1-curr_eps_t, curr_eps_t])
            explore_arms = np.random.choice([j for j in range(self.k)], self.num_sims)
            t_regrets = np.zeros(self.num_sims)
             
            for sim in range(self.num_sims):
                if flips[sim] == 1:     # explore
                    arm = explore_arms[sim]
                    rewards = self.bandit_instance.get_sample_arm(arm)
                else:                   # exploit
                    arm, rewards = self.bandit_instance.get_sample_p(self.last_p[sim])
                
                self.empirical_rewards[sim,:,arm] = \
                    (self.empirical_rewards[sim,:,arm]*self.num_samples[sim,arm] + rewards)/(self.num_samples[sim,arm]+1)
                self.num_samples[sim,arm] += 1
                
                best_p = solve_cvx(self.empirical_rewards[sim], self.k, self.n)
                if best_p is None:
                    print("Not Found")
                    best_p = self.last_p[sim]
                self.last_p[sim] = best_p
                nsw = self.bandit_instance.get_nsw(best_p)
                regret = self.opt_NSW - nsw
                t_regrets[sim] = regret
            
            mean_regret = np.mean(t_regrets)
            self.mean_regrets[t] = mean_regret

        return self.mean_regrets, self.eps_t
                

def main():
    bandit_instance = NSW_Bandit(6, 3)
    T = 1000
    eps_greedy = epsilon_greedy(bandit_instance, T=T)
    mean_regrets, eps_t = eps_greedy.run()
    t_arr = np.arange(T-1)
    plt.plot(t_arr, mean_regrets[1:])
    plt.savefig("regret.png")
    plt.plot(t_arr, eps_t[1:])
    plt.savefig("eps.png")

if __name__ == "__main__":
    main()

