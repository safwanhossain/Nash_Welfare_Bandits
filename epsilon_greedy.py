#!/usr/bin/python3
import numpy as np
from solvers import *
from nsw_bandit import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures 
import sys, csv

NUM_CORES = 10

class epsilon_greedy:
    def __init__(self, bandit_instance, c, num_sims=100, T=1000):
        self.bandit_instance = bandit_instance
        self.bandit_instance.set_default_mu()
        self.num_sims = num_sims
        self.T, self.c = T, c
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
            self.last_p[sim] = np.clip(self.last_p[sim]/np.sum(self.last_p[sim]), 0, 1)


    def run_sim(self, tup):
        sim_id, flip, arm, last_p, empirical_rewards, num_samples, k, n = tup
        if flip == 1:           # explore
            rewards = self.bandit_instance.get_sample_arm(arm)
        else:                   # exploit
            arm, rewards = self.bandit_instance.get_sample_p(last_p)
        
        empirical_rewards[:,arm] = \
            (empirical_rewards[:,arm]*num_samples[arm] + rewards)/(num_samples[arm]+1)
        num_samples[arm] += 1
        regret, best_p = None, solve_cvx(empirical_rewards, k, n)
        if best_p is not None:
            arm_p = np.zeros(self.k)
            arm_p[arm] = 1
            nsw = self.bandit_instance.get_nsw(arm_p)
            regret = self.opt_NSW - nsw
        return sim_id, best_p, regret


    def run(self):
        self.mean_regrets = np.ones(self.T)
        self.std_regrets = np.ones(self.T)
        self.explore_ratio = np.ones(self.T)

        for t in tqdm(range(1, self.T)):
            curr_eps_t = self.c * \
                    np.power(t, -1/3)*(np.power(2*self.n, 2/3))*(np.power(3*self.k*np.log(t), 1/3))
            curr_eps_t = np.clip(curr_eps_t, 0, 1)
            self.eps_t[t] = curr_eps_t
            flips = np.random.choice(2, self.num_sims, p=[1-curr_eps_t, curr_eps_t])
            self.explore_ratio[t] = sum(flips)/len(flips)
            explore_arms = np.random.choice([j for j in range(self.k)], self.num_sims)
            t_regrets = []
           
            parallel_inputs = []
            for sim in range(self.num_sims):
                tup = (sim, flips[sim], explore_arms[sim], self.last_p[sim], \
                        self.empirical_rewards[sim], self.num_samples[sim], self.k, self.n)
                parallel_inputs.append(tup)
            
            executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
            futures = [executor.submit(self.run_sim, item) for item in parallel_inputs]
            concurrent.futures.wait(futures)

            t_regrets = []
            for future in futures:
                assert(future.done())
                out_tup = future.result()
                sim_id, best_p, regret = out_tup
                if best_p is not None:
                    self.last_p[sim_id] = best_p
                    t_regrets.append(regret)
                else:
                    print("CVX failed (OPT p not found) on sim ", sim_id)
            
            mean_regret = np.mean(t_regrets)
            std_regret = np.sqrt(np.var(t_regrets))
            self.mean_regrets[t] = mean_regret
            self.std_regrets[t] = std_regret
        
        return self.eps_t, self.explore_ratio, self.mean_regrets, self.std_regrets
                
def main():
    c, T = 0.1, 1000
    bandit_instance = NSW_Bandit(6, 1)
    eps_greedy = epsilon_greedy(bandit_instance, c, T=T)
    eps_t, explore_ratio, mean_regrets, std_regrets = eps_greedy.run()
    cumulative_regrets = []

    filename = "eps_greedy_sim100_T5000_c02.csv"
    csv_file = open(filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    for t in range(1, T):
        cumulative_regrets.append(mean_regrets[1:t+1])
        row = [str(t)] + [str(eps_t[t])] + [str(explore_ratio[t])] \
                + [str(mean_regrets[t])] + [str(np.sum(mean_regrets[1:t+1]))] + [str(std_regrets[t])]
        csv_writer.writerow(row)
        csv_file.flush()
       
    #plt.plot(cumulative_regrets)
    #plt.imsave("cum_regrets")
    plt.plot(mean_regrets[1:])
    plt.savefig("mean_regrets")
    
if __name__ == "__main__":
    main()

