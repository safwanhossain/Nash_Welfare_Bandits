import numpy as np
from solvers import *
from nsw_bandit import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
import sys, csv

NUM_CORES = 20

class Uniform:
    def __init__(self, bandit_instance, L,  T=1000, num_sims=100,):
        self.bandit_instance = bandit_instance
        self.num_sims = num_sims
        self.T, self.L = T, L
        self.opt_p = self.bandit_instance.get_opt_p(cvx=True)
        self.opt_NSW = self.bandit_instance.get_opt_nsw()
        self.n, self.k = bandit_instance.n, bandit_instance.k
        self.best_p = np.ones((self.num_sims, self.k))

    def run_sim(self, tup):
        sim_id = tup
        rewards = None
        empirical_rewards = np.ones((self.n,self.k)) * 0.0001
        for l in tqdm(range(1, self.L)):
            print("before")
            rewards = self.bandit_instance.get_sample_all_arms()
            print(rewards)
            empirical_rewards = empirical_rewards + rewards
        empirical_rewards=empirical_rewards/self.L
        best_p = solve_cvx(empirical_rewards, self.k, self.n)
        return sim_id,  best_p

    def run(self):
    #    self.mean_regrets = np.zeros(self.T)
    #    self.std_regrets = np.zeros(self.T)
    #    self.explore_ratio = np.zeros(self.T)
        parallel_inputs = []
        for sim in range(self.num_sims):
            parallel_inputs.append(sim)
            executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
            futures = [executor.submit(self.run_sim, item) for item in parallel_inputs]
            concurrent.futures.wait(futures)
        for future in futures:
            assert(future.done())
            out_tup = future.result()
            sim_id, best_p = out_tup
            if best_p is not None:
                self.best_p[sim_id] = best_p
            else:
                print("CVX failed (OPT p not found) on sim ", sim_id)

    #return self.best_p

def main():
    num_sims, T = 2, 5
    n, k = 1, 5
    L=2
    bandit_instance = NSW_Bandit(n, k)
    mu_instance = load_i_instance_nk(n,k,0)
    bandit_instance.set_mu_matrix(mu_instance)
    print(bandit_instance.set_mu_matrix)
    filename = "uniform"+str(n)+"_k"+str(k)+"_sim"+\
            str(num_sims)+"_T"+str(T).replace(".", "")+".csv"
    print("Results wil be saved to: ", filename)
    uniform = Uniform(bandit_instance, L, T=T, num_sims=num_sims)
    uniform.run()

    """
    cumulative_regrets = []

    csv_file = open(filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')

    for t in range(1, T):
        cumulative_regrets.append(mean_regrets[1:t+1])
        row = [str(t)] + [str(eps_t[t])] + [str(explore_ratio[t])] \
                + [str(mean_regrets[t])] + [str(np.sum(mean_regrets[1:t+1]))] + [str(std_regrets[t])]
        csv_writer.writerow(row)
        csv_file.flush()
    """
if __name__ == "__main__":
    main()
