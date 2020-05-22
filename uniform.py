import numpy as np
from solvers import *
from nsw_bandit import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
import sys, csv

NUM_CORES = 20

class Uniform:
    def __init__(self, bandit_instance, c, T=1000, num_sims=100):
        self.bandit_instance = bandit_instance
        self.num_sims = num_sims
        self.c = c
        self.T = T
        self.n, self.k = bandit_instance.n, bandit_instance.k
        self.L= int(np.floor( self.c*np.power(self.T*self.n/self.k, 2/3)*\
                (np.power(2*np.log(self.n*self.T*self.k), 1/3))))
        self.opt_p = self.bandit_instance.get_opt_p(cvx=True)
        self.opt_NSW = self.bandit_instance.get_opt_nsw()
        self.best_p = np.ones((self.num_sims, self.k))

    def run_sim(self, sim_id):
        rewards = None
        regret=None
        empirical_rewards = np.ones((self.n,self.k)) * 0.0001
        for l in range(1, self.L+1):
            rewards = self.bandit_instance.get_sample_all_arms()
            empirical_rewards = empirical_rewards + rewards
        empirical_rewards=empirical_rewards/self.L
        best_p = solve_cvx(empirical_rewards, self.k, self.n)
        if best_p is not None:
            nsw = self.bandit_instance.get_nsw(best_p)
            regret = self.opt_NSW - nsw
        else:
            print("CVX failed (OPT p not found) on sim ")
        return  regret

    def run(self):
        self.mean_regrets = np.zeros(self.T)
        self.std_regrets = np.zeros(self.T)
        next=0;
        for arm in range(self.k):
            arm_p = np.zeros(self.k)
            arm_p[arm] = 1
            nsw = self.bandit_instance.get_nsw(arm_p)
            regret = self.opt_NSW - nsw
            self.mean_regrets[next:next+self.L]=regret
            next=next+self.L


        regrets = []
        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(self.run_sim, item) for item in  range(1,self.num_sims+1)]
        concurrent.futures.wait(futures)
        for future in futures:
            assert(future.done())
            out_tup = future.result()
            regret = out_tup
            regrets.append(regret)
        mean_regret = np.mean(regrets)
        std_regret = np.sqrt(np.var(regrets))
        self.mean_regrets[self.L*self.k:self.T] = mean_regret
        self.std_regrets[self.L*self.k:self.T] = std_regret
        return self.mean_regrets, self.std_regrets

def main():
    c, num_sims, T = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    n, k = int(sys.argv[4]), int(sys.argv[5])
    print("L;",L)
    bandit_instance = NSW_Bandit(n, k)
    mu_instance = load_i_instance_nk(n,k,0)
    bandit_instance.set_mu_matrix(mu_instance)
    print(bandit_instance.set_mu_matrix)
    filename = "uniform"+str(n)+"_k"+str(k)+"_sim"+\
            str(num_sims)+"_T"+str(T)+"_c"+str(c).replace(".", "")+".csv"
    print("Results wil be saved to: ", filename)
    uniform = Uniform(bandit_instance, L, T=T, num_sims=num_sims)
    mean_regrets, std_regrets =uniform.run()


    cumulative_regrets = []

    csv_file = open(filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')

    for t in range(1, T):
        cumulative_regrets.append(mean_regrets[1:t+1])
        row = [str(t)] +  [str(mean_regrets[t])] + [str(np.sum(mean_regrets[1:t+1]))] + [str(std_regrets[t])]
        csv_writer.writerow(row)
        csv_file.flush()

if __name__ == "__main__":
    main()
