import numpy as np
from solvers import *
from nsw_bandit import *
from tqdm import tqdm
#import matplotlib.pyplot as plt
import concurrent.futures
import sys, csv

NUM_CORES = 25

class UCB:
    def __init__(self, bandit_instance, c, num_sims=100, T=1000):
        self.bandit_instance = bandit_instance
        self.num_sims = num_sims
        self.T, self.c = T, c
        self.opt_p = self.bandit_instance.get_opt_p(cvx=True)
        self.opt_NSW = self.bandit_instance.get_opt_nsw()
        self.n, self.k = bandit_instance.n, bandit_instance.k
        print("Optimal NSW: ", self.opt_NSW)
        print("Optimal p:", self.opt_p)

        self.empirical_rewards = np.ones((num_sims,self.n,self.k)) * 0.0001
        self.num_samples = np.ones((num_sims, self.k))
        self.last_p = np.random.uniform(size=(num_sims, self.k))
        for sim in range(self.num_sims):
            self.last_p[sim] = np.clip(self.last_p[sim]/np.sum(self.last_p[sim]), 0, 1)

    def initial_run_sim(self,sim_id):
        rewards=[]
        rewards=self.bandit_instance.get_sample_all_arms()
        #best_p = solve_cvx(rewards,self. k, self.n)
        #best_p = solve_torch_ucb(rewards, num_samples, self.k, self.n,self.k+1)
        #best_p = solve_cvx_ucb(rewards, num_samples, self.k, self.n, self.k+1)
        #best_p = solve_torch_ucb(rewards, self.k, self.n)
        #print("ID:",id, "Rewards:", rewards, "best p:", best_p)
        return sim_id,rewards#, best_p

    def run_sim(self, tup,t):
        sim_id, last_p, empirical_rewards, num_samples= tup
        regret = None
        arm, rewards = self.bandit_instance.get_sample_p(last_p)
        nsw = self.bandit_instance.get_nsw(last_p)
        regret = self.opt_NSW - nsw
        empirical_rewards[:,arm] = \
            (empirical_rewards[:,arm]*num_samples[arm] + rewards)/(num_samples[arm]+1)
        best_p = solve_torch_ucb(empirical_rewards, num_samples,self.k, self.n, t)
        #best_p = solve_cvx(empirical_rewards, self.k, self.n)
        # best_p = solve_cvx_ucb(empirical_rewards, num_samples, self.k, self.n, t)
        #print("CVX",best_p_1)
        #print("Torch", best_p)
        #best_p = solve_torch_ucb(empirical_rewards, self.k,self.n)
        #best_p=solve_cvx(empirical_rewards, self.k, self.n)
        return sim_id, arm, rewards, best_p, regret

    def run(self):
        self.mean_regrets = np.zeros(self.T)
        self.std_regrets = np.zeros(self.T)
        self.explore_ratio = np.zeros(self.T)
        """
        Initially Pull each arm once
        """
        """
        parallel_inputs = []
        for sim in range(self.num_sims):
            parallel_inputs.append(sim)
        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(self.initial_run_sim, item) for item in parallel_inputs]
        concurrent.futures.wait(futures)
        for future in futures:
            assert(future.done())
            out_tup = future.result()
            sim_id,  rewards= out_tup
            #print("ID:",sim_id,"\n Rewrda", rewards)
            self.empirical_rewards[sim_id,:,:] =   rewards
            #self.last_p[sim_id,:]=best_p
        for arm in range(self.k):
            arm_p = np.zeros(self.k)
            arm_p[arm] = 1
            nsw = self.bandit_instance.get_nsw(arm_p)
            regret = self.opt_NSW - nsw
            print(regret)
            self.mean_regrets[arm]=regret
            self.std_regrets[arm] = 0
            """
        """
        End Initially Pull each arm once
        """

        for t in tqdm(range(self.k, self.T)):
            t_regrets = []

            parallel_inputs = []
            for sim in range(self.num_sims):
                tup = (sim,  self.last_p[sim].copy(), \
                self.empirical_rewards[sim].copy(), self.num_samples[sim].copy())
                parallel_inputs.append(tup)

            executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
            futures = [executor.submit(self.run_sim, item,t) for item in parallel_inputs]
            concurrent.futures.wait(futures)

            t_regrets = []
            for future in futures:
                assert(future.done())
                out_tup = future.result()
                sim_id, arm, rewards, best_p, regret = out_tup

                self.empirical_rewards[sim_id,:,arm] = \
                    (self.empirical_rewards[sim_id,:,arm]*self.num_samples[sim_id,arm] + rewards)/(self.num_samples[sim_id,arm]+1)
                self.num_samples[sim_id,arm] += 1

                if best_p is not None:
                    self.last_p[sim_id] = best_p
                    t_regrets.append(regret)
                else:
                    print("Torch failed (OPT p not found) on sim ", sim_id)
                #print(t, "Opt P", self.opt_p)
                #print(t, "Curr P", best_p)
                #print("Number of samples", self.num_samples[sim_id])
                #print("Empirical Rewards", self.empirical_rewards[sim])
            mean_regret = np.mean(t_regrets)
            #print("Mean regret", mean_regret)
            std_regret = np.sqrt(np.var(t_regrets))
            self.mean_regrets[t] = mean_regret
            self.std_regrets[t] = std_regret

        return self.mean_regrets, self.std_regrets



def main():
    for i in range(10):
        c, num_sims, T = 0.50, 50, 4000
        n, k = 5, 8
        bandit_instance = NSW_Bandit(n, k)
        mu_instance = load_i_instance_nk(n,k,i)
        print(mu_instance)

        bandit_instance.set_mu_matrix(mu_instance)
        filename = "UCB"+str(n)+"_k"+str(k)+"_instance"+str(i)+"_sim"+\
            str(num_sims)+"_T"+str(T)+"_c"+str(c).replace(".", "")+".csv"
        print("Results wil be saved to: ", filename)
        ucb = UCB(bandit_instance, c, T=T, num_sims=num_sims)
        mean_regrets, std_regrets = ucb.run()
        print(mean_regrets)
        cumulative_regrets = []

        csv_file = open(filename, mode='w')
        csv_writer = csv.writer(csv_file, delimiter=',')
        for t in range(1, T):
            cumulative_regrets.append(mean_regrets[1:t+1])
            row = [str(t)] + [str(mean_regrets[t])] + [str(np.sum(mean_regrets[1:t+1]))] + [str(std_regrets[t])]
            csv_writer.writerow(row)
            csv_file.flush()

if __name__ == "__main__":
    main()
