import numpy as np
import sys, csv
from nsw_bandit import *
from epsilon_greedy import *
from uniform import *
from UCB import *
from tqdm import tqdm
from utils import *
    
algos = ["uniform", "eps_greedy", "UCB"]
c_vals_eps = [0.05, 0.1, 0.13, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4]
c_vals_unif = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
n_range = [2,4,6]
k_range = [2,4,6,8,10]
T = 800
num_sims = 20
num_instances = 10

def main():
    algo = sys.argv[1]
    print("Sweeping c values in for ", algo)
    assert algo in algos
    filename = algo + "_grid_search.csv"
    
    csv_file = open(filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(["n", "k", "c", "avg_cum_regret"])

    if algo == "uniform":
        c_vals = c_vals_unif
    elif algo == "eps_greedy":
        c_vals = c_vals_eps

    results_dict = {}
    for n in n_range:
        for k in k_range:
            for c in c_vals:
                print("N:", n, "K:", k, "C:", c)
                cum_regrets = []
                for i in range(num_instances):
                    bandit = NSW_Bandit(n,k)
                    mu_mat = load_i_instance_nk(n,k,i)
                    bandit.set_mu_matrix(mu_mat)
                    if algo == "uniform":
                        uniform = Uniform(bandit, c, T, num_sims) 
                        mean_regrets, _ = uniform.run()
                    elif algo == "eps_greedy":
                        eps_greedy = epsilon_greedy(bandit, c, T=T, num_sims=num_sims)
                        _, _, mean_regrets, _ = eps_greedy.run()
                    elif algo == "UCB":
                        ucb = UCB(bandit, c, T, num_sims)
                        mean_regrets, _ = ucb.run()
                    cum_regrets.append(np.sum(mean_regrets))
                results_dict[(n,k,c)] = np.mean(cum_regrets)
                csv_writer.writerow([str(n), str(k), str(c), results_dict[(n,k,c)]])
   
    print("Finished!") 

if __name__ == "__main__":
    main()


    
