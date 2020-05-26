# Nash Welfare for Multi-Armed Bandits

This repository contains code for Fairness in Multi-Armed Bandits. We measure regret in terms of Nash Welfare and consider three algorithms to minimize regret. For each, we learn we provide
an upper bound to the regret in the paper and in the experiments, we learn a scalar parameter (using grid search) incorporating the constants. In our experiments, we vary over both n, the number
of agents and k, the number of arms over multiple instances. Each data point (n,k,instance) is the average of 50 simulations, and we run until time horizon T=4000. The MAB problem is encapsulated
in the NSW\_bandit class - nsw\_bandit.py 

* Uniform Explolation: The code is contained in uniform.py. We use c=0.05 (mention what c here means for all algos)
* Epsilon Greedy: The code in contained in eps\_greedy.py. We use c=0.05 on all instances
* UCB: The code in contained in UCB.py. We use c=0.5.


