import cvxpy as cp
import torch.optim as optim 
import torch
from utils import *
def solve_cvx(mu_matrix, k, n):
    """ Note you can only use this to solve the original log-concace objective (uniform exploration
    and epsion greedy). This can't be used to solve UCB
    """
    def get_nsw_log(p):
        total = 0
        for i in range(n):
            total += cp.log(cp.matmul(p, mu_matrix[i,:]))
        return total
    
    p = cp.Variable(k)
    objective = cp.Maximize(get_nsw_log(p))
    constraints = [cp.sum(p) == 1, p >= 0]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return normalize_p(p.value)

def solve_torch(mu_matrix, k, n):
    """ This is using the torch optimizers to solve the problem using SGD. This specific function
    is largely for testing, as the original problem can be solved using cvx. However, if this tests
    well, we can possibly use it for UCB
    """
    def get_nsw_prod(_p, _mu_matrix):
        total = 1.0
        for i in range(n):
            total *= torch.matmul(_p, _mu_matrix[i,:])
        return -1*total

    p = torch.rand(k, requires_grad=True)
    mu_matrix = torch.from_numpy(mu_matrix).float()

    optimizer = optim.Adam([p], lr=0.01)
    prev, diff, eps = p, 1, 0.001
    while diff > eps:
        prev = p.clone().detach()
        out_val = get_nsw_prod(p, mu_matrix)
        optimizer.zero_grad()
        out_val.backward()
        optimizer.step()
        
        with torch.no_grad():
            for i in range(len(p)):
                p[i] = p[i].clamp_(0,1)
            p = p/torch.sum(p)
            p.requires_grad = True
        diff = torch.sum(torch.pow(p-prev, 2))
    return p.clone().detach().numpy()








