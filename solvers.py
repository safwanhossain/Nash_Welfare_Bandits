import cvxpy as cp
import torch.optim as optim
import torch
from utils import *
import tqdm as tqdm

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

def solve_cvx_ucb(mu_matrix, num_samples, k, n, t, c=0.5):
    """ Note you can only use this to solve the original log-concace objective (uniform exploration
    and epsion greedy). This can't be used to solve UCB
    """
    def get_nsw_log(p):
        nsw = 1
        total = 0
        for i in range(n):
            nsw *= cp.matmul(p, mu_matrix[i,:])
        radius = 0.5*n*cp.sqrt(cp.log(t))*cp.matmul(p, 1/cp.sqrt(num_samples ))
        total=nsw+radius
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
        total = 1
        for i in range(n):
            total *= torch.matmul(_p, _mu_matrix[i,:])
            #out = torch.matmul(_p, _mu_matrix[i,:])
            #if out == 0:
            #    out = out + 0.001
        total = torch.log(total)
        return -1*total

    p = torch.rand(k, requires_grad=True)
    mu_matrix = torch.from_numpy(mu_matrix).float()
    prev = 2*torch.ones(k)
    i, eps = 0, 1e-8

    while torch.sum(torch.pow(prev-p,2)) > eps and i < 2000:
        optimizer = optim.SGD([p], lr=0.005)
        prev = p.clone().detach()
        out_val = get_nsw_prod(p, mu_matrix)
        optimizer.zero_grad()
        out_val.backward()
        optimizer.step()

        out_p = project_fast(p.detach().numpy(), k)
        if out_p is None:
            print("Error")
            print("prev: ", prev)
            print("Outval: ", out_val)
            print("P: ", p)
            print("i: ", i)
        if out_p is not None:
            p = torch.from_numpy(out_p).float()
        p.requires_grad = True
        i += 1

    print("Finished at: ", i, "with p:", p)
    return p.detach().numpy()

def solve_torch_ucb(mu_matrix, num_samples, k, n, t, c=0.5):

    """ This is using the torch optimizers to solve the problem using SGD. This specific function
    is largely for testing, as the original problem can be solved using cvx. However, if this tests
    well, we can possibly use it for UCB
    """
    def get_nsw_prod(_p, _mu_matrix,_num_samples,_n,_t):
        nsw = torch.tensor([1]).float()
        radius = torch.tensor([0]).float()
        total = torch.tensor([0]).float()
        for i in range(n):
            nsw*=torch.matmul(_p, _mu_matrix[i,:])
        radius = 0.5*_n*torch.sqrt(torch.log(torch.tensor(_t).float()))*torch.matmul(_p, 1/torch.sqrt(_num_samples ))
        total=torch.log(nsw+radius)
        return -1*total

    p = torch.rand(k, requires_grad=True)
    mu_matrix = torch.from_numpy(mu_matrix).float()
    num_samples=torch.from_numpy(num_samples).float()
    prev = 2*torch.ones(k)
    i, eps = 0, 1e-8

    while torch.sum(torch.pow(prev-p,2)) > eps and i < 3000:
        prev = p.clone().detach()
        optimizer = optim.SGD([p], lr=0.001)
        out_val = get_nsw_prod(p, mu_matrix,num_samples,n,t)
        #out_val = get_nsw_prod(p, mu_matrix)
        optimizer.zero_grad()
        out_val.backward()
        optimizer.step()
        with torch.no_grad():
            out_p = torch.from_numpy(project_fast(p.detach().numpy(), k)).float()
            if out_p is not None:
                p = out_p
        p.requires_grad = True
        i += 1

    return p.detach().numpy()

def project(x, k):
    y = cp.Variable(k)
    objective = cp.Minimize(cp.pnorm(x-y, 2))
    constraints = [cp.sum(y) == 1, y >= 0]
    problem = cp.Problem(objective, constraints)
    try:
        result = problem.solve(verbose=False)
    except:
        print(x)
    return normalize_p(y.value)

def project_fast(p, k):
    p_sorted = np.sort(p)[::-1]
    satisfied_j = []
    for j, p_s in enumerate(p_sorted):
        curr_sum = 0
        for r in range(j+1):
            curr_sum += p_sorted[r]
        result = p_s - (1/(j+1))*(curr_sum - 1)
        if result > 0:
            satisfied_j.append(j+1)
    try:
        rho = max(satisfied_j)
    except:
        print("EXCEPTION")
        return None

    curr_sum = 0
    for i in range(rho):
        curr_sum += p_sorted[i]
    theta = (1/(rho))*(curr_sum - 1)
    out_p = np.ones(len(p))
    for i in range(len(p)):
        out_p[i] = max(p[i] - theta, 0)
    return out_p

if __name__ == "__main__":
    #simple_solver()

    p = np.array([0,0.9])
    out1 = project(p, 2)
    out2 = project_fast(p, 2)
    print("Out cvx: ", out1)
    print("Out fast: ", out2)
    print("\n")

    p = np.array([0.8,0])
    out1 = project(p, 2)
    out2 = project_fast(p, 2)
    print("Out cvx: ", out1)
    print("Out fast: ", out2)
    print("\n")

    p = np.array([0,0])
    out1 = project(p, 2)
    out2 = project_fast(p, 2)
    print("Out cvx: ", out1)
    print("Out fast: ", out2)
    print("\n")

    p = np.array([0.45812088, 0.8020209,  0.551274,   0.944007,   0.40380913])
    out1 = project(p, 5)
    out2 = project_fast(p, 5)
    print("Out cvx: ", out1)
    print("Out fast: ", out2)
    print("\n")
