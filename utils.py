import numpy as np
from six.moves import cPickle as pickle

FILENAME = "./data/mu_instance.pkl"
MAX_N = 10
MAX_K = 15
MAX_INSTANCES = 100

def normalize_p(p):
    p = np.clip(p, 0, 1)
    p = p / np.sum(p)
    return p

def create_and_save_instance_data():
    def save_dict(all_dict):
        with open(FILENAME, 'wb') as f:
            pickle.dump(all_dict, f)
   
    all_instance_dict = {}
    for n in range(1,MAX_N+1):
        for k in range(1, MAX_K+1):
            mu_matrix = np.random.uniform(size=(MAX_INSTANCES, n, k))
            key_name = "n_"+str(n)+"k_"+str(k)
            all_instance_dict[key_name] = mu_matrix
    save_dict(all_instance_dict)

def load_all_instance_nk(n,k):
    with open(FILENAME, 'rb') as f:
        all_dict = pickle.load(f)
    key = "n_"+str(n)+"k_"+str(k)
    return all_dict[key]

def load_i_instance_nk(n,k,i):
    assert(0 <= i < MAX_INSTANCES) 
    all_instance = load_all_instance_nk(n,k)
    return all_instance[i]


