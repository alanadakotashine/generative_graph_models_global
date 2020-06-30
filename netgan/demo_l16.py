
# coding: utf-8

# In[1]:

import sys 

#sys.path.append('../')
#sys.path.insert(0, 'netgan/')

#for p in sys.path:
#    print(p)



from netgan_mod import *
import tensorflow as tf
import utils
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import time
from matplotlib.colors import Normalize

import random



class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def walk_generator(walker,walk_type):
    if walk_type == 'walk':
        return walker.walk
    else:
        return walker.combo_walk



#get_ipython().magic(u'matplotlib inline')


# #### Load the data

# In[2]:

if __name__ == "__main__":
    random.seed()
    print('entered program')
    netgan_seed = random.randint(1,1000000000)
    seed = random.randint(1,1000000000)
    num_disc_iters = sys.argv[1]
    walk_type = sys.argv[2]
    wstate = sys.argv[3]
    path = sys.argv[4]
    down_projection_size = sys.argv[5]
    num_units_gen = sys.argv[6]
    num_units_disc = sys.argv[7]
    model = sys.argv[8]
    name = sys.argv[9]
    continue_training = False
    start_iter = str(0)
    if len(sys.argv) > 10:
        netgan_seed = int(sys.argv[10])
        seed = int(sys.argv[11])
        start_iter = sys.argv[12]
        continue_training = True
    netgan_seed = str(netgan_seed)
    params = [name,netgan_seed, seed, num_disc_iters, walk_type, wstate, path, down_projection_size, num_units_gen, num_units_disc, model, start_iter]
    with open(path+'/netgan_params.txt', 'w') as f:
        for item in params:
            f.write("%s\n" % item)
    f.close()

    num_disc_iters = int(num_disc_iters)
    wstate = int(wstate)
    netgan_seed = int(netgan_seed)
    seed = int(seed)
    start_iter = int(start_iter)
    down_projection_size = int(down_projection_size)
    num_units_gen = int(num_units_gen)
    num_units_disc = int(num_units_disc)
    
    A = np.loadtxt('../data/'+name+'.txt')

    G = nx.from_numpy_matrix(A)

    

    #Run algorithm on largest connected component
    _A_obs = sp.csr_matrix(A)
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc,:][:,lcc]
    _N = _A_obs.shape[0]

    num_edges = _A_obs.sum()


    val_share = .1
    test_share = 0.05


    # #### Separate the edges into train, test, validation

    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(_A_obs, val_share, test_share, seed, undirected=True, connected=True, asserts=True, set_ops=False)

    assert(train_ones.shape[0] + val_ones.shape[0] + test_ones.shape[0] == num_edges)


    
    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()
    np.savetxt(path+'/'+name+'_largest.txt',A)
    np.savetxt(path+'/'+name+'_val_edges.txt',val_ones)
    np.savetxt(path+'/'+name+'_val_non_edges.txt',val_zeros)
    np.savetxt(path+'/'+name+'_test_edges.txt',test_ones)
    np.savetxt(path+'/'+name+'_test_non_edges.txt',test_zeros)
    np.savetxt(path+'/'+name+'_training.txt',train_graph.todense())


    #length of the walk
    rw_len, data_size = 16, 16
    #batch_size is how many walks are fed into the discriminator
    #we normalize using the length of the walk to make sure the number 
    #of edges seen by the discriminator is constant across walk lengths
    num_edges_in_samples = 2700
    batch_size = num_edges_in_samples/(rw_len-1)



    #Construct walker
    walker = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)
    walk_gen = walk_generator(walker,walk_type)
    #Test to make sure there are num_edges_in_samples number of edges in samples
    #generated from call to walk_generator
    smpls = next(walk_gen())
    num_edges_per_walk = smpls.shape[1]-1
    num_walks = smpls.shape[0]
    assert(num_edges_in_samples == num_edges_per_walk * num_walks)
    

    netgan = NetGAN(_N, data_size, walk_gen, gpu_id=0, use_gumbel=True, disc_iters=num_disc_iters,
                    W_down_discriminator_size=down_projection_size, W_down_generator_size=down_projection_size,
                    l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                    generator_layers=[num_units_gen], discriminator_layers=[num_units_disc], temp_start=5, 
                    learning_rate=0.0003, batch_size=batch_size, wasserstein_penalty=10, seed=netgan_seed, wstate = wstate)


    # Define the stopping criterion
    stopping_criterion = None


    if stopping_criterion == "val": # use val criterion for early stopping
        #flag for val
        stopping = -1
    elif stopping_criterion == "spec+val":
        stopping = -2
    elif stopping_criterion == "eo":  #use eo criterion for early stopping
        stopping = .95# set the target edge overlap here
    else:
        stopping = None #If stopping is None, train for max_new_iterations 


    eval_every, plot_every = 500, 500
    max_new_iterations = 15001

    # In[ ]:
    if continue_training:
        log_dict = netgan.train(A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros,eval_every=eval_every,plot_every=plot_every,max_patience=5,start_iter=start_iter,max_iters=start_iter+max_new_iterations,path=path,continue_training=True,model_name=model,save_directory = path, stopping = stopping)
    else:
        log_dict = netgan.train(A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros,
                                eval_every=eval_every, plot_every=plot_every, max_patience=5, max_iters=max_new_iterations,path=path, model_name=model,save_directory=path, stopping = stopping)

